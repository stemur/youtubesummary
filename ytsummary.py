import argparse
import json
import re
import os
import ollama
import requests
import requests.exceptions
import xml.etree.ElementTree as ET

from urllib.parse import urlparse, parse_qs
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from html import unescape

def format_duration(seconds):
    try:
        seconds = int(seconds)
        if seconds < 0:
            raise ValueError("Duration cannot be negative.")
        
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    except (ValueError, TypeError):
        return "Invalid duration"

def load_config(filename="config.json") -> dict:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Config file '{filename}' not found; using defaults.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing '{filename}': {e}. Using defaults.")
        return {}

def get_video_info(html: str) -> dict:
    pattern = r"ytInitialPlayerResponse\s*=\s*({.*?});"
    match = re.search(pattern, html)
    if not match:
        raise Exception("No video details found in the video page.")

    video_details_json = match.group(1)
    try:
        video_details = json.loads(video_details_json)
    except json.JSONDecodeError:
        raise Exception("Failed to parse video details JSON.")
        exit(0)

    details = video_details.get("videoDetails", {})
    title = details.get("title", "YouTube Video - Title Not Found")
    channel = details.get("author", "Unknown Channel")
    duration = details.get("lengthSeconds", 0)

    print(f"Channel: {channel} \nTitle: {title} \nDuration: {format_duration(int(duration))}\n")

def get_youtube_transcript(html: str, lang: str = "en") -> list:
    try:
        pattern = r'"captionTracks":(\[.*?\])'
        match = re.search(pattern, html)
        if not match:
            print("No caption tracks found in the video page.")
            return None
        caption_tracks_json = match.group(1)
        caption_tracks = json.loads(caption_tracks_json)
        
        # Parse the JSON to extract the transcript url
        track_url = None
        for track in caption_tracks:
            if track.get("languageCode") == lang:
                track_url = track.get("baseUrl")
                break
        if not track_url:
            raise Exception(f"No transcript available in language '{lang}'.")

        try:
            transcript_response = requests.get(track_url, timeout=10)
        except requests.exceptions.Timeout:
            print("Transcript request timed out.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Network error during transcript fetch: {e}")
            return None

        if transcript_response.status_code != 200:
            raise Exception("Could not fetch transcript from the caption track URL.")
            return None
        xml_data = transcript_response.text

        # Parse the XML to extract transcript text, start, and duration
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            print(f"Transcript XML parse error: {e}")
            return None

        transcript = []
        for child in root.findall('text'):
            text = child.text or ""
            start = child.attrib.get("start")
            dur = child.attrib.get("dur")
            transcript.append({"text": unescape(text).strip(), "start": start, "duration": dur})

        return transcript
    except Exception as e:
        print(f"Could not retrieve transcript: {e}")
        return None

def summarize_transcript_with_metrics(transcript: str, model: str, prompt_template: str, temperature: float) -> tuple[str, dict]:
    prompt = prompt_template.format(transcript=transcript)    
    # Generate a response without streaming to obtain performance metrics
    try:
        result = ollama.generate(model=model, prompt=prompt, options={"temperature": temperature}, stream=False)
    except Exception as e:
        raise RuntimeError(f"Ollama summarization error: {e}")
    
    # Extract summary text
    summary = result.get("response", "")
    
    # Extract performance metrics (nanosecond values)
    total_duration_ns    = result.get("total_duration")
    load_duration_ns     = result.get("load_duration")
    prompt_tokens        = result.get("prompt_eval_count")
    prompt_duration_ns   = result.get("prompt_eval_duration")
    response_tokens      = result.get("eval_count")
    response_duration_ns = result.get("eval_duration")
    
    # Calculate tokens per second for the response generation
    if response_tokens and response_duration_ns and response_duration_ns > 0:
        tokens_per_sec = response_tokens * 1e9 / response_duration_ns
    else:
        tokens_per_sec = None

    metrics = {
        "Total Duration (s)": total_duration_ns / 1e9 if total_duration_ns else None,
        "Load Duration (s)": load_duration_ns / 1e9 if load_duration_ns else None,
        "Prompt Tokens": prompt_tokens,
        "Prompt Eval Duration (s)": prompt_duration_ns / 1e9 if prompt_duration_ns else None,
        "Response Tokens": response_tokens,
        "Response Eval Duration (s)": response_duration_ns / 1e9 if response_duration_ns else None,
        "Tokens per Second": tokens_per_sec,
    }
    
    return summary, metrics

def list_models(default_model: str):
    try:
        models = ollama.list()
        console = Console()
        table = Table(title="Available Ollama Models")
        table.add_column("Model Name", no_wrap=True)
        table.add_column("Parameters", no_wrap=True)
        table.add_column("Size (bytes)", no_wrap=True)
        table.add_column("Default Model", no_wrap=True)
        
        for model in models.get("models", []):
            name = model.get("model", "N/A")
            size = model.get("size", 0)
            params = model.get("details", {})
            psize = params.get("parameter_size", "N/A")
            size_str = f"{size:,}" if size else "0"
            is_default = "[green]Default[/green]" if name == default_model else ""
            table.add_row(name, str(psize), size_str, is_default)
            # table.add_row(name, str(psize), size_str)
        
        if not models.get("models"):
            console.print("[yellow]No models available.[/yellow]")
        else:
            console.print(table)
    except KeyError as e:
        print(f"Error parsing model details: Missing key {e}")
    except Exception as e:
        print(f"Error listing models: {e}")

def check_status():
    # Check if Ollama is currently running by printing loaded models.
    try:
        status = ollama.ps()
        models = status.get("models")
        if models:
            print("Ollama is running with the following models loaded:")
            for m in models:
                print(f"- {m.get('name')}")
        else:
            print("Ollama is running, but No models are currently loaded.")
    except Exception as e:
        print(f"Ollama is currently offline.")

def main():
    # Load the config items from config.json
    config = load_config("config.json")
    default_model = config.get("default_model", "llama3.2:latest")
    default_prompt = config.get("default_prompt", "Please provide a concise summary of the following transcript:\n\n{transcript}\n\nSummary: Include some bullet points or key takeaways.")
    default_temperature = config.get("default_temperature", 0.25)
    default_language = config.get("default_language", "en")

    # Validate default_temperature from config
    console = Console()
    if not isinstance(default_temperature, (int, float)) or not (0.0 <= default_temperature <= 1.0):
        console.print(f"[yellow]Invalid default_temperature '{default_temperature}' in config; using fallback 0.25[/yellow]")
        default_temperature = 0.25

    # Ensure the transcripts folder exists if needed.
    os.makedirs("transcripts", exist_ok=True)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Summarize a transcript using a specified Ollama model. "
                    "Optionally, supply a YouTube URL to automatically extract the transcript."
    )

    # Standalone mutually exclusive options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--list", action="store_true", help="List all available Ollama models and exit")
    group.add_argument("-s", "--status", action="store_true", help="Show if Ollama is currently running and exit")

    parser.add_argument("-v", "--verbose", action="store_true", help="Output performance metrics along with the summary")
    parser.add_argument("-m", "--model", type=str, default=default_model, help="Name of the model to use (default: llama3.2:latest)")
    parser.add_argument("-t", "--temperature", type=float, default=default_temperature, help=f"Temperature for summarization (default: {default_temperature})")
    parser.add_argument("-u", "--url", type=str, help="URL to a YouTube video to extract transcript")
    parser.add_argument("transcript_file", type=str, nargs="?", default="transcript.txt",
                        help="Path to the transcript file (default: transcript.txt)")
    parser.add_argument("-x", "--extract", type=str, help="Extract transcript from a YouTube video to a file and exit")

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.url and args.extract:
        parser.error("Cannot use --url and --extract together.")
    # Validate temperature bounds
    if not (0.0 <= args.temperature <= 1.0):
        parser.error("Temperature must be between 0.0 and 1.0.")

    # If standalone options are used, process and exit.
    if args.list:
        list_models(default_model)
        exit(0)
    if args.status:
        check_status()
        exit(0)
    
    # Determine transcript file path.
    # If transcript_file does not include a directory, assume it's in the "transcripts" folder.
    if args.extract:
        if os.path.dirname(args.extract) == "":
            transcript_path = os.path.join("transcripts", args.extract)
        else:
            transcript_path = args.extract
    elif args.transcript_file:
        transcript_path = os.path.join("transcripts", args.transcript_file)
    else:
        transcript_path = "transcripts/transcript.txt"

    # If a YouTube URL is provided, extract the transcript from YouTube; otherwise, use the file
    if args.url:
        # Pre-validate YouTube URL
        parsed_url = urlparse(args.url)
        hostname = parsed_url.hostname.lower() if parsed_url.hostname else ""
        valid_hosts = ("www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com")
        if hostname not in valid_hosts:
            print(f"Invalid YouTube URL: {args.url}\nPlease provide a valid YouTube link.")
            exit(1)
        # Validate YouTube video ID
        if hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
            params = parse_qs(parsed_url.query)
            video_id = params.get("v", [None])[0]
        else:  # youtu.be
            video_id = parsed_url.path.lstrip("/")
        if not video_id or not re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
            print(
                f"Invalid YouTube video ID in URL: {args.url}\n"
                "Please ensure it matches the format 'https://www.youtube.com/watch?v=VIDEO_ID'."
            )
            exit(1)
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
            task = progress.add_task("Downloading video page...", total=None)
            try:
                response = requests.get(args.url, timeout=10)
            except requests.exceptions.RequestException as e:
                console = Console()
                console.print(f"[red]Network error fetching video page:[/red] {e}")
                exit(1)
            if response.status_code != 200:
                print(f"Failed to fetch YouTube video page: {response.status_code}")
                exit(1)
            html = response.text
            progress.update(task, description="Extracting transcript...", refresh=True)
            try:
                video_info = get_video_info(html)
            except Exception as e:
                console = Console()
                console.print(f"[red]Error extracting video metadata:[/red] {e}")
                exit(1)
            transcript_text = get_youtube_transcript(html, default_language)
            if not transcript_text:
                print("Failed to retrieve transcript from YouTube.")
                exit(1)
            full_transcript = " ".join([entry["text"] for entry in transcript_text])
            progress.update(task, description="Transcript extraction complete.", refresh=True)
    else:
        try:
            with open(transcript_path, "r", encoding="utf-8") as file:
                full_transcript = file.read()
        except IOError as e:
            print(f"File error: {e}")
            exit(1)

    if args.extract:
        try:
            with open(transcript_path, "w", encoding="utf-8") as file:
                file.write(full_transcript)
        except IOError as e:
            print(f"File error: {e}")
            exit(1)
        print(f"Transcript extracted from YouTube and saved to '{transcript_path}'.")
        # exit(0)
    
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task(f"Summarizing with model {args.model}...", total=None)
        try:
            summary, metrics = summarize_transcript_with_metrics(full_transcript, model=args.model, prompt_template=default_prompt, temperature=args.temperature)
        except Exception as e:
            console = Console()
            console.print(f"[red]Summarization failed:[/red] {e}")
            exit(1)
        if not summary.strip():
            console.print("[yellow]Warning: Model returned an empty summary.[/yellow]")
        progress.update(task, description="Summarization complete.", refresh=True)

    print("\n")
    print(summary)
    
    if args.verbose:
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
