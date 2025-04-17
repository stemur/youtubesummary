import argparse
import json
import re
import os
import ollama
import requests
import xml.etree.ElementTree as ET

from urllib.parse import urlparse, parse_qs
from rich.console import Console
from rich.table import Table
from html import unescape

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"

def load_config(filename="config.json") -> dict:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            config = json.load(file)
        return config
    except Exception as e:
        print(f"Warning: Could not load config file '{filename}': {e}")
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

    print(f"Channel: {channel} \nTitle: {title}, \nDuration: {format_duration(int(duration))}\n")

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

        transcript_response = requests.get(track_url)
        if transcript_response.status_code != 200:
            raise Exception("Could not fetch transcript from the caption track URL.")
            return None
        xml_data = transcript_response.text

        # Parse the XML to extract transcript text, start, and duration
        root = ET.fromstring(xml_data)
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

def summarize_transcript_with_metrics(transcript: str, model: str, prompt_template: str, temperature: float) -> (str, dict):
    prompt = prompt_template.format(transcript=transcript)    
    # Generate a response without streaming to obtain performance metrics
    result = ollama.generate(model=model, prompt=prompt, options={"temperature": temperature}, stream=False)
    
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
        response = requests.get(args.url)
        if response.status_code != 200:
            print(f"Failed to fetch YouTube video page: {html.status_code}")
            exit(1)

        html = response.text
        video_info = get_video_info(html)

        transcript_text = get_youtube_transcript(html, default_language)
        full_transcript = " ".join([entry["text"] for entry in transcript_text])
        if not full_transcript:
            print("Failed to retrieve transcript from YouTube.")
            exit(1)
    else:
        try:
            with open(transcript_path, "r", encoding="utf-8") as file:
                full_transcript = file.read()
        except FileNotFoundError:
            print(f"Transcript file '{transcript_path}' not found.")
            exit(1)

    if args.extract:
        with open(transcript_path, "w", encoding="utf-8") as file:
            file.write(full_transcript)
        print(f"Transcript extracted from YouTube and saved to '{transcript_path}'.")
        # exit(0)
    
    print(f'Summarizing the transcript using Model: {args.model}')
    summary, metrics = summarize_transcript_with_metrics(full_transcript, model=args.model, prompt_template=default_prompt, temperature=args.temperature)
    
    print("\n")
    print(summary)
    
    if args.verbose:
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()

