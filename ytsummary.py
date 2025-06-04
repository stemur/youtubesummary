import argparse
import json
import re
import os
import ollama
import requests
import requests.exceptions
import xml.etree.ElementTree as ET
import time

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
    try:
        # First pattern: standard format
        pattern = r"ytInitialPlayerResponse\s*=\s*({.*?});"
        match = re.search(pattern, html)
        
        if not match:
            # Alternative pattern: sometimes it's in a different format
            pattern = r'{"playerConfig":.*?"videoDetails":({.*?}),"videoQuality"'
            match = re.search(pattern, html)
            
        if not match:
            raise Exception("No video details found in the video page.")

        video_details_json = match.group(1)
        
        try:
            video_details = json.loads(video_details_json)
            
            # If we got the full player response, navigate to videoDetails
            if "videoDetails" in video_details:
                details = video_details.get("videoDetails", {})
            else:
                # We already have videoDetails directly
                details = video_details
                
            title = details.get("title", "YouTube Video - Title Not Found")
            channel = details.get("author", "Unknown Channel")
            duration = details.get("lengthSeconds", 0)

            print(f"Channel: {channel} \nTitle: {title} \nDuration: {format_duration(int(duration))}\n")
            return details
            
        except json.JSONDecodeError as e:
            # Try to clean up the JSON string if it's malformed
            video_details_json = video_details_json.replace('\n', '').replace('\r', '')
            try:
                video_details = json.loads(video_details_json)
                details = video_details.get("videoDetails", {})
                
                title = details.get("title", "YouTube Video - Title Not Found")
                channel = details.get("author", "Unknown Channel")
                duration = details.get("lengthSeconds", 0)

                print(f"Channel: {channel} \nTitle: {title} \nDuration: {format_duration(int(duration))}\n")
                return details
                
            except json.JSONDecodeError:
                raise Exception(f"Failed to parse video details JSON: {e}")

    except Exception as e:
        print(f"Error in get_video_info: {str(e)}")
        print("Attempting to continue anyway...")
        # Return a minimal dict to allow the process to continue
        return {
            "title": "Unknown Title",
            "author": "Unknown Channel",
            "lengthSeconds": "0"
        }

def get_youtube_transcript(html: str, lang: str = "en", max_retries: int = 3, delay: float = 1.0) -> list:
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

        # Add headers and retry mechanism for transcript fetch
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Retry loop for transcript fetch
        for attempt in range(max_retries):
            try:
                transcript_response = requests.get(
                    track_url, 
                    headers=headers, 
                    timeout=15
                )
                
                if transcript_response.status_code == 200:
                    xml_data = transcript_response.text
                    
                    # Validate XML content
                    if not xml_data or '<transcript>' not in xml_data:
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))
                            continue
                        raise Exception("Invalid transcript XML received")
                    
                    # Parse the XML
                    root = ET.fromstring(xml_data)
                    transcript = []
                    
                    for child in root.findall('text'):
                        text = child.text or ""
                        start = child.attrib.get("start")
                        dur = child.attrib.get("dur")
                        if text.strip():  # Only add non-empty entries
                            transcript.append({
                                "text": unescape(text).strip(),
                                "start": start,
                                "duration": dur
                            })
                    
                    if not transcript:
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))
                            continue
                        raise Exception("Empty transcript received")
                    
                    print(f"Successfully retrieved transcript with {len(transcript)} entries")
                    return transcript
                    
                else:
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    raise Exception(f"Failed to fetch transcript: HTTP {transcript_response.status_code}")
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                raise Exception("Transcript request timed out after all retries")
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                raise Exception(f"Network error during transcript fetch: {e}")
                
            except ET.ParseError as e:
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                raise Exception(f"Transcript XML parse error: {e}")

        return None

    except Exception as e:
        print(f"Could not retrieve transcript: {e}")
        return None

def get_youtube_chapters(html: str) -> list:
    try:
        # Look for chapter data in the video page HTML
        pattern = r'"chapterRenderer":({.*?})(?=,(?:"chapterRenderer"|[}\]]))'
        matches = re.finditer(pattern, html)
        
        chapters = []
        for match in matches:
            chapter_data = json.loads(match.group(1))
            
            # Extract time and title
            time_str = chapter_data.get("timeRangeStartMillis", 0)
            title = chapter_data.get("title", {}).get("simpleText", "")
            
            if time_str and title:
                # Convert milliseconds to seconds
                time_seconds = int(time_str) / 1000
                chapters.append({
                    "time": time_seconds,
                    "title": title,
                    "timestamp": format_duration(time_seconds)
                })
        
        # Sort chapters by time
        chapters.sort(key=lambda x: x["time"])
        print(f'Chapters available: {len(chapters)}')
        return chapters if chapters else None
        
    except Exception as e:
        print(f"Error extracting chapters: {e}")
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

def setup_argument_parser(default_model: str, default_temperature: float) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a transcript using a specified Ollama model. "
                    "Optionally, supply a YouTube URL to automatically extract the transcript."
    )

    # Standalone mutually exclusive options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--list", action="store_true", help="List all available Ollama models and exit")
    group.add_argument("-s", "--status", action="store_true", help="Show if Ollama is currently running and exit")

    parser.add_argument("-v", "--verbose", action="store_true", help="Output performance metrics along with the summary")
    parser.add_argument("-M", "--markdown", action="store_true",
                        help="Save summary and transcript as Markdown with embedded timestamps")
    parser.add_argument("-m", "--model", type=str, default=default_model, help=f"Name of the model to use (default: {default_model})")
    parser.add_argument("-t", "--temperature", type=float, default=default_temperature, 
                        help=f"Temperature for summarization (default: {default_temperature})")
    parser.add_argument("-u", "--url", type=str, help="URL to a YouTube video to extract transcript")
    parser.add_argument("transcript_file", type=str, nargs="?", default="transcript.txt",
                        help="Path to the transcript file (default: transcript.txt)")
    parser.add_argument("-x", "--extract", type=str, help="Extract transcript from a YouTube video to a file and exit")

    return parser

def validate_youtube_url(url: str) -> tuple[bool, str]:
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname.lower() if parsed_url.hostname else ""
    valid_hosts = ("www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com")
    
    if hostname not in valid_hosts:
        return False, f"Invalid YouTube URL: {url}\nPlease provide a valid YouTube link."

    # Extract and validate video ID
    if hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        params = parse_qs(parsed_url.query)
        video_id = params.get("v", [None])[0]
    else:  # youtu.be
        video_id = parsed_url.path.lstrip("/")

    if not video_id or not re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
        return False, f"Invalid YouTube video ID in URL: {url}\nPlease ensure it matches the format 'https://www.youtube.com/watch?v=VIDEO_ID'."

    return True, video_id

def get_transcript_path(args) -> str:
    if args.extract:
        return os.path.join("transcripts", args.extract) if os.path.dirname(args.extract) == "" else args.extract
    elif args.transcript_file:
        return os.path.join("transcripts", args.transcript_file)
    return "transcripts/transcript.txt"

def process_youtube_url(url: str, default_language: str, max_retries: int = 3, delay: float = 2.0) -> tuple[str, list, list, str]:
    console = Console()
    
    def fetch_with_retry(url: str) -> str:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': f'{default_language},en-US;q=0.9,en;q=0.8'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    content = response.text
                    if 'ytInitialPlayerResponse' in content:
                        # Add a fixed delay after successful page load
                        time.sleep(3)  # Wait 3 seconds for everything to load
                        return content
                    else:
                        console.print("[yellow]Page loaded but missing YouTube player data, retrying...[/yellow]")
                else:
                    console.print(f"[yellow]Received status code {response.status_code}, retrying...[/yellow]")
                
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                raise Exception(f"Failed to fetch YouTube page after {max_retries} attempts")
                
            except requests.exceptions.RequestException as e:
                console.print(f"[yellow]Request failed: {e}, retrying...[/yellow]")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                raise
    
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task("Downloading video page...", total=None)
        try:
            # Fetch page with retry mechanism
            html = fetch_with_retry(url)
            
            # Debug check for key content
            if '"captionTracks"' not in html:
                console.print("[yellow]Warning: Caption tracks data not found in page HTML[/yellow]")
            
            progress.update(task, description="Extracting content...", refresh=True)
            
            # Sequential content extraction with validation
            video_info = get_video_info(html)
            if not video_info:
                console.print("[yellow]Warning: Could not extract complete video information, but continuing...[/yellow]")
            
            # Add a small delay before fetching transcript
            time.sleep(1)
            
            progress.update(task, description="Fetching transcript...", refresh=True)
            transcript_text = get_youtube_transcript(html, default_language, max_retries=max_retries, delay=delay)
            if not transcript_text:
                raise Exception("Failed to retrieve transcript from YouTube")
            
            progress.update(task, description="Checking for chapters...", refresh=True)
            chapters = get_youtube_chapters(html)
            
            full_transcript = " ".join([entry["text"] for entry in transcript_text])
            if not full_transcript.strip():
                raise Exception("Empty transcript received")
                
            progress.update(task, description="Content extraction complete.", refresh=True)
            return full_transcript, transcript_text, chapters, html
            
        except Exception as e:
            console.print(f"[red]Error processing YouTube URL:[/red] {e}")
            # Print additional debug info
            if 'html' in locals():
                console.print("[yellow]Debug: Page content indicators:[/yellow]")
                console.print(f"- Contains player data: {'ytInitialPlayerResponse' in html}")
                console.print(f"- Contains caption tracks: {'captionTracks' in html}")
            exit(1)

def save_markdown_output(summary: str, transcript_text: list, chapters: list, video_id: str, md_path: str):
    try:
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write("# Summary\n\n")
            md_file.write(summary + "\n\n")
            
            # Add chapters section if available
            if chapters:
                md_file.write("## Chapters\n\n")
                for chapter in chapters:
                    time_seconds = int(chapter["time"])
                    link_url = f"https://www.youtube.com/watch?v={video_id}&t={time_seconds}s"
                    md_file.write(f"- [`{chapter['timestamp']}`]({link_url}) {chapter['title']}\n")
                md_file.write("\n")
            else:
                md_file.write("## Chapters\n\n")
                md_file.write("No chapters available.\n")
                md_file.write("\n")
            
            if transcript_text:
                md_file.write("## Transcript\n\n")
                for entry in transcript_text:
                    ts = format_duration(float(entry.get("start", 0)))
                    text = entry.get("text", "")
                    start_seconds = int(float(entry.get("start", 0)))
                    link_url = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
                    md_file.write(f"- [`{ts}`]({link_url}) {text}\n")
        print(f"Markdown output saved to '{md_path}'.")
    except IOError as e:
        print(f"Error writing Markdown file: {e}")

def main():
    # Load configuration
    config = load_config("config.json")
    default_model = config.get("default_model", "llama3.2:latest")
    default_prompt = config.get("default_prompt", "Please provide a concise summary of the following transcript:\n\n{transcript}\n\nSummary: Include some bullet points or key takeaways.")
    default_temperature = config.get("default_temperature", 0.25)
    default_language = config.get("default_language", "en")

    # Validate temperature and ensure transcripts directory exists
    console = Console()
    if not isinstance(default_temperature, (int, float)) or not (0.0 <= default_temperature <= 1.0):
        console.print(f"[yellow]Invalid default_temperature '{default_temperature}' in config; using fallback 0.25[/yellow]")
        default_temperature = 0.25

    os.makedirs("transcripts", exist_ok=True)

    # Parse and validate arguments
    parser = setup_argument_parser(default_model, default_temperature)
    args = parser.parse_args()

    if args.url and args.extract:
        parser.error("Cannot use --url and --extract together.")
    if not (0.0 <= args.temperature <= 1.0):
        parser.error("Temperature must be between 0.0 and 1.0.")

    # Handle standalone options
    if args.list:
        list_models(default_model)
        return
    if args.status:
        check_status()
        return

    # Process transcript
    transcript_path = get_transcript_path(args)
    transcript_text = None
    video_id = None

    if args.url:
        is_valid, result = validate_youtube_url(args.url)
        if not is_valid:
            print(result)
            return
        video_id = result
        full_transcript, transcript_text, chapters, _ = process_youtube_url(
            args.url, 
            default_language,
            max_retries=5,  
            delay=3.0       # Increased delay between retries
        )
    else:
        chapters = None
        try:
            with open(transcript_path, "r", encoding="utf-8") as file:
                full_transcript = file.read()
        except IOError as e:
            print(f"File error: {e}")
            return

    if args.extract:
        try:
            with open(transcript_path, "w", encoding="utf-8") as file:
                file.write(full_transcript)
            print(f"Transcript extracted from YouTube and saved to '{transcript_path}'.")
        except IOError as e:
            print(f"File error: {e}")
            return

    # Generate summary
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task(f"Summarizing with model {args.model}...", total=None)
        try:
            summary, metrics = summarize_transcript_with_metrics(
                full_transcript, 
                model=args.model, 
                prompt_template=default_prompt, 
                temperature=args.temperature
            )
            if not summary.strip():
                console.print("[yellow]Warning: Model returned an empty summary.[/yellow]")
            progress.update(task, description="Summarization complete.", refresh=True)
        except Exception as e:
            console.print(f"[red]Summarization failed:[/red] {e}")
            return

    # Output results
    print("\n")
    print(summary)
    
    if args.verbose:
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    if args.markdown:
        md_filename = f"{video_id}.md" if args.url else "summary.md"
        md_path = os.path.join("transcripts", md_filename)
        save_markdown_output(summary, transcript_text, chapters, video_id, md_path)

if __name__ == "__main__":
    main()
