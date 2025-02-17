import argparse
import json
import re
import os
import ollama
import yt_dlp

from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from rich.console import Console
from rich.table import Table

def load_config(filename="config.json") -> dict:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            config = json.load(file)
        return config
    except Exception as e:
        print(f"Warning: Could not load config file '{filename}': {e}")
        return {}

def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    
    # Check for shortened URL (youtu.be)
    if 'youtu.be' in parsed_url.netloc:
        return parsed_url.path.strip('/')
    
    # Check for standard URL with query parameter ?v=
    query_params = parse_qs(parsed_url.query)
    if 'v' in query_params:
        return query_params['v'][0]
    
    # Check for embed URL format
    embed_match = re.search(r'/embed/([^?&/]+)', url)
    if embed_match:
        return embed_match.group(1)
    
    # If no format matched, return None
    return None

def get_video_info(url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info

def get_youtube_transcript(video_id: str):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Could not retrieve transcript: {e}")
        return None

def summarize_transcript_with_metrics(transcript: str, model: str, prompt_template: str) -> (str, dict):
    prompt = prompt_template.format(transcript=transcript)    
    # Generate a response without streaming to obtain performance metrics
    result = ollama.generate(model=model, prompt=prompt, stream=False)
    
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

def list_models():
    try:
        models = ollama.list()
        console = Console()
        table = Table(title = "Available Ollama Models")
        table.add_column("Model Name", no_wrap=True)
        table.add_column("Parameters", no_wrap=True)
        table.add_column("Size (bytes)", no_wrap=True)
        for model in models.get("models", []):
            name = model.get("model")
            size = model.get("size")
            params = model.get("details", [])
            psize = params.get("parameter_size")
            size_str = f"{size:,}" if size else "0"
            table.add_row(name, str(psize), str(size_str))
        console.print(table)
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
    parser.add_argument("-u", "--url", type=str, help="URL to a YouTube video to extract transcript")
    parser.add_argument("transcript_file", type=str, nargs="?", default="transcript.txt",
                        help="Path to the transcript file (default: transcript.txt)")
    parser.add_argument("-x", "--extract", type=str, help="Extract transcript from a YouTube video to a file and exit")

    args = parser.parse_args()

    # If standalone options are used, process and exit.
    if args.list:
        list_models()
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

    # Ensure the transcripts folder exists if needed.
    os.makedirs("transcripts", exist_ok=True)

    # If a YouTube URL is provided, extract the transcript from YouTube; otherwise, use the file
    if args.url:
        video_info = get_video_info(args.url)
        title = video_info.get("title", "YouTube Video - Title Not Found")
        duration = video_info.get("duration", 0)
        video_id = extract_video_id(args.url)
        if not video_id:
            print("Failed to extract video ID from the URL.")
            exit(1)
        transcript_text = get_youtube_transcript(video_id)
        if not transcript_text:
            print("Failed to retrieve transcript from YouTube.")
            exit(1)
    else:
        try:
            # with open(args.transcript_file, "r", encoding="utf-8") as file:
            with open(transcript_path, "r", encoding="utf-8") as file:
                transcript_text = file.read()
        except FileNotFoundError:
            print(f"Transcript file '{transcript_path}' not found.")
            exit(1)

    if args.extract:
        full_transcript = " ".join([entry["text"] for entry in transcript_text])
        with open(transcript_path, "w", encoding="utf-8") as file:
            file.write(full_transcript)
        print(f"Transcript extracted from YouTube and saved to '{transcript_path}'.")
        exit(0)
    
    print(f'Summarizing the transcript using Model: {args.model}')
    summary, metrics = summarize_transcript_with_metrics(transcript_text, model=args.model, prompt_template=default_prompt)
    
    print("Summary:")
    if args.url:
        print(f"Video Title: {title}")
        # Convert duration to minutes and seconds for friendly output
        minutes, seconds = divmod(duration, 60)
        print(f"Video Duration: {minutes}m {seconds}s\n")
        
    print(summary)
    
    if args.verbose:
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()

