import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import argparse
import ollama

from rich.console import Console
from rich.table import Table

def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.
    Handles different formats like:
      - Standard URL: https://www.youtube.com/watch?v=VIDEO_ID
      - Shortened URL: https://youtu.be/VIDEO_ID
      - Embed URL: https://www.youtube.com/embed/VIDEO_ID
    """
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

def get_youtube_transcript(video_id: str):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Could not retrieve transcript: {e}")
        return None


def summarize_transcript_with_metrics(transcript: str, model: str = "llama3.2:latest") -> (str, dict):
    prompt = (
        "Please provide a concise summary of the following transcript:\n\n"
        f"{transcript}\n\nSummary:"
    )
    
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
            # print(f"- {name} \t (Params: {psize}) \t (Size: {size} bytes)")
        console.print(table)
    except Exception as e:
        print(f"Error listing models: {e}")

def check_status():
    """
    Check if Ollama is currently running by printing loaded models.
    """
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
    parser = argparse.ArgumentParser(
        description="Summarize a transcript using a specified Ollama model. "
                    "Optionally, supply a YouTube URL to automatically extract the transcript."
    )

    # Standalone mutually exclusive options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--list", action="store_true", help="List all available Ollama models and exit")
    group.add_argument("-s", "--status", action="store_true", help="Show if Ollama is currently running and exit")

    parser.add_argument("-v", "--verbose", action="store_true", help="Output performance metrics along with the summary")
    parser.add_argument("-m", "--model", type=str, default="llama3.2:latest", help="Name of the model to use (default: llama3.2:latest)")
    parser.add_argument("-u", "--url", type=str, help="URL to a YouTube video to extract transcript")
    parser.add_argument("transcript_file", type=str, nargs="?", default="transcript.txt",
                        help="Path to the transcript file (default: transcript.txt)")

    args = parser.parse_args()

    # If standalone options are used, process and exit.
    if args.list:
        list_models()
        exit(0)
    if args.status:
        check_status()
        exit(0)

    # If a YouTube URL is provided, extract the transcript from YouTube; otherwise, use the file
    if args.url:
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
            with open(args.transcript_file, "r", encoding="utf-8") as file:
                transcript_text = file.read()
        except FileNotFoundError:
            print(f"Transcript file '{args.transcript_file}' not found.")
            exit(1)
    
    print(f'Summarizing the transcript using Model: {args.model}')
    summary, metrics = summarize_transcript_with_metrics(transcript_text, model=args.model)
    
    print("Summary:")
    print(summary)
    
    if args.verbose:
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()

