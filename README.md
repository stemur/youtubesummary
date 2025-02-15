# YouTubeSummary

YouTubeSummary is a command-line tool that extracts the transcript from a YouTube video (or uses a local transcript file), summarizes it using an Ollama model, and (optionally) displays performance metrics such as tokens per second. It also provides options to list available Ollama models and to check whether Ollama is currently running.

## Features

- **Transcript Extraction:**  
  Retrieve the transcript from a given YouTube URL using the [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api). If no URL is provided, a local transcript file is used.

- **Summarization:**  
  Summarize the transcript using an Ollama model (default: `llama2`). The tool builds a prompt asking the model for a concise summary.

- **Performance Metrics:**  
  When enabled, display metrics such as total duration, load duration, prompt evaluation tokens/duration, response tokens/duration, and tokens per second.

- **Model Management:**  
  Easily list all available Ollama models in a nicely formatted table and check the current status of the Ollama server.

## Command-Line Options

- **`-u, --url`**  
  Specify a YouTube video URL. The tool will extract the video ID and fetch the transcript. If not provided, the transcript is read from a local file.

- **`-m, --model`**  
  Specify the name of the Ollama model to use (default is `llama3.2:latest`). For example:  
  ```bash
  -m "llama2-uncensored"

- **`-v, --verbose`**  
  Enable verbose mode to display performance metrics.

- **`-x, --extract`**  
    Specify the name of the transcript file the extract of the transcript from the YouTube video will be saved to.
    
- **`transcript_file`**  
  The path to a transcript file to summarize. If not provided, the tool will attempt to extract the transcript from a YouTube video.

## Standalone Command-Line Options  

- **`-l --list`**  
  List all available Ollama models.

- **`-s --status`**  
    Check the current status of the Ollama server.

    