# YouTubeSummary

A hobby CLI tool to fetch YouTube transcripts and summarize them with Ollama & LLMs.

**⚠️ Disclaimer:** This is a personal side project. No guarantees, no support—use at your own risk!

YouTubeSummary is a command-line tool that extracts the transcript from a YouTube video (or uses a local transcript file), summarizes it using an Ollama model, and (optionally) displays performance metrics such as tokens per second. It also provides options to list available Ollama models and to check whether Ollama is currently running.

## Features

- **Transcript Extraction:**  
  Retrieve the transcript by directly fetching and parsing YouTube’s caption XML via HTTP requests. If no URL is provided, a local transcript file is used.

- **Summarization:**  
  Summarize the transcript using an Ollama model (default: `gemma3:27b`). The tool builds a prompt asking the model for a concise summary.

- **Performance Metrics:**  
  When enabled, display metrics such as total duration, load duration, prompt evaluation tokens/duration, response tokens/duration, and tokens per second.

- **Model Management:**  
  Easily list all available Ollama models in a nicely formatted table and check the current status of the Ollama server.

  - **Progress Indicators:** Show real-time progress spinners for video download, transcript extraction, and summarization using Rich.
  - **URL & Video ID Validation:** Pre-validate YouTube URLs and ensure the video ID matches YouTube's 11-character format.
  - **Enhanced Error Handling:** Catch network timeouts, parsing errors, and file I/O issues, providing clear, user-friendly messages.
  - **Markdown Output:** Optionally save the summary and transcript as a Markdown file with clickable timestamps via `-M/--markdown`.

## Command-Line Options

- **`-u, --url`**  
  Specify a YouTube video URL. The tool will extract the video ID and fetch the transcript. If not provided, the transcript is read from a local file.

- **`-m, --model`**  
  Specify the name of the Ollama model to use (default is `gemma3:27b`). For example:  
  ```bash
  -m "llama2-uncensored"

- **`-t, --temperature`**  
  Specify the temperature value to be used when the LLM performs the analysis and generates the summary output.

- **`-v, --verbose`**  
  Enable verbose mode to display performance metrics.

- **`-x, --extract`**  
    Specify the name of the transcript file the extract of the transcript from the YouTube video will be saved to.
    
- **`-M, --markdown`**  
    Save summary and transcript as Markdown with embedded clickable timestamps (outputs to `transcripts/VIDEO_ID.md`).

- **`transcript_file`**  
  The path to a transcript file to summarize. If not provided, the tool will attempt to extract the transcript from a YouTube video.

## Standalone Command-Line Options  

- **`-l --list`**  
  List all available Ollama models.

- **`-s --status`**  
    Check the current status of the Ollama server.

## Configuration File  
To personalize the tool, you can create a configuration file named `config.json` in the same directory as the script. The file should contain the following fields:  
```json
{
  "default_model": "gemma3:27b",
  "default_prompt": "Please provide a concise summary of the following transcript: {transcript} Summary: Include a bullet point for each item of interest and a key takeaway at the end of the summary.",
  "default_temperature": 0.75
}
```
For the `default_prompt` field, you can use the `{transcript}` placeholder to include the transcript in the prompt.


## License

This project is licensed under the [Apache License 2.0](LICENSE).
