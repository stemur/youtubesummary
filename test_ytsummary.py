import os
import unittest
import tempfile
import json
from unittest.mock import patch

from ytsummary import (
    extract_video_id,
    load_config,
    summarize_transcript_with_metrics
)

class TestYTSummary(unittest.TestCase):
    def test_extract_video_id(self):
        # Standard URL
        url1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url1), "dQw4w9WgXcQ")
        
        # Shortened URL
        url2 = "https://youtu.be/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url2), "dQw4w9WgXcQ")
        
        # Embed URL
        url3 = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url3), "dQw4w9WgXcQ")
        
        # URL with extra parameters
        url4 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s"
        self.assertEqual(extract_video_id(url4), "dQw4w9WgXcQ")
        
        # Invalid URL should return None
        url_invalid = "https://www.example.com"
        self.assertIsNone(extract_video_id(url_invalid))

    def test_load_config(self):
        # Create a temporary config file with known content.
        config_data = {
            "default_model": "test_model",
            "default_prompt": "Test prompt: {transcript}"
        }
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            json.dump(config_data, tmp)
            tmp_path = tmp.name

        try:
            config = load_config(tmp_path)
            self.assertEqual(config.get("default_model"), "test_model")
            self.assertEqual(config.get("default_prompt"), "Test prompt: {transcript}")
        finally:
            os.remove(tmp_path)

    @patch("ytsummary.ollama.generate")
    def test_summarize_transcript_with_metrics(self, mock_generate):
        # Simulate a response from ollama.generate
        mock_response = {
            "response": "Test summary",
            "total_duration": 5000000000,      # 5 seconds in nanoseconds
            "load_duration": 100000000,        # 0.1 seconds in nanoseconds
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200000000, # 0.2 seconds in nanoseconds
            "eval_count": 50,
            "eval_duration": 3000000000        # 3 seconds in nanoseconds
        }
        mock_generate.return_value = mock_response
        
        transcript = "This is a test transcript."
        model = "test_model"
        prompt_template = "Summarize: {transcript}"
        
        summary, metrics = summarize_transcript_with_metrics(transcript, model, prompt_template)
        
        self.assertEqual(summary, "Test summary")
        self.assertAlmostEqual(metrics["Total Duration (s)"], 5.0)
        self.assertAlmostEqual(metrics["Load Duration (s)"], 0.1)
        self.assertEqual(metrics["Prompt Tokens"], 10)
        self.assertAlmostEqual(metrics["Prompt Eval Duration (s)"], 0.2)
        self.assertEqual(metrics["Response Tokens"], 50)
        self.assertAlmostEqual(metrics["Response Eval Duration (s)"], 3.0)
        # Calculate tokens per second: 50 * 1e9 / 3000000000 = ~16.67 tokens/s
        self.assertAlmostEqual(metrics["Tokens per Second"], 16.666666666666668)

if __name__ == '__main__':
    unittest.main()
    