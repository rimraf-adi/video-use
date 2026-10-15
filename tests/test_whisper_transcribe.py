import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "helpers"))

import transcribe
import pack_transcripts
import render


class TestWhisperTranscribe(unittest.TestCase):
    def test_whisper_output_schema_compatibility(self):
        """Verify the Whisper output dict schema matches what pack_transcripts and render expect."""
        # Simulated Whisper Word objects
        class MockWord:
            def __init__(self, word, start, end, prob=0.95):
                self.word = word
                self.start = start
                self.end = end
                self.probability = prob

        class MockSegment:
            def __init__(self, text, words):
                self.text = text
                self.words = words

        class MockInfo:
            language = "hi"
            language_probability = 0.98
            duration = 3.5

        mock_segments = [
            MockSegment("नमस्ते जग", [
                MockWord("नमस्ते", 0.5, 1.2),
                MockWord("जग", 1.5, 2.0),
            ]),
            MockSegment("हा एक व्हिडिओ आहे", [
                MockWord("हा", 2.2, 2.4),
                MockWord("एक", 2.45, 2.7),
                MockWord("व्हिडिओ", 2.75, 3.1),
                MockWord("आहे.", 3.15, 3.4),
            ]),
        ]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (mock_segments, MockInfo())

        with patch("transcribe.get_whisper_model", return_value=mock_model):
            payload = transcribe.call_whisper(
                audio_path=Path("dummy.wav"),
                model_name="small",
                language="hi",
            )

        self.assertEqual(payload["engine"], "whisper")
        self.assertEqual(payload["language_code"], "hi")
        self.assertIn("words", payload)

        words = payload["words"]
        # Should have words and spacing
        word_texts = [w["text"] for w in words if w["type"] == "word"]
        self.assertEqual(word_texts, ["नमस्ते", "जग", "हा", "एक", "व्हिडिओ", "आहे."])

        # Verify pack_transcripts can group into phrases
        phrases = pack_transcripts.group_into_phrases(words, silence_threshold=0.5)
        self.assertTrue(len(phrases) >= 1)
        self.assertIn("नमस्ते", phrases[0]["text"])

        # Verify chunk_words in render
        chunks = render.chunk_words([w for w in words if w["type"] == "word"])
        self.assertTrue(len(chunks) >= 1)

    def test_language_normalization(self):
        """Check that language names like 'hindi', 'marathi', 'english' normalize properly."""
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "mr"
        mock_info.language_probability = 0.99
        mock_info.duration = 1.0
        mock_model.transcribe.return_value = ([], mock_info)

        with patch("transcribe.get_whisper_model", return_value=mock_model):
            payload = transcribe.call_whisper(
                audio_path=Path("dummy.wav"),
                language="marathi",
            )
            # Verify normalize called model with 'mr'
            mock_model.transcribe.assert_called_once()
            args, kwargs = mock_model.transcribe.call_args
            self.assertEqual(kwargs["language"], "mr")

    def test_transcribe_one_cached(self):
        """transcribe_one should return immediately if transcript file exists."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            edit = tmp / "edit"
            transcripts = edit / "transcripts"
            transcripts.mkdir(parents=True)
            existing_json = transcripts / "test_clip.json"
            existing_json.write_text(json.dumps({"words": [], "engine": "whisper"}))

            vid = tmp / "test_clip.mp4"
            vid.touch()

            # Should not call whisper or ffmpeg
            res = transcribe.transcribe_one(
                video=vid,
                edit_dir=edit,
                verbose=False,
            )
            self.assertEqual(res, existing_json)


if __name__ == "__main__":
    unittest.main()
