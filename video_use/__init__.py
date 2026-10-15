"""video-use: Conversation-driven video editor with embedded Whisper."""

import sys
from pathlib import Path

# Ensure helpers can be imported
HELPERS_DIR = Path(__file__).resolve().parent.parent / "helpers"
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

from transcribe import call_whisper, transcribe_one, transcript_path, DEFAULT_MODEL
from pack_transcripts import group_into_phrases, pack_one_file, render_markdown
from timeline_view import words_in_range, find_silences
from render import build_master_srt, chunk_words

__all__ = [
    "call_whisper",
    "transcribe_one",
    "transcript_path",
    "DEFAULT_MODEL",
    "group_into_phrases",
    "pack_one_file",
    "render_markdown",
    "words_in_range",
    "find_silences",
    "build_master_srt",
    "chunk_words",
]
