"""Transcribe a video with embedded Whisper (default) or ElevenLabs Scribe.

Extracts mono 16kHz audio via ffmpeg, runs Whisper with word-level timestamps
and VAD filtering, and writes the full response to <edit_dir>/transcripts/<video_stem>.json.

Supports English (en), Hindi (hi), Marathi (mr), or auto-detection.

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --language hi
    python helpers/transcribe.py <video_path> --language mr
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --model small
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small")
SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"

_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_MODEL_LOCK = threading.Lock()


def load_api_key(required: bool = False) -> str:
    """Load ELEVENLABS_API_KEY from .env or environment if present."""
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "ELEVENLABS_API_KEY":
                    return v.strip().strip('"').strip("'")
    v = os.environ.get("ELEVENLABS_API_KEY", "")
    if not v and required:
        sys.exit("ELEVENLABS_API_KEY not found in .env or environment (required for --engine scribe)")
    return v


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def resolve_device_and_compute(
    device: str | None = None,
    compute_type: str | None = None,
) -> tuple[str, str]:
    """Resolve compute device (cpu/cuda) and quantization type."""
    if device is None or device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if compute_type is None or compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def get_whisper_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
):
    """Load or return cached WhisperModel instance."""
    key = (model_name, device, compute_type)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            from faster_whisper import WhisperModel
            _MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        return _MODEL_CACHE[key]


def call_whisper(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    vad_filter: bool = True,
) -> dict:
    """Transcribe audio using faster-whisper with word timestamps.

    Supports 'en', 'hi', 'mr', or auto-detection if language is None.
    Produces words list matching the schema expected by pack_transcripts.py
    and render.py.
    """
    dev, comp = resolve_device_and_compute(device, compute_type)
    model = get_whisper_model(model_name, device=dev, compute_type=comp)

    # Normalize language: e.g. 'hindi' -> 'hi', 'marathi' -> 'mr', 'english' -> 'en'
    lang_map = {
        "hindi": "hi",
        "marathi": "mr",
        "english": "en",
    }
    norm_lang = lang_map.get(language.lower(), language) if language else None

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        language=norm_lang,
        vad_filter=vad_filter,
    )

    words: list[dict] = []
    full_text_parts: list[str] = []
    prev_end: float | None = None

    for seg in segments:
        seg_text = (seg.text or "").strip()
        if seg_text:
            full_text_parts.append(seg_text)

        seg_words = seg.words or []
        for w in seg_words:
            w_text = (w.word or "").strip()
            if not w_text:
                continue

            w_start = round(float(w.start), 3)
            w_end = round(float(w.end), 3)

            # Insert spacing token if gap between words >= 0.05s
            if prev_end is not None and w_start > prev_end + 0.01:
                words.append({
                    "type": "spacing",
                    "text": " ",
                    "start": round(prev_end, 3),
                    "end": w_start,
                })

            words.append({
                "type": "word",
                "text": w_text,
                "start": w_start,
                "end": w_end,
                "speaker_id": "S0",
                "probability": round(float(w.probability), 3),
            })
            prev_end = w_end

    full_text = " ".join(full_text_parts)
    detected_lang = getattr(info, "language", norm_lang or "en")
    lang_prob = round(float(getattr(info, "language_probability", 1.0)), 3)
    duration = round(float(getattr(info, "duration", prev_end or 0.0)), 2)

    return {
        "text": full_text,
        "language_code": detected_lang,
        "language_probability": lang_prob,
        "duration": duration,
        "engine": "whisper",
        "model": model_name,
        "words": words,
    }


def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    """Optional fallback to ElevenLabs Scribe."""
    import requests

    data: dict[str, str] = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    payload["engine"] = "scribe"
    return payload


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands."""
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str = "",
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
    model: str = DEFAULT_MODEL,
    engine: str = "whisper",
    device: str | None = None,
    compute_type: str | None = None,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    n_tracks = count_audio_tracks(video)
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)

        peak = peak_dbfs(audio)
        if peak < -60.0:
            raise RuntimeError(
                f"track {audio_track + 1} of {video.name} is silent "
                f"(peak {peak:.1f} dBFS) - skipping transcription. "
                + (f"The file has {n_tracks} audio tracks; try --audio-track "
                   + " or ".join(str(i) for i in range(n_tracks) if i != audio_track) + "."
                   if n_tracks > 1 else "Check the source audio.")
            )

        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            engine_desc = f"Whisper ({model})" if engine == "whisper" else "ElevenLabs Scribe"
            lang_desc = f" [{language}]" if language else " [auto-detect]"
            print(f"  transcribing {video.stem}.wav ({size_mb:.1f} MB) via {engine_desc}{lang_desc}", flush=True)

        if engine == "scribe":
            scribe_key = api_key or load_api_key(required=True)
            payload = call_scribe(audio, scribe_key, language, num_speakers)
        else:
            payload = call_whisper(
                audio_path=audio,
                model_name=model,
                language=language,
                device=device,
                compute_type=compute_type,
            )

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        lang_detected = payload.get("language_code", "unknown")
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s (lang: {lang_detected})")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with embedded Whisper (or Scribe)")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language code ('en', 'hi', 'mr'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Whisper model (default: {DEFAULT_MODEL}, choices: tiny, base, small, medium, large-v3, large-v3-turbo)",
    )
    ap.add_argument(
        "--engine",
        type=str,
        choices=["whisper", "scribe"],
        default="whisper",
        help="Transcription engine (default: whisper)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('cpu', 'cuda', 'auto')",
    )
    ap.add_argument(
        "--compute-type",
        type=str,
        default="auto",
        help="Compute type ('int8', 'float16', 'float32', 'auto')",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known (used by scribe engine).",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe. OBS writes the game on track 0 "
             "and the mic on track 1.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    api_key = load_api_key(required=(args.engine == "scribe"))

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        language=args.language,
        num_speakers=args.num_speakers,
        audio_track=args.audio_track,
        model=args.model,
        engine=args.engine,
        device=args.device,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()
