"""Batch-transcribe every video in a directory with parallel workers using embedded Whisper.

Walks <videos_dir> for common video extensions, runs Whisper (or Scribe) on
each, writes transcripts to <videos_dir>/edit/transcripts/<name>.json.

Supports English (en), Hindi (hi), Marathi (mr), or auto-detection.

Cached per-file: any source that already has a transcript is skipped.

Usage:
    python helpers/transcribe_batch.py <videos_dir>
    python helpers/transcribe_batch.py <videos_dir> --language hi
    python helpers/transcribe_batch.py <videos_dir> --language mr
    python helpers/transcribe_batch.py <videos_dir> --language en
    python helpers/transcribe_batch.py <videos_dir> --model small
    python helpers/transcribe_batch.py <videos_dir> --workers 2
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transcribe import DEFAULT_MODEL, load_api_key, transcribe_one, transcript_path

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )
    return videos


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel batch transcription of a videos directory with Whisper")
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language code ('en', 'hi', 'mr'). Omit to auto-detect per file.",
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
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers (used by scribe engine).",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe (OBS: 0 = game, 1 = mic).",
    )
    args = ap.parse_args()

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")

    already_cached = [v for v in videos
                      if transcript_path(edit_dir, v, args.audio_track).exists()]
    pending = [v for v in videos if v not in already_cached]

    print(f"found {len(videos)} videos ({len(already_cached)} cached, {len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return

    api_key = load_api_key(required=(args.engine == "scribe"))

    print(f"transcribing {len(pending)} files with {args.workers} workers using {args.engine} ({args.model})")
    t0 = time.time()

    errors: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                transcribe_one,
                video=v,
                edit_dir=edit_dir,
                api_key=api_key,
                language=args.language,
                num_speakers=args.num_speakers,
                verbose=False,
                audio_track=args.audio_track,
                model=args.model,
                engine=args.engine,
            ): v
            for v in pending
        }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                out = fut.result()
                print(f"  + {v.stem}  →  {out.name}")
            except Exception as e:
                errors.append((v, str(e)))
                print(f"  x {v.stem}  FAILED: {e}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
