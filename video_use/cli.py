"""Unified CLI for video-use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HELPERS_DIR = Path(__file__).resolve().parent.parent / "helpers"
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="video-use",
        description="video-use: Automated AI video editor with embedded Whisper",
    )
    sub = ap.add_subparsers(dest="command", help="Editor commands")

    # transcribe
    tr_parser = sub.add_parser("transcribe", help="Transcribe video(s) using embedded Whisper")
    tr_parser.add_argument("target", type=Path, help="Video file or directory of videos")
    tr_parser.add_argument("--language", type=str, default=None, help="Language code ('en', 'hi', 'mr')")
    tr_parser.add_argument("--model", type=str, default="small", help="Whisper model (tiny, base, small, medium, large-v3-turbo)")
    tr_parser.add_argument("--edit-dir", type=Path, default=None, help="Edit output directory")
    tr_parser.add_argument("--workers", type=int, default=2, help="Parallel workers for batch transcription")

    # pack
    pack_parser = sub.add_parser("pack", help="Pack transcripts into takes_packed.md")
    pack_parser.add_argument("--edit-dir", type=Path, default=Path("edit"), help="Edit directory")
    pack_parser.add_argument("--silence", type=float, default=0.5, help="Silence split threshold (seconds)")

    # view
    view_parser = sub.add_parser("view", help="Render timeline filmstrip + waveform composite")
    view_parser.add_argument("video", type=Path, help="Video path")
    view_parser.add_argument("start", type=float, help="Start time in seconds")
    view_parser.add_argument("end", type=float, help="End time in seconds")
    view_parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    view_parser.add_argument("--n-frames", type=int, default=10, help="Number of frames in filmstrip")

    # render
    rn_parser = sub.add_parser("render", help="Render video from EDL")
    rn_parser.add_argument("edl", type=Path, help="Path to edl.json")
    rn_parser.add_argument("-o", "--output", type=Path, required=True, help="Output mp4 path")
    rn_parser.add_argument("--preview", action="store_true", help="Fast 720p preview render")
    rn_parser.add_argument("--build-subtitles", action="store_true", help="Generate master.srt before render")
    rn_parser.add_argument("--no-subtitles", action="store_true", help="Skip burning subtitles")

    # grade
    gr_parser = sub.add_parser("grade", help="Apply color grade filter")
    gr_parser.add_argument("input", type=Path, help="Input video")
    gr_parser.add_argument("-o", "--output", type=Path, required=True, help="Output video")
    gr_parser.add_argument("--preset", type=str, default=None, help="Grade preset name")
    gr_parser.add_argument("--filter", type=str, default=None, help="Custom ffmpeg vf filter")

    args = ap.parse_args()

    if not args.command:
        ap.print_help()
        sys.exit(0)

    if args.command == "transcribe":
        target = args.target.resolve()
        if target.is_dir():
            import transcribe_batch
            sys.argv = [
                "transcribe_batch.py",
                str(target),
                *(["--language", args.language] if args.language else []),
                "--model", args.model,
                *(["--edit-dir", str(args.edit_dir)] if args.edit_dir else []),
                "--workers", str(args.workers),
            ]
            transcribe_batch.main()
        else:
            import transcribe
            sys.argv = [
                "transcribe.py",
                str(target),
                *(["--language", args.language] if args.language else []),
                "--model", args.model,
                *(["--edit-dir", str(args.edit_dir)] if args.edit_dir else []),
            ]
            transcribe.main()

    elif args.command == "pack":
        import pack_transcripts
        sys.argv = ["pack_transcripts.py", "--edit-dir", str(args.edit_dir), "--silence", str(args.silence)]
        pack_transcripts.main()

    elif args.command == "view":
        import timeline_view
        sys.argv = [
            "timeline_view.py",
            str(args.video),
            str(args.start),
            str(args.end),
            *(["-o", str(args.output)] if args.output else []),
            "--n-frames", str(args.n_frames),
        ]
        timeline_view.main()

    elif args.command == "render":
        import render
        sys.argv = [
            "render.py",
            str(args.edl),
            "-o", str(args.output),
            *(["--preview"] if args.preview else []),
            *(["--build-subtitles"] if args.build_subtitles else []),
            *(["--no-subtitles"] if args.no_subtitles else []),
        ]
        render.main()

    elif args.command == "grade":
        import grade
        sys.argv = [
            "grade.py",
            str(args.input),
            "-o", str(args.output),
            *(["--preset", args.preset] if args.preset else []),
            *(["--filter", args.filter] if args.filter else []),
        ]
        grade.main()


if __name__ == "__main__":
    main()
