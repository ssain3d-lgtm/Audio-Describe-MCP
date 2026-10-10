"""Command line: audio-describe describe FILE | audio-describe compare A B [B ...] [--semantic]"""
from __future__ import annotations

import argparse
import sys

from .core import Describer, format_compare, format_describe
from .signature import AudioError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="audio-describe", description="Describe sounds as text and a spectrogram image.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("describe", help="ID, numeric signature and spectrogram PNG of one audio file")
    p.add_argument("path")
    p.add_argument("--start", type=float, default=0.0, help="start of the part to analyse, seconds")
    p.add_argument("--end", type=float, default=None, help="end of the part, seconds")
    p.add_argument("--no-spectrogram", action="store_true")
    p = sub.add_parser("compare", help="how two sounds differ (or which candidates are closest to the first)")
    p.add_argument("a")
    p.add_argument("b", nargs="+")
    p.add_argument("--semantic", action="store_true", help="also compare with the embedding model (needs the [semantic] extra)")
    args = ap.parse_args(argv)
    d = Describer()
    try:
        if args.cmd == "describe":
            print(format_describe(d.describe(args.path, args.start, args.end, not args.no_spectrogram)))
        else:
            print(format_compare(d.compare(args.a, args.b if len(args.b) > 1 else args.b[0], args.semantic)))
    except (AudioError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
