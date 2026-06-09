"""CLI for offline CFR training experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poker_bot.cfr.kuhn import (
    format_report,
    train_kuhn,
    write_json_report,
    write_markdown_report,
)

DEFAULT_OUTPUT_DIR = Path("benchmark-runs/cfr")


def build_parser():
    parser = argparse.ArgumentParser(description="Run offline CFR training.")
    parser.add_argument(
        "--game",
        default="kuhn",
        choices=("kuhn",),
        help="CFR game abstraction to train.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="CFR iterations to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for default CFR reports.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.iterations < 0:
        print("cfr-train: --iterations must be non-negative", file=sys.stderr)
        return 2

    report = train_kuhn(args.iterations)
    output_dir = Path(args.output_dir)
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else output_dir / f"{args.game}-{args.iterations}.json"
    )
    output_markdown = (
        Path(args.output_markdown)
        if args.output_markdown is not None
        else output_dir / f"{args.game}-{args.iterations}.md"
    )
    write_json_report(report, output_json)
    write_markdown_report(report, output_markdown)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
