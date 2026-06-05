"""Packaged entry point for the terminal simulator."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulator import main as simulator_main  # noqa: E402


def main(argv=None):
    simulator_main(argv)
