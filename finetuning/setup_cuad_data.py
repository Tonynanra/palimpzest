#!/usr/bin/env python
"""Download the local CUAD dataset using the shared data implementation."""

import argparse
from pathlib import Path

try:
    from .cuad_data import DEFAULT_RAW_DATA_DIR
    from .cuad_data import setup_cuad_data as setup_cuad_data
except ImportError:
    from cuad_data import DEFAULT_RAW_DATA_DIR
    from cuad_data import setup_cuad_data as setup_cuad_data

DEFAULT_DATA_DIR = DEFAULT_RAW_DATA_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the local CUAD dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory for raw CUAD data")
    return parser.parse_args(argv)


if __name__ == "__main__":
    setup_cuad_data(parse_args().output_dir)
