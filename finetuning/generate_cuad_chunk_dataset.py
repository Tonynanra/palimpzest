"""CLI and import adapter for the shared CUAD chunk generator."""

import argparse
from pathlib import Path

try:
    from .cuad_data import DEFAULT_CHUNK_DATA_DIR, DEFAULT_RAW_DATA_DIR
    from .cuad_data import DEFAULT_CHUNK_SIZE as DEFAULT_CHUNK_SIZE
    from .cuad_data import REPO_ROOT as REPO_ROOT
    from .cuad_data import SPLIT_TO_FILENAME as SPLIT_TO_FILENAME
    from .cuad_data import _article_exceeds_chunk_size as _article_exceeds_chunk_size
    from .cuad_data import _chunk_paragraph as _chunk_paragraph
    from .cuad_data import _chunk_title as _chunk_title
    from .cuad_data import _project_answers as _project_answers
    from .cuad_data import chunk_article as chunk_article
    from .cuad_data import chunk_articles as chunk_articles
    from .cuad_data import generate_chunked_split as generate_chunked_split
    from .cuad_data import load_split as load_split
    from .cuad_data import write_split as write_split
except ImportError:
    from cuad_data import DEFAULT_CHUNK_DATA_DIR, DEFAULT_RAW_DATA_DIR
    from cuad_data import DEFAULT_CHUNK_SIZE as DEFAULT_CHUNK_SIZE
    from cuad_data import REPO_ROOT as REPO_ROOT
    from cuad_data import SPLIT_TO_FILENAME as SPLIT_TO_FILENAME
    from cuad_data import _article_exceeds_chunk_size as _article_exceeds_chunk_size
    from cuad_data import _chunk_paragraph as _chunk_paragraph
    from cuad_data import _chunk_title as _chunk_title
    from cuad_data import _project_answers as _project_answers
    from cuad_data import chunk_article as chunk_article
    from cuad_data import chunk_articles as chunk_articles
    from cuad_data import generate_chunked_split as generate_chunked_split
    from cuad_data import load_split as load_split
    from cuad_data import write_split as write_split

DEFAULT_INPUT_DIR = DEFAULT_RAW_DATA_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_CHUNK_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chunked CUAD dataset files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the raw CUAD JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the chunked CUAD JSON files will be written",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum number of characters per chunk",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional ProcessPoolExecutor worker count for long contracts",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_TO_FILENAME),
        default=["train", "test"],
        help="Which CUAD splits to generate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        output_path = generate_chunked_split(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split=split,
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
        )
        print(f"Wrote chunked {split} split to {output_path}")


if __name__ == "__main__":
    main()
