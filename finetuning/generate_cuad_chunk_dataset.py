import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "testdata" / "cuad-data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "testdata" / "cuad-chunk"
DEFAULT_CHUNK_SIZE = 65536
SPLIT_TO_FILENAME = {
    "train": "train_separate_questions.json",
    "test": "test.json",
}


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


def _project_answers(
    context: str,
    answers: List[Dict[str, Any]],
    chunk_start: int,
    chunk_end: int,
) -> List[Dict[str, Any]]:
    projected_answers = []
    for answer in answers:
        answer_text = answer.get("text", "")
        answer_start = int(answer.get("answer_start", 0))
        answer_end = answer_start + len(answer_text)

        overlap_start = max(answer_start, chunk_start)
        overlap_end = min(answer_end, chunk_end)
        if overlap_start >= overlap_end:
            continue

        projected_answers.append(
            {
                "text": context[overlap_start:overlap_end],
                "answer_start": overlap_start - chunk_start,
            }
        )

    return projected_answers


def _chunk_paragraph(paragraph: Dict[str, Any], chunk_size: int, paragraph_index: int) -> List[Dict[str, Any]]:
    context = paragraph["context"]
    qas = paragraph["qas"]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    num_chunks = max(1, math.ceil(len(context) / chunk_size))
    chunked_paragraphs = []
    for chunk_index in range(num_chunks):
        chunk_start = chunk_index * chunk_size
        chunk_end = min(len(context), chunk_start + chunk_size)
        chunk_context = context[chunk_start:chunk_end]

        chunk_qas = []
        for qa in qas:
            projected_answers = _project_answers(context, qa.get("answers", []), chunk_start, chunk_end)
            chunk_qas.append(
                {
                    "answers": projected_answers,
                    "id": f"{qa['id']}_chunk_{chunk_index}",
                    "question": qa["question"],
                    "is_impossible": len(projected_answers) == 0,
                }
            )

        chunked_paragraphs.append(
            {
                "qas": chunk_qas,
                "context": chunk_context,
                "chunk_index": chunk_index,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "source_paragraph_index": paragraph_index,
            }
        )

    return chunked_paragraphs


def _chunk_title(article_title: str, paragraph_index: int, chunk_index: int) -> str:
    return f"{article_title} [p{paragraph_index}-chunk{chunk_index}]"


def chunk_article(article: Dict[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
    chunked_articles = []
    for paragraph_index, paragraph in enumerate(article["paragraphs"]):
        chunked_paragraphs = _chunk_paragraph(paragraph, chunk_size=chunk_size, paragraph_index=paragraph_index)
        for chunked_paragraph in chunked_paragraphs:
            chunked_articles.append(
                {
                    "title": _chunk_title(
                        article["title"],
                        paragraph_index=paragraph_index,
                        chunk_index=chunked_paragraph["chunk_index"],
                    ),
                    "paragraphs": [chunked_paragraph],
                }
            )

    return chunked_articles


def _article_exceeds_chunk_size(article: Dict[str, Any], chunk_size: int) -> bool:
    return any(len(paragraph["context"]) > chunk_size for paragraph in article["paragraphs"])


def chunk_articles(
    articles: List[Dict[str, Any]],
    chunk_size: int,
    max_workers: Optional[int] = None,
    executor_cls=ProcessPoolExecutor,
) -> List[Dict[str, Any]]:
    chunked_articles: List[List[Dict[str, Any]] | None] = [None] * len(articles)
    long_articles = [
        (index, article)
        for index, article in enumerate(articles)
        if _article_exceeds_chunk_size(article, chunk_size)
    ]

    for index, article in enumerate(articles):
        if not _article_exceeds_chunk_size(article, chunk_size):
            chunked_articles[index] = chunk_article(article, chunk_size)

    if long_articles:
        with executor_cls(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(chunk_article, article, chunk_size): index
                for index, article in long_articles
            }
            for future, index in future_to_index.items():
                chunked_articles[index] = future.result()

    return [
        chunked_article
        for article_chunks in chunked_articles
        if article_chunks is not None
        for chunked_article in article_chunks
    ]


def load_split(input_dir: Path, split: str) -> Dict[str, Any]:
    path = input_dir / SPLIT_TO_FILENAME[split]
    with path.open() as f:
        return json.load(f)


def write_split(output_dir: Path, split: str, payload: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / SPLIT_TO_FILENAME[split]
    with output_path.open("w") as f:
        json.dump(payload, f)
    return output_path


def generate_chunked_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_workers: Optional[int] = None,
) -> Path:
    raw_payload = load_split(input_dir, split)
    chunked_payload = {
        "version": raw_payload.get("version"),
        "data": chunk_articles(raw_payload["data"], chunk_size=chunk_size, max_workers=max_workers),
    }
    return write_split(output_dir, split, chunked_payload)


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
