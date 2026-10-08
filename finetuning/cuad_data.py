"Shared CUAD acquisition, file I/O, chunking, and annotation aggregation."

import json
import math
import os
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

try:
    from .cuad_schema import normalize_category_name
except ImportError:
    from cuad_schema import normalize_category_name

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR = REPO_ROOT / "testdata" / "cuad" / "raw"
DEFAULT_CHUNK_DATA_DIR = REPO_ROOT / "testdata" / "cuad" / "chunks"
DEFAULT_SFT_DATA_DIR = REPO_ROOT / "testdata" / "cuad" / "sft" / "train-dev"
DEFAULT_SFT_TEST_DATA_DIR = REPO_ROOT / "testdata" / "cuad" / "sft" / "test"
DEFAULT_CHUNK_SIZE = 65536
SPLIT_TO_FILENAME = {
    "train": "train_separate_questions.json",
    "test": "test.json",
}


def resolve_cuad_data_dir(data_dir=None, dataset_mode="cuad-data"):
    if data_dir is not None:
        return str(data_dir)
    if dataset_mode == "cuad-data":
        return str(DEFAULT_RAW_DATA_DIR)
    if dataset_mode == "cuad-chunk":
        return str(DEFAULT_CHUNK_DATA_DIR)
    raise ValueError(f"Unsupported CUAD dataset mode: {dataset_mode}")


def iter_cuad_paragraphs(
    split="test", data_dir=None, dataset_mode="cuad-data",
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Read paragraphs with the same text normalization as the flat loader."""
    data_dir = resolve_cuad_data_dir(data_dir, dataset_mode)
    filename = SPLIT_TO_FILENAME["train" if split == "train" else "test"]
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        setup_script = (
            "finetuning/setup_cuad_data.py"
            if dataset_mode == "cuad-data"
            else "finetuning/generate_cuad_chunk_dataset.py"
        )
        raise FileNotFoundError(
            f"CUAD data file not found at {path}. "
            f"Please run './.venv/bin/python {setup_script}' from the repository root first."
        )
    with open(path) as handle:
        payload = json.load(handle)
    for article in payload["data"]:
        title = article.get("title", "").strip()
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            qas = [
                {
                    "id": qa["id"],
                    "question": qa["question"].strip(),
                    "answers": qa.get("answers", []),
                }
                for qa in paragraph["qas"]
            ]
            if qas:
                yield title, {**paragraph, "context": context, "qas": qas}


def load_cuad_data(split="test", data_dir=None, dataset_mode="cuad-data"):
    """Return the existing flat QA row shape for callers that need annotations."""
    dataset = []
    for title, paragraph in iter_cuad_paragraphs(split, data_dir, dataset_mode):
        for qa in paragraph["qas"]:
            dataset.append({
                "id": qa["id"],
                "title": title,
                "context": paragraph["context"],
                "question": qa["question"],
                "answers": qa["answers"],
                "dataset_mode": dataset_mode,
                "chunk_index": paragraph.get("chunk_index"),
                "chunk_start": paragraph.get("chunk_start"),
                "chunk_end": paragraph.get("chunk_end"),
                "source_paragraph_index": paragraph.get("source_paragraph_index"),
            })
    return dataset


def load_cuad_records(split="test", data_dir=None, dataset_mode="cuad-data"):
    """Group ordered annotations by title, retaining first-annotation metadata."""
    records = {}
    for title, paragraph in iter_cuad_paragraphs(split, data_dir, dataset_mode):
        if title not in records:
            records[title] = {
                "contract_id": paragraph["qas"][0]["id"],
                "title": title,
                "contract": paragraph["context"],
                "dataset_mode": dataset_mode,
                "chunk_index": paragraph.get("chunk_index"),
                "chunk_start": paragraph.get("chunk_start"),
                "chunk_end": paragraph.get("chunk_end"),
                "source_paragraph_index": paragraph.get("source_paragraph_index"),
                "qas": [],
            }
        records[title]["qas"].extend(paragraph["qas"])
    return records


def _sample_titles(titles, num_contracts, seed):
    titles = list(titles)
    np.random.default_rng(seed=seed).shuffle(titles)
    return titles[:num_contracts]


def select_records(records, num_contracts, seed=42):
    return [records[title] for title in _sample_titles(records, num_contracts, seed)]


def build_label_records(records, selected_categories, dataset_mode):
    """Aggregate category spans without changing answer order or duplicates."""
    selected_category_set = set(selected_categories)
    labels = []
    for record in records:
        label = {key: record[key] for key in ("contract_id", "title", "contract")}
        if dataset_mode == "cuad-chunk":
            label.update({
                key: record.get(key)
                for key in ("dataset_mode", "chunk_index", "chunk_start", "chunk_end", "source_paragraph_index")
            })
        label.update({category: [] for category in selected_categories})
        for qa in record["qas"]:
            category = normalize_category_name(qa["id"].split("__")[-1].split("_")[0].strip())
            if category not in selected_category_set:
                continue
            answers = qa.get("answers", [])
            answer_texts = [answer["text"] for answer in answers] if isinstance(answers, list) else answers.get("text", [])
            label[category].extend(answer_texts)
        labels.append(label)
    return labels


def get_unique_contracts(dataset):
    """Return titles in their first-annotation order."""
    return list(dict.fromkeys(row["title"] for row in dataset))


def filter_by_contracts(dataset, contract_titles):
    return [row for row in dataset if row["title"] in contract_titles]


def sample_contracts(dataset, num_contracts, seed=42):
    sampled_titles = _sample_titles(get_unique_contracts(dataset), num_contracts, seed)
    return filter_by_contracts(dataset, sampled_titles), sampled_titles


class CuadJsonlDataset:
    def __init__(self, path: Path, limit: int | None = None):
        self.path = path
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if limit is not None:
            if limit <= 0:
                raise ValueError("dataset limit must be positive")
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No SFT examples found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


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


def setup_cuad_data(output_dir=DEFAULT_RAW_DATA_DIR):
    data_dir = str(output_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Download CUAD data zip file
    data_url = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
    zip_path = os.path.join(data_dir, "data.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading CUAD data from {data_url}...")
        urllib.request.urlretrieve(data_url, zip_path)
        print(f"Downloaded to {zip_path}")
    else:
        print(f"Data already downloaded at {zip_path}")
    
    # Extract the zip file
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"Extracted data to {data_dir}")
    
    # Download the dataset loading script (for reference, not actually used)
    script_url = "https://huggingface.co/datasets/theatticusproject/cuad-qa/resolve/main/cuad-qa.py"
    script_path = os.path.join(data_dir, "cuad-qa.py")
    
    if not os.path.exists(script_path):
        print(f"Downloading CUAD dataset script from {script_url}...")
        urllib.request.urlretrieve(script_url, script_path)
        print(f"Downloaded to {script_path}")
    else:
        print(f"Script already exists at {script_path}")
    
    # List extracted files
    print("\nExtracted files:")
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            file_path = os.path.join(data_dir, file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  - {file} ({size:.2f} MB)")
    
    print(f"\nSetup complete! CUAD data is ready in {data_dir}.")
    print("\nTo use this data in your scripts, update the data loading to:")
    print(f"  - train data: {os.path.join(data_dir, 'train_separate_questions.json')}")
    print(f"  - test data: {os.path.join(data_dir, 'test.json')}")
    
    return data_dir



