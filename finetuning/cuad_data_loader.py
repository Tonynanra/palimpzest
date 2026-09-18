"""
Shared CUAD data loading utilities to replace HuggingFace datasets.
All CUAD scripts should import from this module.
"""

import json
import os
from pathlib import Path

import numpy as np

# Default data directory
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = str(REPO_ROOT / "cuad-data")
FALLBACK_DATA_DIR = str(REPO_ROOT / "testdata" / "cuad-data")
DEFAULT_CHUNK_DATA_DIR = str(REPO_ROOT / "cuad-chunk")
FALLBACK_CHUNK_DATA_DIR = str(REPO_ROOT / "testdata" / "cuad-chunk")


def resolve_cuad_data_dir(data_dir=None, dataset_mode="cuad-data"):
    if data_dir is not None:
        return str(data_dir)

    if dataset_mode == "cuad-data":
        default_dir = DEFAULT_DATA_DIR
        fallback_dir = FALLBACK_DATA_DIR
    elif dataset_mode == "cuad-chunk":
        default_dir = DEFAULT_CHUNK_DATA_DIR
        fallback_dir = FALLBACK_CHUNK_DATA_DIR
    else:
        raise ValueError(f"Unsupported CUAD dataset mode: {dataset_mode}")

    return default_dir if os.path.exists(default_dir) else fallback_dir


def load_cuad_data(split="test", data_dir=None, dataset_mode="cuad-data"):
    """
    Load CUAD dataset from local JSON files.
    
    Args:
        split: "train" or "test"
        data_dir: Directory containing CUAD JSON files (default: "cuad-data")
        dataset_mode: "cuad-data" or "cuad-chunk"
    
    Returns:
        List of dictionaries with CUAD data in flat format
    """
    data_dir = resolve_cuad_data_dir(data_dir=data_dir, dataset_mode=dataset_mode)
    
    if split == "train":
        file_path = os.path.join(data_dir, "train_separate_questions.json")
    else:
        file_path = os.path.join(data_dir, "test.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CUAD data file not found at {file_path}. "
            f"Please run 'python setup_cuad_data.py' first to download the data."
        )
    
    with open(file_path) as f:
        raw_data = json.load(f)

    # Convert to flat format
    dataset = []
    for article in raw_data["data"]:
        title = article.get("title", "").strip()
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                dataset.append({
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"].strip(),
                    "answers": qa.get("answers", []),
                    "dataset_mode": dataset_mode,
                    "chunk_index": paragraph.get("chunk_index"),
                    "chunk_start": paragraph.get("chunk_start"),
                    "chunk_end": paragraph.get("chunk_end"),
                    "source_paragraph_index": paragraph.get("source_paragraph_index"),
                })
    
    return dataset


def get_unique_contracts(dataset):
    """Get list of unique contract titles from dataset."""
    contract_titles = []
    for row in dataset:
        if row["title"] not in contract_titles:
            contract_titles.append(row["title"])
    return contract_titles


def filter_by_contracts(dataset, contract_titles):
    """Filter dataset to only include specified contracts."""
    return [row for row in dataset if row["title"] in contract_titles]


def sample_contracts(dataset, num_contracts, seed=42):
    """
    Sample a subset of contracts from the dataset.
    
    Args:
        dataset: CUAD dataset
        num_contracts: Number of contracts to sample
        seed: Random seed for reproducibility
    
    Returns:
        Filtered dataset with only the sampled contracts
    """
    contract_titles = get_unique_contracts(dataset)
    
    # Shuffle and sample
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(contract_titles)
    sampled_titles = contract_titles[:num_contracts]
    
    return filter_by_contracts(dataset, sampled_titles), sampled_titles
