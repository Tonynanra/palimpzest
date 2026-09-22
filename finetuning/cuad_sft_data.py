"""Build deterministic conversational SFT examples from the local CUAD data.

This module deliberately reuses the CUAD loader, category definitions, label
aggregation, and Palimpzest prompt factory.  The only SFT-specific behavior is
the generation of several one-convert views with randomized requested fields.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .cuad_data_loader import load_cuad_data
    from .cuad_demo import (
        _category_output_column,
        get_category_by_name,
        get_category_names,
        get_label_df,
    )
except ImportError:
    from cuad_data_loader import load_cuad_data
    from cuad_demo import (
        _category_output_column,
        get_category_by_name,
        get_category_names,
        get_label_df,
    )

from palimpzest.constants import Cardinality, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import create_schema_from_fields
from palimpzest.prompts.prompt_factory import PromptFactory

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK_DATA_DIR = REPO_ROOT / "testdata" / "cuad-chunk"
DEFAULT_RAW_DATA_DIR = REPO_ROOT / "testdata" / "cuad-data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "finetuning" / "artifacts" / "cuad-sft"

INPUT_FIELD = {
    "name": "contract",
    "type": str,
    "desc": "The content of the the contract to be analyzed",
}


@dataclass(frozen=True)
class ViewSpec:
    name: str
    min_categories: int
    max_categories: int


VIEW_SPECS = (
    ViewSpec("singleton", 1, 1),
    ViewSpec("singleton", 1, 1),
    ViewSpec("small", 2, 5),
    ViewSpec("small", 2, 5),
    ViewSpec("medium", 6, 15),
    ViewSpec("large", 16, 30),
    ViewSpec("all", 41, 41),
)


@dataclass(frozen=True)
class SFTDataConfig:
    chunk_data_dir: Path = DEFAULT_CHUNK_DATA_DIR
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR
    split: str = "train"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 42
    train_fraction: float = 0.9
    views_per_chunk: int = len(VIEW_SPECS)

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise ValueError(f"Unsupported CUAD split: {self.split}")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1")
        if self.views_per_chunk != len(VIEW_SPECS):
            raise ValueError(
                f"views_per_chunk must be {len(VIEW_SPECS)} so every category-size regime is represented"
            )


class _TextPromptModel:
    """Minimal model capability object needed by PromptFactory for text prompts."""

    @staticmethod
    def is_llama_model() -> bool:
        return False


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_source_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _source_hash_by_title(raw_data_dir: Path, split: str) -> dict[str, str]:
    """Map each original contract title to a stable full-source content hash."""
    raw_rows = load_cuad_data(split=split, data_dir=str(raw_data_dir), dataset_mode="cuad-data")
    source_hashes: dict[str, str] = {}
    for row in raw_rows:
        title = row["title"]
        digest = _stable_hash(_normalize_source_text(row["context"]))
        previous = source_hashes.setdefault(title, digest)
        if previous != digest:
            raise ValueError(f"Source title {title!r} maps to multiple full contract contents")
    return source_hashes


def _base_source_title(chunk_title: str) -> str:
    return chunk_title.split(" [p", 1)[0]


def _coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [str(answer) for answer in value if answer is not None]
    return [str(value)]


def _ordered_view_categories(
    categories: list[str],
    spec: ViewSpec,
    rng: np.random.Generator,
    category_cursor: int,
) -> tuple[list[str], int]:
    if spec.min_categories == spec.max_categories:
        count = spec.min_categories
    else:
        count = int(rng.integers(spec.min_categories, spec.max_categories + 1))

    if count == len(categories):
        selected = list(categories)
        rng.shuffle(selected)
        return selected, category_cursor

    # Walk a shuffled category cycle so singleton and small-set views do not
    # repeatedly select the same early categories for every contract.
    selected = [categories[(category_cursor + offset) % len(categories)] for offset in range(count)]
    category_cursor = (category_cursor + count) % len(categories)
    rng.shuffle(selected)
    return selected, category_cursor


def _make_prompt_factory() -> PromptFactory:
    return PromptFactory(
        PromptStrategy.MAP_NO_REASONING,
        _TextPromptModel(),
        Cardinality.ONE_TO_ONE,
    )


def render_cuad_messages(contract: str, selected_categories: list[str]) -> list[dict[str, Any]]:
    """Render the same no-reasoning map prompt used by Palimpzest."""
    if not selected_categories:
        raise ValueError("selected_categories must not be empty")

    input_schema = create_schema_from_fields([copy.deepcopy(INPUT_FIELD)])
    candidate = DataRecord(
        input_schema(contract=contract),
        source_indices=["cuad-sft-0"],
    )

    output_fields = [
        copy.deepcopy(_category_output_column(get_category_by_name(category)))
        for category in selected_categories
    ]
    output_schema = create_schema_from_fields(output_fields)
    return _make_prompt_factory().create_messages(
        candidate,
        selected_categories,
        project_cols=["contract"],
        output_schema=output_schema,
        output_field_order="provided",
    )


def build_json_target(
    selected_categories: list[str],
    labels_by_category: dict[str, list[str]],
) -> str:
    """Build the exact JSON completion consumed by the one-to-one parser."""
    target = {
        category: list(labels_by_category.get(category, []))
        for category in selected_categories
    }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":")) + "\n---"


def _split_source_hashes(
    source_hashes: Iterable[str],
    train_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    sources = sorted(set(source_hashes))
    if len(sources) < 2:
        raise ValueError("At least two source contracts are required to create train/dev splits")

    rng = np.random.default_rng(seed)
    rng.shuffle(sources)
    train_count = int(round(len(sources) * train_fraction))
    train_count = min(max(train_count, 1), len(sources) - 1)
    return set(sources[:train_count]), set(sources[train_count:])


def _metadata_for_title(rows: list[dict[str, Any]], title: str) -> dict[str, Any]:
    matching = [row for row in rows if row["title"] == title]
    if not matching:
        raise KeyError(f"No CUAD rows found for title {title!r}")
    first = matching[0]
    return {
        "contract_id": first["id"],
        "title": title,
        "contract": first["context"],
        "dataset_mode": first.get("dataset_mode"),
        "chunk_index": first.get("chunk_index"),
        "chunk_start": first.get("chunk_start"),
        "chunk_end": first.get("chunk_end"),
        "source_paragraph_index": first.get("source_paragraph_index"),
    }


def build_sft_examples(config: SFTDataConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build train/dev SFT rows and a reproducibility manifest."""
    categories = get_category_names()
    chunk_rows = load_cuad_data(
        split=config.split,
        data_dir=str(config.chunk_data_dir),
        dataset_mode="cuad-chunk",
    )
    titles = sorted({row["title"] for row in chunk_rows})
    if not titles:
        raise ValueError("No CUAD chunk rows were loaded")

    labels_df = get_label_df(
        num_contracts=len(titles),
        seed=config.seed,
        selected_categories=categories,
        split=config.split,
        dataset_mode="cuad-chunk",
        data_dir=str(config.chunk_data_dir),
    )
    labels_by_title = {
        row["title"]: {
            category: _coerce_answers(row.get(category, []))
            for category in categories
        }
        for _, row in labels_df.iterrows()
    }

    source_hashes_by_original_title = _source_hash_by_title(config.raw_data_dir, config.split)
    title_to_metadata = {
        title: _metadata_for_title(chunk_rows, title)
        for title in titles
    }
    title_to_source_hash: dict[str, str] = {}
    for title in titles:
        original_title = _base_source_title(title)
        title_to_source_hash[title] = source_hashes_by_original_title.get(
            original_title,
            _stable_hash(_normalize_source_text(title_to_metadata[title]["contract"])),
        )

    train_sources, dev_sources = _split_source_hashes(
        title_to_source_hash.values(),
        config.train_fraction,
        config.seed,
    )

    examples: list[dict[str, Any]] = []
    for title in titles:
        source_hash = title_to_source_hash[title]
        dataset_split = "train" if source_hash in train_sources else "dev"
        metadata = title_to_metadata[title]
        labels = labels_by_title.get(title, {category: [] for category in categories})
        title_seed = int(source_hash[:16], 16) ^ config.seed
        rng = np.random.default_rng(title_seed)
        shuffled_categories = list(categories)
        rng.shuffle(shuffled_categories)
        category_cursor = 0

        for view_index, spec in enumerate(VIEW_SPECS):
            selected_categories, category_cursor = _ordered_view_categories(
                shuffled_categories,
                spec,
                rng,
                category_cursor,
            )
            response = build_json_target(selected_categories, labels)
            messages = render_cuad_messages(metadata["contract"], selected_categories)
            examples.append(
                {
                    "id": f"{source_hash}-{view_index}",
                    "split": dataset_split,
                    "source_hash": source_hash,
                    "source_title": _base_source_title(title),
                    "chunk_title": title,
                    "contract_id": metadata["contract_id"],
                    "contract": metadata["contract"],
                    "chunk_index": metadata["chunk_index"],
                    "chunk_start": metadata["chunk_start"],
                    "chunk_end": metadata["chunk_end"],
                    "view": spec.name,
                    "view_index": view_index,
                    "categories": selected_categories,
                    "labels": {
                        category: list(labels.get(category, []))
                        for category in selected_categories
                    },
                    "messages": messages,
                    "response": response,
                }
            )

    manifest = {
        "schema_version": 1,
        "dataset_mode": "cuad-chunk",
        "source_split": config.split,
        "chunk_data_dir": str(config.chunk_data_dir),
        "raw_data_dir": str(config.raw_data_dir),
        "seed": config.seed,
        "train_fraction": config.train_fraction,
        "views": [asdict(spec) for spec in VIEW_SPECS],
        "categories": categories,
        "source_contracts": len(set(title_to_source_hash.values())),
        "chunk_records": len(titles),
        "train_source_hashes": sorted(train_sources),
        "dev_source_hashes": sorted(dev_sources),
        "train_examples": sum(example["split"] == "train" for example in examples),
        "dev_examples": sum(example["split"] == "dev" for example in examples),
    }
    return examples, manifest


def write_sft_artifacts(
    examples: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                if example["split"] == split:
                    handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build randomized CUAD Qwen SFT data")
    parser.add_argument("--chunk-data-dir", type=Path, default=DEFAULT_CHUNK_DATA_DIR)
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SFTDataConfig(
        chunk_data_dir=args.chunk_data_dir,
        raw_data_dir=args.raw_data_dir,
        split=args.split,
        output_dir=args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    examples, manifest = build_sft_examples(config)
    write_sft_artifacts(examples, manifest, config.output_dir)
    print(
        f"Wrote {manifest['train_examples']} train and {manifest['dev_examples']} dev examples "
        f"to {config.output_dir}"
    )


if __name__ == "__main__":
    main()
