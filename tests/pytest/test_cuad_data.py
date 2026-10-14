import hashlib
import json
import sys
from pathlib import Path

import pytest

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import cuad_data  # noqa: E402
import cuad_data_loader  # noqa: E402
import cuad_demo  # noqa: E402
import cuad_sft_data  # noqa: E402
import train_cuad_sft  # noqa: E402


def _qa(category, texts=(), prefix="first"):
    return {
        "id": f"{prefix}__{category}_0",
        "question": f"  What is {category}?  ",
        "answers": [{"text": text, "answer_start": 0} for text in texts],
    }


def _write_data(directory, articles, split="test"):
    directory.mkdir(parents=True, exist_ok=True)
    filename = "train_separate_questions.json" if split == "train" else "test.json"
    (directory / filename).write_text(json.dumps({"data": articles}), encoding="utf-8")
    return directory


@pytest.fixture
def grouped_data_dir(tmp_path):
    dict_answers = _qa("Change Of Control", prefix="second")
    dict_answers["answers"] = {"text": ["yes", "yes"]}
    return _write_data(
        tmp_path / "data",
        [
            {
                "title": "  Contract B  ",
                "paragraphs": [
                    {"context": "ignored paragraph", "qas": []},
                    {
                        "context": "  First text  ",
                        "chunk_index": 2,
                        "chunk_start": 10,
                        "chunk_end": 22,
                        "source_paragraph_index": 1,
                        "qas": [_qa("Parties", ["Acme", "Acme"]), _qa("Document Name")],
                    },
                    {
                        "context": "Second text",
                        "qas": [_qa("Parties", ["Beta"], prefix="second"), dict_answers],
                    },
                ],
            },
            {
                "title": "Contract A",
                "paragraphs": [{"context": "A text", "qas": [_qa("Parties", ["A"])]}],
            },
            {
                "title": "Contract B",
                "paragraphs": [{"context": "Third text", "qas": [_qa("Parties", ["Gamma"], prefix="third")]}],
            },
            {
                "title": "Contract C",
                "paragraphs": [{"context": "C text", "qas": [_qa("Parties", ["C"])]}],
            },
            {"title": "No annotations", "paragraphs": [{"context": "not a record", "qas": []}]},
        ],
    )


def test_grouping_keeps_first_annotation_metadata_and_all_ordered_answers(grouped_data_dir):
    records = cuad_data.load_cuad_records(data_dir=grouped_data_dir, dataset_mode="cuad-chunk")

    assert list(records) == ["Contract B", "Contract A", "Contract C"]
    record = records["Contract B"]
    assert {key: value for key, value in record.items() if key != "qas"} == {
        "contract_id": "first__Parties_0",
        "title": "Contract B",
        "contract": "First text",
        "dataset_mode": "cuad-chunk",
        "chunk_index": 2,
        "chunk_start": 10,
        "chunk_end": 22,
        "source_paragraph_index": 1,
    }
    labels = cuad_data.build_label_records(
        [record], ["Parties", "Change of Control", "Document Name"], "cuad-chunk",
    )[0]
    assert list(labels) == [
        "contract_id", "title", "contract", "dataset_mode", "chunk_index", "chunk_start",
        "chunk_end", "source_paragraph_index", "Parties", "Change of Control", "Document Name",
    ]
    assert labels["Parties"] == ["Acme", "Acme", "Beta", "Gamma"]
    assert labels["Change of Control"] == ["yes", "yes"]
    assert labels["Document Name"] == []


def test_flat_loader_preserves_traversal_order_and_public_shape(grouped_data_dir):
    rows = cuad_data_loader.load_cuad_data(data_dir=grouped_data_dir, dataset_mode="cuad-chunk")

    assert [row["title"] for row in rows] == [
        "Contract B", "Contract B", "Contract B", "Contract B", "Contract A", "Contract B", "Contract C",
    ]
    assert rows[0] == {
        "id": "first__Parties_0",
        "title": "Contract B",
        "context": "First text",
        "question": "What is Parties?",
        "answers": [{"text": "Acme", "answer_start": 0}, {"text": "Acme", "answer_start": 0}],
        "dataset_mode": "cuad-chunk",
        "chunk_index": 2,
        "chunk_start": 10,
        "chunk_end": 22,
        "source_paragraph_index": 1,
    }
    assert rows[2]["context"] == "Second text"
    assert rows[5]["context"] == "Third text"


def test_seeded_sampling_and_benchmark_projection_keep_title_order(grouped_data_dir):
    records = cuad_data.load_cuad_records(data_dir=grouped_data_dir)
    sampled = cuad_data.select_records(records, num_contracts=2, seed=42)
    assert [record["title"] for record in sampled] == ["Contract C", "Contract A"]

    dataset = cuad_demo.CUADDataset(num_contracts=2, seed=42, data_dir=str(grouped_data_dir), split="test")
    assert dataset.dataset == [
        {
            "contract_id": "first__Parties_0", "title": "Contract C", "contract": "C text",
            "dataset_mode": "cuad-data", "chunk_index": None,
        },
        {
            "contract_id": "first__Parties_0", "title": "Contract A", "contract": "A text",
            "dataset_mode": "cuad-data", "chunk_index": None,
        },
    ]
    labels = cuad_demo.get_label_df(
        num_contracts=2, seed=42, selected_categories=["Parties"], data_dir=str(grouped_data_dir),
    )
    assert list(labels.columns) == ["contract_id", "title", "contract", "Parties"]
    assert labels["title"].tolist() == ["Contract C", "Contract A"]
    assert labels["Parties"].tolist() == [["C"], ["A"]]


def test_sft_preserves_whitespace_hash_fallback_and_duplicate_sibling_ids(tmp_path):
    raw_dir = _write_data(
        tmp_path / "raw",
        [{"title": " Contract A ", "paragraphs": [
            {"context": " Whole  contract \n text ", "qas": [_qa("Parties", ["Acme"])]},
            {"context": "ignored without annotations", "qas": []},
        ]}],
    )
    titles = ["Contract A [p0-chunk0]", "Contract A [p0-chunk1]", "Contract A  [p0-chunk2]"]
    chunk_dir = _write_data(
        tmp_path / "chunks",
        [{"title": title, "paragraphs": [{
            "context": f" Chunk {index} ", "chunk_index": index,
            "chunk_start": index * 10, "chunk_end": index * 10 + 7,
            "qas": [_qa("Parties", ["Acme"], prefix=f"chunk-{index}")],
        }]} for index, title in enumerate(titles)],
    )
    examples, manifest = cuad_sft_data.build_sft_examples(cuad_sft_data.SFTDataConfig(
        raw_data_dir=raw_dir, chunk_data_dir=chunk_dir, split="test", output_dir=tmp_path / "output",
    ))
    by_title = {title: [row for row in examples if row["chunk_title"] == title] for title in titles}
    source_hash = hashlib.sha256(b"Whole contract text").hexdigest()
    fallback_hash = hashlib.sha256(b"Chunk 2").hexdigest()
    assert {row["source_hash"] for row in by_title[titles[0]] + by_title[titles[1]]} == {source_hash}
    assert {row["source_hash"] for row in by_title[titles[2]]} == {fallback_hash}
    assert {row["source_title"] for row in by_title[titles[2]]} == {"Contract A "}
    assert [row["id"] for row in by_title[titles[0]]] == [row["id"] for row in by_title[titles[1]]]
    assert [row["categories"] for row in by_title[titles[0]]] == [row["categories"] for row in by_title[titles[1]]]
    assert len(examples) == 21
    assert len({row["id"] for row in examples}) == 14
    assert {row["split"] for row in examples} == {"test"}
    assert manifest["source_contracts"] == 2
    assert manifest["chunk_records"] == 3
    assert manifest["test_examples"] == 21
    assert manifest["train_source_hashes"] == manifest["dev_source_hashes"] == []


def test_raw_source_hashing_rejects_conflicting_annotated_paragraphs(tmp_path):
    raw_dir = _write_data(tmp_path, [{"title": "Same title", "paragraphs": [
        {"context": "First content", "qas": [_qa("Parties")]},
        {"context": "Second content", "qas": [_qa("Parties")]},
    ]}])
    with pytest.raises(ValueError, match="maps to multiple full contract contents"):
        cuad_sft_data._source_hash_by_title(raw_dir, "test")


def test_jsonl_loader_keeps_duplicate_ids_blank_line_handling_and_limits(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('\n{"id":"same","value":1}\n  \n{"id":"same","value":2}\n', encoding="utf-8")
    dataset = cuad_data.CuadJsonlDataset(path)

    assert train_cuad_sft.CuadJsonlDataset is cuad_data.CuadJsonlDataset
    assert len(dataset) == 2
    assert dataset[0] == {"id": "same", "value": 1}
    assert dataset[1] == {"id": "same", "value": 2}
    assert cuad_data.CuadJsonlDataset(path, limit=1).rows == [{"id": "same", "value": 1}]
    for limit in (0, -1):
        with pytest.raises(ValueError, match="dataset limit must be positive"):
            cuad_data.CuadJsonlDataset(path, limit=limit)


def test_jsonl_loader_validates_the_whole_file_before_applying_limit(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id":"valid"}\ninvalid json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        cuad_data.CuadJsonlDataset(path, limit=1)
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="No SFT examples found"):
        cuad_data.CuadJsonlDataset(path)
