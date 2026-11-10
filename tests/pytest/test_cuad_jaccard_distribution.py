import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import plot_cuad_jaccard_distribution as jaccard_plot  # noqa: E402


@pytest.fixture
def saved_evaluation(tmp_path):
    first = {
        "id": "shared-chunk-id",
        "contract": "Alpha beta gamma unreturned Acme Corp",
        "categories": ["Document Name", "Agreement Date", "Parties"],
        "labels": {
            "Document Name": ["Alpha, beta/gamma!", "unreturned", "Alpha, beta/gamma!"],
            "Agreement Date": [],
            "Parties": ["Acme Corp"],
        },
    }
    second = copy.deepcopy(first)
    second.update(contract="second sample", labels={"Document Name": ["second sample"]})
    outputs = {
        "vanilla": [
            {"Document Name": ["beta GAMMA", "irrelevant", "Alpha"], "Agreement Date": ["false positive"]},
            {"Document Name": ["second"]},
        ],
        "sft": [
            {"Document Name": ["alpha beta gamma"], "Parties": ["Acme Corp with more words"]},
            {"Document Name": "second sample"},
        ],
    }
    for role, predictions in outputs.items():
        records = [
            {
                "row_index": index,
                "model_role": role,
                "model_id": "saved/model",
                "mode": "all",
                "row": row,
                "completion": json.dumps(prediction),
            }
            for index, (row, prediction) in enumerate(zip((first, second), predictions, strict=True), start=1)
        ]
        # Pairing must use row indices even when SFT capture order differs and IDs repeat.
        if role == "sft":
            records.reverse()
        (tmp_path / f"{role}.raw.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
        )
    (tmp_path / "comparison.json").write_text(
        json.dumps({"rows": 2, "model_id": "saved/model", "mode": "all"}), encoding="utf-8",
    )
    return tmp_path


def test_best_match_per_gold_span_keeps_duplicates_and_uses_pure_jaccard(saved_evaluation):
    scores = jaccard_plot.collect_scores(saved_evaluation)

    assert scores["vanilla"] == pytest.approx([2 / 3, 0, 2 / 3, 0, 0.5])
    assert scores["sft"] == pytest.approx([1, 0, 1, 2 / 5, 1])


def test_malformed_and_non_object_completions_keep_gold_spans_at_zero(saved_evaluation):
    for role, completion in (("vanilla", "not JSON"), ("sft", "[]")):
        path = saved_evaluation / f"{role}.raw.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        next(record for record in records if record["row_index"] == 1)["completion"] = completion
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    scores = jaccard_plot.collect_scores(saved_evaluation)

    assert scores["vanilla"] == [0, 0, 0, 0, 0.5]
    assert scores["sft"] == [0, 0, 0, 0, 1]


def test_mismatched_inputs_are_rejected(saved_evaluation):
    path = saved_evaluation / "sft.raw.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["row"]["contract"] = "different contract"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="embedded inputs differ"):
        jaccard_plot.collect_scores(saved_evaluation)
