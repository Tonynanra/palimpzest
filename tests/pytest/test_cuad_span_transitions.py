import copy
import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import evaluate_cuad_span_transitions as span_eval  # noqa: E402

MODEL_ID = "Qwen/Qwen3.8-27B"


def _row(labels=None, chunk_index=0):
    categories = span_eval.get_category_names()
    all_labels = {category: [] for category in categories}
    all_labels.update(labels or {})
    contract = " ".join(span for spans in all_labels.values() for span in spans)
    return {
        "id": "shared-id-6",
        "split": "test",
        "source_hash": "shared-source",
        "source_title": "Sample contract",
        "chunk_title": f"Sample contract [p0-chunk{chunk_index}]",
        "contract_id": f"contract-chunk-{chunk_index}",
        "contract": contract,
        "chunk_index": chunk_index,
        "chunk_start": chunk_index * 65536,
        "chunk_end": chunk_index * 65536 + len(contract),
        "view": "all",
        "view_index": 6,
        "categories": categories,
        "labels": all_labels,
        "messages": [{"role": "user", "content": contract}],
    }


def _write_evaluation(directory, rows, vanilla=None, sft=None):
    comparison = {"mode": "all", "model_id": MODEL_ID, "rows": len(rows)}
    for role, outputs in (("vanilla", vanilla), ("sft", sft)):
        comparison[role] = {"mode": "all", "model_id": MODEL_ID, "rows": len(rows)}
        records = []
        for index, row in enumerate(rows, start=1):
            output = outputs[index - 1] if outputs is not None else {}
            records.append(
                {
                    "row_index": index,
                    "model_role": role,
                    "model_id": MODEL_ID,
                    "checkpoint": "/fake/adapter" if role == "sft" else None,
                    "mode": "all",
                    "row": row,
                    "completion": output if isinstance(output, str) else json.dumps(output),
                }
            )
        _write_records(directory / f"{role}.raw.jsonl", records)
    (directory / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")


def _write_records(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _eight_cell_inputs(chunk_index=0):
    row = _row(
        {
            "Document Name": ["alpha", "beta", "gamma", "delta"],
            "Agreement Date": ["epsilon", "zeta", "eta", "theta"],
        },
        chunk_index,
    )
    vanilla = {"Document Name": ["alpha", "beta", "epsilon", "zeta"]}
    sft = {"Document Name": ["alpha", "gamma", "epsilon", "eta"]}
    return row, vanilla, sft


def test_all_eight_cells_and_pooled_rates(tmp_path):
    row, vanilla, sft = _eight_cell_inputs()
    _write_evaluation(tmp_path, [row], [vanilla], [sft])
    report = span_eval.score_evaluation(tmp_path)

    category = report["per_category"]["Document Name"]
    assert category["counts"] == dict.fromkeys("abcdefgh", 1)
    assert category["support"] == {"gold_positive": 4, "gold_negative": 4, "total": 8}
    for rate in category["rates"].values():
        assert rate == {"numerator": 1, "denominator": 2, "value": 0.5}

    aggregate = report["aggregate"]
    assert aggregate["counts"] == {"a": 1, "b": 1, "c": 1, "d": 5, "e": 317, "f": 1, "g": 1, "h": 1}
    assert aggregate["support"] == {"gold_positive": 8, "gold_negative": 320, "total": 328}
    assert aggregate["rates"]["fn_recovery"]["value"] == pytest.approx(1 / 6)
    assert aggregate["rates"]["fp_correction"]["value"] == 0.5
    assert aggregate["rates"]["tp_loss"]["value"] == 0.5
    assert aggregate["rates"]["new_fp_introduction"]["value"] == pytest.approx(1 / 318)
    assert report["metadata"]["parse_failures"] == {"vanilla": 0, "sft": 0}
    for cell in "abcdefgh":
        assert aggregate["counts"][cell] == sum(result["counts"][cell] for result in report["per_category"].values())


def test_duplicate_and_shared_annotations_count_once_per_chunk_category(tmp_path):
    row = _row({"Document Name": ["alpha", "alpha"], "Agreement Date": ["alpha", "beta", "beta"]})
    _write_evaluation(tmp_path, [row], [{"Document Name": ["alpha"]}], [{"Document Name": ["alpha", "alpha"]}])
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    report = span_eval.score_evaluation(tmp_path)

    assert report["per_category"]["Document Name"]["counts"] == {
        "a": 1,
        "b": 0,
        "c": 0,
        "d": 0,
        "e": 1,
        "f": 0,
        "g": 0,
        "h": 0,
    }
    assert report["per_category"]["Agreement Date"]["counts"]["d"] == 2
    assert report["aggregate"]["support"] == {"gold_positive": 3, "gold_negative": 79, "total": 82}
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_pairing_uses_row_index_and_keeps_repeated_ids_across_chunks(tmp_path):
    first, vanilla, sft = _eight_cell_inputs()
    second, _, _ = _eight_cell_inputs(chunk_index=1)
    _write_evaluation(tmp_path, [first, second], [vanilla, vanilla], [sft, sft])
    path = tmp_path / "sft.raw.jsonl"
    _write_records(path, list(reversed(_read_records(path))))

    report = span_eval.score_evaluation(tmp_path)

    assert report["metadata"]["rows"] == 2
    assert report["per_category"]["Document Name"]["counts"] == dict.fromkeys("abcdefgh", 2)
    assert report["aggregate"]["support"] == {"gold_positive": 16, "gold_negative": 640, "total": 656}


@pytest.mark.parametrize(
    "category,span,prediction,cell",
    [
        (
            "Document Name",
            "The warranty shall remain in effect for a period of ten months.",
            "remain in effect for a period of ten months.",
            "b",
        ),
        ("Document Name", "alpha beta gamma delta epsilon zeta eta theta iota kappa", "alpha", "d"),
        ("Document Name", " ".join(f"w{i}" for i in range(20)), "w0 w1 w2", "b"),
        ("Document Name", "License/Grant: TEN Months.", "license grant ten months", "b"),
        ("Parties", "Acme", "Acme " + " ".join(f"w{i}" for i in range(20)), "b"),
        ("Document Name", "Acme", "Acme " + " ".join(f"w{i}" for i in range(20)), "d"),
    ],
)
def test_detection_preserves_existing_matching(category, span, prediction, cell, tmp_path):
    _write_evaluation(tmp_path, [_row({category: [span]})], [{category: [prediction]}])
    result = span_eval.score_evaluation(tmp_path)["per_category"][category]
    assert result["counts"][cell] == 1
    assert result["support"]["gold_positive"] == 1


@pytest.mark.parametrize(
    "output,failed",
    [
        ("not JSON", 1),
        ("[]", 1),
        ("false", 1),
        ({}, 0),
        ({"Document Name": None}, 0),
        ({"Document Name": "null"}, 0),
        ({"Unrequested key": ["alpha"]}, 0),
    ],
)
def test_parse_failures_and_missing_values_yield_empty_predictions(tmp_path, output, failed):
    _write_evaluation(tmp_path, [_row({"Document Name": ["alpha"]})], [output])
    report = span_eval.score_evaluation(tmp_path)
    assert report["per_category"]["Document Name"]["counts"]["d"] == 1
    assert report["metadata"]["parse_failures"] == {"vanilla": failed, "sft": 0}


def test_prediction_normalization_handles_scalar_and_null_list_items(tmp_path):
    _write_evaluation(
        tmp_path, [_row({"Document Name": ["123"]})], [{"Document Name": 123}], [{"Document Name": [None, 123]}]
    )
    report = span_eval.score_evaluation(tmp_path)
    assert report["per_category"]["Document Name"]["counts"]["a"] == 1


@pytest.mark.parametrize(
    "problem,message",
    [
        ("missing_comparison", "Completed comparison.json"),
        ("comparison_rows", "sft rows"),
        ("comparison_mode", "full-category 'all'"),
        ("missing_rows", "incomplete coverage"),
        ("extra_rows", "incomplete coverage"),
        ("duplicate_index", "duplicate row_index"),
        ("invalid_index", "invalid row_index"),
        ("invalid_jsonl", "invalid JSONL"),
        ("non_object", "expected a JSON object"),
        ("model_id", "mismatched model_id"),
        ("model_role", "mismatched model_role"),
        ("mode", "mismatched mode"),
        ("embedded_row", "mismatched embedded input rows"),
        ("subset_categories", "full-category 'all'"),
        ("missing_labels", "full-category 'all'"),
        ("view", "full-category 'all'"),
        ("empty_label", "nonempty strings"),
        ("completion", "completion must be a string"),
    ],
)
def test_invalid_or_incomplete_inputs_are_rejected(tmp_path, problem, message):
    _write_evaluation(tmp_path, [_row({"Document Name": ["alpha"]})])
    comparison_path = tmp_path / "comparison.json"
    raw_path = tmp_path / "sft.raw.jsonl"
    records = _read_records(raw_path)
    if problem == "missing_comparison":
        comparison_path.unlink()
    elif problem.startswith("comparison_"):
        comparison = json.loads(comparison_path.read_text())
        if problem == "comparison_rows":
            comparison["sft"]["rows"] = 2
        else:
            comparison["mode"] = "singleton"
        comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    elif problem == "invalid_jsonl":
        raw_path.write_text('{"row_index":', encoding="utf-8")
    else:
        if problem == "missing_rows":
            records = []
        elif problem == "extra_rows":
            extra = copy.deepcopy(records[0])
            extra["row_index"] = 2
            records.append(extra)
        elif problem == "duplicate_index":
            records.append(copy.deepcopy(records[0]))
        elif problem == "invalid_index":
            records[0]["row_index"] = True
        elif problem == "non_object":
            records = [[]]
        elif problem in ("model_id", "model_role", "mode"):
            records[0][problem] = "mismatch"
        elif problem == "embedded_row":
            records[0]["row"]["contract"] = "different input"
        elif problem == "completion":
            records[0]["completion"] = None
        else:
            row = records[0]["row"]
            if problem == "subset_categories":
                row["categories"] = ["Document Name"]
            elif problem == "missing_labels":
                del row["labels"]["Agreement Date"]
            elif problem == "view":
                row["view"] = "singleton"
            elif problem == "empty_label":
                row["labels"]["Document Name"] = [""]
            vanilla_path = tmp_path / "vanilla.raw.jsonl"
            vanilla_records = _read_records(vanilla_path)
            vanilla_records[0]["row"] = copy.deepcopy(row)
            _write_records(vanilla_path, vanilla_records)
        _write_records(raw_path, records)

    with pytest.raises(ValueError, match=message):
        span_eval.score_evaluation(tmp_path)
    assert not list(tmp_path.glob("span_transitions.*"))


def test_mixed_checkpoints_are_rejected(tmp_path):
    _write_evaluation(tmp_path, [_row(), _row(chunk_index=1)])
    path = tmp_path / "sft.raw.jsonl"
    records = _read_records(path)
    records[1]["checkpoint"] = "/different/adapter"
    _write_records(path, records)
    with pytest.raises(ValueError, match="mismatched checkpoint"):
        span_eval.score_evaluation(tmp_path)


def test_cli_writes_new_reports_and_preserves_existing_inputs(tmp_path, capsys):
    row, vanilla, sft = _eight_cell_inputs()
    _write_evaluation(tmp_path, [row], [vanilla], [sft])
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    span_eval.main(["--evaluation-dir", str(tmp_path)])

    assert json.loads(capsys.readouterr().out)["support"]["total"] == 328
    for name, content in original.items():
        assert (tmp_path / name).read_bytes() == content
    assert {path.name for path in tmp_path.iterdir()} - set(original) == {
        "span_transitions.json",
        "span_transitions.md",
        "span_transitions.png",
    }
    report = json.loads((tmp_path / "span_transitions.json").read_text())
    assert report["metadata"]["matching"] == {"iou_threshold": 0.15, "parties_substring_exception": True}
    markdown = (tmp_path / "span_transitions.md").read_text()
    assert "## Gold-positive cases (Y = 1)" in markdown
    assert "## Gold-negative cases (Y = 0)" in markdown
    assert "| Predict 1 | a = 1 | b = 1 |" in markdown
    assert "| Predict 1 | h = 1 | g = 1 |" in markdown
    assert "| FN recovery | c/(c+d) | 1 / 6 | 16.67% |" in markdown
    assert "| TP loss | b/(a+b) | 1 / 2 | 50.00% |" in markdown
    assert "| New FP introduction | f/(e+f) | 1 / 318 | 0.31% |" in markdown
    assert "![Overall span transition rates](span_transitions.png)" in markdown


def test_cli_rejects_incomplete_capture_without_writing_reports(tmp_path, capsys):
    _write_evaluation(tmp_path, [_row()])
    (tmp_path / "sft.raw.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        span_eval.main(["--evaluation-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert "incomplete coverage" in capsys.readouterr().err
    assert not list(tmp_path.glob("span_transitions.*"))


@pytest.fixture
def captured_figure(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    captured = {}
    original_savefig = Figure.savefig

    def capture(fig, *args, **kwargs):
        captured["figure"] = fig
        original_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return captured


def test_plot_order_values_colors_labels_and_dimensions(tmp_path, captured_figure):
    from matplotlib.colors import to_hex
    from PIL import Image

    report = {
        "aggregate": span_eval._summarize_counts(
            {
                "a": 60,
                "b": 20,
                "c": 15,
                "d": 5,
                "e": 90,
                "f": 10,
                "g": 3,
                "h": 7,
            }
        )
    }
    path = tmp_path / "plot.png"
    span_eval._plot_rates(report, path)
    fig = captured_figure["figure"]
    ax = fig.axes[0]

    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "FN recovery",
        "FP correction",
        "TP loss",
        "New FP introduction",
    ]
    assert [bar.get_height() for bar in ax.patches] == pytest.approx([75, 30, 25, 10])
    assert [to_hex(bar.get_facecolor()) for bar in ax.patches] == [
        "#0072b2",
        "#0072b2",
        "#d55e00",
        "#d55e00",
    ]
    assert [text.get_text() for text in ax.texts] == [
        "75.00%\n15/20",
        "30.00%\n3/10",
        "25.00%\n20/80",
        "10.00%\n10/100",
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Improvements", "Regressions"]
    assert ax.get_ylim() == (0, 100)
    assert ax.get_ylabel() == "Rate (%)"
    assert ax.get_title() == "Vanilla → SFT span transitions"
    assert fig.get_size_inches().tolist() == [10, 5]
    with Image.open(path) as image:
        assert image.size == (1800, 900)


def test_zero_denominators_are_null_and_na_without_numeric_heights(tmp_path, captured_figure):
    _write_evaluation(tmp_path, [_row()])
    report = span_eval.score_evaluation(tmp_path)
    assert report["aggregate"]["support"] == {"gold_positive": 0, "gold_negative": 0, "total": 0}
    for rate in report["aggregate"]["rates"].values():
        assert rate == {"numerator": 0, "denominator": 0, "value": None}

    span_eval.write_reports(report, tmp_path)

    ax = captured_figure["figure"].axes[0]
    assert len(ax.patches) == 4
    assert all(math.isnan(bar.get_height()) for bar in ax.patches)
    assert [text.get_text() for text in ax.texts] == ["N/A\n0/0"] * 4
    assert (tmp_path / "span_transitions.md").read_text().count("| N/A |") == 4
    content = (tmp_path / "span_transitions.json").read_text()
    assert '"value": null' in content
    assert "NaN" not in content
