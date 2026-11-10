import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import evaluate_cuad_sft as live_eval  # noqa: E402
import evaluate_cuad_sft_from_raw as raw_eval  # noqa: E402


@pytest.fixture
def saved_evaluation(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    rows = [
        {
            "id": "shared-chunk-id",
            "chunk_index": index,
            "view": "all",
            "contract": contract,
            "categories": ["Document Name", "Agreement Date"],
            "labels": labels,
        }
        for index, contract, labels in (
            (0, "Café alpha", {"Document Name": ["Café alpha"], "Agreement Date": []}),
            (1, "Beta contract 2024", {"Document Name": ["Beta contract"], "Agreement Date": ["2024"]}),
        )
    ]
    rows = live_eval._prepare_rows(rows, "canonical", None)
    outputs = {
        False: ['{"Document Name":["Café alpha"],"Unexpected":["hallucination"]}', "not JSON"],
        True: [
            '{"Document Name":"Café alpha","Agreement Date":null}',
            '{"Document Name":["Beta contract","hallucination"],"Agreement Date":["2024"]}',
        ],
    }

    def generate(model, tokenizer, row, max_new_tokens, adapter_enabled, **kwargs):
        return live_eval.RowCompletion(
            outputs[adapter_enabled][row["chunk_index"]],
            {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": 0},
            # Deliberately differs from completion to verify the evaluator text is replayed.
            raw_completion="unparseable raw token stream",
            prompt="saved prompt",
        )

    monkeypatch.setattr(live_eval, "_completion_for_row", generate)
    monkeypatch.setattr(
        live_eval, "_load_adapter_model", lambda *args: (object(), SimpleNamespace(name_or_path="saved/tokenizer")),
    )
    monkeypatch.setattr(live_eval, "_release_model", lambda model: None)
    logger = live_eval.InferenceWalltimeLogger(source / "inference_walltime.jsonl")
    try:
        comparison = live_eval._run_comparison(
            "saved/model", checkpoint, rows, "canonical", 16, False,
            timing_logger=logger, raw_output_dir=source,
        )
    finally:
        logger.close()
    comparison.update(
        mode="canonical", data_split="test", rows=len(rows), model_id="saved/model",
        inference_walltime_log=str(source / "inference_walltime.jsonl"),
    )
    (source / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    return source, comparison


def test_replay_matches_original_artifacts_and_all_numbers(monkeypatch, tmp_path, saved_evaluation):
    source, expected = saved_evaluation
    live_eval._write_reports(expected, source / "comparison.json", source / "sft_plots")
    original_bytes = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Replay attempted to load a model, tokenizer, dataset, or generate text")

    for name in ("_load_base_model", "_load_adapter_model", "_load_merged_model", "_load_tokenizer", "_load_rows", "_completion_for_row"):
        monkeypatch.setattr(live_eval, name, forbidden)

    destination = tmp_path / "replay"
    actual = raw_eval.replay(source, destination)
    expected = {**expected, "inference_walltime_log": str(destination / "inference_walltime.jsonl")}
    assert actual == expected
    assert (actual["vanilla"]["tp"], actual["vanilla"]["fp"], actual["vanilla"]["fn"]) == (1, 0, 2)
    assert (actual["sft"]["tp"], actual["sft"]["fp"], actual["sft"]["fn"]) == (3, 1, 0)
    assert actual["vanilla"]["parse_failures"] == 1
    assert actual["vanilla"]["extra_keys"] == 1
    assert json.loads((destination / "comparison.json").read_text()) == expected
    assert {path.relative_to(destination) for path in destination.rglob("*") if path.is_file()} == set(original_bytes)
    assert len(original_bytes) == 12
    for name, contents in original_bytes.items():
        assert (source / name).read_bytes() == contents
        if name != Path("comparison.json"):
            assert (destination / name).read_bytes() == contents


def test_scores_are_recomputed_and_custom_filenames_pair_by_index(tmp_path, saved_evaluation):
    source, expected = saved_evaluation
    tampered = json.loads((source / "comparison.json").read_text())
    tampered["vanilla"]["f1"] = -100
    tampered["sft"]["tp"] = -100
    tampered["delta"]["f1"] = -100
    (source / "comparison.json").write_text(json.dumps(tampered), encoding="utf-8")
    raw_paths = []
    for role in ("vanilla", "sft"):
        path = tmp_path / f"{role}_raw.jsonl"
        lines = (source / f"{role}.raw.jsonl").read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(reversed(lines) if role == "sft" else lines) + "\n", encoding="utf-8")
        raw_paths.append(path)
    destination = tmp_path / "replay"
    actual = raw_eval.replay(source, destination, *raw_paths)
    assert actual == {**expected, "inference_walltime_log": str(destination / "inference_walltime.jsonl")}
    for role, path in zip(("vanilla", "sft"), raw_paths, strict=True):
        assert (destination / f"{role}.raw.jsonl").read_bytes() == path.read_bytes()


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing_row", "incomplete coverage"),
        ("duplicate_index", "duplicate row_index"),
        ("extra_index", "extra rows"),
        ("invalid_index", "invalid row_index"),
        ("model_role", "mismatched model_role"),
        ("model_id", "mismatched model_id"),
        ("mode", "mismatched mode"),
        ("checkpoint", "mismatched checkpoint"),
        ("completion", "completion must be a string"),
        ("input", "embedded inputs differ"),
    ],
)
def test_invalid_captures_fail_before_writing(tmp_path, saved_evaluation, problem, message):
    source, _ = saved_evaluation
    path = source / "sft.raw.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if problem == "missing_row":
        records.pop()
    elif problem == "duplicate_index":
        records[1]["row_index"] = 1
    elif problem == "extra_index":
        records[1]["row_index"] = 3
    elif problem == "invalid_index":
        records[1]["row_index"] = True
    elif problem == "input":
        records[1]["row"]["contract"] = "different contract"
    elif problem == "completion":
        records[1]["completion"] = None
    else:
        records[1][problem] = "different metadata"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    destination = tmp_path / "replay"
    with pytest.raises(ValueError, match=message):
        raw_eval.replay(source, destination)
    assert not destination.exists()


def test_source_directory_cannot_be_overwritten(saved_evaluation):
    source, _ = saved_evaluation
    before = (source / "comparison.json").read_bytes()
    with pytest.raises(ValueError, match="different directory"):
        raw_eval.replay(source, source)
    assert (source / "comparison.json").read_bytes() == before
