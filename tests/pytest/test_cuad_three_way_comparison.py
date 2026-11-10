import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import compare_cuad_inference as comparison  # noqa: E402

LOCAL_MODEL = "Qwen/test-model"
API_MODEL = "openai/test-model"


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.fixture
def saved_results(tmp_path):
    vanilla_sft_dir = tmp_path / "vanilla-sft"
    api_result_dir = tmp_path / "api"
    vanilla_sft_dir.mkdir()
    api_result_dir.mkdir()
    row = {
        "id": "shared-chunk-id",
        "split": "test",
        "source_hash": "hash",
        "contract_id": "contract-1",
        "contract": "Alpha Corporation agreement dated June 2020.",
        "chunk_index": 0,
        "view": "all",
        "categories": ["Document Name", "Agreement Date"],
        "labels": {"Document Name": ["Alpha Corporation"], "Agreement Date": ["June 2020"]},
        "messages": [{"role": "user", "content": "Analyze contract"}],
    }
    completions = {
        "vanilla": '{"Document Name":["Alpha Corporation"],"Agreement Date":[]}',
        "sft": '{"Document Name":["Alpha Corporation"],"Agreement Date":["June 2020"]}',
        "api": '{"Document Name":["Other Corporation"],"Agreement Date":["June 2020"]}',
    }
    vanilla_timing = {
        "model_id": LOCAL_MODEL,
        "mode": "all",
        "rows": 1,
        "inference_walltime_seconds": 10.0,
        "average_row_seconds": 10.0,
        "rows_per_second": 0.1,
    }
    sft_timing = {
        "model_id": LOCAL_MODEL,
        "mode": "all",
        "rows": 1,
        "inference_walltime_seconds": 8.0,
        "average_row_seconds": 8.0,
        "rows_per_second": 0.125,
    }
    api_timing = {
        "model_id": API_MODEL,
        "mode": "all",
        "rows": 1,
        "inference_walltime_seconds": 6.0,
        "average_row_seconds": 6.0,
        "rows_per_second": 1 / 6,
    }
    _write_json(
        vanilla_sft_dir / "comparison.json",
        {
            "vanilla": vanilla_timing,
            "sft": sft_timing,
            "rows": 1,
            "model_id": LOCAL_MODEL,
            "mode": "all",
            "data_split": "test",
        },
    )
    _write_json(
        api_result_dir / "openrouter.json",
        {
            "status": "complete",
            "rows": 1,
            "responses_completed": 1,
            "model_id": API_MODEL,
            "mode": "all",
            "data_split": "test",
            "metrics": api_timing,
        },
    )
    for role in ("vanilla", "sft"):
        _write_jsonl(
            vanilla_sft_dir / f"{role}.raw.jsonl",
            [
                {
                    "row_index": 1,
                    "model_role": role,
                    "model_id": LOCAL_MODEL,
                    "checkpoint": "adapter" if role == "sft" else None,
                    "mode": "all",
                    "generation_kwargs": {},
                    "row": row,
                    "completion": completions[role],
                }
            ],
        )
    _write_jsonl(
        api_result_dir / "openrouter.raw.jsonl",
        [
            {
                "row_index": 1,
                "model_role": "openrouter",
                "model_id": API_MODEL,
                "checkpoint": None,
                "mode": "all",
                "generation_kwargs": {},
                "row": row,
                "completion": completions["api"],
            }
        ],
    )
    return vanilla_sft_dir, api_result_dir


def test_compare_recomputes_metrics_preserves_inference_timing_and_writes_plots(tmp_path, saved_results):
    vanilla_sft_dir, api_result_dir = saved_results
    output_dir = tmp_path / "artifacts" / "cuad-three-way-comparison"

    result = comparison.compare_results(vanilla_sft_dir, api_result_dir, output_dir)

    assert result["vanilla"]["tp"] == 1 and result["vanilla"]["fn"] == 1
    assert result["sft"]["tp"] == 2 and result["sft"]["f1"] == 1.0
    assert result["api"]["tp"] == 1 and result["api"]["fp"] == 1
    assert result["vanilla"]["inference_walltime_seconds"] == 10.0
    assert result["sft"]["inference_walltime_seconds"] == 8.0
    assert result["api"]["inference_walltime_seconds"] == 6.0
    assert result["api_minus_sft"]["f1"] == result["api"]["f1"] - result["sft"]["f1"]
    assert result["api_minus_sft"]["per_category"]["Document Name"]["f1"] < 0
    assert "delta" not in result
    assert json.loads((output_dir / "comparison.json").read_text()) == result
    assert all((output_dir / f"{role}.json").is_file() for role in comparison.ROLES)
    assert {path.name for path in (output_dir / "sft_plots").iterdir()} == {
        "aggregate_metrics.png",
        "per_category_precision.png",
        "per_category_recall.png",
        "per_category_f1.png",
        "per_category_f1_delta.png",
        "format_metrics.png",
    }
    assert all(path.stat().st_size > 0 for path in (output_dir / "sft_plots").iterdir())


@pytest.mark.parametrize("failure", ["incomplete_api", "different_split", "different_input"])
def test_invalid_sources_fail_before_writing(tmp_path, saved_results, failure):
    vanilla_sft_dir, api_result_dir = saved_results
    if failure == "incomplete_api":
        summary_path = api_result_dir / "openrouter.json"
        summary = json.loads(summary_path.read_text())
        summary["status"] = "failed"
        _write_json(summary_path, summary)
    elif failure == "different_split":
        summary_path = api_result_dir / "openrouter.json"
        summary = json.loads(summary_path.read_text())
        summary["data_split"] = "dev"
        _write_json(summary_path, summary)
    else:
        raw_path = api_result_dir / "openrouter.raw.jsonl"
        record = json.loads(raw_path.read_text())
        record["row"]["contract"] = "A different contract"
        _write_jsonl(raw_path, [record])

    output_dir = tmp_path / "artifacts" / "cuad-three-way-comparison"
    with pytest.raises(ValueError):
        comparison.compare_results(vanilla_sft_dir, api_result_dir, output_dir)
    assert not output_dir.exists()


def test_cli_requires_exactly_two_directory_arguments(saved_results):
    vanilla_sft_dir, api_result_dir = saved_results
    parser = comparison.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([str(vanilla_sft_dir)])
    parsed = parser.parse_args([str(vanilla_sft_dir), str(api_result_dir)])
    assert parsed.vanilla_sft_dir == vanilla_sft_dir
    assert parsed.api_result_dir == api_result_dir
    with pytest.raises(SystemExit):
        parser.parse_args([str(vanilla_sft_dir), str(api_result_dir), "extra"])


def test_output_directory_is_directly_under_artifacts():
    assert (
        Path(comparison.__file__).resolve().parent / "artifacts" / "cuad-three-way-comparison"
    ) == comparison.OUTPUT_DIR
