import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import cuad_sft_data  # noqa: E402
import evaluate_cuad_sft  # noqa: E402
from train_cuad_sft import WalltimeLoggingCallback, _tokenize_pair  # noqa: E402


def test_json_target_preserves_requested_field_order():
    target = cuad_sft_data.build_json_target(
        ["Agreement Date", "Document Name"],
        {
            "Agreement Date": ["March 27, 2020."],
            "Document Name": [],
        },
    )

    assert target == '{"Agreement Date":["March 27, 2020."],"Document Name":[]}\n---'


def test_view_sampler_is_deterministic_and_uses_unique_categories():
    categories = ["A", "B", "C", "D", "E"]
    spec = cuad_sft_data.ViewSpec("small", 2, 4)
    first = cuad_sft_data._ordered_view_categories(
        categories,
        spec,
        __import__("numpy").random.default_rng(42),
        0,
    )
    second = cuad_sft_data._ordered_view_categories(
        categories,
        spec,
        __import__("numpy").random.default_rng(42),
        0,
    )

    assert first == second
    assert 2 <= len(first[0]) <= 4
    assert len(set(first[0])) == len(first[0])


def test_prompt_factory_provided_order_is_used_for_sft():
    messages = cuad_sft_data.render_cuad_messages(
        "Agreement Date: March 27, 2020.",
        ["Agreement Date", "Document Name"],
    )
    prompt = "\n".join(message["content"] for message in messages if message["role"] == "user")
    assert prompt.index("- Agreement Date:") < prompt.index("- Document Name:")


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        del tokenize
        ids = []
        for message in messages:
            if message["role"] == "assistant":
                ids.append(999)
            ids.extend([len(message["role"]), len(message["content"])])
        if add_generation_prompt:
            ids.append(999)
        return ids


def test_token_mask_supervises_only_assistant_completion():
    input_ids, labels = _tokenize_pair(
        _FakeTokenizer(),
        {
            "id": "example",
            "messages": [{"role": "user", "type": "text", "content": "contract"}],
            "response": "{}\n---",
        },
        max_seq_length=32,
    )

    assert len(input_ids) == len(labels)
    assert labels[:3] == [-100, -100, -100]
    assert any(label != -100 for label in labels[3:])


def _sample_metrics(role: str) -> dict:
    return {
        "model_role": role,
        "model_id": "Qwen/Qwen3.5-4B",
        "precision": 0.5 if role == "vanilla" else 0.75,
        "recall": 0.4 if role == "vanilla" else 0.6,
        "f1": 0.444 if role == "vanilla" else 0.666,
        "tp": 2,
        "fp": 2 if role == "vanilla" else 1,
        "fn": 3 if role == "vanilla" else 2,
        "parse_success_rate": 0.9 if role == "vanilla" else 1.0,
        "grounding_rate": 0.8 if role == "vanilla" else 0.95,
        "missing_key_rate": 0.1 if role == "vanilla" else 0.0,
        "extra_key_rate": 0.05 if role == "vanilla" else 0.01,
        "per_category": {
            "Agreement Date": {
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5 if role == "vanilla" else 0.8,
                "tp": 1,
                "fp": 1,
                "fn": 1,
            },
        },
    }


def test_evaluator_cli_has_no_comparison_or_plot_directory_flags():
    option_strings = {
        option
        for action in evaluate_cuad_sft.build_parser()._actions
        for option in action.option_strings
    }

    assert "--compare-baseline" not in option_strings
    assert "--plot-dir" not in option_strings


def test_metric_delta_is_sft_minus_vanilla():
    delta = evaluate_cuad_sft._metric_delta(
        _sample_metrics("vanilla"),
        _sample_metrics("sft"),
    )

    assert delta["f1"] == pytest.approx(0.222)
    assert delta["per_category"]["Agreement Date"]["f1"] == pytest.approx(0.3)


def test_adapter_comparison_evaluates_same_rows_in_both_modes(monkeypatch, tmp_path):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    calls = []

    monkeypatch.setattr(evaluate_cuad_sft, "_load_adapter_model", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(evaluate_cuad_sft, "_release_model", lambda model: None)

    def fake_evaluate(model, tokenizer, rows, max_new_tokens, model_role, model_id, mode=None, adapter_enabled=None, **kwargs):
        del kwargs
        calls.append((model_role, adapter_enabled, id(rows), mode))
        return evaluate_cuad_sft.EvaluationResult(_sample_metrics(model_role), [])

    monkeypatch.setattr(evaluate_cuad_sft, "_evaluate_rows", fake_evaluate)
    rows = [{"id": "row-1"}]
    result = evaluate_cuad_sft._run_comparison(
        "Qwen/Qwen3.5-4B",
        checkpoint,
        rows,
        "all",
        128,
        False,
    )

    assert calls == [("vanilla", False, id(rows), "all"), ("sft", True, id(rows), "all")]
    assert result["delta"]["f1"] == pytest.approx(0.222)


def test_merged_comparison_releases_vanilla_before_sft(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(evaluate_cuad_sft, "_load_base_model", lambda *args, **kwargs: calls.append("base") or object())
    monkeypatch.setattr(evaluate_cuad_sft, "_load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluate_cuad_sft, "_load_merged_model", lambda *args, **kwargs: calls.append("merged") or (object(), object()))
    monkeypatch.setattr(evaluate_cuad_sft, "_release_model", lambda model: calls.append("release"))
    monkeypatch.setattr(
        evaluate_cuad_sft,
        "_evaluate_rows",
        lambda *args, **kwargs: evaluate_cuad_sft.EvaluationResult(_sample_metrics(args[4]), []),
    )

    result = evaluate_cuad_sft._run_comparison(
        "Qwen/Qwen3.5-4B",
        tmp_path / "merged",
        [{"id": "row-1"}],
        "all",
        128,
        False,
    )

    assert calls == ["base", "release", "merged", "release"]
    assert result["sft"]["model_role"] == "sft"


def test_comparison_reports_and_plots_are_written(tmp_path):
    comparison = {
        "mode": "all",
        "rows": 1,
        "vanilla": _sample_metrics("vanilla"),
        "sft": _sample_metrics("sft"),
        "delta": evaluate_cuad_sft._metric_delta(_sample_metrics("vanilla"), _sample_metrics("sft")),
    }
    output = tmp_path / "comparison.json"
    plot_dir = tmp_path / "sft_plots"

    evaluate_cuad_sft._write_reports(comparison, output, plot_dir)

    assert (tmp_path / "vanilla.json").exists()
    assert (tmp_path / "sft.json").exists()
    assert output.exists()
    assert {
        "aggregate_metrics.png",
        "per_category_precision.png",
        "per_category_recall.png",
        "per_category_f1.png",
        "per_category_f1_delta.png",
        "format_metrics.png",
    } == {path.name for path in plot_dir.iterdir()}


def test_training_walltime_callback_logs_checkpoints(tmp_path):
    log_path = tmp_path / "training_walltime.jsonl"
    callback = WalltimeLoggingCallback(log_path)
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=0, epoch=0.0)
    control = SimpleNamespace()

    callback.on_train_begin(args, state, control)
    state.global_step = 10
    state.epoch = 1.0
    callback.on_save(args, state, control)
    callback.on_train_end(args, state, control)

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["train_start", "checkpoint", "train_end"]
    assert events[1]["checkpoint"].endswith("checkpoint-10")
    assert events[1]["elapsed_seconds"] >= 0


def test_inference_walltime_is_logged_periodically(monkeypatch, tmp_path):
    log_path = tmp_path / "inference_walltime.jsonl"
    logger = evaluate_cuad_sft.InferenceWalltimeLogger(log_path)
    monkeypatch.setattr(evaluate_cuad_sft, "_completion_for_row", lambda *args, **kwargs: '{"Document Name":[]}')
    rows = [
        {
            "id": "row-1",
            "categories": ["Document Name"],
            "labels": {"Document Name": []},
            "contract": "contract",
        },
        {
            "id": "row-2",
            "categories": ["Document Name"],
            "labels": {"Document Name": []},
            "contract": "contract",
        },
    ]

    try:
        result = evaluate_cuad_sft._evaluate_rows(
            object(),
            object(),
            rows,
            8,
            "vanilla",
            "Qwen/Qwen3.5-4B",
            mode="all",
            timing_logger=logger,
            timing_log_every=1,
        )
    finally:
        logger.close()

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "inference_start",
        "inference_progress",
        "inference_progress",
    ]
    assert result.metrics["inference_walltime_seconds"] >= 0
    assert result.metrics["rows_per_second"] >= 0
