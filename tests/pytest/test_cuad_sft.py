import json
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import cuad_sft_data  # noqa: E402
import evaluate_cuad_sft  # noqa: E402
import train_cuad_sft  # noqa: E402
from train_cuad_sft import DEFAULT_OUTPUT_DIR, TrainConfig, WalltimeLoggingCallback, _tokenize_pair  # noqa: E402


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


def test_training_cli_smoke_controls_and_output_default():
    args = train_cuad_sft.parse_args(["--max-steps", "1", "--eval-limit", "1"])

    assert args.max_steps == 1
    assert args.eval_limit == 1
    assert args.output_dir == DEFAULT_OUTPUT_DIR

    explicit = train_cuad_sft.parse_args(["--output-dir", "/tmp/cuad-smoke"])
    assert explicit.output_dir == Path("/tmp/cuad-smoke")

    config = TrainConfig.from_mapping({"model": "Qwen/Qwen3.5-4B"})
    assert config.output_dir == DEFAULT_OUTPUT_DIR


def test_sft_yaml_does_not_define_output_directory():
    config_text = (FINETUNING_DIR / "sft_config.yaml").read_text(encoding="utf-8")
    assert "output:" not in config_text
    assert "cuad-sft-run" not in config_text


def test_evaluator_cli_rejects_non_positive_smoke_values():
    parser = evaluate_cuad_sft.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "run", "--checkpoint", "checkpoint", "--data-dir", "data", "--limit", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "run", "--checkpoint", "checkpoint", "--data-dir", "data", "--max-new-tokens", "0"])


def test_training_passes_smoke_limits_to_trainer(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = '{"id":"row"}\n'
    (data_dir / "train.jsonl").write_text(rows * 2, encoding="utf-8")
    (data_dir / "dev.jsonl").write_text(rows * 3, encoding="utf-8")

    captured = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class FakeTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def train(self, resume_from_checkpoint=None):
            captured["resume_from_checkpoint"] = resume_from_checkpoint
            return SimpleNamespace(metrics={})

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    import transformers

    monkeypatch.setattr(transformers, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        train_cuad_sft,
        "_load_model_and_tokenizer",
        lambda config: (object(), FakeTokenizer(), ["q_proj"]),
    )
    monkeypatch.setattr(train_cuad_sft.CuadTrainer, "build_base", staticmethod(lambda: FakeTrainer))

    config = TrainConfig(data_dir=data_dir, output_dir=tmp_path / "output")
    train_cuad_sft.train(config, max_steps=1, eval_limit=1)

    assert captured["training_args"]["max_steps"] == 1
    assert len(captured["trainer"]["train_dataset"]) == 2
    assert len(captured["trainer"]["eval_dataset"]) == 1


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
    monkeypatch.setattr(
        evaluate_cuad_sft, "_completion_for_row",
        lambda *args, **kwargs: evaluate_cuad_sft.RowCompletion('{"Document Name":[]}', {}),
    )
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


@pytest.fixture
def raw_generation():
    torch = pytest.importorskip("torch")
    from transformers import BatchEncoding

    class Tokenizer:
        pad_token_id = 0
        special_tokens = {1: "<|im_start|>", 2: "<|im_end|>"}

        def __init__(self, source):
            self.name_or_path = source
            self.decode_calls = []

        @staticmethod
        def encode(text):
            return [ord(char) + 10 for char in text]

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "enable_thinking": False,
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }
            ids = []
            for message in messages:
                ids.extend([1, *self.encode(message["role"] + "\n" + message["content"]), 2])
            ids.extend([1, *self.encode("assistant\n")])
            input_ids = torch.tensor([ids])
            return BatchEncoding({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=None):
            self.decode_calls.append((skip_special_tokens, clean_up_tokenization_spaces))
            return "".join(
                self.special_tokens[token] if token in self.special_tokens else chr(token - 10)
                for token in ids.tolist()
                if not (skip_special_tokens and token in self.special_tokens)
            )

    class Model:
        device = "cpu"

        def __init__(self, tokenizer, outputs):
            self.tokenizer = tokenizer
            self.outputs = iter(outputs)
            self.generation_calls = []
            self.adapter_enabled = True

        @contextmanager
        def disable_adapter(self):
            previous = self.adapter_enabled
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = previous

        def generate(self, **kwargs):
            self.generation_calls.append({**kwargs, "adapter_enabled": self.adapter_enabled})
            output = next(self.outputs)
            if isinstance(output, Exception):
                raise output
            completion_ids = torch.tensor([self.tokenizer.encode(output) + [2]])
            return torch.cat((kwargs["input_ids"], completion_ids), dim=-1)

    def make(outputs, source="fake/tokenizer"):
        tokenizer = Tokenizer(source)
        return Model(tokenizer, outputs), tokenizer

    return make


@pytest.fixture
def raw_file_handles(monkeypatch):
    original_open = Path.open
    handles = []

    def tracking_open(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        if path.name.endswith(".raw.jsonl") and mode == "w":
            handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)
    return handles


def _raw_evaluation_rows():
    rows = []
    for chunk_index, contract in enumerate(("Café contract", "Second contract")):
        row = {
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
            "categories": ["Document Name", "Agreement Date"],
            "labels": {"Document Name": [contract.split()[0]], "Agreement Date": []},
            "messages": [{"role": "user", "content": "Original unsorted prompt: " + contract}],
        }
        row["response"] = cuad_sft_data.build_json_target(row["categories"], row["labels"])
        rows.append(row)
    return rows


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluator_cli_raw_output_capture_is_opt_in(enabled):
    argv = ["--run-dir", "run", "--data-dir", "data"]
    if enabled:
        argv.append("--save-raw-outputs")
    assert evaluate_cuad_sft.build_parser().parse_args(argv).save_raw_outputs is enabled


@pytest.mark.parametrize("save_raw_outputs", [False, True])
def test_raw_capture_decodes_one_generation_without_changing_parser_text(raw_generation, save_raw_outputs):
    response = ' \nPreamble β\n{"Document Name":["Café"]}\n---  \n'
    model, tokenizer = raw_generation([response])
    result = evaluate_cuad_sft._completion_for_row(
        model, tokenizer, _raw_evaluation_rows()[0], 128, save_raw_outputs=save_raw_outputs,
    )

    assert len(model.generation_calls) == 1
    call = model.generation_calls[0]
    assert result.generation_kwargs == {"max_new_tokens": 128, "do_sample": False, "pad_token_id": 0}
    assert all(call[key] == value for key, value in result.generation_kwargs.items())
    assert call["attention_mask"].shape == call["input_ids"].shape
    assert result.completion == response
    if save_raw_outputs:
        assert result.raw_completion == response + "<|im_end|>"
        assert result.prompt == "<|im_start|>user\nOriginal unsorted prompt: Café contract<|im_end|><|im_start|>assistant\n"
        assert tokenizer.decode_calls == [(True, None), (False, False), (False, False)]
    else:
        assert result.raw_completion is None
        assert result.prompt is None
        assert tokenizer.decode_calls == [(True, None)]


@pytest.mark.parametrize("checkpoint_kind", ["adapter", "merged"])
def test_raw_comparison_records_preserve_rows_provenance_and_metrics(
    monkeypatch, tmp_path, raw_generation, raw_file_handles, checkpoint_kind,
):
    checkpoint = tmp_path / checkpoint_kind
    checkpoint.mkdir()
    if checkpoint_kind == "adapter":
        (checkpoint / "adapter_config.json").write_text("{}")
    original_rows = _raw_evaluation_rows()
    rows = evaluate_cuad_sft._prepare_rows(original_rows, "canonical", None)
    outputs = [' \n{"Document Name":["Café"],"Agreement Date":[]}\n---  \n', "Malformed {\nβ"]
    loaded = []

    def load(outputs, source):
        model, tokenizer = raw_generation(outputs, source)
        loaded.append((model, tokenizer))
        return model, tokenizer

    monkeypatch.setattr(evaluate_cuad_sft, "_release_model", lambda model: None)
    monkeypatch.setattr(
        evaluate_cuad_sft, "_load_adapter_model",
        lambda *args: load(outputs * 2, str(checkpoint)),
    )
    monkeypatch.setattr(evaluate_cuad_sft, "_load_base_model", lambda *args: load(outputs, "fake/base")[0])
    monkeypatch.setattr(evaluate_cuad_sft, "_load_tokenizer", lambda *args: loaded[-1][1])
    monkeypatch.setattr(evaluate_cuad_sft, "_load_merged_model", lambda *args: load(outputs, str(checkpoint)))

    disabled = evaluate_cuad_sft._run_comparison("fake/base", checkpoint, rows, "canonical", 128, False)
    assert not list(tmp_path.rglob("*.raw.jsonl"))
    assert all(tokenizer.decode_calls == [(True, None)] * len(model.generation_calls) for model, tokenizer in loaded)
    loaded.clear()
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    for role in ("vanilla", "sft"):
        (output_dir / f"{role}.raw.jsonl").write_text("stale record\n")
    raw_file_handles.clear()
    enabled = evaluate_cuad_sft._run_comparison(
        "fake/base", checkpoint, rows, "canonical", 128, False, raw_output_dir=output_dir,
    )

    assert sum(len(model.generation_calls) for model, _ in loaded) == 2 * len(rows)
    assert len(raw_file_handles) == 2
    assert all(handle.closed for handle in raw_file_handles)
    timing_keys = {"inference_walltime_seconds", "average_row_seconds", "rows_per_second"}
    for role in ("vanilla", "sft"):
        path = output_dir / f"{role}.raw.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [record["row_index"] for record in records] == [1, 2]
        assert [record["row"] for record in records] == rows
        assert rows[0]["id"] == rows[1]["id"]
        assert original_rows[0]["categories"] == ["Document Name", "Agreement Date"]
        for record, response in zip(records, outputs):
            assert record["model_role"] == role
            assert record["model_id"] == "fake/base"
            assert record["checkpoint"] == (str(checkpoint) if role == "sft" else None)
            expected_source = "fake/base" if checkpoint_kind == "merged" and role == "vanilla" else str(checkpoint)
            assert record["tokenizer_source"] == expected_source
            assert record["mode"] == "canonical"
            assert record["generation_kwargs"] == {"max_new_tokens": 128, "do_sample": False, "pad_token_id": 0}
            assert record["enable_thinking"] is False
            assert record["row"]["categories"] == ["Agreement Date", "Document Name"]
            assert "Original unsorted prompt" not in record["prompt"]
            assert record["row"]["messages"][-1]["content"] in record["prompt"]
            assert record["prompt"].endswith("<|im_start|>assistant\n")
            assert record["raw_completion"] == response + "<|im_end|>"
            assert record["completion"] == response
        assert "Café" in path.read_text(encoding="utf-8")
        assert enabled[role]["parse_failures"] == 1
        assert enabled[role]["tp"] == 1
        assert enabled[role]["fn"] == 1
        assert {key: value for key, value in enabled[role].items() if key not in timing_keys} == {
            key: value for key, value in disabled[role].items() if key not in timing_keys
        }
    assert enabled["delta"]["f1"] == disabled["delta"]["f1"]
    if checkpoint_kind == "adapter":
        assert [call["adapter_enabled"] for call in loaded[0][0].generation_calls] == [False, False, True, True]


@pytest.mark.parametrize("failure", ["generation", "evaluation"])
def test_raw_records_are_flushed_before_parsing_and_files_close_after_failure(
    monkeypatch, tmp_path, raw_generation, raw_file_handles, failure,
):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "sft.raw.jsonl").write_text("stale SFT output\n")
    raw_file_handles.clear()
    model, tokenizer = raw_generation(["{}", RuntimeError("generation failed")])
    monkeypatch.setattr(evaluate_cuad_sft, "_load_adapter_model", lambda *args: (model, tokenizer))
    original_parser = evaluate_cuad_sft.get_json_from_answer
    parser_calls = []

    def parse_after_flush(*args):
        lines = (output_dir / "vanilla.raw.jsonl").read_text().splitlines()
        assert len(lines) == len(parser_calls) + 1
        assert json.loads(lines[-1])["completion"] == args[0]
        parser_calls.append(args[0])
        return original_parser(*args)

    monkeypatch.setattr(evaluate_cuad_sft, "get_json_from_answer", parse_after_flush)
    if failure == "evaluation":
        def fail_evaluation(*args):
            raise RuntimeError("evaluation failed")

        monkeypatch.setattr(evaluate_cuad_sft, "evaluate_entry", fail_evaluation)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        evaluate_cuad_sft._run_comparison(
            "fake/base", checkpoint, _raw_evaluation_rows(), "all", 128, False, raw_output_dir=output_dir,
        )

    assert parser_calls == ["{}"]
    assert len((output_dir / "vanilla.raw.jsonl").read_text().splitlines()) == 1
    assert (output_dir / "sft.raw.jsonl").read_text() == ""
    assert len(raw_file_handles) == 2
    assert all(handle.closed for handle in raw_file_handles)


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_raw_output_io_errors_propagate_and_close_files(monkeypatch, tmp_path, raw_generation, operation):
    class FailingOutput(StringIO):
        def write(self, value):
            if operation == "write":
                raise OSError("raw write failed")
            return super().write(value)

        def flush(self):
            if operation == "flush":
                raise OSError("raw flush failed")
            return super().flush()

    writer = FailingOutput()
    original_open = Path.open

    def fail_raw_output(path, *args, **kwargs):
        if path.name == "vanilla.raw.jsonl":
            return writer
        return original_open(path, *args, **kwargs)

    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    model, tokenizer = raw_generation(["{}"])
    monkeypatch.setattr(Path, "open", fail_raw_output)
    monkeypatch.setattr(evaluate_cuad_sft, "_load_adapter_model", lambda *args: (model, tokenizer))
    with pytest.raises(OSError, match=f"raw {operation} failed"):
        evaluate_cuad_sft._run_comparison(
            "fake/base", checkpoint, _raw_evaluation_rows(), "all", 128, False, raw_output_dir=tmp_path,
        )
    assert writer.closed
