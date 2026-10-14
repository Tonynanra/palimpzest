"""CPU-only coverage for model selection and run configuration handoff."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import cuad_sft_config  # noqa: E402
import evaluate_cuad_sft  # noqa: E402
import merge_cuad_sft_adapter  # noqa: E402
import serve_cuad_sft  # noqa: E402
import train_cuad_sft  # noqa: E402

MODEL_IDS = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.8-27B",
)


def _save_run(tmp_path, model_id=MODEL_IDS[0], **settings):
    config = cuad_sft_config.TrainConfig(
        model_id=model_id,
        output_dir=tmp_path / "run",
        **settings,
    )
    cuad_sft_config.save_config_snapshot(config)
    return config


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_yaml_selects_each_supported_model(tmp_path, model_id):
    source = tmp_path / "sft.yaml"
    source.write_text(f"model: {model_id}\n", encoding="utf-8")

    config = train_cuad_sft._load_config(source)

    assert config.model_id == model_id
    assert cuad_sft_config.SUPPORTED_MODEL_IDS == MODEL_IDS


@pytest.mark.parametrize(
    "mapping",
    [{}, {"model": None}, {"model": 9}, {"model": {"id": MODEL_IDS[0]}}, {"model": "other/model"}],
)
def test_invalid_yaml_model_requires_a_valid_cli_override(mapping):
    with pytest.raises(ValueError):
        cuad_sft_config.TrainConfig.from_mapping(mapping)

    config = cuad_sft_config.TrainConfig.from_mapping(mapping, model_id=MODEL_IDS[1])
    assert config.model_id == MODEL_IDS[1]


def test_invalid_cli_model_does_not_fall_back_to_yaml():
    with pytest.raises(ValueError):
        cuad_sft_config.TrainConfig.from_mapping({"model": MODEL_IDS[0]}, model_id="other/model")


def test_snapshot_records_effective_defaults_and_runtime_without_changing_source(tmp_path):
    source = tmp_path / "sft.yaml"
    source.write_text(
        f"model: {MODEL_IDS[0]}\ndata:\n  dir: custom/data\ntraining:\n  learning_rate: 0.0002\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    config = replace(
        train_cuad_sft._load_config(source, model_id=MODEL_IDS[1]),
        output_dir=tmp_path / "run",
    )
    cuad_sft_config.save_config_snapshot(config)
    checkpoint = config.output_dir / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (config.output_dir / "adapter").mkdir()

    snapshot = cuad_sft_config.save_config_snapshot(
        config,
        max_steps=5,
        eval_limit=2,
        resume_from_checkpoint=str(checkpoint),
    )
    saved = yaml.safe_load(snapshot.read_text(encoding="utf-8"))

    assert snapshot == config.output_dir / "config_snapshot.yaml"
    assert saved["model"] == MODEL_IDS[1]
    assert saved["data"]["dir"] == str(config.data_dir.resolve())
    assert saved["training"]["learning_rate"] == 0.0002
    assert saved["training"]["max_seq_length"] == config.max_seq_length
    assert saved["training"]["lora_rank"] == 32
    assert saved["training"]["flash_attention"] is True
    assert saved["runtime"] == {
        "output_dir": str(config.output_dir),
        "max_steps": 5,
        "eval_limit": 2,
        "resume_from_checkpoint": str(checkpoint),
    }
    assert source.read_bytes() == original
    assert list(config.output_dir.rglob("config_snapshot.yaml")) == [snapshot]
    assert (config.output_dir / "adapter").is_dir()
    assert checkpoint.is_dir()
    assert cuad_sft_config.load_run_config(config.output_dir) == replace(config, data_dir=config.data_dir.resolve())


def test_run_can_be_moved_without_retaining_original_output_directory(tmp_path):
    config = _save_run(tmp_path, MODEL_IDS[2])
    moved = tmp_path / "moved-run"
    config.output_dir.rename(moved)

    restored = cuad_sft_config.load_run_config(moved)

    assert restored.model_id == MODEL_IDS[2]
    assert restored.output_dir == moved


def test_run_configuration_requires_a_snapshot_at_the_run_root(tmp_path):
    adapter = tmp_path / "run" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "config_snapshot.yaml").write_text(f"model: {MODEL_IDS[0]}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        cuad_sft_config.load_run_config(adapter.parent)


def test_reusing_a_run_for_another_model_preserves_its_snapshot(tmp_path):
    config = _save_run(tmp_path)
    snapshot = config.output_dir / "config_snapshot.yaml"
    original = snapshot.read_bytes()

    with pytest.raises(ValueError, match="[Mm]odel"):
        cuad_sft_config.save_config_snapshot(replace(config, model_id=MODEL_IDS[1]))

    assert snapshot.read_bytes() == original


@pytest.mark.parametrize("new_run", [False, True])
def test_resume_model_mismatch_is_checked_before_writing(tmp_path, new_run):
    config = _save_run(tmp_path)
    snapshot = config.output_dir / "config_snapshot.yaml"
    original = snapshot.read_bytes()
    checkpoint = config.output_dir / "checkpoint-4"
    checkpoint.mkdir()
    output_dir = tmp_path / "new-run" if new_run else config.output_dir

    with pytest.raises(ValueError, match="[Mm]odel"):
        cuad_sft_config.save_config_snapshot(
            replace(config, model_id=MODEL_IDS[2], output_dir=output_dir),
            resume_from_checkpoint=str(checkpoint),
        )

    assert snapshot.read_bytes() == original
    if new_run:
        assert not (output_dir / "config_snapshot.yaml").exists()


@pytest.mark.parametrize("new_run", [False, True])
def test_matching_resume_model_records_runtime_controls(tmp_path, new_run):
    config = _save_run(tmp_path, MODEL_IDS[2])
    checkpoint = config.output_dir / "checkpoint-1"
    checkpoint.mkdir()

    output_dir = tmp_path / "new-run" if new_run else config.output_dir
    snapshot = cuad_sft_config.save_config_snapshot(
        replace(config, output_dir=output_dir),
        max_steps=3,
        resume_from_checkpoint=str(checkpoint),
    )

    assert yaml.safe_load(snapshot.read_text())["runtime"]["resume_from_checkpoint"] == str(checkpoint)
    assert list(output_dir.rglob("config_snapshot.yaml")) == [snapshot]


@pytest.mark.parametrize("failure_stage", ["data", "model"])
def test_failed_training_keeps_snapshot_before_data_or_model_loading(monkeypatch, tmp_path, failure_stage):
    config = cuad_sft_config.TrainConfig(model_id=MODEL_IDS[1], output_dir=tmp_path / "run")
    snapshot = config.output_dir / "config_snapshot.yaml"
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TrainingArguments=object, set_seed=lambda seed: None),
    )

    def fail_loading(*args, **kwargs):
        assert snapshot.exists()
        saved = yaml.safe_load(snapshot.read_text())
        assert saved["model"] == MODEL_IDS[1]
        assert saved["runtime"]["max_steps"] == 1
        assert saved["runtime"]["eval_limit"] == 2
        raise RuntimeError(f"failed {failure_stage} loading")

    if failure_stage == "data":
        monkeypatch.setattr(train_cuad_sft, "CuadJsonlDataset", fail_loading)
    else:
        monkeypatch.setattr(train_cuad_sft, "CuadJsonlDataset", lambda *args, **kwargs: [])
        monkeypatch.setattr(train_cuad_sft, "_load_model_and_tokenizer", fail_loading)

    with pytest.raises(RuntimeError, match=f"failed {failure_stage} loading"):
        train_cuad_sft.train(config, max_steps=1, eval_limit=2)

    assert list(config.output_dir.rglob("config_snapshot.yaml")) == [snapshot]


def test_training_main_forwards_cli_model_and_runtime_controls(monkeypatch, tmp_path):
    source = tmp_path / "sft.yaml"
    source.write_text(f"model: {MODEL_IDS[0]}\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    assert train_cuad_sft.parse_args([]).model_id is None
    args = train_cuad_sft.parse_args(
        [
            "--config", str(source),
            "--model-id", MODEL_IDS[2],
            "--output-dir", str(output_dir),
            "--max-steps", "2",
            "--eval-limit", "1",
        ]
    )
    captured = {}

    def fake_train(config, **kwargs):
        captured.update(config=config, controls=kwargs)
        return config.output_dir / "adapter"

    monkeypatch.setattr(train_cuad_sft, "parse_args", lambda: args)
    monkeypatch.setattr(train_cuad_sft, "train", fake_train)
    train_cuad_sft.main()

    assert captured["config"].model_id == MODEL_IDS[2]
    assert captured["config"].output_dir == output_dir
    assert captured["controls"] == {"max_steps": 2, "eval_limit": 1, "resume_from_checkpoint": None}
    assert yaml.safe_load(source.read_text())["model"] == MODEL_IDS[0]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_downstream_defaults_follow_the_saved_model_and_run_layout(tmp_path, model_id):
    config = _save_run(tmp_path, model_id, lora_rank=64, flash_attention=False)
    run = str(config.output_dir)
    evaluation = evaluate_cuad_sft.parse_args(["--run-dir", run, "--data-dir", "test/data"])
    merge = merge_cuad_sft_adapter.parse_args(["--run-dir", run])
    serve = serve_cuad_sft.parse_args(["--run-dir", run])

    assert evaluation.model_id == merge.model_id == serve.model_id == model_id
    assert evaluation.checkpoint == merge.adapter_dir == serve.adapter_dir == config.output_dir / "adapter"
    assert evaluation.output == config.output_dir / "comparison.json"
    assert merge.output_dir == config.output_dir / "merged"
    assert evaluation.flash_attention is False
    assert serve.max_lora_rank == 64
    alias = "cuad-" + model_id.split("/")[1].lower().replace(".", "")
    assert serve.served_model_name == serve.adapter_name == alias


@pytest.mark.parametrize("module", [evaluate_cuad_sft, merge_cuad_sft_adapter, serve_cuad_sft])
def test_downstream_cli_requires_run_and_has_no_independent_model_selection(module):
    parser = module.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--model-id" not in options
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_downstream_artifact_and_serving_overrides_remain_available(tmp_path):
    config = _save_run(tmp_path, MODEL_IDS[1])
    run = str(config.output_dir)
    checkpoint = tmp_path / "checkpoint-7"
    output = tmp_path / "reports" / "comparison.json"
    evaluation = evaluate_cuad_sft.parse_args(
        ["--run-dir", run, "--data-dir", "test/data", "--checkpoint", str(checkpoint),
         "--output", str(output), "--no-flash-attention"]
    )
    merge = merge_cuad_sft_adapter.parse_args(
        ["--run-dir", run, "--adapter-dir", str(checkpoint), "--output-dir", str(tmp_path / "export")]
    )
    serve = serve_cuad_sft.parse_args(
        ["--run-dir", run, "--adapter-dir", str(checkpoint), "--adapter-name", "custom-adapter",
         "--served-model-name", "custom-base", "--max-lora-rank", "128"]
    )

    assert evaluation.checkpoint == merge.adapter_dir == serve.adapter_dir == checkpoint
    assert evaluation.output == output
    assert evaluation.flash_attention is False
    assert merge.output_dir == tmp_path / "export"
    command = serve_cuad_sft.build_command(serve)
    assert command[2] == MODEL_IDS[1]
    assert command[command.index("--lora-modules") + 1] == f"custom-adapter={checkpoint}"
    assert command[command.index("--served-model-name") + 1] == "custom-base"
    assert command[command.index("--max-lora-rank") + 1] == "128"
    assert yaml.safe_load(command[command.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}


def test_serving_merged_checkpoint_uses_explicit_export_and_no_thinking(tmp_path):
    config = _save_run(tmp_path, MODEL_IDS[2])
    merged = tmp_path / "merged-export"
    args = serve_cuad_sft.parse_args(["--run-dir", str(config.output_dir), "--merged-dir", str(merged)])

    command = serve_cuad_sft.build_command(args)

    assert command[2] == str(merged)
    assert "--enable-lora" not in command
    assert yaml.safe_load(command[command.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert list(config.output_dir.rglob("config_snapshot.yaml")) == [config.output_dir / "config_snapshot.yaml"]


def test_evaluation_main_uses_snapshot_after_source_config_changes(monkeypatch, tmp_path):
    source = tmp_path / "sft.yaml"
    source.write_text(f"model: {MODEL_IDS[1]}\ntraining:\n  flash_attention: false\n", encoding="utf-8")
    config = replace(cuad_sft_config.load_training_config(source), output_dir=tmp_path / "run")
    cuad_sft_config.save_config_snapshot(config)
    source.write_text(f"model: {MODEL_IDS[0]}\n", encoding="utf-8")
    argv = ["--run-dir", str(config.output_dir), "--data-dir", "holdout"]
    args = evaluate_cuad_sft.parse_args(argv)
    captured = {}

    def fake_comparison(*args, **kwargs):
        captured["comparison_args"] = args
        captured["comparison_kwargs"] = kwargs
        return {"vanilla": {}, "sft": {}, "delta": {}}

    def fake_reports(comparison, output, plot_dir):
        captured.update(report=comparison, output=output, plot_dir=plot_dir)

    monkeypatch.setattr(evaluate_cuad_sft, "parse_args", lambda: args)
    monkeypatch.setattr(evaluate_cuad_sft, "_load_rows", lambda *args: [])
    monkeypatch.setattr(evaluate_cuad_sft, "_run_comparison", fake_comparison)
    monkeypatch.setattr(evaluate_cuad_sft, "_write_reports", fake_reports)
    evaluate_cuad_sft.main()

    assert captured["comparison_args"][0] == MODEL_IDS[1]
    assert captured["comparison_args"][1] == config.output_dir / "adapter"
    assert captured["comparison_args"][5] is False
    assert captured["report"]["model_id"] == MODEL_IDS[1]
    assert captured["output"] == config.output_dir / "comparison.json"
    assert captured["plot_dir"] == config.output_dir / "sft_plots"
    assert (config.output_dir / "inference_walltime.jsonl").exists()


def test_merge_main_routes_saved_model_and_default_artifacts(monkeypatch, tmp_path):
    config = _save_run(tmp_path, MODEL_IDS[2])
    args = merge_cuad_sft_adapter.parse_args(["--run-dir", str(config.output_dir)])
    calls = []
    monkeypatch.setattr(merge_cuad_sft_adapter, "parse_args", lambda: args)
    monkeypatch.setattr(merge_cuad_sft_adapter, "merge_adapter", lambda *args: calls.append(args) or args[2])

    merge_cuad_sft_adapter.main()

    assert calls == [(MODEL_IDS[2], config.output_dir / "adapter", config.output_dir / "merged")]


def test_slurm_executes_with_its_fixed_model_and_run_directory(tmp_path):
    checkout = tmp_path / "checkout"
    interpreter = checkout / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    calls_path = tmp_path / "calls.jsonl"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with Path(os.environ['CUAD_STUB_LOG']).open('a') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "if args[0] == '-':\n"
        "    sys.stdin.read()\n"
        "elif args[0] == 'finetuning/train_cuad_sft.py':\n"
        "    adapter = Path(args[-1]) / 'adapter'\n"
        "    adapter.mkdir(parents=True)\n"
        "    (adapter / 'adapter_config.json').write_text('{}')\n"
        "elif args[0] == 'finetuning/evaluate_cuad_sft.py':\n"
        "    output = Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True)\n"
        "    output.write_text('{}')\n"
        "else:\n"
        "    raise SystemExit('unexpected pipeline command: ' + repr(args))\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    commands = tmp_path / "commands"
    commands.mkdir()
    nvidia_smi = commands / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nprintf 'NVIDIA H200, 9.0, 141312 MiB, GPU-stub\\n'\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    train_data = checkout / "testdata" / "cuad" / "sft" / "train-dev"
    test_data = checkout / "testdata" / "cuad" / "sft" / "test"
    for path in (train_data / "train.jsonl", train_data / "dev.jsonl", test_data / "test.jsonl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"id":"existing-row"}\n', encoding="utf-8")
    environment = os.environ | {
        "SLURM_SUBMIT_DIR": str(checkout),
        "SLURM_JOB_ID": "12345",
        "CUAD_STUB_LOG": str(calls_path),
        "PATH": str(commands) + os.pathsep + os.environ.get("PATH", ""),
    }
    earlier_output = tmp_path / "earlier-output"

    subprocess.run(
        [
            "bash", str(FINETUNING_DIR / "run_h200_cuad_sft.slurm"),
            "--model-id", MODEL_IDS[1], "--output-dir", str(earlier_output), "--max-steps", "1",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert [call[0] for call in calls] == [
        "-", "finetuning/train_cuad_sft.py", "finetuning/evaluate_cuad_sft.py",
    ]
    run_dir = checkout / "finetuning" / "artifacts" / "cuad-sft-runs" / "12345"
    training = calls[1]
    assert training[training.index("--model-id") + 1] == MODEL_IDS[1]
    assert training[training.index("--output-dir") + 1] == str(earlier_output)
    assert training[-2:] == ["--output-dir", str(run_dir)]
    parsed_training = train_cuad_sft.parse_args(training[1:])
    assert parsed_training.model_id == MODEL_IDS[2]
    assert parsed_training.output_dir == run_dir
    evaluation = calls[2]
    assert evaluation[evaluation.index("--run-dir") + 1] == str(run_dir)
    assert evaluation[evaluation.index("--checkpoint") + 1] == str(run_dir / "adapter")
    assert evaluation[evaluation.index("--data-dir") + 1] == str(test_data)
    assert evaluation[evaluation.index("--split") + 1] == "test"
    assert evaluation[evaluation.index("--output") + 1] == str(run_dir / "evaluation-test" / "comparison.json")
    assert "--model-id" not in evaluation
    assert (run_dir / "adapter" / "adapter_config.json").exists()
    assert (run_dir / "evaluation-test" / "comparison.json").exists()
    assert not earlier_output.exists()
