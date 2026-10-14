"""Shared CUAD training configuration and run-root snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MODEL_IDS = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.8-27B",
)
DEFAULT_MODEL_ID = SUPPORTED_MODEL_IDS[0]
DEFAULT_MAX_SEQ_LENGTH = 32768
DEFAULT_OUTPUT_DIR = Path("finetuning/artifacts/cuad-sft-run")
CONFIG_SNAPSHOT_NAME = "config_snapshot.yaml"


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = DEFAULT_MODEL_ID
    data_dir: Path = Path("testdata/cuad/sft/train-dev")
    output_dir: Path = DEFAULT_OUTPUT_DIR
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    seed: int = 42
    logging_steps: int = 1
    save_total_limit: int = 3
    qlora: bool = False
    gradient_checkpointing: bool = True
    flash_attention: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any], *, model_id: str | None = None) -> TrainConfig:
        if not isinstance(mapping, dict):
            raise ValueError("CUAD SFT configuration must be a YAML mapping")
        data = mapping.get("data", {})
        training = mapping.get("training", {})
        values = {
            "model_id": mapping.get("model") if model_id is None else model_id,
            "data_dir": Path(data.get("dir", cls.data_dir)),
            "max_seq_length": int(training.get("max_seq_length", cls.max_seq_length)),
            "per_device_train_batch_size": int(training.get("per_device_train_batch_size", cls.per_device_train_batch_size)),
            "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", cls.per_device_eval_batch_size)),
            "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", cls.gradient_accumulation_steps)),
            "num_train_epochs": float(training.get("num_train_epochs", cls.num_train_epochs)),
            "learning_rate": float(training.get("learning_rate", cls.learning_rate)),
            "warmup_ratio": float(training.get("warmup_ratio", cls.warmup_ratio)),
            "weight_decay": float(training.get("weight_decay", cls.weight_decay)),
            "lora_rank": int(training.get("lora_rank", cls.lora_rank)),
            "lora_alpha": int(training.get("lora_alpha", cls.lora_alpha)),
            "lora_dropout": float(training.get("lora_dropout", cls.lora_dropout)),
            "seed": int(training.get("seed", cls.seed)),
            "logging_steps": int(training.get("logging_steps", cls.logging_steps)),
            "save_total_limit": int(training.get("save_total_limit", cls.save_total_limit)),
            "qlora": bool(training.get("qlora", cls.qlora)),
            "gradient_checkpointing": bool(training.get("gradient_checkpointing", cls.gradient_checkpointing)),
            "flash_attention": bool(training.get("flash_attention", cls.flash_attention)),
        }
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.model_id, str) or self.model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"model must be one of {SUPPORTED_MODEL_IDS}; got {self.model_id!r}")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.per_device_train_batch_size != 1 or self.per_device_eval_batch_size != 1:
            raise ValueError("The CUAD SFT implementation is configured for batch size 1 on one GPU")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.num_train_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("num_train_epochs and learning_rate must be positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")


def load_training_config(path: Path, *, model_id: str | None = None) -> TrainConfig:
    with path.open(encoding="utf-8") as handle:
        return TrainConfig.from_mapping(yaml.safe_load(handle), model_id=model_id)


def load_run_config(run_dir: Path) -> TrainConfig:
    """Read the sole snapshot at the run root, independent of the source YAML."""
    return replace(load_training_config(run_dir / CONFIG_SNAPSHOT_NAME), output_dir=run_dir)


def save_config_snapshot(
    config: TrainConfig,
    *,
    max_steps: int | None = None,
    eval_limit: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> Path:
    """Persist effective settings, rejecting model mismatches before any write."""
    config.validate()
    snapshot_path = config.output_dir / CONFIG_SNAPSHOT_NAME
    previous_roots = []
    if snapshot_path.exists():
        previous_roots.append(config.output_dir)
    if resume_from_checkpoint is not None:
        previous_roots.append(Path(resume_from_checkpoint).resolve().parent)
    for run_dir in previous_roots:
        previous = load_run_config(run_dir)
        if previous.model_id != config.model_id:
            raise ValueError(
                f"Model mismatch for run {run_dir}: saved {previous.model_id}, requested {config.model_id}"
            )

    training = asdict(config)
    for name in ("model_id", "data_dir", "output_dir"):
        training.pop(name)
    snapshot = {
        "model": config.model_id,
        "data": {"dir": str(config.data_dir.resolve())},
        "training": training,
        "runtime": {
            "output_dir": str(config.output_dir.resolve()),
            "max_steps": max_steps,
            "eval_limit": eval_limit,
            "resume_from_checkpoint": resume_from_checkpoint,
        },
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)
    return snapshot_path
