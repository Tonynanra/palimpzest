"""Single-GPU LoRA SFT trainer for Qwen3.5 on CUAD."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("cuad-sft")
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_SEQ_LENGTH = 32768

# Qwen3.5 dense text layers contain both the Qwen attention projections and
# Gated DeltaNet projections.  Resolve only suffixes that are actually present
# in the loaded text-only model and fail closed if none are found.
TARGET_MODULE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = DEFAULT_MODEL_ID
    data_dir: Path = Path("finetuning/artifacts/cuad-sft")
    output_dir: Path = Path("finetuning/artifacts/cuad-sft-run")
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
    def from_mapping(cls, mapping: dict[str, Any]) -> TrainConfig:
        model = mapping.get("model", {})
        data = mapping.get("data", {})
        training = mapping.get("training", {})
        output = mapping.get("output", {})
        values = {
            "model_id": model.get("id", mapping.get("model_id", cls.model_id)),
            "data_dir": Path(data.get("dir", mapping.get("data_dir", cls.data_dir))),
            "output_dir": Path(output.get("dir", mapping.get("output_dir", cls.output_dir))),
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
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.per_device_train_batch_size != 1 or self.per_device_eval_batch_size != 1:
            raise ValueError("The CUAD SFT implementation is configured for batch size 1 on one L40S")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.num_train_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("num_train_epochs and learning_rate must be positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")


class CuadJsonlDataset:
    def __init__(self, path: Path):
        self.path = path
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"No SFT examples found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def apply_sft_chat_template(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    """Apply Qwen's no-thinking template, with compatibility for simpler tokenizers."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return tokenizer.apply_chat_template(messages, **kwargs)


def _text_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for message in raw_messages:
        if message.get("type", "text") != "text":
            raise ValueError("CUAD SFT only supports text prompt messages")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Each SFT prompt message must contain string content")
        messages.append({"role": message["role"], "content": content})
    return messages


def _tokenize_pair(tokenizer: Any, row: dict[str, Any], max_seq_length: int) -> tuple[list[int], list[int]]:
    def _ids(value: Any) -> list[int]:
        if hasattr(value, "input_ids") or isinstance(value, dict):
            value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError("Expected one tokenized conversation per SFT example")
            value = value[0]
        return [int(token) for token in value]

    messages = _text_messages(row["messages"])
    response = row["response"]
    prompt_ids = apply_sft_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = apply_sft_chat_template(
        tokenizer,
        messages + [{"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = _ids(prompt_ids)
    full_ids = _ids(full_ids)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Qwen chat template did not produce an assistant continuation prefix; "
            "inspect the tokenizer template before training"
        )
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"SFT example {row.get('id')} has {len(full_ids)} tokens, "
            f"exceeding max_seq_length={max_seq_length}; refusing to truncate"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return full_ids, labels


class CuadDataCollator:
    def __init__(self, tokenizer: Any, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        tokenized = [_tokenize_pair(self.tokenizer, row, self.max_seq_length) for row in rows]
        max_length = max(len(input_ids) for input_ids, _ in tokenized)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        labels = []
        attention_mask = []
        completion_starts = []
        sequence_lengths = []
        for row_input_ids, row_labels in tokenized:
            padding = max_length - len(row_input_ids)
            input_ids.append(row_input_ids + [pad_id] * padding)
            labels.append(row_labels + [-100] * padding)
            attention_mask.append([1] * len(row_input_ids) + [0] * padding)
            completion_starts.append(next(index for index, label in enumerate(row_labels) if label != -100))
            sequence_lengths.append(len(row_input_ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "completion_start": torch.tensor(completion_starts, dtype=torch.long),
            "sequence_length": torch.tensor(sequence_lengths, dtype=torch.long),
        }


class CuadTrainer:
    """Lazy wrapper so importing this module does not require Transformers."""

    @staticmethod
    def build_base():
        from transformers import Trainer

        class _Trainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                del num_items_in_batch
                completion_start = int(inputs.pop("completion_start")[0].item())
                sequence_length = int(inputs.pop("sequence_length")[0].item())
                labels = inputs.pop("labels")
                keep = sequence_length - completion_start + 1
                labels_for_loss = labels[:, sequence_length - keep : sequence_length]
                outputs = model(
                    **inputs,
                    labels=labels_for_loss,
                    logits_to_keep=keep,
                )
                return (outputs.loss, outputs) if return_outputs else outputs.loss

        return _Trainer


class WalltimeLoggingCallback:
    """Write durable walltime events at train start, checkpoints, and train end."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.started_at = None
        self.started_perf = None

    def _write(self, event: str, state: Any, **fields: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "global_step": int(getattr(state, "global_step", 0)),
            **fields,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_perf = time.perf_counter()
        self._write("train_start", state, started_at_utc=self.started_at)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del control, kwargs
        elapsed = time.perf_counter() - self.started_perf if self.started_perf is not None else None
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self._write(
            "checkpoint",
            state,
            checkpoint=str(checkpoint),
            elapsed_seconds=elapsed,
            epoch=getattr(state, "epoch", None),
        )

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        elapsed = time.perf_counter() - self.started_perf if self.started_perf is not None else None
        self._write("train_end", state, elapsed_seconds=elapsed)


def resolve_lora_targets(model: Any) -> list[str]:
    """Return the supported Qwen3.5 text projection suffixes present in `model`."""
    import torch

    linear_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    targets = [suffix for suffix in TARGET_MODULE_SUFFIXES if suffix in linear_suffixes]
    if not targets:
        raise RuntimeError(
            "No Qwen3.5 LoRA target modules were found. "
            f"Observed linear suffixes: {sorted(linear_suffixes)}"
        )
    forbidden = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(part in name.lower().split(".") for part in ("visual", "vision", "image", "video"))
        and name.rsplit(".", 1)[-1] in targets
    }
    if forbidden:
        raise RuntimeError(f"Refusing to target vision modules: {sorted(forbidden)[:10]}")
    return targets


def _load_config(path: Path) -> TrainConfig:
    with path.open(encoding="utf-8") as handle:
        return TrainConfig.from_mapping(yaml.safe_load(handle) or {})


def _load_model_and_tokenizer(config: TrainConfig) -> tuple[Any, Any, list[str]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3.5 CUAD SFT")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This entrypoint is single-GPU by design; detected {torch.cuda.device_count()} GPUs"
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    if config.flash_attention and importlib.util.find_spec("flash_attn") is not None:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif config.flash_attention:
        LOGGER.warning("flash_attn is unavailable; falling back to SDPA attention")
    if config.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.config.use_cache = False

    targets = resolve_lora_targets(model)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, targets


def train(config: TrainConfig, resume_from_checkpoint: str | None = None) -> Path:
    from transformers import TrainingArguments, set_seed

    config.validate()
    set_seed(config.seed)
    train_dataset = CuadJsonlDataset(config.data_dir / "train.jsonl")
    eval_dataset = CuadJsonlDataset(config.data_dir / "dev.jsonl")
    model, tokenizer, targets = _load_model_and_tokenizer(config)
    collator = CuadDataCollator(tokenizer, config.max_seq_length)
    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_dataset)
            / (config.per_device_train_batch_size * config.gradient_accumulation_steps)
        ),
    )
    total_steps = max(1, math.ceil(config.num_train_epochs * steps_per_epoch))
    warmup_steps = int(round(config.warmup_ratio * total_steps))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if config.qlora else "adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=1.0,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )
    trainer = CuadTrainer.build_base()(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[WalltimeLoggingCallback(config.output_dir / "training_walltime.jsonl")],
    )
    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    adapter_dir = config.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    with (config.output_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_id": config.model_id,
                "max_seq_length": config.max_seq_length,
                "lora_targets": targets,
                "config": config.__dict__ | {
                    "data_dir": str(config.data_dir),
                    "output_dir": str(config.output_dir),
                },
                "train_examples": len(train_dataset),
                "dev_examples": len(eval_dataset),
                "train_metrics": train_output.metrics,
                "walltime_log": str(config.output_dir / "training_walltime.jsonl"),
            },
            handle,
            indent=2,
            default=str,
        )
    return adapter_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen3.5 CUAD LoRA adapter")
    parser.add_argument("--config", type=Path, default=Path("finetuning/sft_config.yaml"))
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    config = _load_config(args.config)
    adapter_dir = train(config, resume_from_checkpoint=args.resume_from_checkpoint)
    LOGGER.info("Saved adapter to %s", adapter_dir)


if __name__ == "__main__":
    main()
