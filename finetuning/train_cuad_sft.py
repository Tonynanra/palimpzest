"""Single-GPU LoRA SFT trainer for Qwen3.8-27B-FP8 on CUAD."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("cuad-sft")
DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_MAX_SEQ_LENGTH = 36864
DEFAULT_OUTPUT_DIR = Path("finetuning/artifacts/cuad-sft-run")

# Qwen3.8's dense text backbone contains both the Qwen attention projections and
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
    # PyTorch does not implement dropout for the Float8 activations exposed by
    # this fine-grained FP8 checkpoint's native PEFT path.
    lora_dropout: float = 0.0
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
        values = {
            "model_id": model.get("id", mapping.get("model_id", cls.model_id)),
            "data_dir": Path(data.get("dir", mapping.get("data_dir", cls.data_dir))),
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
            raise ValueError("The CUAD SFT implementation is configured for batch size 1 on one GH200")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.num_train_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("num_train_epochs and learning_rate must be positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")


class CuadJsonlDataset:
    def __init__(self, path: Path, limit: int | None = None):
        self.path = path
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if limit is not None:
            if limit <= 0:
                raise ValueError("dataset limit must be positive")
            self.rows = self.rows[:limit]
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
    """Write synchronized timing events and exclude Triton compilation from steady state."""

    def __init__(
        self,
        log_path: Path,
        *,
        full_run_steps: int | None = None,
        train_examples: int | None = None,
        gradient_accumulation_steps: int | None = None,
    ):
        self.log_path = log_path
        self.started_at = None
        self.started_perf = None
        self.full_run_steps = full_run_steps
        self.train_examples = train_examples
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step_started_perf = None
        self.substep_started_perf = None
        self.microbatch_index = 0
        self.optimizer_step_seconds: list[float] = []
        self.optimizer_step_records: list[dict[str, Any]] = []
        self._triton_cache_before_step: dict[str, tuple[int, int]] | None = None
        self._triton_cache_tracking_limit = 32
        self.summary: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Remain compatible with new optional TrainerCallback lifecycle hooks."""
        if name.startswith("on_"):
            return lambda args, state, control, **kwargs: control
        raise AttributeError(name)

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

    @staticmethod
    def _synchronize_cuda() -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _cuda_memory() -> dict[str, int] | None:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }

    @staticmethod
    def _compile_debug() -> dict[str, Any]:
        try:
            from torch._dynamo.utils import counters

            return {
                str(group): {str(key): int(value) for key, value in values.items() if value}
                for group, values in counters.items()
                if values
            }
        except Exception:
            return {}

    @staticmethod
    def _triton_cache_root() -> Path | None:
        configured = os.environ.get("TRITON_CACHE_DIR")
        root = Path(configured) if configured else Path.home() / ".triton" / "cache"
        try:
            if root.exists() or configured:
                return root
        except OSError:
            return None
        return None

    @classmethod
    def _triton_cache_snapshot(cls) -> dict[str, tuple[int, int]] | None:
        """Return file size/mtime state for Triton's JIT cache, if trackable."""
        root = cls._triton_cache_root()
        if root is None:
            return None
        snapshot: dict[str, tuple[int, int]] = {}
        try:
            if not root.exists():
                return snapshot
            for path in root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            return None
        return snapshot

    @staticmethod
    def _triton_cache_diff(
        before: dict[str, tuple[int, int]] | None,
        after: dict[str, tuple[int, int]] | None,
    ) -> dict[str, Any]:
        if before is None or after is None:
            return {
                "triton_cache_tracking_available": False,
                "includes_triton_kernel_compile": False,
                "triton_cache_new_file_count": None,
                "triton_cache_modified_file_count": None,
            }
        new_files = set(after) - set(before)
        modified_files = {path for path in set(after) & set(before) if after[path] != before[path]}
        return {
            "triton_cache_tracking_available": True,
            "includes_triton_kernel_compile": bool(new_files or modified_files),
            "triton_cache_new_file_count": len(new_files),
            "triton_cache_modified_file_count": len(modified_files),
        }

    def _process_elapsed(self) -> float | None:
        return time.perf_counter() - self.started_perf if self.started_perf is not None else None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        model = kwargs.get("model")
        del control
        self._synchronize_cuda()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_perf = time.perf_counter()
        fields: dict[str, Any] = {
            "started_at_utc": self.started_at,
            "planned_optimizer_steps": getattr(state, "max_steps", None),
            "full_run_optimizer_steps": self.full_run_steps,
            "train_examples": self.train_examples,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "model_class": type(model).__name__ if model is not None else None,
            "model_is_quantized": bool(getattr(model, "is_quantized", False)) if model is not None else None,
            "model_is_compiled": bool(hasattr(model, "_orig_mod")) if model is not None else None,
            "cuda_memory": self._cuda_memory(),
            "torch_compile_counters": self._compile_debug(),
            "triton_cache_dir": str(self._triton_cache_root()) if self._triton_cache_root() else None,
        }
        self._write("train_start", state, **fields)
        LOGGER.info(
            "timing stage=train_loop_start planned_steps=%s full_run_steps=%s model=%s quantized=%s compiled=%s cuda_memory=%s",
            fields["planned_optimizer_steps"],
            self.full_run_steps,
            fields["model_class"],
            fields["model_is_quantized"],
            fields["model_is_compiled"],
            fields["cuda_memory"],
        )

    def on_substep_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        self._synchronize_cuda()
        self.substep_started_perf = time.perf_counter()
        self.microbatch_index += 1

    def on_substep_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        if self.substep_started_perf is None:
            return
        self._synchronize_cuda()
        elapsed = time.perf_counter() - self.substep_started_perf
        self.substep_started_perf = None
        if self.microbatch_index <= 8:
            self._write(
                "microbatch",
                state,
                microbatch_index=self.microbatch_index,
                microbatch_seconds=elapsed,
                process_elapsed_seconds=self._process_elapsed(),
                includes_first_step_overhead=self.microbatch_index <= 4,
            )

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        self._synchronize_cuda()
        if len(self.optimizer_step_records) < self._triton_cache_tracking_limit:
            self._triton_cache_before_step = self._triton_cache_snapshot()
        else:
            self._triton_cache_before_step = None
        self.step_started_perf = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        if self.step_started_perf is None:
            return
        self._synchronize_cuda()
        elapsed = time.perf_counter() - self.step_started_perf
        self.step_started_perf = None
        triton_cache = self._triton_cache_diff(
            self._triton_cache_before_step,
            self._triton_cache_snapshot() if self._triton_cache_before_step is not None else None,
        )
        self._triton_cache_before_step = None
        self.optimizer_step_seconds.append(elapsed)
        global_step = int(getattr(state, "global_step", 0))
        is_first_step = global_step == 1
        self.optimizer_step_records.append(
            {
                "global_step": global_step,
                "optimizer_step_seconds": elapsed,
                **triton_cache,
            }
        )
        self._write(
            "optimizer_step",
            state,
            optimizer_step_seconds=elapsed,
            process_elapsed_seconds=self._process_elapsed(),
            includes_first_step_overhead=is_first_step,
            cuda_memory=self._cuda_memory(),
            **triton_cache,
        )
        if is_first_step or global_step <= 3 or global_step % 100 == 0:
            LOGGER.info(
                "timing optimizer_step=%s seconds=%.3f first_step_overhead=%s triton_compile=%s cache_new=%s cache_modified=%s cuda_memory=%s",
                global_step,
                elapsed,
                is_first_step,
                triton_cache["includes_triton_kernel_compile"],
                triton_cache["triton_cache_new_file_count"],
                triton_cache["triton_cache_modified_file_count"],
                self._cuda_memory(),
            )

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del control, kwargs
        self._synchronize_cuda()
        elapsed = self._process_elapsed()
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
        self._synchronize_cuda()
        elapsed = self._process_elapsed()
        records = self.optimizer_step_records
        first_step = records[0]["optimizer_step_seconds"] if records else None
        compile_records = [record for record in records if record["includes_triton_kernel_compile"]]
        warmup_records = [
            record
            for index, record in enumerate(records)
            if index == 0 or record["includes_triton_kernel_compile"]
        ]
        steady_records = [
            record
            for index, record in enumerate(records)
            if index > 0
            and record["triton_cache_tracking_available"]
            and not record["includes_triton_kernel_compile"]
        ]
        steady_steps = [record["optimizer_step_seconds"] for record in steady_records]
        steady_mean = sum(steady_steps) / len(steady_steps) if steady_steps else None
        estimated_full = None
        estimate_status = "insufficient_clean_steady_state_samples"
        tracking_available = bool(records) and all(
            record["triton_cache_tracking_available"] for record in records
        )
        if steady_mean is not None and self.full_run_steps is not None and tracking_available:
            one_time_seconds = sum(record["optimizer_step_seconds"] for record in warmup_records)
            remaining_steps = max(self.full_run_steps - len(warmup_records), 0)
            estimated_full = one_time_seconds + remaining_steps * steady_mean
            estimate_status = "clean_steady_state_with_compile_steps_excluded"
        self.summary = {
            "train_loop_walltime_seconds": elapsed,
            "observed_optimizer_steps": len(records),
            "first_optimizer_step_seconds": first_step,
            "steady_optimizer_step_count": len(steady_steps),
            "steady_optimizer_step_mean_seconds": steady_mean,
            "triton_compile_optimizer_step_count": len(compile_records),
            "triton_compile_optimizer_steps": [record["global_step"] for record in compile_records],
            "triton_compile_step_seconds": sum(
                record["optimizer_step_seconds"] for record in compile_records
            ),
            "warmup_optimizer_step_count": len(warmup_records),
            "warmup_optimizer_step_seconds": sum(
                record["optimizer_step_seconds"] for record in warmup_records
            ),
            "triton_cache_tracking_available": tracking_available,
            "full_run_optimizer_steps": self.full_run_steps,
            "estimated_full_training_loop_seconds": estimated_full,
            "estimate_status": estimate_status,
            "estimate_method": (
                "one-time first/compile steps plus clean steady-state mean; "
                "Triton compile steps are never multiplied"
            ),
            "torch_compile_counters": self._compile_debug(),
        }
        self._write(
            "train_end",
            state,
            **self.summary,
            cuda_memory=self._cuda_memory(),
        )
        LOGGER.info("timing summary %s", self.summary)


def resolve_lora_targets(model: Any) -> list[str]:
    """Return supported Qwen3.8 text projection suffixes present in `model`."""
    import torch

    linear_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    targets = [suffix for suffix in TARGET_MODULE_SUFFIXES if suffix in linear_suffixes]
    if not targets:
        raise RuntimeError(
            "No Qwen3.8 LoRA target modules were found. "
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


def _log_trainable_parameters(model: Any) -> None:
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        return
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    percentage = 100.0 * trainable / total if total else 0.0
    LOGGER.info(
        "trainable params: %s || all params: %s || trainable%%: %.4f",
        trainable,
        total,
        percentage,
    )


def _load_model_and_tokenizer(config: TrainConfig) -> tuple[Any, Any, list[str]]:
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils.quantization_config import FineGrainedFP8Config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3.8 CUAD SFT")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This entrypoint is single-GPU by design; detected {torch.cuda.device_count()} GPUs"
        )

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info(
        "timing stage=tokenizer_load seconds=%.3f model_id=%s vocab_size=%s",
        time.perf_counter() - load_started,
        config.model_id,
        getattr(tokenizer, "vocab_size", None),
    )

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
    elif config.model_id.lower().endswith("-fp8"):
        # Transformers' compressed FP8 kernels are inference-only: the
        # fine-grained matmul has no autograd formula.  GH200 has ample memory
        # to dequantize this 27B checkpoint to BF16 before attaching LoRA,
        # preserving the requested base model while making SFT trainable.
        model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        LOGGER.info("Qwen3.8 FP8 checkpoint will be dequantized to BF16 for LoRA training")

    model_load_started = time.perf_counter()
    # AutoModelForCausalLM unwraps Qwen3.8's composite vision-language config
    # into its text-only Qwen3_5ForCausalLM backbone for this text dataset.
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    LOGGER.info(
        "timing stage=model_load seconds=%.3f model_class=%s quantized=%s dequantized_for_training=%s hf_quantizer=%s",
        time.perf_counter() - model_load_started,
        type(model).__name__,
        bool(getattr(model, "is_quantized", False)),
        bool(config.model_id.lower().endswith("-fp8") and not getattr(model, "is_quantized", False)),
        type(getattr(model, "hf_quantizer", None)).__name__ if getattr(model, "hf_quantizer", None) else None,
    )
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
    adapter_started = time.perf_counter()
    if hasattr(model, "add_adapter"):
        # Transformers' native PEFT integration marks the base model as
        # adapter-enabled.  That is required for FP8 checkpoints because the
        # generic get_peft_model wrapper is rejected by the FP8 training guard.
        model.add_adapter(lora_config, adapter_name="default")
        model.set_adapter("default")
    else:
        if getattr(model, "is_quantized", False):
            raise RuntimeError(
                "The installed Transformers version cannot attach a native LoRA adapter to this FP8 checkpoint; "
                "install Transformers >= 5.8 and PEFT >= 0.20."
            )
        from peft import get_peft_model

        model = get_peft_model(model, lora_config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    LOGGER.info(
        "timing stage=lora_attach seconds=%.3f adapter_api=%s target_modules=%s",
        time.perf_counter() - adapter_started,
        "transformers_native" if hasattr(model, "_hf_peft_config_loaded") else "peft_wrapper",
        targets,
    )
    _log_trainable_parameters(model)
    return model, tokenizer, targets


def train(
    config: TrainConfig,
    resume_from_checkpoint: str | None = None,
    *,
    max_steps: int | None = None,
    eval_limit: int | None = None,
) -> Path:
    from transformers import TrainingArguments, set_seed

    config.validate()
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if eval_limit is not None and eval_limit <= 0:
        raise ValueError("eval_limit must be positive")
    run_started = time.perf_counter()
    set_seed(config.seed)
    dataset_started = time.perf_counter()
    train_dataset = CuadJsonlDataset(config.data_dir / "train.jsonl")
    eval_dataset = CuadJsonlDataset(config.data_dir / "dev.jsonl", limit=eval_limit)
    LOGGER.info(
        "timing stage=dataset_load seconds=%.3f train_examples=%s dev_examples=%s eval_limit=%s data_dir=%s",
        time.perf_counter() - dataset_started,
        len(train_dataset),
        len(eval_dataset),
        eval_limit,
        config.data_dir,
    )
    model_started = time.perf_counter()
    model, tokenizer, targets = _load_model_and_tokenizer(config)
    LOGGER.info(
        "timing stage=model_and_adapter_ready seconds=%.3f process_seconds=%.3f",
        time.perf_counter() - model_started,
        time.perf_counter() - run_started,
    )
    collator = CuadDataCollator(tokenizer, config.max_seq_length)
    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_dataset)
            / (config.per_device_train_batch_size * config.gradient_accumulation_steps)
        ),
    )
    full_run_steps = max(1, math.ceil(config.num_train_epochs * steps_per_epoch))
    total_steps = max_steps or full_run_steps
    warmup_steps = int(round(config.warmup_ratio * total_steps))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer_started = time.perf_counter()
    timing_callback = WalltimeLoggingCallback(
        config.output_dir / "training_walltime.jsonl",
        full_run_steps=full_run_steps,
        train_examples=len(train_dataset),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
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
        callbacks=[timing_callback],
    )
    LOGGER.info(
        "timing stage=trainer_init seconds=%.3f steps_per_epoch=%s smoke_steps=%s full_run_steps=%s",
        time.perf_counter() - trainer_started,
        steps_per_epoch,
        max_steps,
        full_run_steps,
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
                "steps_per_epoch": steps_per_epoch,
                "full_run_optimizer_steps": full_run_steps,
                "runtime_overrides": {
                    "max_steps": max_steps,
                    "eval_limit": eval_limit,
                },
                "train_metrics": train_output.metrics,
                "timing_summary": timing_callback.summary,
                "walltime_log": str(config.output_dir / "training_walltime.jsonl"),
            },
            handle,
            indent=2,
            default=str,
        )
    return adapter_dir


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen3.8-27B-FP8 CUAD LoRA adapter")
    parser.add_argument("--config", type=Path, default=Path("finetuning/sft_config.yaml"))
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=None,
        help="Optional optimizer-step cap for a short smoke run.",
    )
    parser.add_argument(
        "--eval-limit",
        type=_positive_int,
        default=None,
        help="Optional limit on dev rows evaluated during training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Training artifact directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    config = _load_config(args.config)
    config = replace(config, output_dir=args.output_dir)
    adapter_dir = train(
        config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_steps=args.max_steps,
        eval_limit=args.eval_limit,
    )
    LOGGER.info("Saved adapter to %s", adapter_dir)


if __name__ == "__main__":
    main()
