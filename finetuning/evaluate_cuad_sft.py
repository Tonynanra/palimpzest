"""Default vanilla-vs-SFT generation evaluation for CUAD."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .cuad_demo import evaluate_entry
    from .cuad_sft_data import _TextPromptModel, render_cuad_messages
    from .train_cuad_sft import CuadJsonlDataset, _text_messages, apply_sft_chat_template
except ImportError:
    from cuad_demo import evaluate_entry
    from cuad_sft_data import _TextPromptModel, render_cuad_messages
    from train_cuad_sft import CuadJsonlDataset, _text_messages, apply_sft_chat_template

from palimpzest.constants import Cardinality
from palimpzest.query.generators.generators import get_json_from_answer


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    records: list[dict[str, Any]]


class InferenceWalltimeLogger:
    """Write periodic inference walltime events as JSONL."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _model_kwargs(use_flash_attention: bool) -> dict[str, Any]:
    import torch

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    if use_flash_attention and importlib.util.find_spec("flash_attn") is not None:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    return model_kwargs


def _validate_single_cuda_device() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUAD SFT evaluation expects exactly one CUDA GPU")


def _load_tokenizer(model_id: str, tokenizer_source: Path | str | None = None):
    from transformers import AutoTokenizer

    source = str(tokenizer_source) if tokenizer_source is not None else model_id
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(model_id: str, use_flash_attention: bool = True):
    from transformers import AutoModelForCausalLM

    _validate_single_cuda_device()
    model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs(use_flash_attention))
    model.eval()
    return model


def _load_adapter_model(model_id: str, checkpoint: Path, use_flash_attention: bool = True):
    from peft import PeftModel

    base = _load_base_model(model_id, use_flash_attention)
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.eval()
    tokenizer = _load_tokenizer(
        model_id,
        checkpoint if (checkpoint / "tokenizer_config.json").exists() else None,
    )
    return model, tokenizer


def _load_merged_model(checkpoint: Path, model_id: str, use_flash_attention: bool = True):
    from transformers import AutoModelForCausalLM

    _validate_single_cuda_device()
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **_model_kwargs(use_flash_attention))
    model.eval()
    tokenizer = _load_tokenizer(
        model_id,
        checkpoint if (checkpoint / "tokenizer_config.json").exists() else None,
    )
    return model, tokenizer


def _load_model(model_id: str, checkpoint: Path, use_flash_attention: bool = True):
    """Backward-compatible loader for callers that need one SFT model."""
    if (checkpoint / "adapter_config.json").exists():
        return _load_adapter_model(model_id, checkpoint, use_flash_attention)
    return _load_merged_model(checkpoint, model_id, use_flash_attention)


def _adapter_context(model: Any, adapter_enabled: bool | None):
    if adapter_enabled is False and hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


def _completion_for_row(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    max_new_tokens: int,
    adapter_enabled: bool | None = None,
) -> str:
    import torch

    messages = _text_messages(row["messages"])
    prompt_inputs = apply_sft_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs.get("attention_mask")
    with _adapter_context(model, adapter_enabled), torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids = generated[0, input_ids.shape[-1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def _normalize_predictions(value: Any) -> list[str]:
    if value is None or value == "" or value == "null":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _select_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "all":
        return [row for row in rows if row["view"] == "all"]
    if mode == "singleton":
        return [row for row in rows if row["view"] == "singleton"]
    if mode == "grouped":
        return [row for row in rows if row["view"] in {"small", "medium", "large"}]
    if mode == "randomized":
        return rows
    if mode == "canonical":
        return [row for row in rows if row["view"] == "all"]
    raise ValueError(f"Unsupported evaluation mode: {mode}")


def _canonicalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-render stored views in the normal alphabetical PZ field order."""
    canonical_rows = []
    for row in rows:
        categories = sorted(row["categories"])
        canonical = dict(row)
        canonical["categories"] = categories
        canonical["messages"] = render_cuad_messages(row["contract"], categories)
        canonical_rows.append(canonical)
    return canonical_rows


def _prepare_rows(rows: list[dict[str, Any]], mode: str, limit: int | None) -> list[dict[str, Any]]:
    selected_rows = _select_rows(rows, mode)
    if limit is not None:
        selected_rows = selected_rows[:limit]
    if mode == "canonical":
        selected_rows = _canonicalize_rows(selected_rows)
    return selected_rows


def _finalize_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _evaluate_rows(
    model: Any,
    tokenizer: Any,
    selected_rows: list[dict[str, Any]],
    max_new_tokens: int,
    model_role: str,
    model_id: str,
    mode: str | None = None,
    adapter_enabled: bool | None = None,
    timing_logger: InferenceWalltimeLogger | None = None,
    timing_log_every: int = 10,
) -> EvaluationResult:
    tp = fp = fn = 0
    parse_failures = missing_keys = extra_keys = 0
    expected_key_count = parsed_key_count = 0
    grounded_predictions = total_predictions = 0
    per_category = defaultdict(lambda: [0, 0, 0])
    records = []
    started_perf = time.perf_counter()
    total_rows = len(selected_rows)
    if timing_logger is not None:
        timing_logger.write(
            "inference_start",
            model_role=model_role,
            model_id=model_id,
            rows=total_rows,
        )

    for row_index, row in enumerate(selected_rows, start=1):
        row_started_perf = time.perf_counter()
        completion = _completion_for_row(model, tokenizer, row, max_new_tokens, adapter_enabled)
        parse_failed = False
        try:
            parsed = get_json_from_answer(completion, _TextPromptModel(), Cardinality.ONE_TO_ONE)
            if not isinstance(parsed, dict):
                raise ValueError("completion JSON is not an object")
        except Exception:
            parsed = {}
            parse_failures += 1
            parse_failed = True

        expected_categories = set(row["categories"])
        expected_key_count += len(expected_categories)
        parsed_key_count += len(parsed)
        missing_keys += len(expected_categories - set(parsed))
        extra_keys += len(set(parsed) - expected_categories)
        for category in row["categories"]:
            labels = list(row.get("labels", {}).get(category, []))
            predictions = _normalize_predictions(parsed.get(category))
            entry_tp, entry_fp, entry_fn = evaluate_entry(labels, predictions, "Parties" in category)
            tp += entry_tp
            fp += entry_fp
            fn += entry_fn
            category_counts = per_category[category]
            category_counts[0] += entry_tp
            category_counts[1] += entry_fp
            category_counts[2] += entry_fn
            for prediction in predictions:
                total_predictions += 1
                if prediction and prediction in row["contract"]:
                    grounded_predictions += 1
            records.append(
                {
                    "id": row["id"],
                    "category": category,
                    "labels": labels,
                    "predictions": predictions,
                    "parse_failed": parse_failed,
                }
            )

        if timing_logger is not None and (row_index % timing_log_every == 0 or row_index == total_rows):
            elapsed = time.perf_counter() - started_perf
            timing_logger.write(
                "inference_progress",
                model_role=model_role,
                model_id=model_id,
                completed_rows=row_index,
                total_rows=total_rows,
                elapsed_seconds=elapsed,
                last_row_seconds=time.perf_counter() - row_started_perf,
                rows_per_second=row_index / elapsed if elapsed > 0 else None,
            )

    elapsed = time.perf_counter() - started_perf
    metrics = _finalize_counts(tp, fp, fn)
    metrics.update(
        {
            "model_role": model_role,
            "model_id": model_id,
            "mode": mode,
            "rows": len(selected_rows),
            "inference_walltime_seconds": elapsed,
            "average_row_seconds": elapsed / len(selected_rows) if selected_rows else 0.0,
            "rows_per_second": len(selected_rows) / elapsed if elapsed > 0 else 0.0,
            "parse_failures": parse_failures,
            "parse_success_rate": 1.0 - parse_failures / len(selected_rows) if selected_rows else 0.0,
            "missing_keys": missing_keys,
            "missing_key_rate": missing_keys / expected_key_count if expected_key_count else 0.0,
            "extra_keys": extra_keys,
            "extra_key_rate": extra_keys / parsed_key_count if parsed_key_count else 0.0,
            "grounding_rate": grounded_predictions / total_predictions if total_predictions else 1.0,
            "per_category": {
                category: _finalize_counts(*counts)
                for category, counts in sorted(per_category.items())
            },
        }
    )
    return EvaluationResult(metrics=metrics, records=records)


def evaluate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    mode: str,
    max_new_tokens: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate one model; retained for programmatic callers and unit tests."""
    selected_rows = _prepare_rows(rows, mode, limit)
    return _evaluate_rows(model, tokenizer, selected_rows, max_new_tokens, "model", "unknown", mode=mode).metrics


def _release_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _metric_delta(vanilla: dict[str, Any], sft: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in (
        "precision",
        "recall",
        "f1",
        "parse_success_rate",
        "grounding_rate",
        "missing_key_rate",
        "extra_key_rate",
        "inference_walltime_seconds",
        "average_row_seconds",
        "rows_per_second",
        "tp",
        "fp",
        "fn",
    ):
        vanilla_value = vanilla.get(key)
        sft_value = sft.get(key)
        delta[key] = sft_value - vanilla_value if isinstance(vanilla_value, (int, float)) and isinstance(sft_value, (int, float)) else None

    categories = sorted(set(vanilla.get("per_category", {})) | set(sft.get("per_category", {})))
    delta["per_category"] = {}
    for category in categories:
        vanilla_category = vanilla.get("per_category", {}).get(category, {})
        sft_category = sft.get("per_category", {}).get(category, {})
        delta["per_category"][category] = {
            key: sft_category.get(key) - vanilla_category.get(key)
            if isinstance(vanilla_category.get(key), (int, float)) and isinstance(sft_category.get(key), (int, float))
            else None
            for key in ("precision", "recall", "f1", "tp", "fp", "fn")
        }
    return delta


def _plot_comparison(comparison: dict[str, Any], plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    vanilla = comparison["vanilla"]
    sft = comparison["sft"]
    labels = ["precision", "recall", "f1"]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([position - width / 2 for position in x], [vanilla.get(key) or 0.0 for key in labels], width, label="Vanilla")
    ax.bar([position + width / 2 for position in x], [sft.get(key) or 0.0 for key in labels], width, label="SFT")
    ax.set_xticks(x, [label.upper() for label in labels])
    ax.set_ylim(0, 1)
    ax.set_title("Aggregate extraction metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "aggregate_metrics.png", dpi=180)
    plt.close(fig)

    categories = sorted(set(vanilla.get("per_category", {})) | set(sft.get("per_category", {})))
    for metric in ("precision", "recall", "f1"):
        fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(categories))))
        positions = list(range(len(categories)))
        values_vanilla = [vanilla.get("per_category", {}).get(category, {}).get(metric) or 0.0 for category in categories]
        values_sft = [sft.get("per_category", {}).get(category, {}).get(metric) or 0.0 for category in categories]
        ax.barh([position - width / 2 for position in positions], values_vanilla, width, label="Vanilla")
        ax.barh([position + width / 2 for position in positions], values_sft, width, label="SFT")
        ax.set_yticks(positions, categories)
        ax.set_xlim(0, 1)
        ax.set_title(f"Per-category {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"per_category_{metric}.png", dpi=180)
        plt.close(fig)

    f1_deltas = [comparison["delta"]["per_category"].get(category, {}).get("f1") or 0.0 for category in categories]
    order = sorted(range(len(categories)), key=lambda index: f1_deltas[index])
    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(categories))))
    ax.barh([categories[index] for index in order], [f1_deltas[index] for index in order])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("SFT minus vanilla per-category F1")
    fig.tight_layout()
    fig.savefig(plot_dir / "per_category_f1_delta.png", dpi=180)
    plt.close(fig)

    format_labels = ["parse success", "grounding", "missing-key rate", "extra-key rate"]
    format_keys = ["parse_success_rate", "grounding_rate", "missing_key_rate", "extra_key_rate"]
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = list(range(len(format_labels)))
    ax.bar([position - width / 2 for position in positions], [vanilla.get(key) or 0.0 for key in format_keys], width, label="Vanilla")
    ax.bar([position + width / 2 for position in positions], [sft.get(key) or 0.0 for key in format_keys], width, label="SFT")
    ax.set_xticks(positions, format_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Formatting and grounding metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "format_metrics.png", dpi=180)
    plt.close(fig)


def _write_reports(comparison: dict[str, Any], output_path: Path, plot_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "vanilla.json").write_text(json.dumps(comparison["vanilla"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_path.parent / "sft.json").write_text(json.dumps(comparison["sft"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_comparison(comparison, plot_dir)


def _load_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "dev"):
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            rows.extend(CuadJsonlDataset(path).rows)
    if not rows:
        raise FileNotFoundError(f"No train.jsonl or dev.jsonl found in {data_dir}")
    return rows


def _run_comparison(
    model_id: str,
    checkpoint: Path,
    selected_rows: list[dict[str, Any]],
    mode: str,
    max_new_tokens: int,
    use_flash_attention: bool,
    timing_logger: InferenceWalltimeLogger | None = None,
    timing_log_every: int = 10,
) -> dict[str, Any]:
    if (checkpoint / "adapter_config.json").exists():
        model, tokenizer = _load_adapter_model(model_id, checkpoint, use_flash_attention)
        vanilla_result = _evaluate_rows(model, tokenizer, selected_rows, max_new_tokens, "vanilla", model_id, mode=mode, adapter_enabled=False, timing_logger=timing_logger, timing_log_every=timing_log_every)
        sft_result = _evaluate_rows(model, tokenizer, selected_rows, max_new_tokens, "sft", model_id, mode=mode, adapter_enabled=True, timing_logger=timing_logger, timing_log_every=timing_log_every)
        _release_model(model)
    else:
        vanilla_model = _load_base_model(model_id, use_flash_attention)
        vanilla_result = _evaluate_rows(vanilla_model, _load_tokenizer(model_id), selected_rows, max_new_tokens, "vanilla", model_id, mode=mode, timing_logger=timing_logger, timing_log_every=timing_log_every)
        _release_model(vanilla_model)
        sft_model, sft_tokenizer = _load_merged_model(checkpoint, model_id, use_flash_attention)
        sft_result = _evaluate_rows(sft_model, sft_tokenizer, selected_rows, max_new_tokens, "sft", model_id, mode=mode, timing_logger=timing_logger, timing_log_every=timing_log_every)
        _release_model(sft_model)
    return {"vanilla": vanilla_result.metrics, "sft": sft_result.metrics, "delta": _metric_delta(vanilla_result.metrics, sft_result.metrics)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare vanilla and SFT Qwen3.5 CUAD performance")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--mode", choices=["all", "singleton", "grouped", "randomized", "canonical"], default="all")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timing-log-every", type=int, default=10)
    parser.add_argument("--no-flash-attention", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    selected_rows = _prepare_rows(_load_rows(args.data_dir), args.mode, args.limit)
    output_path = args.output or (args.checkpoint.parent / "comparison.json")
    if args.timing_log_every <= 0:
        raise ValueError("--timing-log-every must be positive")
    timing_log_path = output_path.parent / "inference_walltime.jsonl"
    timing_logger = InferenceWalltimeLogger(timing_log_path)
    try:
        comparison = _run_comparison(
            args.model_id,
            args.checkpoint,
            selected_rows,
            args.mode,
            args.max_new_tokens,
            not args.no_flash_attention,
            timing_logger=timing_logger,
            timing_log_every=args.timing_log_every,
        )
    finally:
        timing_logger.close()
    comparison.update(
        {
            "mode": args.mode,
            "rows": len(selected_rows),
            "model_id": args.model_id,
            "inference_walltime_log": str(timing_log_path),
        }
    )
    _write_reports(comparison, output_path, output_path.parent / "sft_plots")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
