from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from cuad_demo import (
    CUADDataset,
    build_cuad_query,
    compute_metrics,
    evaluate_entry,
    get_label_df,
    handle_empty_preds,
    labels_to_long,
    normalize_predictions_to_long,
    resolve_selected_categories,
    validate_category_names,
)
from dotenv import load_dotenv

import palimpzest as pz

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "opt-profiling-data"


@dataclass(frozen=True)
class OpenRouterConfig:
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env_var: str = "OPENROUTER_API_KEY"
    extra_headers: dict[str, str] | None = None
    provider: str = "openrouter"


@dataclass(frozen=True)
class ExecutionConfig:
    execution_strategy: str = "parallel"
    optimizer_strategy: str = "none"
    max_workers: int = 64
    reasoning_effort: str = "disable"
    progress: bool | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    selected_categories: list[str]
    difficulty_buckets: dict[str, list[str]]
    category_to_difficulty: dict[str, str]
    job_mode: str
    models: list[str]
    openrouter: OpenRouterConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    dataset_mode: str = "cuad-chunk"
    data_dir: Path | None = None
    num_contracts: int = 100
    seed: int = 42
    split: str = "test"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    exp_name: str = "cuad-qwen-openrouter"
    run_id: str = field(default_factory=lambda: uuid4().hex[:8])
    verbose: bool = False
    convert_mode: str = "separate-converts"


@dataclass(frozen=True)
class BenchmarkJob:
    job_id: str
    model_id: str
    categories: list[str]
    job_mode: str


@dataclass
class JobResult:
    job: BenchmarkJob
    predictions_long: pd.DataFrame
    predictions_wide: pd.DataFrame
    stats: dict[str, Any]
    execution_strategy: str
    max_workers: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the native pz CUAD OpenRouter benchmark")
    parser.add_argument("--config", required=True, type=Path, help="Path to the benchmark YAML config")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose execution output")
    return parser.parse_args()


def load_benchmark_config(config_path: str | Path, verbose_override: bool | None = None) -> BenchmarkConfig:
    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    _reject_legacy_worker_config(raw_config)

    categories_config = raw_config.get("categories", {})
    selected_categories = _resolve_categories_from_config(categories_config)
    validate_category_names(selected_categories)
    difficulty_buckets = _resolve_difficulty_buckets(categories_config)
    category_to_difficulty = _build_category_to_difficulty_map(selected_categories, difficulty_buckets)

    job_mode = raw_config.get("job_mode") or raw_config.get("execution", {}).get("job_mode", "grouped")
    if job_mode not in {"per-category", "grouped"}:
        raise ValueError(f"Unsupported job mode: {job_mode}")

    models = raw_config.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(model, str) for model in models):
        raise ValueError("Config field 'models' must be a non-empty list of model ids")

    dataset_mode = raw_config.get("dataset_mode") or raw_config.get("dataset", {}).get("mode", "cuad-chunk")
    if dataset_mode not in {"cuad-data", "cuad-chunk"}:
        raise ValueError(f"Unsupported dataset mode: {dataset_mode}")
    data_dir_value = raw_config.get("data_dir") or raw_config.get("dataset", {}).get("data_dir")
    data_dir = Path(data_dir_value) if data_dir_value else None

    endpoint_config = raw_config.get("openrouter")
    endpoint_provider = "openrouter"
    if endpoint_config is None and raw_config.get("endpoint") is not None:
        endpoint_config = raw_config.get("endpoint")
        endpoint_provider = str(endpoint_config.get("provider", "openai-compatible"))
    openrouter = _parse_openrouter_config(endpoint_config, provider=endpoint_provider)
    execution = _parse_execution_config(raw_config.get("execution", {}))

    convert_mode = raw_config.get("convert_mode", "separate-converts")
    if convert_mode not in {"separate-converts", "one-convert"}:
        raise ValueError("convert_mode must be 'separate-converts' or 'one-convert'")

    output_dir = Path(raw_config.get("output_dir", DEFAULT_OUTPUT_DIR))
    verbose = raw_config.get("verbose", False) if verbose_override is None else verbose_override
    return BenchmarkConfig(
        selected_categories=selected_categories,
        difficulty_buckets=difficulty_buckets,
        category_to_difficulty=category_to_difficulty,
        job_mode=job_mode,
        convert_mode=convert_mode,
        models=models,
        openrouter=openrouter,
        execution=execution,
        dataset_mode=dataset_mode,
        data_dir=data_dir,
        num_contracts=int(raw_config.get("num_contracts", 100)),
        seed=int(raw_config.get("seed", 42)),
        split=raw_config.get("split", "test"),
        output_dir=output_dir,
        exp_name=raw_config.get("exp_name", "cuad-qwen-openrouter"),
        run_id=raw_config.get("run_id", uuid4().hex[:8]),
        verbose=verbose,
    )


def _reject_legacy_worker_config(raw_config: dict[str, Any]) -> None:
    if "backends" in raw_config:
        raise ValueError("Config field 'backends' is unsupported for the native pz runner")
    if "nodes" in raw_config:
        raise ValueError("Config field 'nodes' is unsupported for the native pz runner")


def _parse_openrouter_config(
    openrouter_config: dict[str, Any] | None,
    provider: str = "openrouter",
) -> OpenRouterConfig:
    if not isinstance(openrouter_config, dict):
        raise ValueError("Config field 'openrouter' or 'endpoint' must be defined for the native pz runner")

    api_base = openrouter_config.get("api_base", "https://openrouter.ai/api/v1")
    api_key_env_var = openrouter_config.get("api_key_env_var", "OPENROUTER_API_KEY")
    extra_headers = openrouter_config.get("extra_headers")
    if extra_headers is not None and (
        not isinstance(extra_headers, dict)
        or not all(isinstance(key, str) and isinstance(val, str) for key, val in extra_headers.items())
    ):
        raise ValueError("OpenRouter extra_headers must be a mapping of strings to strings")

    if not isinstance(api_base, str) or not api_base:
        raise ValueError("OpenRouter config must define a non-empty api_base")
    if not isinstance(api_key_env_var, str) or not api_key_env_var:
        raise ValueError("OpenRouter config must define a non-empty api_key_env_var")

    return OpenRouterConfig(
        api_base=api_base.rstrip("/"),
        api_key_env_var=api_key_env_var,
        extra_headers=dict(extra_headers) if extra_headers else None,
        provider=provider,
    )


def _parse_execution_config(execution_config: dict[str, Any]) -> ExecutionConfig:
    execution_strategy = execution_config.get("execution_strategy", "parallel")
    optimizer_strategy = execution_config.get("optimizer_strategy", "none")
    max_workers = int(execution_config.get("max_workers", 64))
    reasoning_effort = execution_config.get("reasoning_effort", "disable")
    progress = execution_config.get("progress")

    if execution_strategy not in {"parallel", "sequential"}:
        raise ValueError(f"Unsupported execution strategy: {execution_strategy}")
    if optimizer_strategy not in {"none", "pareto"}:
        raise ValueError(f"Unsupported optimizer strategy: {optimizer_strategy}")
    if max_workers < 1:
        raise ValueError("Execution max_workers must be positive")
    if progress is not None and not isinstance(progress, bool):
        raise ValueError("Execution progress must be a boolean when provided")

    return ExecutionConfig(
        execution_strategy=execution_strategy,
        optimizer_strategy=optimizer_strategy,
        max_workers=max_workers,
        reasoning_effort=reasoning_effort,
        progress=progress,
    )


def _resolve_categories_from_config(categories_config: dict[str, Any]) -> list[str]:
    explicit_categories = categories_config.get("selected") or categories_config.get("explicit")
    if explicit_categories is not None:
        return resolve_selected_categories(selected_categories=explicit_categories)

    buckets = categories_config.get("buckets", categories_config)
    if any(key in buckets for key in ("easy", "medium", "hard")):
        return resolve_selected_categories(difficulty_buckets=buckets)

    return resolve_selected_categories()


def _resolve_difficulty_buckets(categories_config: dict[str, Any]) -> dict[str, list[str]]:
    buckets = categories_config.get("buckets", categories_config)
    resolved_buckets = {
        difficulty: list(buckets.get(difficulty, []))
        for difficulty in ("easy", "medium", "hard")
    }

    all_bucket_categories = [category for values in resolved_buckets.values() for category in values]
    if all_bucket_categories:
        validate_category_names(all_bucket_categories)
    return resolved_buckets


def _build_category_to_difficulty_map(
    selected_categories: list[str],
    difficulty_buckets: dict[str, list[str]],
) -> dict[str, str]:
    category_to_difficulty: dict[str, str] = {}
    for difficulty in ("easy", "medium", "hard"):
        for category in difficulty_buckets.get(difficulty, []):
            if category in category_to_difficulty:
                raise ValueError(
                    f"CUAD category {category!r} appears in multiple difficulty buckets: "
                    f"{category_to_difficulty[category]!r} and {difficulty!r}"
                )
            category_to_difficulty[category] = difficulty

    missing_categories = [category for category in selected_categories if category not in category_to_difficulty]
    if missing_categories:
        raise ValueError(
            "Selected CUAD categories are missing from difficulty buckets: "
            f"{missing_categories}"
        )
    return {category: category_to_difficulty[category] for category in selected_categories}


def generate_jobs(config: BenchmarkConfig) -> list[BenchmarkJob]:
    jobs = []
    for model_id in config.models:
        if config.job_mode == "grouped":
            jobs.append(
                BenchmarkJob(
                    job_id=_job_id(model_id, "selected"),
                    model_id=model_id,
                    categories=config.selected_categories,
                    job_mode="grouped",
                )
            )
        else:
            for category in config.selected_categories:
                jobs.append(
                    BenchmarkJob(
                        job_id=_job_id(model_id, category),
                        model_id=model_id,
                        categories=[category],
                        job_mode="per-category",
                    )
                )
    return jobs


def build_openrouter_model(config: BenchmarkConfig, model_id: str) -> pz.Model:
    load_dotenv(override=True)
    api_key = os.getenv(config.openrouter.api_key_env_var)
    if not api_key:
        if config.openrouter.provider == "openrouter":
            raise RuntimeError(
                f"OpenRouter config requires {config.openrouter.api_key_env_var} to be set"
            )
        api_key = "fake-api-key"

    model_kwargs: dict[str, Any] = {"api_key": api_key}
    if config.openrouter.extra_headers:
        model_kwargs["extra_headers"] = dict(config.openrouter.extra_headers)

    return pz.Model(
        model_id,
        api_base=config.openrouter.api_base,
        **model_kwargs,
    )


def run_benchmark_job(config: BenchmarkConfig, job: BenchmarkJob) -> JobResult:
    model = build_openrouter_model(config, job.model_id)
    progress = config.execution.progress if config.execution.progress is not None else not config.verbose
    qp_config = pz.QueryProcessorConfig(
        available_models=[model],
        optimizer_strategy=config.execution.optimizer_strategy,
        execution_strategy=config.execution.execution_strategy,
        reasoning_effort=config.execution.reasoning_effort,
        verbose=config.verbose,
        progress=progress,
        max_workers=config.execution.max_workers,
    )

    dataset = CUADDataset(
        split=config.split,
        num_contracts=config.num_contracts,
        seed=config.seed,
        dataset_mode=config.dataset_mode,
        data_dir=str(config.data_dir) if config.data_dir else None,
    )
    query = build_cuad_query(dataset, config.convert_mode, selected_categories=job.categories)
    data_record_collection = query.run(config=qp_config)
    predictions_wide = data_record_collection.to_df()
    predictions_long = normalize_predictions_to_long(
        predictions_wide,
        selected_categories=job.categories,
        metadata={
            "job_id": job.job_id,
            "model_id": job.model_id,
            "job_mode": job.job_mode,
            "run_kind": "native-pz",
            "provider": config.openrouter.provider,
            "api_base": config.openrouter.api_base,
            "execution_strategy": config.execution.execution_strategy,
            "max_workers": config.execution.max_workers,
            "difficulty": None,
        },
    )
    predictions_long["difficulty"] = predictions_long["category"].map(config.category_to_difficulty)
    return JobResult(
        job=job,
        predictions_long=predictions_long,
        predictions_wide=predictions_wide,
        stats=data_record_collection.execution_stats.to_json(),
        execution_strategy=config.execution.execution_strategy,
        max_workers=config.execution.max_workers,
    )


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    build_output_dir(config).mkdir(parents=True, exist_ok=True)
    jobs = generate_jobs(config)
    print(
        "Starting CUAD benchmark: "
        f"job_mode={config.job_mode}, models={len(config.models)}, "
        f"categories={len(config.selected_categories)}, jobs={len(jobs)}, "
        f"convert_mode={config.convert_mode}, "
        f"rows={config.num_contracts}, dataset_mode={config.dataset_mode}, seed={config.seed}, "
        f"execution_strategy={config.execution.execution_strategy}, max_workers={config.execution.max_workers}"
    )

    results = [run_benchmark_job(config, job) for job in jobs]
    predictions_long = pd.concat([result.predictions_long for result in results], ignore_index=True)
    predictions_wide = long_predictions_to_wide(predictions_long)
    label_df = get_label_df(
        num_contracts=config.num_contracts,
        seed=config.seed,
        split=config.split,
        selected_categories=config.selected_categories,
        dataset_mode=config.dataset_mode,
        data_dir=str(config.data_dir) if config.data_dir else None,
    )
    metrics_by_model = {}
    for model_id in config.models:
        model_predictions = predictions_long[predictions_long["model_id"] == model_id]
        base_metrics = compute_metrics(
            label_df,
            long_predictions_to_wide(model_predictions),
            selected_categories=config.selected_categories,
        )
        metrics_by_model[model_id] = augment_metrics_with_difficulty(
            base_metrics,
            label_df,
            model_predictions,
            config.selected_categories,
            config.category_to_difficulty,
        )

    write_outputs(config, results, predictions_long, predictions_wide, metrics_by_model)
    return {"results": results, "metrics": metrics_by_model}


def long_predictions_to_wide(predictions_long: pd.DataFrame) -> pd.DataFrame:
    if predictions_long.empty:
        return pd.DataFrame()
    index_columns = [
        column
        for column in ("contract_id", "title", "dataset_mode", "chunk_index", "chunk_start", "chunk_end", "source_paragraph_index")
        if column in predictions_long.columns
    ]
    wide = (
        predictions_long.pivot_table(
            index=index_columns,
            columns="category",
            values="predictions",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return wide


def write_outputs(
    config: BenchmarkConfig,
    results: list[JobResult],
    predictions_long: pd.DataFrame,
    predictions_wide: pd.DataFrame,
    metrics_by_model: dict[str, Any],
) -> None:
    run_prefix = build_run_prefix(config)
    output_dir = build_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_long.to_json(output_dir / f"{run_prefix}-predictions-long.json", orient="records", indent=2)
    predictions_wide.to_json(output_dir / f"{run_prefix}-predictions-wide.json", orient="records", indent=2)
    with open(output_dir / f"{run_prefix}-metrics.json", "w") as f:
        json.dump(_json_safe(metrics_by_model), f, indent=2)
    with open(output_dir / f"{run_prefix}-resolved-run.json", "w") as f:
        json.dump(_json_safe(build_resolved_run_metadata(config)), f, indent=2)

    job_stats = {
        result.job.job_id: {
            "model_id": result.job.model_id,
            "categories": result.job.categories,
            "category_to_difficulty": {
                category: config.category_to_difficulty[category]
                for category in result.job.categories
            },
            "job_mode": result.job.job_mode,
            "run_kind": "native-pz",
            "provider": config.openrouter.provider,
            "api_base": config.openrouter.api_base,
            "execution_strategy": result.execution_strategy,
            "max_workers": result.max_workers,
            "stats": result.stats,
        }
        for result in results
    }
    with open(output_dir / f"{run_prefix}-job-stats.json", "w") as f:
        json.dump(_json_safe(job_stats), f, indent=2)
    generate_benchmark_plots(
        output_dir,
        run_prefix,
        metrics_by_model,
        results,
        config.category_to_difficulty,
    )


def _job_id(model_id: str, category_or_group: str) -> str:
    return _safe_name(f"{model_id}-{category_or_group}")


def build_run_prefix(config: BenchmarkConfig) -> str:
    return _safe_name(
        f"{config.exp_name}-{config.job_mode}-n{config.num_contracts}-seed{config.seed}-{config.run_id}"
    )


def build_output_dir(config: BenchmarkConfig) -> Path:
    return config.output_dir / build_run_prefix(config)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def generate_benchmark_plots(
    output_dir: Path,
    run_prefix: str,
    metrics_by_model: dict[str, Any],
    results: list[JobResult],
    category_to_difficulty: dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = build_plot_rows(metrics_by_model, results, category_to_difficulty)
    difficulty_order = [difficulty for difficulty in ("easy", "medium", "hard") if difficulty in plot_rows]
    if not difficulty_order:
        return

    plot_specs = [
        ("runtime_seconds", "Runtime", "Runtime (s)", "runtime-by-difficulty"),
        ("precision", "Precision", "Precision", "precision-by-difficulty"),
        ("accuracy", "Accuracy", "Accuracy", "accuracy-by-difficulty"),
        ("f1", "F1 Score", "F1 Score", "f1-by-difficulty"),
    ]
    for metric_key, title, ylabel, suffix in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        render_grouped_bar_plot(ax, plot_rows, difficulty_order, metric_key, title, ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / f"{run_prefix}-{suffix}.png", dpi=200)
        plt.close(fig)


def build_plot_rows(
    metrics_by_model: dict[str, Any],
    results: list[JobResult],
    category_to_difficulty: dict[str, str],
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    runtime_samples_by_model_and_difficulty = aggregate_runtime_samples_by_difficulty(results, category_to_difficulty)
    plot_rows: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for model_id, model_metrics in metrics_by_model.items():
        per_category = model_metrics.get("per_category", {})
        per_difficulty_metric_samples = _collect_metric_samples_by_difficulty(per_category)
        difficulty_keys = set(model_metrics.get("per_difficulty", {})) | set(per_difficulty_metric_samples)
        for difficulty in difficulty_keys:
            plot_rows.setdefault(difficulty, {})
            difficulty_metric_samples = per_difficulty_metric_samples.get(difficulty, {})
            plot_rows[difficulty][model_id] = {
                "runtime_seconds": _summarize_numeric_samples(
                    runtime_samples_by_model_and_difficulty.get(model_id, {}).get(difficulty, [])
                ),
                "precision": _summarize_numeric_samples(
                    difficulty_metric_samples.get("precision", [])
                ),
                "accuracy": _summarize_numeric_samples(
                    difficulty_metric_samples.get("accuracy", [])
                ),
                "f1": _summarize_numeric_samples(
                    difficulty_metric_samples.get("f1", [])
                ),
            }
    return plot_rows


def aggregate_runtime_samples_by_difficulty(
    results: list[JobResult],
    category_to_difficulty: dict[str, str],
) -> dict[str, dict[str, list[float]]]:
    runtime_by_model_and_difficulty: dict[str, dict[str, list[float]]] = {}
    for result in results:
        total_runtime = float(result.stats.get("total_execution_time", 0.0) or 0.0)
        difficulty_counts: dict[str, int] = {}
        for category in result.job.categories:
            difficulty = category_to_difficulty.get(category)
            if difficulty is None:
                continue
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        if not difficulty_counts:
            continue

        total_categories = sum(difficulty_counts.values())
        model_runtime = runtime_by_model_and_difficulty.setdefault(result.job.model_id, {})
        for difficulty, count in difficulty_counts.items():
            allocated_runtime = total_runtime * (count / total_categories)
            per_category_runtime = allocated_runtime / count if count > 0 else 0.0
            runtime_samples = model_runtime.setdefault(difficulty, [])
            runtime_samples.extend([per_category_runtime] * count)
    return runtime_by_model_and_difficulty


def render_grouped_bar_plot(
    ax: Any,
    plot_rows: dict[str, dict[str, dict[str, dict[str, float | None]]]],
    difficulty_order: list[str],
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    models = sorted({model_id for difficulty_data in plot_rows.values() for model_id in difficulty_data})
    if not models:
        return

    x_positions = np.arange(len(difficulty_order))
    width = 0.8 / max(len(models), 1)
    cmap = plt.get_cmap("tab10")
    colors = {model_id: cmap(idx % 10) for idx, model_id in enumerate(models)}

    for idx, model_id in enumerate(models):
        offsets = x_positions - 0.4 + (idx + 0.5) * width
        values = [
            _coerce_plot_value(plot_rows.get(difficulty, {}).get(model_id, {}).get(metric_key, {}).get("mean"))
            for difficulty in difficulty_order
        ]
        errors = [
            _coerce_plot_error_value(plot_rows.get(difficulty, {}).get(model_id, {}).get(metric_key, {}).get("std"))
            for difficulty in difficulty_order
        ]
        ax.bar(
            offsets,
            values,
            width=width,
            label=model_id,
            color=colors.get(model_id),
            yerr=errors,
            capsize=4,
            ecolor=colors.get(model_id),
        )

    ax.set_title(title)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([difficulty.title() for difficulty in difficulty_order])
    ax.legend(title="Model")


def _coerce_plot_value(value: float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and np.isnan(value):
        return 0.0
    return float(value)


def _coerce_plot_error_value(value: float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and np.isnan(value):
        return 0.0
    return max(float(value), 0.0)


def _accuracy_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else 0.0


def _collect_metric_samples_by_difficulty(
    per_category: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    samples: dict[str, dict[str, list[float]]] = {}
    for category_metrics in per_category.values():
        difficulty = category_metrics.get("difficulty")
        if difficulty is None:
            continue
        difficulty_samples = samples.setdefault(
            difficulty,
            {"precision": [], "accuracy": [], "f1": []},
        )
        precision = category_metrics.get("precision")
        if precision is not None and not pd.isna(precision):
            difficulty_samples["precision"].append(float(precision))
        difficulty_samples["accuracy"].append(
            _accuracy_from_counts(
                int(category_metrics.get("tp", 0)),
                int(category_metrics.get("fp", 0)),
                int(category_metrics.get("fn", 0)),
            )
        )
        f1 = category_metrics.get("f1")
        if f1 is not None and not pd.isna(f1):
            difficulty_samples["f1"].append(float(f1))
    return samples


def _summarize_numeric_samples(samples: list[float]) -> dict[str, float | None]:
    cleaned = [float(sample) for sample in samples if sample is not None and not pd.isna(sample)]
    if not cleaned:
        return {"mean": None, "std": None}
    if len(cleaned) == 1:
        return {"mean": cleaned[0], "std": 0.0}
    return {
        "mean": float(np.mean(cleaned)),
        "std": float(np.std(cleaned, ddof=0)),
    }


def compute_metrics_from_long(
    label_long: pd.DataFrame,
    pred_long: pd.DataFrame,
    selected_categories: list[str],
) -> dict[str, Any]:
    merged = label_long.merge(
        pred_long[["contract_id", "category", "predictions"]],
        on=["contract_id", "category"],
        how="left",
    )

    aggregate_tp, aggregate_fp, aggregate_fn = 0, 0, 0
    per_category = {}
    for category in selected_categories:
        category_rows = merged[merged["category"] == category]
        category_tp, category_fp, category_fn = 0, 0, 0
        substr_ok = "Parties" in category

        for _, row in category_rows.iterrows():
            labels = row["labels"]
            preds = row["predictions"] if isinstance(row["predictions"], list) else []
            entry_tp, entry_fp, entry_fn = _evaluate_metric_row(labels, preds, substr_ok)
            category_tp += entry_tp
            category_fp += entry_fp
            category_fn += entry_fn

        per_category[category] = _finalize_metric_counts(category_tp, category_fp, category_fn)
        aggregate_tp += category_tp
        aggregate_fp += category_fp
        aggregate_fn += category_fn

    metrics = _finalize_metric_counts(aggregate_tp, aggregate_fp, aggregate_fn)
    metrics["per_category"] = per_category
    return metrics


def augment_metrics_with_difficulty(
    base_metrics: dict[str, Any],
    label_df: pd.DataFrame,
    model_predictions: pd.DataFrame,
    selected_categories: list[str],
    category_to_difficulty: dict[str, str],
) -> dict[str, Any]:
    label_long = labels_to_long(label_df, selected_categories)
    label_long["difficulty"] = label_long["category"].map(category_to_difficulty)

    model_predictions = model_predictions.copy()
    model_predictions["difficulty"] = model_predictions["category"].map(category_to_difficulty)

    per_difficulty = {}
    for difficulty in ("easy", "medium", "hard"):
        difficulty_categories = [
            category for category in selected_categories if category_to_difficulty.get(category) == difficulty
        ]
        if not difficulty_categories:
            continue
        difficulty_labels = label_long[label_long["difficulty"] == difficulty]
        difficulty_predictions = model_predictions[model_predictions["difficulty"] == difficulty]
        per_difficulty[difficulty] = compute_metrics_from_long(
            difficulty_labels,
            difficulty_predictions,
            difficulty_categories,
        )

    enriched_per_category = {}
    for category, category_metrics in base_metrics["per_category"].items():
        enriched_per_category[category] = {
            **category_metrics,
            "difficulty": category_to_difficulty[category],
        }

    return {
        **base_metrics,
        "per_category": enriched_per_category,
        "per_difficulty": per_difficulty,
        "selected_categories": list(selected_categories),
        "category_to_difficulty": dict(category_to_difficulty),
    }


def build_resolved_run_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "exp_name": config.exp_name,
        "run_id": config.run_id,
        "output_dir": str(build_output_dir(config)),
        "job_mode": config.job_mode,
        "convert_mode": config.convert_mode,
        "dataset_mode": config.dataset_mode,
        "data_dir": str(config.data_dir) if config.data_dir else None,
        "benchmark_unit": "chunk" if config.dataset_mode == "cuad-chunk" else "contract",
        "models": list(config.models),
        "selected_categories": list(config.selected_categories),
        "difficulty_buckets": {key: list(value) for key, value in config.difficulty_buckets.items()},
        "category_to_difficulty": dict(config.category_to_difficulty),
        "num_contracts": config.num_contracts,
        "seed": config.seed,
        "split": config.split,
        "openrouter": {
            "api_base": config.openrouter.api_base,
            "api_key_env_var": config.openrouter.api_key_env_var,
            "extra_headers": dict(config.openrouter.extra_headers or {}),
            "provider": config.openrouter.provider,
        },
        "execution": {
            "execution_strategy": config.execution.execution_strategy,
            "optimizer_strategy": config.execution.optimizer_strategy,
            "max_workers": config.execution.max_workers,
            "reasoning_effort": config.execution.reasoning_effort,
            "progress": config.execution.progress,
        },
        "runner_kind": "native-pz",
    }


def _evaluate_metric_row(labels: list[str], preds: list[str], substr_ok: bool) -> tuple[int, int, int]:
    normalized_preds = handle_empty_preds(preds)
    return evaluate_entry(labels, normalized_preds, substr_ok)


def _finalize_metric_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * (precision * recall) / (precision + recall) if not (np.isnan(precision) or np.isnan(recall)) and precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_arguments()
    config = load_benchmark_config(args.config, verbose_override=args.verbose)
    result = run_benchmark(config)
    for model_id, metrics in result["metrics"].items():
        print(
            f"{model_id}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
