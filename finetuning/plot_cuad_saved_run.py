import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grouped CUAD benchmark plots from a saved run directory."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing *-metrics.json, *-job-stats.json, and *-resolved-run.json",
    )
    parser.add_argument(
        "--output-suffix",
        default="grouped-by-difficulty",
        help="Suffix to append to generated plot filenames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    prefix = infer_run_prefix(run_dir)

    metrics_by_model = load_json(run_dir / f"{prefix}-metrics.json")
    job_stats = load_json(run_dir / f"{prefix}-job-stats.json")
    resolved_run = load_json(run_dir / f"{prefix}-resolved-run.json")
    category_to_difficulty = resolved_run["category_to_difficulty"]

    plot_rows = build_plot_rows(metrics_by_model, job_stats, category_to_difficulty)
    difficulty_order = [difficulty for difficulty in ("easy", "medium", "hard") if difficulty in plot_rows]
    if not difficulty_order:
        raise ValueError(f"No difficulty buckets found in {run_dir}")

    plot_specs = [
        ("runtime_seconds", "Runtime", "Runtime (s)", "runtime"),
        ("precision", "Precision", "Precision", "precision"),
        ("accuracy", "Accuracy", "Accuracy", "accuracy"),
        ("f1", "F1 Score", "F1 Score", "f1"),
    ]
    for metric_key, title, ylabel, short_name in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        render_grouped_bar_plot(ax, plot_rows, difficulty_order, metric_key, title, ylabel)
        fig.tight_layout()
        output_path = run_dir / f"{prefix}-{short_name}-{args.output_suffix}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(output_path)


def infer_run_prefix(run_dir: Path) -> str:
    metrics_files = sorted(run_dir.glob("*-metrics.json"))
    if len(metrics_files) != 1:
        raise ValueError(f"Expected exactly one *-metrics.json file in {run_dir}, found {len(metrics_files)}")
    return metrics_files[0].name[: -len("-metrics.json")]


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def build_plot_rows(
    metrics_by_model: Dict[str, Any],
    job_stats: Dict[str, Any],
    category_to_difficulty: Dict[str, str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]:
    runtime_samples_by_model_and_difficulty = aggregate_runtime_samples_by_difficulty(
        job_stats,
        category_to_difficulty,
    )
    plot_rows = {}  # type: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]
    for model_id, model_metrics in metrics_by_model.items():
        per_category = model_metrics.get("per_category", {})
        per_difficulty_metric_samples = collect_metric_samples_by_difficulty(per_category)
        difficulty_keys = set(model_metrics.get("per_difficulty", {})) | set(per_difficulty_metric_samples)
        for difficulty in difficulty_keys:
            plot_rows.setdefault(difficulty, {})
            difficulty_metric_samples = per_difficulty_metric_samples.get(difficulty, {})
            plot_rows[difficulty][model_id] = {
                "runtime_seconds": summarize_numeric_samples(
                    runtime_samples_by_model_and_difficulty.get(model_id, {}).get(difficulty, [])
                ),
                "precision": summarize_numeric_samples(difficulty_metric_samples.get("precision", [])),
                "accuracy": summarize_numeric_samples(difficulty_metric_samples.get("accuracy", [])),
                "f1": summarize_numeric_samples(difficulty_metric_samples.get("f1", [])),
            }
    return plot_rows


def aggregate_runtime_samples_by_difficulty(
    job_stats: Dict[str, Any],
    category_to_difficulty: Dict[str, str],
) -> Dict[str, Dict[str, List[float]]]:
    runtime_by_model_and_difficulty = {}  # type: Dict[str, Dict[str, List[float]]]
    for job in job_stats.values():
        total_runtime = float(job.get("stats", {}).get("total_execution_time", 0.0) or 0.0)
        categories = job.get("categories", [])
        difficulty_counts = {}  # type: Dict[str, int]
        for category in categories:
            difficulty = category_to_difficulty.get(category)
            if difficulty is None:
                continue
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        if not difficulty_counts:
            continue

        total_categories = sum(difficulty_counts.values())
        model_runtime = runtime_by_model_and_difficulty.setdefault(job["model_id"], {})
        for difficulty, count in difficulty_counts.items():
            allocated_runtime = total_runtime * (count / total_categories)
            per_category_runtime = allocated_runtime / count if count > 0 else 0.0
            runtime_samples = model_runtime.setdefault(difficulty, [])
            runtime_samples.extend([per_category_runtime] * count)
    return runtime_by_model_and_difficulty


def render_grouped_bar_plot(
    ax: Any,
    plot_rows: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    difficulty_order: List[str],
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
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
            coerce_plot_value(plot_rows.get(difficulty, {}).get(model_id, {}).get(metric_key, {}).get("mean"))
            for difficulty in difficulty_order
        ]
        errors = [
            coerce_plot_error_value(plot_rows.get(difficulty, {}).get(model_id, {}).get(metric_key, {}).get("std"))
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


def collect_metric_samples_by_difficulty(
    per_category: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, List[float]]]:
    samples = {}  # type: Dict[str, Dict[str, List[float]]]
    for category_metrics in per_category.values():
        difficulty = category_metrics.get("difficulty")
        if difficulty is None:
            continue
        difficulty_samples = samples.setdefault(
            difficulty,
            {"precision": [], "accuracy": [], "f1": []},
        )
        precision = category_metrics.get("precision")
        if precision is not None and not is_nan_like(precision):
            difficulty_samples["precision"].append(float(precision))
        difficulty_samples["accuracy"].append(
            accuracy_from_counts(
                int(category_metrics.get("tp", 0)),
                int(category_metrics.get("fp", 0)),
                int(category_metrics.get("fn", 0)),
            )
        )
        f1 = category_metrics.get("f1")
        if f1 is not None and not is_nan_like(f1):
            difficulty_samples["f1"].append(float(f1))
    return samples


def summarize_numeric_samples(samples: List[float]) -> Dict[str, Optional[float]]:
    cleaned = [float(sample) for sample in samples if sample is not None and not is_nan_like(sample)]
    if not cleaned:
        return {"mean": None, "std": None}
    if len(cleaned) == 1:
        return {"mean": cleaned[0], "std": 0.0}
    return {
        "mean": float(np.mean(cleaned)),
        "std": float(np.std(cleaned, ddof=0)),
    }


def accuracy_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else 0.0


def coerce_plot_value(value: Optional[float]) -> float:
    if value is None or is_nan_like(value):
        return 0.0
    return float(value)


def coerce_plot_error_value(value: Optional[float]) -> float:
    if value is None or is_nan_like(value):
        return 0.0
    return max(float(value), 0.0)


def is_nan_like(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


if __name__ == "__main__":
    main()
