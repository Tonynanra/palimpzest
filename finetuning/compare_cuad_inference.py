"""Compare saved Vanilla, SFT, and API CUAD inference results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .evaluate_cuad_sft import RowCompletion, _evaluate_rows, _metric_delta
    from .evaluate_cuad_span_transitions import _load_raw_records
except ImportError:
    from evaluate_cuad_sft import RowCompletion, _evaluate_rows, _metric_delta
    from evaluate_cuad_span_transitions import _load_raw_records

ROLES = ("vanilla", "sft", "api")
ROLE_LABELS = {"vanilla": "Vanilla", "sft": "SFT", "api": "API"}
ROLE_COLORS = {"vanilla": "#0072B2", "sft": "#D55E00", "api": "#009E73"}
TIMING_KEYS = ("inference_walltime_seconds", "average_row_seconds", "rows_per_second")
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "cuad-reasoning-comparison"


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _required_string(value: Any, path: Path, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path.name} must declare {key}")
    return value


def _required_rows(value: Any, path: Path) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path.name} must declare a positive integer rows count")
    return value


def _load_sources(
    vanilla_sft_dir: Path, api_result_dir: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    vanilla_sft_dir = vanilla_sft_dir.resolve()
    api_result_dir = api_result_dir.resolve()
    sft_path = vanilla_sft_dir / "comparison.json"
    api_path = api_result_dir / "openrouter.json"
    sft_summary = _load_object(sft_path)
    api_summary = _load_object(api_path)

    rows = _required_rows(sft_summary.get("rows"), sft_path)
    sft_model_id = _required_string(sft_summary.get("model_id"), sft_path, "model_id")
    mode = _required_string(sft_summary.get("mode"), sft_path, "mode")
    data_split = _required_string(sft_summary.get("data_split"), sft_path, "data_split")
    api_rows = _required_rows(api_summary.get("rows"), api_path)
    api_model_id = _required_string(api_summary.get("model_id"), api_path, "model_id")
    api_mode = _required_string(api_summary.get("mode"), api_path, "mode")
    api_split = _required_string(api_summary.get("data_split"), api_path, "data_split")
    if rows != api_rows:
        raise ValueError(f"Result row counts differ: Vanilla/SFT {rows}, API {api_rows}")
    if mode != api_mode:
        raise ValueError(f"Result modes differ: Vanilla/SFT {mode!r}, API {api_mode!r}")
    if data_split != api_split:
        raise ValueError(f"Result data splits differ: Vanilla/SFT {data_split!r}, API {api_split!r}")

    for role in ("vanilla", "sft"):
        metrics = sft_summary.get(role)
        if not isinstance(metrics, dict):
            raise ValueError(f"{sft_path.name} must contain {role} metrics")
        for key, expected in (("model_id", sft_model_id), ("mode", mode), ("rows", rows)):
            if metrics.get(key) != expected:
                raise ValueError(f"{sft_path.name}: {role} metrics have mismatched {key}")

    if api_summary.get("status") != "complete":
        raise ValueError(f"{api_path.name} must describe a complete API run")
    api_metrics = api_summary.get("metrics")
    if not isinstance(api_metrics, dict):
        raise ValueError(f"{api_path.name} must contain API metrics")
    for key, expected in (("model_id", api_model_id), ("mode", mode), ("rows", rows)):
        if api_metrics.get(key) != expected:
            raise ValueError(f"{api_path.name}: API metrics have mismatched {key}")
    if api_summary.get("responses_completed") != rows:
        raise ValueError(f"{api_path.name} must contain all {rows} completed API responses")

    captures = {
        "vanilla": _load_raw_records(
            vanilla_sft_dir / "vanilla.raw.jsonl",
            "vanilla",
            rows,
            sft_model_id,
            mode=mode,
        ),
        "sft": _load_raw_records(
            vanilla_sft_dir / "sft.raw.jsonl",
            "sft",
            rows,
            sft_model_id,
            mode=mode,
        ),
        "api": _load_raw_records(
            api_result_dir / "openrouter.raw.jsonl",
            "openrouter",
            rows,
            api_model_id,
            mode=mode,
        ),
    }
    ordered = {role: [records[index] for index in range(1, rows + 1)] for role, records in captures.items()}
    for row_index in range(1, rows + 1):
        reference_row = ordered["vanilla"][row_index - 1]["row"]
        if ordered["sft"][row_index - 1]["row"] != reference_row:
            raise ValueError(f"Row {row_index}: Vanilla and SFT embedded inputs differ")
        if ordered["api"][row_index - 1]["row"] != reference_row:
            raise ValueError(f"Row {row_index}: API embedded input differs from Vanilla/SFT")

    return {
        "rows": rows,
        "mode": mode,
        "data_split": data_split,
        "model_ids": {"vanilla": sft_model_id, "sft": sft_model_id, "api": api_model_id},
        "timing_metrics": {
            "vanilla": sft_summary["vanilla"],
            "sft": sft_summary["sft"],
            "api": api_metrics,
        },
        "sources": {"vanilla_sft": str(vanilla_sft_dir), "api": str(api_result_dir)},
    }, ordered


def _score_role(
    role: str,
    records: list[dict[str, Any]],
    model_id: str,
    mode: str,
    timing_source: dict[str, Any],
) -> dict[str, Any]:
    result = _evaluate_rows(
        None,
        None,
        [record["row"] for record in records],
        0,
        role,
        model_id,
        mode=mode,
        completions=[RowCompletion(record["completion"], record.get("generation_kwargs", {})) for record in records],
    )
    for key in TIMING_KEYS:
        value = timing_source.get(key)
        if type(value) not in (int, float):
            raise ValueError(f"Saved {role} metrics must declare numeric {key}")
        result.metrics[key] = value
    return result.metrics


def compare_results(
    vanilla_sft_dir: Path,
    api_result_dir: Path,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Score complete saved captures and write reports and plots to output_dir."""
    vanilla_sft_dir = vanilla_sft_dir.resolve()
    api_result_dir = api_result_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir in (vanilla_sft_dir, api_result_dir):
        raise ValueError("Output directory must differ from both source directories")

    source_meta, captures = _load_sources(vanilla_sft_dir, api_result_dir)
    results = {
        role: _score_role(
            role,
            captures[role],
            source_meta["model_ids"][role],
            source_meta["mode"],
            source_meta["timing_metrics"][role],
        )
        for role in ROLES
    }
    comparison = {
        **results,
        "api_minus_sft": _metric_delta(results["sft"], results["api"]),
        "mode": source_meta["mode"],
        "data_split": source_meta["data_split"],
        "rows": source_meta["rows"],
        "model_ids": source_meta["model_ids"],
        "source_directories": source_meta["sources"],
    }
    _write_reports(comparison, output_dir)
    return comparison


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_reports(comparison: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        _write_json(output_dir / f"{role}.json", comparison[role])
    _write_json(output_dir / "comparison.json", comparison)
    _plot_comparison(comparison, output_dir / "sft_plots")


def _plot_comparison(comparison: dict[str, Any], plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    width = 0.8 / len(ROLES)

    def grouped_bars(
        ax: Any, labels: list[str], metric_values: dict[str, list[float]], *, horizontal: bool = False
    ) -> None:
        positions = list(range(len(labels)))
        for index, role in enumerate(ROLES):
            offsets = [position - 0.4 + width * (index + 0.5) for position in positions]
            values = metric_values[role]
            if horizontal:
                ax.barh(offsets, values, height=width, label=ROLE_LABELS[role], color=ROLE_COLORS[role])
            else:
                ax.bar(offsets, values, width=width, label=ROLE_LABELS[role], color=ROLE_COLORS[role])
        if horizontal:
            ax.set_yticks(positions, labels)
        else:
            ax.set_xticks(positions, labels)
        ax.legend()

    metric_labels = ["Precision", "Recall", "F1"]
    aggregate_values = {
        role: [float(comparison[role].get(key) or 0.0) for key in ("precision", "recall", "f1")] for role in ROLES
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_bars(ax, metric_labels, aggregate_values)
    ax.set_ylim(0, 1)
    ax.set_title("Aggregate extraction metrics")
    fig.tight_layout()
    fig.savefig(plot_dir / "aggregate_metrics.png", dpi=180)
    plt.close(fig)

    categories = sorted(set().union(*(comparison[role].get("per_category", {}) for role in ROLES)))
    for metric in ("precision", "recall", "f1"):
        values = {
            role: [
                float(comparison[role].get("per_category", {}).get(category, {}).get(metric) or 0.0)
                for category in categories
            ]
            for role in ROLES
        }
        fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(categories))))
        grouped_bars(ax, categories, values, horizontal=True)
        ax.set_xlim(0, 1)
        ax.set_title(f"Per-category {metric}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"per_category_{metric}.png", dpi=180)
        plt.close(fig)

    deltas = comparison["api_minus_sft"]["per_category"]
    f1_deltas = [float(deltas.get(category, {}).get("f1") or 0.0) for category in categories]
    order = sorted(range(len(categories)), key=lambda index: f1_deltas[index])
    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(categories))))
    ax.barh([categories[index] for index in order], [f1_deltas[index] for index in order], color=ROLE_COLORS["api"])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("API minus SFT per-category F1")
    fig.tight_layout()
    fig.savefig(plot_dir / "per_category_f1_delta.png", dpi=180)
    plt.close(fig)

    format_labels = ["Parse success", "Grounding", "Missing-key rate", "Extra-key rate"]
    format_keys = ["parse_success_rate", "grounding_rate", "missing_key_rate", "extra_key_rate"]
    format_values = {role: [float(comparison[role].get(key) or 0.0) for key in format_keys] for role in ROLES}
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped_bars(ax, format_labels, format_values)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(format_labels, rotation=20, ha="right")
    ax.set_title("Formatting and grounding metrics")
    fig.tight_layout()
    fig.savefig(plot_dir / "format_metrics.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla_sft_dir", type=Path, help="Directory containing the SFT comparison and raw captures")
    parser.add_argument("--api_result_dir", type=Path, help="Directory containing openrouter.json and its raw capture")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        comparison = compare_results(args.vanilla_sft_dir, args.api_result_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(comparison, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
