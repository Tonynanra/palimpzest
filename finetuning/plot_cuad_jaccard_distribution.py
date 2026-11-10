"""Plot best-prediction Jaccard scores per gold span from saved CUAD completions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .cuad_demo import get_jaccard
    from .evaluate_cuad_sft import _normalize_predictions
    from .evaluate_cuad_span_transitions import _load_raw_records, _parse_completion
except ImportError:
    from cuad_demo import get_jaccard
    from evaluate_cuad_sft import _normalize_predictions
    from oevaluate_cuad_span_transitions import _load_raw_records, _parse_completion


def collect_scores(evaluation_dir: Path) -> dict[str, list[float]]:
    """Keep every gold annotation, including duplicates; missing predictions score zero."""
    comparison = json.loads((evaluation_dir / "comparison.json").read_text(encoding="utf-8"))
    if not isinstance(comparison, dict) or type(comparison.get("rows")) is not int or comparison["rows"] < 0:
        raise ValueError("comparison.json must declare a nonnegative integer rows count")
    for key in ("model_id", "mode"):
        if not isinstance(comparison.get(key), str):
            raise ValueError(f"comparison.json must declare {key}")
    captures = {
        role: _load_raw_records(
            evaluation_dir / f"{role}.raw.jsonl", role, comparison["rows"], comparison["model_id"],
            mode=comparison["mode"],
        )
        for role in ("vanilla", "sft")
    }
    scores: dict[str, list[float]] = {role: [] for role in captures}
    for row_index in range(1, comparison["rows"] + 1):
        row = captures["vanilla"][row_index]["row"]
        if row != captures["sft"][row_index]["row"]:
            raise ValueError(f"Row {row_index}: vanilla and SFT embedded inputs differ")
        for role, records in captures.items():
            parsed, _ = _parse_completion(records[row_index]["completion"])
            for category in row["categories"]:
                predictions = _normalize_predictions(parsed.get(category))
                for label in row.get("labels", {}).get(category, []):
                    scores[role].append(max((get_jaccard(label, pred) for pred in predictions), default=0.0))
    return scores


def plot_scores(scores: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    bins = [index / 20 for index in range(21)]
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        for role, label, color in (("vanilla", "Vanilla", "#0072B2"), ("sft", "SFT", "#D55E00")):
            ax.hist(scores[role], bins=bins, color=color, alpha=0.5, label=f"{label} (n={len(scores[role]):,})")
        ax.axvline(0.15, color="#333333", linestyle="--", linewidth=1.5, label="Threshold 0.15")
        ax.axvline(0.5, color="#333333", linestyle=":", linewidth=1.8, label="Threshold 0.5")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.15, 0.25, 0.5, 0.75, 1], ["0", "0.15", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel("Best Jaccard similarity to a prediction in the same category")
        ax.set_ylabel("Gold spans (count)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("CUAD Jaccard distribution: Vanilla vs SFT")
        ax.legend(loc="upper left", bbox_to_anchor=(0.2, 1))
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", type=Path, required=True,
        help="Completed evaluation containing comparison.json and both raw JSONL captures",
    )
    parser.add_argument("--output", type=Path, help="Defaults to evaluation-dir/sft_plots/jaccard_distribution.png")
    args = parser.parse_args(argv)
    output = args.output or args.evaluation_dir / "sft_plots" / "jaccard_distribution.png"
    try:
        scores = collect_scores(args.evaluation_dir)
        plot_scores(scores, output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{output.resolve()} ({len(scores['vanilla']):,} gold spans per model)")


if __name__ == "__main__":
    main()
