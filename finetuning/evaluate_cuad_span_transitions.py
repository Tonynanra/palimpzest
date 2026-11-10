"""Score paired span transitions from completed CUAD raw-output captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .cuad_demo import IOU_THRESH, evaluate_entry
    from .cuad_schema import get_category_names
    from .cuad_sft_data import _TextPromptModel
    from .evaluate_cuad_sft import _normalize_predictions
except ImportError:
    from cuad_demo import IOU_THRESH, evaluate_entry
    from cuad_schema import get_category_names
    from cuad_sft_data import _TextPromptModel
    from evaluate_cuad_sft import _normalize_predictions

from palimpzest.constants import Cardinality
from palimpzest.query.generators.generators import get_json_from_answer

IMPROVEMENT_COLOR = "#0072B2"
REGRESSION_COLOR = "#D55E00"
RATE_SPECS = (
    ("fn_recovery", "FN recovery", "c", "cd"),
    ("fp_correction", "FP correction", "g", "gh"),
    ("tp_loss", "TP loss", "b", "ab"),
    ("new_fp_introduction", "New FP introduction", "f", "ef"),
)
# Keys are (gold label, Vanilla prediction, SFT prediction).
CELL_BY_OUTCOME = {
    (True, True, True): "a",
    (True, True, False): "b",
    (True, False, True): "c",
    (True, False, False): "d",
    (False, False, False): "e",
    (False, False, True): "f",
    (False, True, False): "g",
    (False, True, True): "h",
}


def _summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    rates = {}
    for key, _, numerator_cell, denominator_cells in RATE_SPECS:
        numerator = counts[numerator_cell]
        denominator = sum(counts[cell] for cell in denominator_cells)
        rates[key] = {
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator if denominator else None,
        }
    return {
        "counts": counts,
        "support": {
            "gold_positive": sum(counts[cell] for cell in "abcd"),
            "gold_negative": sum(counts[cell] for cell in "efgh"),
            "total": sum(counts.values()),
        },
        "rates": rates,
    }


def _load_raw_records(
    path: Path,
    role: str,
    expected_rows: int,
    model_id: str,
    *,
    mode: str = "all",
) -> dict[int, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSONL: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
            row_index = record.get("row_index")
            if type(row_index) is not int or row_index < 1:
                raise ValueError(f"{path.name}:{line_number}: invalid row_index")
            if row_index in records:
                raise ValueError(f"{path.name}: duplicate row_index {row_index}")
            for key, expected in (("model_role", role), ("model_id", model_id), ("mode", mode)):
                if record.get(key) != expected:
                    raise ValueError(f"{path.name}: row {row_index} has mismatched {key}")
            if not isinstance(record.get("row"), dict):
                raise ValueError(f"{path.name}: row {row_index} has no embedded input row")
            if not isinstance(record.get("completion"), str):
                raise ValueError(f"{path.name}: row {row_index} completion must be a string")
            if records and record.get("checkpoint") != next(iter(records.values())).get("checkpoint"):
                raise ValueError(f"{path.name}: row {row_index} has mismatched checkpoint")
            records[row_index] = record

    expected_indices = set(range(1, expected_rows + 1))
    if set(records) != expected_indices:
        missing = sorted(expected_indices - set(records))
        extra = sorted(set(records) - expected_indices)
        raise ValueError(f"{path.name}: incomplete coverage; missing rows {missing}, extra rows {extra}")
    return records


def _validate_row(row: dict[str, Any], row_index: int, categories: list[str]) -> None:
    selected = row.get("categories")
    labels = row.get("labels")
    if (
        row.get("view") != "all"
        or not isinstance(selected, list)
        or not all(isinstance(category, str) for category in selected)
        or len(selected) != len(categories)
        or set(selected) != set(categories)
        or not isinstance(labels, dict)
        or set(labels) != set(categories)
    ):
        raise ValueError(f"Row {row_index}: full-category 'all' inputs and labels are required")
    for category in categories:
        spans = labels[category]
        if not isinstance(spans, list) or any(not isinstance(span, str) or not span for span in spans):
            raise ValueError(f"Row {row_index}: {category} labels must be a list of nonempty strings")


def _parse_completion(completion: str) -> tuple[dict[str, Any], bool]:
    try:
        parsed = get_json_from_answer(completion, _TextPromptModel(), Cardinality.ONE_TO_ONE)
        if not isinstance(parsed, dict):
            raise ValueError("completion JSON is not an object")
    except Exception:
        return {}, True
    return parsed, False


def score_evaluation(evaluation_dir: Path) -> dict[str, Any]:
    evaluation_dir = evaluation_dir.resolve()
    comparison_path = evaluation_dir / "comparison.json"
    if not comparison_path.is_file():
        raise ValueError(f"Completed comparison.json is required: {comparison_path}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if not isinstance(comparison, dict) or comparison.get("mode") != "all":
        raise ValueError("comparison.json must describe a full-category 'all' evaluation")
    expected_rows = comparison.get("rows")
    model_id = comparison.get("model_id")
    if type(expected_rows) is not int or expected_rows < 0:
        raise ValueError("comparison.json must declare a nonnegative integer rows count")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("comparison.json must declare model_id")
    records = {}
    for role in ("vanilla", "sft"):
        metrics = comparison.get(role)
        if (
            not isinstance(metrics, dict)
            or type(metrics.get("rows")) is not int
            or metrics["rows"] != expected_rows
            or metrics.get("model_id") != model_id
            or metrics.get("mode") != "all"
        ):
            raise ValueError(f"comparison.json: {role} rows, model_id, and mode must match the evaluation")
        records[role] = _load_raw_records(
            evaluation_dir / f"{role}.raw.jsonl",
            role,
            expected_rows,
            model_id,
        )

    categories = get_category_names()
    per_category = {category: dict.fromkeys("abcdefgh", 0) for category in categories}
    parse_failures = {"vanilla": 0, "sft": 0}
    for row_index in range(1, expected_rows + 1):
        row = records["vanilla"][row_index]["row"]
        if row != records["sft"][row_index]["row"]:
            raise ValueError(f"Row {row_index}: mismatched embedded input rows")
        _validate_row(row, row_index, categories)
        parsed = {}
        for role in ("vanilla", "sft"):
            parsed[role], failed = _parse_completion(records[role][row_index]["completion"])
            parse_failures[role] += int(failed)
        labels = row["labels"]
        candidates = {span for spans in labels.values() for span in spans}
        for category in categories:
            gold = set(labels[category])
            predictions = {role: _normalize_predictions(parsed[role].get(category)) for role in parsed}
            substr_ok = "Parties" in category
            for span in candidates:
                vanilla = evaluate_entry([span], predictions["vanilla"], substr_ok)[0] > 0
                sft = evaluate_entry([span], predictions["sft"], substr_ok)[0] > 0
                cell = CELL_BY_OUTCOME[(span in gold, vanilla, sft)]
                per_category[category][cell] += 1

    aggregate = {cell: sum(counts[cell] for counts in per_category.values()) for cell in "abcdefgh"}
    return {
        "metadata": {
            "evaluation_dir": str(evaluation_dir),
            "source_files": {
                "comparison": str(comparison_path),
                "vanilla": str(evaluation_dir / "vanilla.raw.jsonl"),
                "sft": str(evaluation_dir / "sft.raw.jsonl"),
            },
            "model_id": model_id,
            "mode": "all",
            "rows": expected_rows,
            "checkpoints": {role: records[role][1].get("checkpoint") if expected_rows else None for role in records},
            "parse_failures": parse_failures,
            "matching": {"iou_threshold": IOU_THRESH, "parties_substring_exception": True},
            "candidate_population": "Distinct annotated answer strings within each chunk, across all categories.",
            "gold_membership": "Exact original string membership in the target category's labels.",
        },
        "aggregate": _summarize_counts(aggregate),
        "per_category": {category: _summarize_counts(counts) for category, counts in per_category.items()},
    }


def _plot_rates(report: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    rates = report["aggregate"]["rates"]
    heights = [
        rates[key]["value"] * 100 if rates[key]["value"] is not None else float("nan") for key, _, _, _ in RATE_SPECS
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        positions = list(range(len(RATE_SPECS)))
        ax.bar(positions, heights, color=[IMPROVEMENT_COLOR] * 2 + [REGRESSION_COLOR] * 2)
        ax.set_xticks(positions, [label for _, label, _, _ in RATE_SPECS])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Rate (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_title("Vanilla → SFT span transitions")
        ax.legend(
            handles=[
                Patch(facecolor=IMPROVEMENT_COLOR, label="Improvements"),
                Patch(facecolor=REGRESSION_COLOR, label="Regressions"),
            ]
        )
        for position, (key, _, _, _) in enumerate(RATE_SPECS):
            rate = rates[key]
            value = rate["value"]
            label = f"{value * 100:.2f}%" if value is not None else "N/A"
            label += f"\n{rate['numerator']}/{rate['denominator']}"
            inside = value is not None and value >= 0.9
            ax.annotate(
                label,
                (position, value * 100 if value is not None else 0),
                xytext=(0, -6 if inside else 6),
                textcoords="offset points",
                ha="center",
                va="top" if inside else "bottom",
                color="white" if inside else "black",
            )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)


def _render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    counts = aggregate["counts"]
    metadata = report["metadata"]
    lines = [
        "# CUAD span transitions",
        "",
        f"Model: `{metadata['model_id']}`. Evaluated chunks: {metadata['rows']}.",
        "",
        f"Gold-positive cases: {aggregate['support']['gold_positive']:,}. "
        f"Gold-negative cases: {aggregate['support']['gold_negative']:,}.",
        "",
        "Candidates are distinct annotated answer strings within each chunk. "
        "Each candidate is evaluated separately for every category; unannotated text adds no cases.",
        "",
        f"Matching: Jaccard ≥ {metadata['matching']['iou_threshold']}, including the `Parties` substring exception.",
        "",
        f"Parse failures: Vanilla {metadata['parse_failures']['vanilla']}, SFT {metadata['parse_failures']['sft']}. "
        "Malformed/non-object responses and missing category values produce empty predictions.",
        "",
        "## Gold-positive cases (Y = 1)",
        "",
        "| Vanilla ↓ / SFT → | Predict 1 | Predict 0 |",
        "|---|---:|---:|",
        f"| Predict 1 | a = {counts['a']} | b = {counts['b']} |",
        f"| Predict 0 | c = {counts['c']} | d = {counts['d']} |",
        "",
        "## Gold-negative cases (Y = 0)",
        "",
        "| Vanilla ↓ / SFT → | Predict 1 | Predict 0 |",
        "|---|---:|---:|",
        f"| Predict 1 | h = {counts['h']} | g = {counts['g']} |",
        f"| Predict 0 | f = {counts['f']} | e = {counts['e']} |",
        "",
        "## Transition rates",
        "",
        "| Rate | Formula | Numerator / denominator | Value |",
        "|---|---|---:|---:|",
    ]
    for key, label, numerator_cell, denominator_cells in RATE_SPECS:
        rate = aggregate["rates"][key]
        value = f"{rate['value'] * 100:.2f}%" if rate["value"] is not None else "N/A"
        formula = f"{numerator_cell}/({'+'.join(denominator_cells)})"
        lines.append(f"| {label} | {formula} | {rate['numerator']} / {rate['denominator']} | {value} |")
    lines.extend(
        [
            "",
            "![Overall span transition rates](span_transitions.png)",
            "",
            "Per-category counts, supports, and rates are in [span_transitions.json](span_transitions.json).",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], evaluation_dir: Path) -> None:
    _plot_rates(report, evaluation_dir / "span_transitions.png")
    (evaluation_dir / "span_transitions.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (evaluation_dir / "span_transitions.md").write_text(_render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
        help="Completed all-category evaluation containing comparison.json and both raw JSONL files.",
    )
    args = parser.parse_args(argv)
    try:
        report = score_evaluation(args.evaluation_dir)
        write_reports(report, args.evaluation_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
