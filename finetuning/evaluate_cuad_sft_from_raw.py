"""Replay CUAD evaluation from saved completions, preserving original inference timings."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from .evaluate_cuad_sft import RowCompletion, _evaluate_rows, _metric_delta, _write_reports
    from .evaluate_cuad_span_transitions import _load_raw_records
except ImportError:
    from evaluate_cuad_sft import RowCompletion, _evaluate_rows, _metric_delta, _write_reports
    from evaluate_cuad_span_transitions import _load_raw_records

TIMING_KEYS = ("inference_walltime_seconds", "average_row_seconds", "rows_per_second")


def replay(
    evaluation_dir: Path,
    output_dir: Path,
    vanilla_raw: Path | None = None,
    sft_raw: Path | None = None,
) -> dict[str, Any]:
    """Recompute scores from captures; only timing values come from comparison.json."""
    evaluation_dir = evaluation_dir.resolve()
    output_dir = output_dir.resolve()
    raw_paths = {
        "vanilla": vanilla_raw or evaluation_dir / "vanilla.raw.jsonl",
        "sft": sft_raw or evaluation_dir / "sft.raw.jsonl",
    }
    if output_dir == evaluation_dir or any(path.resolve().parent == output_dir for path in raw_paths.values()):
        raise ValueError("Replay output must use a different directory from the saved inputs")

    reference = json.loads((evaluation_dir / "comparison.json").read_text(encoding="utf-8"))
    if not isinstance(reference, dict) or type(reference.get("rows")) is not int or reference["rows"] < 0:
        raise ValueError("comparison.json must declare a nonnegative integer rows count")
    for key in ("model_id", "mode", "data_split"):
        if not isinstance(reference.get(key), str):
            raise ValueError(f"comparison.json must declare {key}")
    captures = {}
    for role, path in raw_paths.items():
        records = _load_raw_records(path, role, reference["rows"], reference["model_id"], mode=reference["mode"])
        captures[role] = [records[index] for index in sorted(records)]
    for vanilla, sft in zip(captures["vanilla"], captures["sft"], strict=True):
        if vanilla["row"] != sft["row"]:
            raise ValueError(f"Row {vanilla['row_index']}: vanilla and SFT embedded inputs differ")

    # Raw captures contain no elapsed times. Preserve the saved inference values,
    # rather than reporting the much shorter replay duration as model performance.
    timing_bytes = (evaluation_dir / "inference_walltime.jsonl").read_bytes()
    results = {}
    for role, records in captures.items():
        completions = [
            RowCompletion(record["completion"], record.get("generation_kwargs", {}))
            for record in records
        ]
        result = _evaluate_rows(
            None, None, [record["row"] for record in records], 0, role, reference["model_id"],
            mode=reference["mode"], completions=completions,
        )
        for key in TIMING_KEYS:
            value = reference[role][key]
            if type(value) not in (int, float):
                raise ValueError(f"comparison.json: {role}.{key} must be numeric")
            result.metrics[key] = value
        results[role] = result.metrics

    timing_path = output_dir / "inference_walltime.jsonl"
    comparison = {
        **results,
        "delta": _metric_delta(results["vanilla"], results["sft"]),
        "mode": reference["mode"],
        "data_split": reference["data_split"],
        "rows": reference["rows"],
        "model_id": reference["model_id"],
        "inference_walltime_log": str(timing_path),
    }
    _write_reports(comparison, output_dir / "comparison.json", output_dir / "sft_plots")
    timing_path.write_bytes(timing_bytes)
    for role, path in raw_paths.items():
        shutil.copyfile(path, output_dir / f"{role}.raw.jsonl")
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", type=Path, required=True,
        help="Completed evaluation containing comparison.json, inference_walltime.jsonl, and raw captures",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate directory for replayed artifacts")
    parser.add_argument("--vanilla-raw", type=Path, help="Override the default evaluation-dir/vanilla.raw.jsonl")
    parser.add_argument("--sft-raw", type=Path, help="Override the default evaluation-dir/sft.raw.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = replay(args.evaluation_dir, args.output_dir, args.vanilla_raw, args.sft_raw)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
