"""Run standalone CUAD inference through GPT-6 Luna on OpenRouter."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

from dotenv import dotenv_values
from openai import OpenAI

try:
    from .cuad_data import REPO_ROOT
    from .evaluate_cuad_sft import RowCompletion, _evaluate_rows, _load_rows, _positive_int, _prepare_rows
    from .train_cuad_sft import _text_messages
except ImportError:
    from cuad_data import REPO_ROOT
    from evaluate_cuad_sft import RowCompletion, _evaluate_rows, _load_rows, _positive_int, _prepare_rows
    from train_cuad_sft import _text_messages

MODEL_ID = "openai/gpt-6-luna"
API_BASE = "https://openrouter.ai/api/v1"
PRICE_PATH = f"/models/{MODEL_ID}/endpoints"
PRICE_URL = API_BASE + PRICE_PATH
REQUEST_TIMEOUT_SECONDS = 300
PRICE_TIMEOUT_SECONDS = 30
MAX_CONCURRENT_REQUESTS = 64
PROVIDERS = {"OpenAI": "openai", "Azure": "azure"}
TOKEN_RATES = {
    "ordinary_input": "prompt",
    "cached_input": "input_cache_read",
    "cache_write": "input_cache_write",
    "output": "completion",
}
REASONING = {"effort": "medium"}
PLUGINS = [{"id": "context-compression", "enabled": False}]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def _write_json(handle: TextIO, value: dict[str, Any]) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    handle.flush()


def _artifact_paths(output: Path) -> dict[str, Path]:
    output_dir = output.resolve()
    return {
        "summary": output_dir / "openrouter.json",
        "raw": output_dir / "openrouter.raw.jsonl",
        "errors": output_dir / "openrouter.errors.jsonl",
        "walltime": output_dir / "openrouter.inference_walltime.jsonl",
    }


def _open_artifacts(stack: ExitStack, paths: dict[str, Path]) -> dict[str, TextIO]:
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite inference artifact: {path}")
    return {name: stack.enter_context(path.open("x", encoding="utf-8")) for name, path in paths.items()}


def _load_api_key() -> str:
    api_key = dotenv_values(REPO_ROOT / ".env", interpolate=False).get("OPENROUTER_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(f"OPENROUTER_API_KEY is missing or empty in {REPO_ROOT / '.env'}")
    return api_key.strip()


def _load_pricing(client: OpenAI) -> tuple[dict[str, dict[str, Any]], list[str], str]:
    payload = client.get(PRICE_PATH, cast_to=dict, options={"timeout": PRICE_TIMEOUT_SECONDS})
    endpoints = payload.get("data", {}).get("endpoints")
    if not isinstance(endpoints, list):
        raise ValueError("OpenRouter pricing response did not contain an endpoint list")
    selected = {
        endpoint["provider_name"]: endpoint
        for endpoint in endpoints
        if isinstance(endpoint, dict)
        and endpoint.get("provider_name") in PROVIDERS
        and endpoint.get("tag") == PROVIDERS[endpoint["provider_name"]]
        and endpoint.get("service_tier") is None
    }
    ignored = sorted(
        {
            endpoint["tag"]
            for endpoint in endpoints
            if isinstance(endpoint, dict)
            and endpoint.get("provider_name") == "Azure"
            and isinstance(endpoint.get("tag"), str)
            and endpoint["tag"].startswith("azure/")
        }
    )
    return selected, ignored, _utc_now()


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def _decimal_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _applicable_rates(pricing: dict[str, Any], input_tokens: int | None) -> dict[str, Decimal | None]:
    overrides = pricing.get("overrides") or []
    eligible = (
        [
            item
            for item in overrides
            if isinstance(item, dict)
            and type(item.get("min_prompt_tokens")) is int
            and input_tokens is not None
            and item["min_prompt_tokens"] <= input_tokens
        ]
        if isinstance(overrides, list)
        else []
    )
    selected = {**pricing, **max(eligible, key=lambda item: item["min_prompt_tokens"], default={})}
    if input_tokens is None and overrides:
        return dict.fromkeys(TOKEN_RATES)
    return {category: _decimal(selected.get(field)) for category, field in TOKEN_RATES.items()}


def _usage_and_costs(usage: Any, pricing: dict[str, Any] | None) -> tuple[dict[str, int | None], dict[str, Any]]:
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details") or {}
    details = details if isinstance(details, dict) else {}
    values = {
        "input_tokens": usage.get("prompt_tokens"),
        "cached_input_tokens": details.get("cached_tokens"),
        "cache_write_tokens": details.get("cache_write_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }
    tokens = {key: value if type(value) is int and value >= 0 else None for key, value in values.items()}
    reasons = [f"missing or invalid {key}" for key, value in tokens.items() if value is None]
    counts = {category: tokens.get(f"{category}_tokens") for category in TOKEN_RATES}
    if all(tokens[key] is not None for key in ("input_tokens", "cached_input_tokens", "cache_write_tokens")):
        ordinary = tokens["input_tokens"] - tokens["cached_input_tokens"] - tokens["cache_write_tokens"]
        if ordinary >= 0:
            counts["ordinary_input"] = ordinary
        else:
            reasons.append("cached and cache-write token counts exceed input tokens")
            counts.update(dict.fromkeys(("ordinary_input", "cached_input", "cache_write")))
    rates = _applicable_rates(pricing if isinstance(pricing, dict) else {}, tokens["input_tokens"])
    reasons.extend(f"missing or invalid {category} rate" for category, rate in rates.items() if rate is None)
    amounts = {
        category: Decimal(counts[category]) * rate if counts[category] is not None and rate is not None else None
        for category, rate in rates.items()
    }
    costs = {f"{category}_usd": _decimal_text(amount) for category, amount in amounts.items()}
    for key, categories in (
        ("input_total_usd", ("ordinary_input", "cached_input", "cache_write")),
        ("calculated_total_usd", tuple(TOKEN_RATES)),
    ):
        parts = [amounts[category] for category in categories]
        costs[key] = _decimal_text(sum(parts, Decimal(0))) if all(part is not None for part in parts) else None
    costs.update(
        {
            "currency": "USD",
            "reported_billed_cost_credits": usage.get("cost"),
            "reported_cost_details": usage.get("cost_details"),
            "applicable_rates_usd_per_token": {
                f"{category}_usd_per_token": _decimal_text(rate) for category, rate in rates.items()
            },
            "accounting_error": "; ".join(reasons) or None,
        }
    )
    return tokens, costs


def _aggregate(values: list[Any], expected_rows: int, *, unit: str) -> dict[str, Any]:
    known = [value for item in values if (value := _decimal(item)) is not None]
    total = _decimal_text(sum(known, Decimal(0)))
    return {
        "total": total if len(known) == expected_rows else None,
        "known_total": total,
        "missing_rows": expected_rows - len(known),
        "unit": unit,
    }


def _refresh_aggregates(summary: dict[str, Any], records: list[dict[str, Any]], expected_rows: int) -> None:
    summary["usage_totals"] = {
        key: _aggregate([record["tokens"][key] for record in records], expected_rows, unit="tokens")
        for key in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")
    }
    summary["cost_totals"] = {
        key: _aggregate(
            [record["costs"][key] for record in records],
            expected_rows,
            unit="USD" if key.endswith("_usd") else "OpenRouter credits",
        )
        for key in (
            *(f"{category}_usd" for category in TOKEN_RATES),
            "input_total_usd",
            "calculated_total_usd",
            "reported_billed_cost_credits",
        )
    }
    warnings = [
        {"row_index": record["row_index"], "message": record["costs"]["accounting_error"]}
        for record in records
        if record["costs"]["accounting_error"]
    ]
    if warnings:
        summary["usage_accounting_warnings"] = warnings


def _exception_retries(exc: Exception) -> int | None:
    # Read this one SDK header, never credentials or the complete header collection.
    value = getattr(getattr(exc, "request", None), "headers", {}).get("x-stainless-retry-count")
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _request_completion(
    client: OpenAI,
    row_index: int,
    row: dict[str, Any],
    request_options: dict[str, Any],
    mark_request_started: Any,
) -> dict[str, Any]:
    outcome = {
        "row_index": row_index,
        "row": row,
        "messages": None,
        "stage": "request preparation",
        "request_started": False,
        "response_received": False,
        "request_started_at_utc": None,
        "request_completed_at_utc": None,
        "request_latency_seconds": None,
        "sdk_retries": None,
        "response_data": None,
        "response_body": None,
        "error": None,
    }
    try:
        outcome["messages"] = _text_messages(row["messages"])
    except Exception as exc:
        outcome["error"] = exc
        return outcome

    outcome["stage"] = "request"
    outcome["request_started_at_utc"] = _utc_now()
    request_started = time.perf_counter()
    outcome["request_started"] = True
    mark_request_started()
    try:
        raw_response = client.chat.completions.with_raw_response.create(
            model=MODEL_ID,
            messages=outcome["messages"],
            **request_options,
        )
    except Exception as exc:
        outcome.update(
            request_completed_at_utc=_utc_now(),
            request_latency_seconds=time.perf_counter() - request_started,
            sdk_retries=_exception_retries(exc),
            error=exc,
        )
        return outcome

    outcome.update(
        request_completed_at_utc=_utc_now(),
        request_latency_seconds=time.perf_counter() - request_started,
        sdk_retries=raw_response.retries_taken,
        response_received=True,
        response_body=raw_response.text,
        stage="response",
    )
    try:
        outcome["response_data"] = raw_response.http_response.json()
        outcome["response_body"] = None
    except Exception as exc:
        outcome["error"] = exc
    return outcome


def _response_record(
    outcome: dict[str, Any],
    pricing: dict[str, dict[str, Any]],
    retrieved_at: str,
    args: argparse.Namespace,
    generation_kwargs: dict[str, Any],
) -> dict[str, Any]:
    outcome["stage"] = "response"
    response_data = outcome["response_data"]
    if not isinstance(response_data, dict):
        raise ValueError("OpenRouter response was not a JSON object")
    choices = response_data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") or {}
    completion = message.get("content")
    if not isinstance(completion, str):
        raise ValueError("OpenRouter response did not contain string message content")
    response_provider = response_data.get("provider")
    if not isinstance(response_provider, str) or response_provider not in PROVIDERS:
        raise ValueError(f"OpenRouter returned a provider outside the allowed list: {response_provider!r}")

    outcome["stage"] = "accounting"
    endpoint = pricing.get(response_provider)
    usage = response_data.get("usage")
    tokens, costs = _usage_and_costs(usage, endpoint.get("pricing") if endpoint else None)
    return {
        "row_index": outcome["row_index"],
        "model_role": "openrouter",
        "model_id": MODEL_ID,
        "response_model": response_data.get("model"),
        "provider_name": response_provider,
        "response_id": response_data.get("id"),
        **{key: response_data.get(key) for key in ("system_fingerprint", "service_tier", "openrouter_metadata")},
        "checkpoint": None,
        "tokenizer_source": None,
        "mode": args.mode,
        "generation_kwargs": generation_kwargs,
        "enable_thinking": False,
        "request_messages": outcome["messages"],
        "row": outcome["row"],
        "request_started_at_utc": outcome["request_started_at_utc"],
        "request_completed_at_utc": outcome["request_completed_at_utc"],
        "request_latency_seconds": outcome["request_latency_seconds"],
        "sdk_retries": outcome["sdk_retries"],
        "http_attempts": (1 + outcome["sdk_retries"] if type(outcome["sdk_retries"]) is int else None),
        "finish_reason": choice.get("finish_reason"),
        "raw_completion": completion,
        "completion": completion,
        "response": response_data,
        "usage": usage,
        "tokens": tokens,
        "costs": costs,
        "pricing_retrieved_at_utc": retrieved_at,
        "pricing_endpoint": endpoint,
    }


def _error_record(outcome: dict[str, Any], exc: Exception) -> dict[str, Any]:
    retry_count = outcome["sdk_retries"]
    if retry_count is None:
        retry_count = _exception_retries(exc)
    body = getattr(exc, "body", None)
    if body is None:
        body = outcome["response_body"]
    return {
        "stage": outcome["stage"],
        "timestamp_utc": _utc_now(),
        "row_index": outcome["row_index"],
        "row": outcome["row"],
        "message": str(exc),
        "exception_type": type(exc).__name__,
        "http_status": getattr(exc, "status_code", None),
        "request_id": getattr(exc, "request_id", None),
        "response_body": body,
        "response": outcome["response_data"],
        "request_started_at_utc": outcome["request_started_at_utc"],
        "request_completed_at_utc": outcome["request_completed_at_utc"],
        "request_latency_seconds": outcome["request_latency_seconds"],
        "sdk_retries": retry_count,
        "http_attempts": retry_count + 1 if retry_count is not None else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone GPT-6 Luna CUAD inference through OpenRouter")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["all", "singleton", "grouped", "randomized", "canonical"], default="all")
    parser.add_argument("--split", choices=["all", "train", "dev", "test"], default="test")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=8192)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Run output directory",
    )
    parser.add_argument("--timing-log-every", type=_positive_int, default=10)
    return parser


def run(args: argparse.Namespace) -> int:
    paths = _artifact_paths(args.output)
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ExitStack() as stack:
        files = _open_artifacts(stack, paths)
        summary: dict[str, Any] = {
            "status": "running",
            "started_at_utc": _utc_now(),
            "model_id": MODEL_ID,
            "api_base": API_BASE,
            "mode": args.mode,
            "data_split": args.split,
            "data_dir": str(args.data_dir.resolve()),
            "requested_max_new_tokens": args.max_new_tokens,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "sdk_max_retries": None,
            "reasoning": REASONING,
            "provider_candidates": list(PROVIDERS),
            "plugins": PLUGINS,
            "artifact_paths": {key: str(path) for key, path in paths.items()},
            "rows": None,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "requests_attempted": 0,
            "responses_received": 0,
            "responses_completed": 0,
            "metadata_requests": 0,
        }
        _write_json(files["summary"], summary)
        records_by_index: dict[int, dict[str, Any]] = {}
        latencies: list[float] = []
        retry_counts: list[int | None] = []
        inference_started = inference_elapsed = request_latency = None
        row_index = row = response_data = sdk_retries = None
        request_count_lock = Lock()

        def mark_request_started() -> None:
            with request_count_lock:
                summary["requests_attempted"] += 1

        stage = "dataset"
        try:
            rows = _prepare_rows(_load_rows(args.data_dir, args.split), args.mode, args.limit)
            summary["rows"] = len(rows)
            stage = "credentials"
            api_key = _load_api_key()
            stage = "client"
            client = stack.enter_context(OpenAI(api_key=api_key, base_url=API_BASE, timeout=REQUEST_TIMEOUT_SECONDS))
            summary["sdk_max_retries"] = client.max_retries
            stage = "pricing"
            summary["metadata_requests"] = 1
            pricing, ignored, retrieved_at = _load_pricing(client)
            provider = {"only": list(PROVIDERS.values()), "ignore": ignored}
            summary.update(
                {
                    "provider": provider,
                    "pricing_source_url": PRICE_URL,
                    "pricing_retrieved_at_utc": retrieved_at,
                    "pricing_endpoints": pricing,
                    "ignored_azure_endpoints": ignored,
                }
            )
            request_options = {
                "max_completion_tokens": args.max_new_tokens,
                "stream": False,
                "extra_body": {"reasoning": REASONING, "provider": provider, "plugins": PLUGINS},
            }
            generation_kwargs = {
                **{key: value for key, value in request_options.items() if key != "extra_body"},
                **request_options["extra_body"],
            }
            inference_started = time.perf_counter()
            summary["inference_started_at_utc"] = _utc_now()
            _write_jsonl(
                files["walltime"],
                {"event": "inference_start", "timestamp_utc": _utc_now(), "model_id": MODEL_ID, "rows": len(rows)},
            )
            stage = "request batch"
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                for batch_start in range(0, len(rows), MAX_CONCURRENT_REQUESTS):
                    batch = rows[batch_start : batch_start + MAX_CONCURRENT_REQUESTS]
                    futures = {
                        executor.submit(
                            _request_completion,
                            client,
                            batch_start + offset + 1,
                            row_value,
                            request_options,
                            mark_request_started,
                        ): (batch_start + offset + 1, row_value)
                        for offset, row_value in enumerate(batch)
                    }
                    first_error = None

                    for future in as_completed(futures):
                        if future.cancelled():
                            futures.pop(future, None)
                            continue
                        fallback_index, fallback_row = futures[future]
                        try:
                            outcome = future.result()
                        except Exception as exc:
                            outcome = {
                                "row_index": fallback_index,
                                "row": fallback_row,
                                "messages": None,
                                "stage": "request",
                                "request_started": False,
                                "response_received": False,
                                "request_started_at_utc": None,
                                "request_completed_at_utc": None,
                                "request_latency_seconds": None,
                                "sdk_retries": None,
                                "response_data": None,
                                "response_body": None,
                                "error": exc,
                            }
                        futures.pop(future, None)

                        row_index = outcome["row_index"]
                        row = outcome["row"]
                        response_data = outcome["response_data"]
                        request_latency = outcome["request_latency_seconds"]
                        sdk_retries = outcome["sdk_retries"]
                        if outcome["request_started"]:
                            if request_latency is not None:
                                latencies.append(request_latency)
                            retry_counts.append(sdk_retries)
                        if outcome["response_received"]:
                            summary["responses_received"] += 1

                        error = outcome["error"]
                        if error is None:
                            try:
                                record = _response_record(outcome, pricing, retrieved_at, args, generation_kwargs)
                            except Exception as exc:
                                error = exc

                        if error is not None:
                            outcome["error"] = error
                            error_record = _error_record(outcome, error)
                            _write_jsonl(files["errors"], error_record)
                            _write_jsonl(files["walltime"], {"event": "inference_error", **error_record})
                            if first_error is None:
                                first_error = error_record
                            for pending in futures:
                                if not pending.done():
                                    pending.cancel()
                            continue

                        stage = "capture"
                        _write_jsonl(files["raw"], record)
                        records_by_index[row_index] = record
                        summary["responses_completed"] = len(records_by_index)
                        completed = len(records_by_index)
                        if completed % args.timing_log_every == 0 or completed == len(rows):
                            elapsed = time.perf_counter() - inference_started
                            _write_jsonl(
                                files["walltime"],
                                {
                                    "event": "inference_progress",
                                    "timestamp_utc": _utc_now(),
                                    "model_id": MODEL_ID,
                                    "completed_rows": completed,
                                    "total_rows": len(rows),
                                    "elapsed_seconds": elapsed,
                                    "last_request_seconds": request_latency,
                                    "rows_per_second": completed / elapsed if elapsed else None,
                                },
                            )

                    if first_error is not None:
                        summary.update(status="failed", error=first_error)
                        print(f"{first_error['stage']}: {first_error['message']}", file=sys.stderr)
                        break

            inference_elapsed = time.perf_counter() - inference_started
            if summary["status"] != "failed":
                _write_jsonl(
                    files["walltime"],
                    {
                        "event": "inference_end",
                        "timestamp_utc": _utc_now(),
                        "model_id": MODEL_ID,
                        "completed_rows": len(records_by_index),
                        "elapsed_seconds": inference_elapsed,
                        "average_request_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
                        "requests_per_second": len(records_by_index) / inference_elapsed if inference_elapsed else 0.0,
                    },
                )
                stage = "scoring"
                ordered_records = [records_by_index[index] for index in range(1, len(rows) + 1)]
                result = _evaluate_rows(
                    None,
                    None,
                    rows,
                    args.max_new_tokens,
                    "openrouter",
                    MODEL_ID,
                    mode=args.mode,
                    completions=[RowCompletion(record["completion"], generation_kwargs) for record in ordered_records],
                )
                result.metrics.update(
                    {
                        "inference_walltime_seconds": inference_elapsed,
                        "average_row_seconds": inference_elapsed / len(rows) if rows else 0.0,
                        "rows_per_second": len(rows) / inference_elapsed if inference_elapsed else 0.0,
                    }
                )
                summary.update(status="complete", metrics=result.metrics)
        except Exception as exc:
            retry_count = _exception_retries(exc)
            body = getattr(exc, "body", None)
            error = {
                "stage": stage,
                "timestamp_utc": _utc_now(),
                "row_index": row_index,
                "row": row,
                "message": str(exc),
                "exception_type": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None),
                "request_id": getattr(exc, "request_id", None),
                "response_body": body,
                "response": response_data,
                "request_latency_seconds": request_latency,
                "sdk_retries": retry_count,
                "http_attempts": retry_count + 1 if retry_count is not None else None,
            }
            _write_jsonl(files["errors"], error)
            _write_jsonl(files["walltime"], {"event": "inference_error", **error})
            summary.update(status="failed", error=error)
            print(f"{stage}: {exc}", file=sys.stderr)
        finally:
            retries_known = len(retry_counts) == summary["requests_attempted"] and None not in retry_counts
            total_retries = sum(retry_counts) if retries_known else None
            summary.update(
                {
                    "finished_at_utc": _utc_now(),
                    "end_to_end_seconds": time.perf_counter() - started,
                    "inference_walltime_seconds": inference_elapsed
                    if inference_elapsed is not None
                    else (time.perf_counter() - inference_started if inference_started is not None else None),
                    "request_latency_total_seconds": sum(latencies),
                    "average_request_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
                    "sdk_retries": total_retries,
                    "inference_http_attempts": summary["requests_attempted"] + total_retries if retries_known else None,
                    "requests_with_unknown_retry_count": retry_counts.count(None),
                }
            )
            if summary["rows"] is not None:
                _refresh_aggregates(summary, list(records_by_index.values()), summary["rows"])
            _write_json(files["summary"], summary)
        if summary["status"] == "complete":
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if summary["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = run(args)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
