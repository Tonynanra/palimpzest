import copy
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai._base_client as sdk_base
import pytest
from openai import OpenAI, OpenAIError

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import evaluate_cuad_openrouter as runner  # noqa: E402

RATES = {
    "prompt": "0.0000001",
    "input_cache_read": "0.00000001",
    "input_cache_write": "0.000000125",
    "completion": "0.0000005",
}
USAGE = {
    "prompt_tokens": 100,
    "prompt_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 30},
    "completion_tokens": 10,
    "cost": 0.00001395,
    "cost_details": {"upstream_inference_cost": 0.00001395},
    "extra_usage_field": {"keep": True},
}
COMPLETION = '{"Document Name":["Café Alpha","hallucination"],"Agreement Date":["2024"]}'


def rows(count):
    return [
        {
            "id": "shared-id",
            "chunk_index": index,
            "view": "all",
            "contract": "Café Alpha 2024",
            "categories": ["Document Name", "Agreement Date"],
            "labels": {"Document Name": ["Café Alpha"], "Agreement Date": ["2024"]},
            "messages": [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": f"Contract {index}: Café Alpha 2024"},
            ],
        }
        for index in range(count)
    ]


def response_body(completion=COMPLETION, provider="OpenAI"):
    return {
        "id": "gen-test",
        "object": "chat.completion",
        "created": 1,
        "model": runner.MODEL_ID,
        "provider": provider,
        "service_tier": "default",
        "unknown_response_field": [1, 2],
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": completion}}],
        "usage": copy.deepcopy(USAGE),
    }


def catalog():
    return [
        {"provider_name": name, "tag": slug, "service_tier": None, "status": 0, "pricing": dict(RATES)}
        for name, slug in runner.PROVIDERS.items()
    ] + [{"provider_name": "Azure", "tag": f"azure/{region}", "pricing": {"prompt": "0.1"}} for region in ("us", "eu")]


def test_artifact_paths_use_only_the_output_directory(tmp_path):
    output_dir = tmp_path / "luna"
    paths = runner._artifact_paths(output_dir)

    assert paths == {
        "summary": output_dir / "openrouter.json",
        "raw": output_dir / "openrouter.raw.jsonl",
        "errors": output_dir / "openrouter.errors.jsonl",
        "walltime": output_dir / "openrouter.inference_walltime.jsonl",
    }


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "MAX_CONCURRENT_REQUESTS", 1)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "wrong-inherited-key")

    def build(post=None, count=1, metadata_failures=0):
        calls, posts, constructors, sleeps = [], [], [], []
        clock = {"seconds": 0.0}
        endpoint_catalog = catalog()
        selected_rows = rows(count)
        args = runner.build_parser().parse_args(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--output",
                str(tmp_path / "run"),
                "--timing-log-every",
                "1",
            ]
        )
        paths = runner._artifact_paths(args.output)
        monkeypatch.setattr(runner, "_load_rows", lambda *args: copy.deepcopy(selected_rows))
        monkeypatch.setattr(runner.time, "perf_counter", lambda: clock["seconds"])

        def sleep(seconds):
            sleeps.append(seconds)
            clock["seconds"] += seconds

        monkeypatch.setattr(sdk_base.time, "sleep", sleep)

        def serve(request):
            assert request.url.host == "openrouter.ai"
            calls.append(request)
            clock["seconds"] += 0.25
            if request.method == "GET":
                assert request.url.path == "/api/v1" + runner.PRICE_PATH
                if sum(item.method == "GET" for item in calls) <= metadata_failures:
                    return httpx2.Response(502, json={"error": {"message": "Bad Gateway"}})
                return httpx2.Response(200, json={"data": {"endpoints": endpoint_catalog}})
            assert request.url.path == "/api/v1/chat/completions"
            posts.append(request)
            return post(request, len(posts), paths) if post else httpx2.Response(200, json=response_body())

        def make_client(**kwargs):
            constructors.append(kwargs)
            assert "max_retries" not in kwargs
            assert kwargs["api_key"] == "test-key"
            return OpenAI(**kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(serve)))

        monkeypatch.setattr(runner, "OpenAI", make_client)
        return SimpleNamespace(
            args=args,
            paths=paths,
            calls=calls,
            posts=posts,
            constructors=constructors,
            sleeps=sleeps,
            clock=clock,
            catalog=endpoint_catalog,
            rows=selected_rows,
        )

    return build


def test_run_writes_artifacts_inside_output_directory(harness):
    run = harness()
    assert runner.run(run.args) == 0

    assert all(path.is_file() for path in run.paths.values())
    summary, records, errors = saved(run)
    assert summary["artifact_paths"] == {key: str(path) for key, path in run.paths.items()}
    assert len(records) == 1 and errors == []


def saved(run):
    return (
        json.loads(run.paths["summary"].read_text()),
        [json.loads(line) for line in run.paths["raw"].read_text().splitlines()],
        [json.loads(line) for line in run.paths["errors"].read_text().splitlines()],
    )


def test_sdk_retry_success_preserves_routing_accounting_and_latency(harness):
    def post(request, attempt, paths):
        if attempt == 1:
            return httpx2.Response(502, json={"error": {"message": "Bad Gateway"}})
        body = response_body(provider="Azure")
        body["usage"]["cost"] = 0.0000279
        return httpx2.Response(200, json=body)

    run = harness(post)
    # Health and equal-price gates must not prevent OpenRouter from using the other endpoint.
    run.catalog[0]["status"] = -5
    run.catalog[1]["pricing"] = {field: str(float(value) * 2) for field, value in RATES.items()}
    assert runner.run(run.args) == 0
    summary, records, errors = saved(run)
    assert len(run.constructors) == 1
    assert len(run.calls) == 3 and len(run.posts) == 2
    assert [request.headers["x-stainless-retry-count"] for request in run.posts] == ["0", "1"]
    assert summary["sdk_max_retries"] == 2
    assert summary["requests_attempted"] == summary["responses_completed"] == 1
    assert summary["sdk_retries"] == 1 and summary["inference_http_attempts"] == 2
    assert errors == [] and len(records) == 1
    record = records[0]
    assert record["sdk_retries"] == 1 and record["http_attempts"] == 2
    assert record["request_latency_seconds"] == pytest.approx(0.5 + sum(run.sleeps))
    assert record["request_latency_seconds"] > 0.5
    assert summary["end_to_end_seconds"] == pytest.approx(0.75 + sum(run.sleeps))
    payload = json.loads(run.posts[0].content)
    assert payload["model"] == runner.MODEL_ID
    assert payload["max_completion_tokens"] == 8192 and "max_tokens" not in payload
    assert payload["reasoning"] == runner.REASONING
    assert payload["provider"] == {"only": ["openai", "azure"], "ignore": ["azure/eu", "azure/us"]}
    assert record["generation_kwargs"]["provider"] == payload["provider"]
    assert record["request_messages"] == run.rows[0]["messages"]
    assert record["row"] == run.rows[0]
    assert record["raw_completion"] == record["completion"] == COMPLETION
    assert record["response"]["unknown_response_field"] == [1, 2]
    assert record["usage"]["extra_usage_field"] == {"keep": True}
    assert record["pricing_endpoint"]["provider_name"] == "Azure"
    assert float(record["costs"]["calculated_total_usd"]) == pytest.approx(0.0000279)
    assert summary["cost_totals"]["reported_billed_cost_credits"]["total"] == "0.0000279"
    assert "test-key" not in "".join(path.read_text() for path in run.paths.values())


@pytest.mark.parametrize("failure", [502, 400, "connection", "timeout", "unknown"])
def test_terminal_failures_keep_previous_rows_and_record_observed_attempts(harness, failure):
    def post(request, attempt, paths):
        if attempt == 1:
            return httpx2.Response(200, json=response_body())
        assert len(paths["raw"].read_text().splitlines()) == 1  # The previous completion was flushed.
        if failure == "connection":
            raise httpx2.ConnectError("connection failed", request=request)
        if failure == "timeout":
            raise httpx2.ReadTimeout("request timed out", request=request)
        if failure == "unknown":
            raise OpenAIError("local SDK error without retry information")
        return httpx2.Response(failure, json={"error": {"message": "request failed"}})

    run = harness(post, count=3)
    assert runner.run(run.args) == 1
    summary, records, errors = saved(run)
    expected_retries = 0 if failure == 400 else None if failure == "unknown" else 2
    assert len(records) == 1 and len(errors) == 1
    assert len(run.posts) == (4 if expected_retries == 2 else 2)
    assert summary["status"] == "failed"
    assert summary["requests_attempted"] == 2 and summary["responses_completed"] == 1
    assert errors[0]["stage"] == "request" and errors[0]["row_index"] == 2
    assert errors[0]["sdk_retries"] == expected_retries
    assert summary["sdk_retries"] == expected_retries
    assert summary["inference_http_attempts"] == (2 + expected_retries if expected_retries is not None else None)
    assert summary["requests_with_unknown_retry_count"] == (1 if expected_retries is None else 0)
    assert errors[0]["request_latency_seconds"] >= 0.25
    assert summary["usage_totals"]["input_tokens"] == {
        "total": None,
        "known_total": "100",
        "missing_rows": 2,
        "unit": "tokens",
    }
    artifacts = "".join(path.read_text() for path in run.paths.values())
    assert "test-key" not in artifacts and "authorization" not in artifacts.lower()


def test_pricing_lookup_uses_the_same_sdk_defaults(harness):
    run = harness(metadata_failures=1)
    assert runner.run(run.args) == 0
    summary, records, _ = saved(run)
    assert len(run.constructors) == 1 and len(run.calls) == 3
    assert summary["metadata_requests"] == 1  # Logical lookup, independently retried by the SDK.
    assert summary["sdk_retries"] == 0 and summary["inference_http_attempts"] == 1
    assert records[0]["request_latency_seconds"] == pytest.approx(0.25)
    assert summary["end_to_end_seconds"] == pytest.approx(0.75 + sum(run.sleeps))


def test_scores_match_shared_evaluator_and_raw_is_flushed_before_scoring(harness, monkeypatch):
    outputs = [COMPLETION, "not JSON"]
    run = harness(
        lambda request, attempt, paths: httpx2.Response(200, json=response_body(outputs[attempt - 1])), count=2
    )
    original = runner._evaluate_rows

    def score(*args, **kwargs):
        assert [json.loads(line)["completion"] for line in run.paths["raw"].read_text().splitlines()] == outputs
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_evaluate_rows", score)
    assert runner.run(run.args) == 0
    summary, records, errors = saved(run)
    expected = original(
        None,
        None,
        run.rows,
        8192,
        "openrouter",
        runner.MODEL_ID,
        mode="all",
        completions=[runner.RowCompletion(text, {}) for text in outputs],
    ).metrics
    timing_keys = {"inference_walltime_seconds", "average_row_seconds", "rows_per_second"}
    assert {key: value for key, value in summary["metrics"].items() if key not in timing_keys} == {
        key: value for key, value in expected.items() if key not in timing_keys
    }
    assert (summary["metrics"]["tp"], summary["metrics"]["fp"], summary["metrics"]["fn"]) == (2, 1, 2)
    assert summary["metrics"]["parse_failures"] == 1 and len(records) == 2 and errors == []
    assert summary["usage_totals"]["input_tokens"]["total"] == "200"
    assert Decimal(summary["cost_totals"]["calculated_total_usd"]["total"]) == Decimal("0.00002790")


def test_missing_provider_pricing_keeps_response_and_reported_bill(harness):
    run = harness()
    run.catalog[:] = [endpoint for endpoint in run.catalog if endpoint.get("provider_name") != "OpenAI"]
    assert runner.run(run.args) == 0
    summary, records, errors = saved(run)
    assert records[0]["pricing_endpoint"] is None
    assert records[0]["costs"]["calculated_total_usd"] is None
    assert records[0]["costs"]["reported_billed_cost_credits"] == USAGE["cost"]
    assert summary["cost_totals"]["ordinary_input_usd"]["missing_rows"] == 1
    assert summary["usage_accounting_warnings"] and errors == []


def test_missing_dotenv_key_stops_before_network_requests(harness, monkeypatch):
    run = harness()
    (runner.REPO_ROOT / ".env").write_text("", encoding="utf-8")
    assert runner.run(run.args) == 1
    assert run.calls == []
    summary, records, errors = saved(run)
    assert summary["status"] == "failed" and errors[0]["stage"] == "credentials" and records == []


def test_existing_artifacts_are_rejected_without_network_or_overwrite(harness):
    run = harness()
    assert runner.run(run.args) == 0
    original = {path: path.read_bytes() for path in run.paths.values()}
    previous_calls = len(run.calls)
    with pytest.raises(SystemExit) as exc:
        runner.main(["--data-dir", str(run.args.data_dir), "--output", str(run.args.output)])
    assert exc.value.code == 1 and len(run.calls) == previous_calls
    assert {path: path.read_bytes() for path in run.paths.values()} == original


def test_complete_cost_breakdown_and_pricing_override():
    tokens, costs = runner._usage_and_costs(USAGE, RATES)
    assert tokens == {"input_tokens": 100, "cached_input_tokens": 20, "cache_write_tokens": 30, "output_tokens": 10}
    assert float(costs["ordinary_input_usd"]) == pytest.approx(0.000005)
    assert float(costs["cached_input_usd"]) == pytest.approx(0.0000002)
    assert float(costs["cache_write_usd"]) == pytest.approx(0.00000375)
    assert float(costs["output_usd"]) == pytest.approx(0.000005)
    assert float(costs["input_total_usd"]) == pytest.approx(0.00000895)
    assert float(costs["calculated_total_usd"]) == pytest.approx(0.00001395)
    assert costs["accounting_error"] is None
    override = {"min_prompt_tokens": 100, **{key: str(float(value) * 2) for key, value in RATES.items()}}
    _, doubled = runner._usage_and_costs(USAGE, {**RATES, "overrides": [override]})
    assert float(doubled["calculated_total_usd"]) == pytest.approx(0.0000279)


@pytest.mark.parametrize("missing", ["cached_tokens", "cache_write_tokens"])
def test_missing_cache_counts_remain_unknown(missing):
    usage = copy.deepcopy(USAGE)
    del usage["prompt_tokens_details"][missing]
    _, costs = runner._usage_and_costs(usage, RATES)
    assert costs["ordinary_input_usd"] is None and costs["calculated_total_usd"] is None
    assert costs["accounting_error"] is not None
    assert costs["reported_billed_cost_credits"] == USAGE["cost"]


def test_missing_input_cannot_select_a_pricing_tier():
    usage = copy.deepcopy(USAGE)
    del usage["prompt_tokens"]
    _, costs = runner._usage_and_costs(usage, {**RATES, "overrides": [{"min_prompt_tokens": 100, **RATES}]})
    assert costs["calculated_total_usd"] is None and costs["output_usd"] is None
    assert costs["accounting_error"] is not None


def test_inconsistent_input_counts_do_not_produce_input_costs():
    usage = copy.deepcopy(USAGE)
    usage["prompt_tokens"] = 10
    _, costs = runner._usage_and_costs(usage, RATES)
    assert all(costs[key] is None for key in ("ordinary_input_usd", "cached_input_usd", "cache_write_usd"))
    assert float(costs["output_usd"]) == pytest.approx(0.000005)
    assert costs["calculated_total_usd"] is None
