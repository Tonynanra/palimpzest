import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import cuad_demo  # noqa: E402
import cuad_qwen_benchmark as benchmark  # noqa: E402


def _minimal_config_yaml(extra: str = "") -> str:
    return f"""
dataset:
  mode: cuad-chunk
  data_dir: testdata/cuad-chunk
categories:
  selected:
    - Document Name
    - Parties
  buckets:
    easy:
      - Document Name
      - Parties
    medium: []
    hard: []
job_mode: per-category
models:
  - openrouter/qwen/qwen3-0.6b-04-28
openrouter:
  api_base: https://openrouter.ai/api/v1
  api_key_env_var: OPENROUTER_API_KEY
execution:
  execution_strategy: parallel
  optimizer_strategy: none
  max_workers: 4
{extra}
"""


def test_resolve_selected_categories_from_manual_buckets():
    selected = cuad_demo.resolve_selected_categories(
        difficulty_buckets={
            "easy": ["Document Name", "Parties", "Agreement Date"],
            "medium": ["Effective Date", "Expiration Date", "Renewal Term"],
            "hard": ["Non-Compete", "Exclusivity", "Anti-Assignment"],
        },
    )

    assert selected == [
        "Document Name",
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Renewal Term",
        "Non-Compete",
        "Exclusivity",
        "Anti-Assignment",
    ]


def test_get_label_df_selected_categories_ignores_inactive_rows(monkeypatch):
    rows = [
        {
            "id": "contract-1__Document Name_0",
            "title": "Contract A",
            "context": "Contract A text",
            "answers": [{"text": "Contract A"}],
        },
        {
            "id": "contract-1__Parties_0",
            "title": "Contract A",
            "context": "Contract A text",
            "answers": [{"text": "Acme"}],
        },
    ]
    monkeypatch.setattr(cuad_demo, "load_cuad_data", lambda split, data_dir=None, dataset_mode="cuad-data": rows)

    label_df = cuad_demo.get_label_df(num_contracts=1, seed=0, selected_categories=["Document Name"])

    assert list(label_df.columns) == ["contract_id", "title", "contract", "Document Name"]
    assert label_df.loc[0, "Document Name"] == ["Contract A"]


def test_get_label_df_chunk_native_rows_preserve_chunk_metadata(monkeypatch):
    rows = [
        {
            "id": "contract-1__Document Name_chunk_0",
            "title": "Contract A [p0-chunk0]",
            "context": "Contract A chunk",
            "answers": [{"text": "Contract A"}],
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": 16,
            "source_paragraph_index": 0,
        },
        {
            "id": "contract-1__Document Name_chunk_1",
            "title": "Contract A [p0-chunk1]",
            "context": "Contract A second chunk",
            "answers": [],
            "chunk_index": 1,
            "chunk_start": 16,
            "chunk_end": 38,
            "source_paragraph_index": 0,
        },
    ]

    def fake_load(split, data_dir=None, dataset_mode="cuad-data"):
        assert dataset_mode == "cuad-chunk"
        return rows

    monkeypatch.setattr(cuad_demo, "load_cuad_data", fake_load)

    label_df = cuad_demo.get_label_df(
        num_contracts=2,
        seed=0,
        selected_categories=["Document Name"],
        dataset_mode="cuad-chunk",
    )

    assert set(label_df["title"]) == {"Contract A [p0-chunk0]", "Contract A [p0-chunk1]"}
    assert set(label_df["dataset_mode"]) == {"cuad-chunk"}
    assert set(label_df["chunk_index"]) == {0, 1}


def test_build_cuad_query_sequential_uses_only_selected_categories():
    class FakeDataset:
        def __init__(self):
            self.calls = []

        def sem_map(self, cols, depends_on):
            self.calls.append((cols, depends_on))
            return self

    dataset = FakeDataset()
    result = cuad_demo.build_cuad_query(dataset, "sequential", selected_categories=["Document Name", "Parties"])

    assert result is dataset
    assert [call[0][0]["name"] for call in dataset.calls] == ["Document Name", "Parties"]
    assert all(call[1] == ["contract"] for call in dataset.calls)


def test_compute_metrics_scores_selected_categories_only():
    label_df = pd.DataFrame(
        [
            {
                "contract_id": "c1",
                "title": "Contract A",
                "Document Name": ["Contract A"],
                "Parties": ["Acme"],
            }
        ]
    )
    pred_df = pd.DataFrame(
        [
            {
                "contract_id": "c1",
                "title": "Contract A",
                "Document Name": ["Contract A"],
            }
        ]
    )

    metrics = cuad_demo.compute_metrics(label_df, pred_df, selected_categories=["Document Name"])

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert set(metrics["per_category"]) == {"Document Name"}


def test_prediction_normalization_uses_canonical_long_shape():
    pred_df = pd.DataFrame(
        [
            {
                "contract_id": "c1",
                "title": "Contract A",
                "Document Name": ["Contract A"],
                "Parties": None,
            }
        ]
    )

    long_df = cuad_demo.normalize_predictions_to_long(
        pred_df,
        selected_categories=["Document Name", "Parties"],
        metadata={"model_id": "openrouter/qwen/qwen3-0.6b-04-28"},
    )

    assert list(long_df["category"]) == ["Document Name", "Parties"]
    assert long_df.loc[0, "predictions"] == ["Contract A"]
    assert long_df.loc[1, "predictions"] == []
    assert set(long_df.columns) >= {"model_id", "contract_id", "category", "predictions"}


def test_config_validation_and_job_generation(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(_minimal_config_yaml())

    config = benchmark.load_benchmark_config(config_path)
    jobs = benchmark.generate_jobs(config)

    assert config.selected_categories == ["Document Name", "Parties"]
    assert config.job_mode == "per-category"
    assert config.dataset_mode == "cuad-chunk"
    assert config.execution.execution_strategy == "parallel"
    assert len(config.run_id) == 8
    assert [job.categories for job in jobs] == [["Document Name"], ["Parties"]]


def test_config_explicit_categories_override_manual_buckets(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        _minimal_config_yaml(
            extra="""
categories:
  selected:
    - Document Name
  buckets:
    easy:
      - Document Name
      - Parties
    medium:
      - Agreement Date
    hard:
      - Non-Compete
""",
        ),
    )

    config = benchmark.load_benchmark_config(config_path)

    assert config.selected_categories == ["Document Name"]


def test_config_rejects_unknown_category(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        _minimal_config_yaml(
            extra="""
categories:
  selected:
    - Definitely Not A CUAD Category
  buckets:
    easy:
      - Document Name
    medium: []
    hard: []
""",
        ),
    )

    with pytest.raises(ValueError, match="Unknown CUAD categories"):
        benchmark.load_benchmark_config(config_path)


def test_config_rejects_unsupported_job_mode(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(_minimal_config_yaml(extra="job_mode: bonded-everything\n"))

    with pytest.raises(ValueError, match="Unsupported job mode"):
        benchmark.load_benchmark_config(config_path)


def test_config_rejects_unsupported_dataset_mode(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(_minimal_config_yaml(extra="dataset:\n  mode: cuad-contract-ish\n"))

    with pytest.raises(ValueError, match="Unsupported dataset mode"):
        benchmark.load_benchmark_config(config_path)


def test_config_rejects_legacy_backends(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(_minimal_config_yaml(extra="backends:\n  - name: old\n    kind: openrouter\n"))

    with pytest.raises(ValueError, match="Config field 'backends' is unsupported"):
        benchmark.load_benchmark_config(config_path)


def test_config_rejects_legacy_nodes(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(_minimal_config_yaml(extra="nodes:\n  - name: old\n    kind: local\n"))

    with pytest.raises(ValueError, match="Config field 'nodes' is unsupported"):
        benchmark.load_benchmark_config(config_path)


def test_config_parses_openrouter_and_execution_settings(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        _minimal_config_yaml(
            extra="""
openrouter:
  api_base: https://openrouter.ai/api/v1
  api_key_env_var: CUSTOM_OPENROUTER_KEY
  extra_headers:
    HTTP-Referer: https://example.com
execution:
  execution_strategy: sequential
  optimizer_strategy: none
  max_workers: 2
  progress: false
""",
        ),
    )

    config = benchmark.load_benchmark_config(config_path)

    assert config.openrouter.api_key_env_var == "CUSTOM_OPENROUTER_KEY"
    assert config.openrouter.extra_headers == {"HTTP-Referer": "https://example.com"}
    assert config.execution.execution_strategy == "sequential"
    assert config.execution.max_workers == 2
    assert config.execution.progress is False


def test_config_uses_all_bucket_entries_when_selected_is_absent(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        _minimal_config_yaml(
            extra="""
categories:
  buckets:
    easy:
      - Document Name
      - Parties
    medium:
      - Agreement Date
    hard:
      - Non-Compete
""",
        ),
    )

    config = benchmark.load_benchmark_config(config_path)

    assert config.selected_categories == ["Document Name", "Parties", "Agreement Date", "Non-Compete"]


def test_config_rejects_duplicate_categories_across_buckets(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        _minimal_config_yaml(
            extra="""
categories:
  buckets:
    easy:
      - Document Name
    medium:
      - Document Name
    hard:
      - Non-Compete
""",
        ),
    )

    with pytest.raises(ValueError, match="Duplicate CUAD categories"):
        benchmark.load_benchmark_config(config_path)


def test_config_rejects_selected_category_missing_from_buckets(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        _minimal_config_yaml(
            extra="""
categories:
  selected:
    - Document Name
  buckets:
    easy:
      - Parties
    medium: []
    hard: []
""",
        ),
    )

    with pytest.raises(ValueError, match="missing from difficulty buckets"):
        benchmark.load_benchmark_config(config_path)


def test_grouped_job_generation_creates_one_job_per_model():
    config = benchmark.BenchmarkConfig(
        selected_categories=["Document Name", "Parties"],
        difficulty_buckets={"easy": ["Document Name", "Parties"], "medium": [], "hard": []},
        category_to_difficulty={"Document Name": "easy", "Parties": "easy"},
        job_mode="grouped",
        models=["openrouter/qwen/qwen3-0.6b-04-28", "openrouter/qwen/qwen3-4b"],
        openrouter=benchmark.OpenRouterConfig(),
    )

    jobs = benchmark.generate_jobs(config)

    assert len(jobs) == 2
    assert all(job.categories == ["Document Name", "Parties"] for job in jobs)


def test_build_openrouter_model_uses_env_key(monkeypatch):
    monkeypatch.setattr(benchmark, "load_dotenv", lambda override=True: os.environ.__setitem__("OPENROUTER_API_KEY", "dotenv-key"))
    config = benchmark.BenchmarkConfig(
        selected_categories=["Document Name"],
        difficulty_buckets={"easy": ["Document Name"], "medium": [], "hard": []},
        category_to_difficulty={"Document Name": "easy"},
        job_mode="grouped",
        models=["openrouter/qwen/qwen3-0.6b-04-28"],
        openrouter=benchmark.OpenRouterConfig(),
    )

    model = benchmark.build_openrouter_model(config, "openrouter/qwen/qwen3-0.6b-04-28")

    assert model.api_base == "https://openrouter.ai/api/v1"
    assert model.vllm_kwargs["api_key"] == "dotenv-key"


def test_build_openrouter_model_requires_key(monkeypatch):
    monkeypatch.setattr(benchmark, "load_dotenv", lambda override=True: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = benchmark.BenchmarkConfig(
        selected_categories=["Document Name"],
        difficulty_buckets={"easy": ["Document Name"], "medium": [], "hard": []},
        category_to_difficulty={"Document Name": "easy"},
        job_mode="grouped",
        models=["openrouter/qwen/qwen3-0.6b-04-28"],
        openrouter=benchmark.OpenRouterConfig(),
    )

    with pytest.raises(RuntimeError, match="requires OPENROUTER_API_KEY"):
        benchmark.build_openrouter_model(config, "openrouter/qwen/qwen3-0.6b-04-28")


def test_run_benchmark_job_uses_native_pz_execution(monkeypatch):
    config = benchmark.BenchmarkConfig(
        selected_categories=["Document Name"],
        difficulty_buckets={"easy": ["Document Name"], "medium": [], "hard": []},
        category_to_difficulty={"Document Name": "easy"},
        job_mode="grouped",
        models=["openrouter/qwen/qwen3-0.6b-04-28"],
        openrouter=benchmark.OpenRouterConfig(),
        execution=benchmark.ExecutionConfig(execution_strategy="parallel", max_workers=7),
    )
    job = benchmark.BenchmarkJob(
        job_id="job-1",
        model_id="openrouter/qwen/qwen3-0.6b-04-28",
        categories=["Document Name"],
        job_mode="grouped",
    )

    class FakeExecutionStats:
        def to_json(self):
            return {"total_execution_time": 1.5}

    class FakeCollection:
        execution_stats = FakeExecutionStats()

        def to_df(self):
            return pd.DataFrame(
                [
                    {
                        "contract_id": "c1",
                        "title": "Contract A",
                        "dataset_mode": "cuad-chunk",
                        "chunk_index": 0,
                        "Document Name": ["Contract A"],
                    }
                ]
            )

    class FakeQuery:
        def __init__(self):
            self.seen_config = None

        def run(self, config):
            self.seen_config = config
            return FakeCollection()

    fake_query = FakeQuery()
    monkeypatch.setattr(benchmark, "build_openrouter_model", lambda config, model_id: "openrouter/qwen/qwen3-0.6b-04-28")
    monkeypatch.setattr(benchmark, "CUADDataset", lambda **kwargs: object())
    monkeypatch.setattr(benchmark, "build_cuad_query", lambda dataset, mode, selected_categories: fake_query)

    result = benchmark.run_benchmark_job(config, job)

    assert fake_query.seen_config.execution_strategy == "parallel"
    assert fake_query.seen_config.max_workers == 7
    assert result.execution_strategy == "parallel"
    assert result.max_workers == 7
    assert result.predictions_long.loc[0, "run_kind"] == "native-pz"
    assert result.predictions_long.loc[0, "provider"] == "openrouter"


def test_long_predictions_to_wide_round_trips_list_predictions():
    long_df = pd.DataFrame(
        [
            {
                "contract_id": "c1",
                "title": "Contract A",
                "category": "Document Name",
                "predictions": ["Contract A"],
            },
            {
                "contract_id": "c1",
                "title": "Contract A",
                "category": "Parties",
                "predictions": ["Acme"],
            },
        ]
    )

    wide_df = benchmark.long_predictions_to_wide(long_df)

    assert wide_df.loc[0, "Document Name"] == ["Contract A"]
    assert wide_df.loc[0, "Parties"] == ["Acme"]


def test_long_predictions_to_wide_preserves_chunk_metadata():
    long_df = pd.DataFrame(
        [
            {
                "contract_id": "c1__Document Name_chunk_0",
                "title": "Contract A [p0-chunk0]",
                "dataset_mode": "cuad-chunk",
                "chunk_index": 0,
                "category": "Document Name",
                "predictions": ["Contract A"],
            }
        ]
    )

    wide_df = benchmark.long_predictions_to_wide(long_df)

    assert wide_df.loc[0, "dataset_mode"] == "cuad-chunk"
    assert wide_df.loc[0, "chunk_index"] == 0
    assert wide_df.loc[0, "Document Name"] == ["Contract A"]


def test_metrics_are_aggregated_by_difficulty():
    label_df = pd.DataFrame(
        [
            {
                "contract_id": "c1",
                "title": "Contract A",
                "Document Name": ["Contract A"],
                "Parties": ["Acme"],
                "Agreement Date": ["01/01/2020"],
            }
        ]
    )
    pred_long = pd.DataFrame(
        [
            {
                "contract_id": "c1",
                "title": "Contract A",
                "category": "Document Name",
                "predictions": ["Contract A"],
                "difficulty": "easy",
                "model_id": "openrouter/qwen/qwen3-0.6b-04-28",
            },
            {
                "contract_id": "c1",
                "title": "Contract A",
                "category": "Parties",
                "predictions": ["Wrong"],
                "difficulty": "medium",
                "model_id": "openrouter/qwen/qwen3-0.6b-04-28",
            },
            {
                "contract_id": "c1",
                "title": "Contract A",
                "category": "Agreement Date",
                "predictions": ["01/01/2020"],
                "difficulty": "hard",
                "model_id": "openrouter/qwen/qwen3-0.6b-04-28",
            },
        ]
    )

    base_metrics = cuad_demo.compute_metrics(
        label_df,
        benchmark.long_predictions_to_wide(pred_long),
        selected_categories=["Document Name", "Parties", "Agreement Date"],
    )
    metrics = benchmark.augment_metrics_with_difficulty(
        base_metrics,
        label_df,
        pred_long,
        ["Document Name", "Parties", "Agreement Date"],
        {"Document Name": "easy", "Parties": "medium", "Agreement Date": "hard"},
    )

    assert set(metrics["per_difficulty"]) == {"easy", "medium", "hard"}
    assert metrics["per_difficulty"]["easy"]["tp"] == 1
    assert metrics["per_difficulty"]["medium"]["fn"] == 1
    assert metrics["per_difficulty"]["hard"]["tp"] == 1
    assert metrics["per_category"]["Document Name"]["difficulty"] == "easy"


def test_build_resolved_run_metadata_contains_openrouter_and_execution():
    config = benchmark.BenchmarkConfig(
        selected_categories=["Document Name"],
        difficulty_buckets={"easy": ["Document Name"], "medium": [], "hard": []},
        category_to_difficulty={"Document Name": "easy"},
        job_mode="grouped",
        models=["openrouter/qwen/qwen3-0.6b-04-28"],
        openrouter=benchmark.OpenRouterConfig(api_key_env_var="OPENROUTER_API_KEY"),
        execution=benchmark.ExecutionConfig(execution_strategy="parallel", max_workers=8),
    )

    metadata = benchmark.build_resolved_run_metadata(config)

    assert metadata["selected_categories"] == ["Document Name"]
    assert metadata["dataset_mode"] == "cuad-chunk"
    assert metadata["benchmark_unit"] == "chunk"
    assert metadata["category_to_difficulty"] == {"Document Name": "easy"}
    assert metadata["run_id"] == config.run_id
    assert metadata["openrouter"]["api_base"] == "https://openrouter.ai/api/v1"
    assert metadata["execution"]["max_workers"] == 8
    assert metadata["runner_kind"] == "native-pz"


def test_runtime_samples_are_aggregated_by_difficulty_for_grouped_job():
    results = [
        benchmark.JobResult(
            job=benchmark.BenchmarkJob(
                job_id="job-1",
                model_id="openrouter/qwen/qwen3-0.6b-04-28",
                categories=["Document Name", "Non-Transferable License"],
                job_mode="grouped",
            ),
            predictions_long=pd.DataFrame(),
            predictions_wide=pd.DataFrame(),
            stats={"total_execution_time": 30.0},
            execution_strategy="parallel",
            max_workers=8,
        )
    ]

    runtime = benchmark.aggregate_runtime_samples_by_difficulty(
        results,
        {"Document Name": "easy", "Non-Transferable License": "medium"},
    )

    assert runtime["openrouter/qwen/qwen3-0.6b-04-28"]["easy"] == [15.0]
    assert runtime["openrouter/qwen/qwen3-0.6b-04-28"]["medium"] == [15.0]


def test_build_plot_rows_contains_runtime_precision_accuracy_and_f1():
    metrics_by_model = {
        "openrouter/qwen/qwen3-0.6b-04-28": {
            "per_category": {
                "Document Name": {"precision": 1.0, "f1": 0.8, "tp": 4, "fp": 0, "fn": 1, "difficulty": "easy"},
                "Non-Transferable License": {"precision": 0.5, "f1": 0.4, "tp": 1, "fp": 1, "fn": 2, "difficulty": "medium"},
            },
            "per_difficulty": {
                "easy": {"precision": 1.0, "f1": 0.8, "tp": 4, "fp": 0, "fn": 1},
                "medium": {"precision": 0.5, "f1": 0.4, "tp": 1, "fp": 1, "fn": 2},
            }
        }
    }
    results = [
        benchmark.JobResult(
            job=benchmark.BenchmarkJob(
                job_id="job-1",
                model_id="openrouter/qwen/qwen3-0.6b-04-28",
                categories=["Document Name", "Non-Transferable License"],
                job_mode="grouped",
            ),
            predictions_long=pd.DataFrame(),
            predictions_wide=pd.DataFrame(),
            stats={"total_execution_time": 30.0},
            execution_strategy="parallel",
            max_workers=8,
        )
    ]

    plot_rows = benchmark.build_plot_rows(
        metrics_by_model,
        results,
        {"Document Name": "easy", "Non-Transferable License": "medium"},
    )

    assert plot_rows["easy"]["openrouter/qwen/qwen3-0.6b-04-28"]["runtime_seconds"]["mean"] == 15.0
    assert plot_rows["easy"]["openrouter/qwen/qwen3-0.6b-04-28"]["runtime_seconds"]["std"] == 0.0
    assert plot_rows["easy"]["openrouter/qwen/qwen3-0.6b-04-28"]["precision"]["mean"] == 1.0
    assert plot_rows["easy"]["openrouter/qwen/qwen3-0.6b-04-28"]["accuracy"]["mean"] == 0.8
    assert plot_rows["easy"]["openrouter/qwen/qwen3-0.6b-04-28"]["f1"]["mean"] == 0.8


def test_generate_benchmark_plots_writes_pngs(tmp_path):
    metrics_by_model = {
        "openrouter/qwen/qwen3-0.6b-04-28": {
            "per_category": {
                "Document Name": {"precision": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0, "difficulty": "easy"},
                "Non-Transferable License": {"precision": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 0, "difficulty": "medium"},
            },
            "per_difficulty": {
                "easy": {"precision": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
                "medium": {"precision": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 0},
            }
        }
    }
    results = [
        benchmark.JobResult(
            job=benchmark.BenchmarkJob(
                job_id="job-1",
                model_id="openrouter/qwen/qwen3-0.6b-04-28",
                categories=["Document Name", "Non-Transferable License"],
                job_mode="grouped",
            ),
            predictions_long=pd.DataFrame(),
            predictions_wide=pd.DataFrame(),
            stats={"total_execution_time": 20.0},
            execution_strategy="parallel",
            max_workers=8,
        )
    ]

    benchmark.generate_benchmark_plots(
        tmp_path,
        "plot-test",
        metrics_by_model,
        results,
        {"Document Name": "easy", "Non-Transferable License": "medium"},
    )

    assert (tmp_path / "plot-test-runtime-by-difficulty.png").exists()
    assert (tmp_path / "plot-test-precision-by-difficulty.png").exists()
    assert (tmp_path / "plot-test-accuracy-by-difficulty.png").exists()
    assert (tmp_path / "plot-test-f1-by-difficulty.png").exists()


def test_write_outputs_records_native_run_metadata(tmp_path):
    config = benchmark.BenchmarkConfig(
        selected_categories=["Document Name"],
        difficulty_buckets={"easy": ["Document Name"], "medium": [], "hard": []},
        category_to_difficulty={"Document Name": "easy"},
        job_mode="grouped",
        models=["openrouter/qwen/qwen3-0.6b-04-28"],
        openrouter=benchmark.OpenRouterConfig(),
        output_dir=tmp_path,
        exp_name="native-run",
    )
    result = benchmark.JobResult(
        job=benchmark.BenchmarkJob(
            job_id="job-1",
            model_id="openrouter/qwen/qwen3-0.6b-04-28",
            categories=["Document Name"],
            job_mode="grouped",
        ),
        predictions_long=pd.DataFrame(
            [
                {
                    "contract_id": "c1",
                    "title": "Contract A",
                    "dataset_mode": "cuad-chunk",
                    "chunk_index": 0,
                    "category": "Document Name",
                    "predictions": ["Contract A"],
                    "model_id": "openrouter/qwen/qwen3-0.6b-04-28",
                }
            ]
        ),
        predictions_wide=pd.DataFrame(
            [
                {
                    "contract_id": "c1",
                    "title": "Contract A",
                    "dataset_mode": "cuad-chunk",
                    "chunk_index": 0,
                    "Document Name": ["Contract A"],
                }
            ]
        ),
        stats={"total_execution_time": 1.0},
        execution_strategy="parallel",
        max_workers=8,
    )

    benchmark.write_outputs(
        config,
        [result],
        result.predictions_long,
        result.predictions_wide,
        {
            "openrouter/qwen/qwen3-0.6b-04-28": {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "per_category": {"Document Name": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0}},
                "per_difficulty": {"easy": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0}},
            }
        },
    )

    prefix = benchmark.build_run_prefix(config)
    output_dir = benchmark.build_output_dir(config)
    with open(output_dir / f"{prefix}-job-stats.json") as f:
        job_stats = json.load(f)
    with open(output_dir / f"{prefix}-resolved-run.json") as f:
        resolved_run = json.load(f)

    assert job_stats["job-1"]["run_kind"] == "native-pz"
    assert job_stats["job-1"]["provider"] == "openrouter"
    assert job_stats["job-1"]["execution_strategy"] == "parallel"
    assert resolved_run["runner_kind"] == "native-pz"
    assert resolved_run["openrouter"]["api_key_env_var"] == "OPENROUTER_API_KEY"
    assert resolved_run["output_dir"] == str(output_dir)
