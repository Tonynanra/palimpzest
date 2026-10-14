# CUAD Qwen SFT

This document describes the current CUAD supervised fine-tuning workflow in this repository.
The implementation trains a Qwen3.5-4B, Qwen3.5-9B, or Qwen3.8-27B LoRA/QLoRA adapter on the local CUAD chunk dataset and evaluates it through the existing Palimpzest JSON extraction pipeline.

## Workflow

```text
testdata/cuad/chunks + testdata/cuad/raw
        │
        ▼
cuad_sft_data.py
        │  randomized one-convert views
        ▼
testdata/cuad/sft/train-dev/{train,dev}.jsonl
        │
        ▼
train_cuad_sft.py
        │  selected Qwen model + PEFT LoRA/QLoRA
        ▼
finetuning/artifacts/cuad-sft-runs/{RUN}/
        │  config_snapshot.yaml, adapter/, checkpoint-*/
        │
        ├── evaluate_cuad_sft.py
        ├── merge_cuad_sft_adapter.py
        └── serve_cuad_sft.py
```

Each training example requests a randomized subset of CUAD fields and returns exactly those fields as JSON:

```json
{"Agreement Date":["March 27, 2020."],"Document Name":[]}
```

The assistant completion ends with `---`, matching the existing Palimpzest convert parser.

## Files

| File | Purpose |
|---|---|
| [cuad_data.py](cuad_data.py) | Shared download, JSON/JSONL loading, chunk generation, grouping, labels, and seeded sampling. |
| [cuad_schema.py](cuad_schema.py) | CUAD categories, normalization, validation, and output-field descriptions. |
| [setup_cuad_data.py](setup_cuad_data.py) | Downloads raw CUAD data without depending on the historical research directory. |
| [cuad_sft_data.py](cuad_sft_data.py) | Builds deterministic train/dev and isolated test JSONL examples. |
| [sft_config.yaml](sft_config.yaml) | Default Qwen training configuration. |
| [cuad_sft_config.py](cuad_sft_config.py) | Shared model selection, configuration parsing, and run-root snapshots. |
| [train_cuad_sft.py](train_cuad_sft.py) | Loads the selected model, attaches LoRA/QLoRA, masks prompt tokens, and trains. |
| [evaluate_cuad_sft.py](evaluate_cuad_sft.py) | Generates predictions and reports CUAD extraction metrics. |
| [merge_cuad_sft_adapter.py](merge_cuad_sft_adapter.py) | Merges a PEFT adapter into a standalone checkpoint. |
| [serve_cuad_sft.py](serve_cuad_sft.py) | Launches adapter or merged-model vLLM serving. |
| [cuad_qwen_benchmark.py](cuad_qwen_benchmark.py) | Existing Palimpzest benchmark, extended with local endpoint and `convert_mode` support. |
| [generate_cuad_chunk_dataset.py](generate_cuad_chunk_dataset.py) | CLI adapter for `cuad-chunk` generation. SFT consumes the generated files; it does not regenerate them automatically. |
| [../src/palimpzest/prompts/prompt_factory.py](../src/palimpzest/prompts/prompt_factory.py) | Existing prompt factory with an opt-in provided-field-order mode for SFT data. |

Raw data, chunks, and generated SFT datasets are ignored under `testdata/cuad/`.
Checkpoints, models, logs, metrics, and plots are stored under `finetuning/artifacts/`.
See [the dataset layout and relocation mapping](../testdata/README.md#cuad) for existing run records.

## Dataset construction

The builder reads the existing `cuad-chunk` file once and aggregates labels directly
from ordered annotations grouped by title. Benchmark loading shares the same
grouping and label logic. This avoids flattening and regrouping annotations or
converting labels through a DataFrame; text chunking, source hashes, seeded views,
prompts, and artifact serialization retain their existing behavior.
Train/dev assignment is made at source-contract content-hash level, before chunks are separated, preventing chunks from the same source contract from crossing the split.

Seven views are generated per chunk:

1. Singleton field.
2. Singleton field.
3. Random 2–5-field subset.
4. Random 2–5-field subset.
5. Random 6–15-field subset.
6. Random 16–30-field subset.
7. All 41 CUAD fields.

Each subset is shuffled. Empty labels are retained so the model learns when to return `[]`.

### Data-builder CLI

```bash
./.venv/bin/python finetuning/cuad_sft_data.py \
  --chunk-data-dir testdata/cuad/chunks \
  --raw-data-dir testdata/cuad/raw \
  --split train \
  --output-dir testdata/cuad/sft/train-dev \
  --seed 42 \
  --train-fraction 0.9
```

| Option | Default | Meaning |
|---|---|---|
| `--chunk-data-dir` | `testdata/cuad/chunks` | Chunk-native CUAD input directory. |
| `--raw-data-dir` | `testdata/cuad/raw` | Raw CUAD directory used for source hashing. |
| `--split` | `train` | CUAD source split: `train` or `test`. |
| `--output-dir` | `testdata/cuad/sft/train-dev` for `train`; `testdata/cuad/sft/test` for `test` | JSONL and manifest directory; an explicit override takes precedence. |
| `--seed` | `42` | Split and view-sampling seed. |
| `--train-fraction` | `0.9` | Source-contract fraction assigned to training. |

Outputs:

- `train.jsonl`
- `dev.jsonl`
- `manifest.json`

When `--split test` is selected, the builder defaults to `testdata/cuad/sft/test`,
containing `test.jsonl` and `manifest.json`. The holdout is used only for standalone
test evaluation; training reads `testdata/cuad/sft/train-dev`.

```bash
./.venv/bin/python finetuning/cuad_sft_data.py --split test
```

The Slurm workflow reuses non-empty train/dev and test JSONL files already present
at these locations, building only missing bundles.

## Training configuration

The checked-in [sft_config.yaml](sft_config.yaml) currently contains:

```yaml
model: Qwen/Qwen3.5-4B

data:
  dir: testdata/cuad/sft/train-dev

training:
  max_seq_length: 36864
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  learning_rate: 0.0001
  warmup_ratio: 0.03
  lora_rank: 32
  lora_alpha: 64
  lora_dropout: 0.05
  qlora: false
  gradient_checkpointing: true
  flash_attention: true
```

The training artifact directory is owned by `train_cuad_sft.py`, not this YAML file. It defaults to `finetuning/artifacts/cuad-sft-run` and can be changed with `--output-dir`.

The supported `model` values are `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, and `Qwen/Qwen3.8-27B`. Training's optional `--model-id` overrides this YAML value before model validation. Downstream commands read the effective selection from the saved run, so editing the input YAML after training does not change inference's model.

The H200 workflow uses `qlora: false`, FlashAttention-3, and one visible CUDA device. The base model is loaded in BF16 and only the LoRA adapter parameters are trainable. When `training.flash_attention` is true, the trainer and evaluator request `flash_attention_3` when `flash_attn_interface` is installed and warn before falling back to SDPA when it is unavailable.

The trainer:

- Uses Qwen3.5’s text-only `AutoModelForCausalLM` path.
- Uses `enable_thinking=False` in the chat template.
- Masks all user/system prompt tokens with `-100`.
- Trains only the assistant JSON completion and delimiter.
- Uses Qwen3.5 attention, Gated DeltaNet, and MLP projections as LoRA targets.
- Uses `logits_to_keep` so vocabulary logits are materialized only for completion tokens.
- Refuses to truncate an example that exceeds `max_seq_length`.
- Requires one visible CUDA device.

### Training CLI

```bash
./.venv/bin/python finetuning/train_cuad_sft.py \
  --config finetuning/sft_config.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --output-dir finetuning/artifacts/cuad-sft-runs/demo
```

For a one-step smoke run on the same training path:

```bash
./.venv/bin/python finetuning/train_cuad_sft.py \
  --config finetuning/sft_config.yaml \
  --max-steps 1 \
  --eval-limit 1 \
  --output-dir /tmp/cuad-sft-smoke
```

| Option | Default | Meaning |
|---|---|---|
| `--config` | `finetuning/sft_config.yaml` | YAML training configuration. |
| `--model-id` | unset | Override YAML `model` with one of the three supported full model IDs. |
| `--resume-from-checkpoint` | unset | Hugging Face Trainer checkpoint to resume. |
| `--max-steps` | unset | Optional optimizer-step cap for a smoke run. |
| `--eval-limit` | unset | Optional limit on dev rows evaluated during training. |
| `--output-dir` | `finetuning/artifacts/cuad-sft-run` | Training artifact directory; owned by the CLI/script rather than YAML. |
| `--log-level` | `INFO` | Python logging level. |

The trainer writes checkpoints, `adapter/`, tokenizer files, and `training_manifest.json` under the selected output directory.
Before loading weights it writes one `config_snapshot.yaml` at that directory's root, containing the effective model, resolved data/training settings, and runtime controls (`output_dir`, `max_steps`, `eval_limit`, and `resume_from_checkpoint`). No snapshot copies are created inside checkpoints, adapters, or merged exports. Reusing a run directory or resuming a checkpoint with a different saved model is rejected before the snapshot is overwritten. Downstream commands require the snapshot; legacy runs are outside this workflow.
It also writes `training_walltime.jsonl` with `train_start`, `checkpoint`, and `train_end` events.
Each checkpoint event records the checkpoint path, global step, epoch, UTC timestamp, and elapsed training seconds.

## Evaluation CLI

```bash
./.venv/bin/python finetuning/evaluate_cuad_sft.py \
  --run-dir finetuning/artifacts/cuad-sft-runs/demo \
  --data-dir testdata/cuad/sft/test \
  --split test \
  --mode all \
  --output finetuning/artifacts/cuad-sft-runs/demo/evaluation-test/comparison.json
```

Every invocation compares the vanilla base model against the supplied SFT checkpoint. There is no comparison flag and no plot-directory flag.
The output directory contains `vanilla.json`, `sft.json`, `comparison.json`, and a fixed `sft_plots/` subdirectory.

| Option | Default | Meaning |
|---|---|---|
| `--run-dir` | required | Run root containing `config_snapshot.yaml`. |
| `--checkpoint` | run root/`adapter` | Adapter directory or merged model directory. |
| `--data-dir` | required | Directory containing `train.jsonl`, `dev.jsonl`, or `test.jsonl`. |
| `--mode` | `all` | `all`, `singleton`, `grouped`, `randomized`, or `canonical`. |
| `--split` | `all` | Read all JSONL files or only `train`, `dev`, or `test`. |
| `--max-new-tokens` | `8192` | Generation limit. |
| `--limit` | unset | Optional row limit for a smoke evaluation. |
| `--output` | run root/`comparison.json` | Comparison JSON output path. The vanilla and SFT reports are written beside it. |
| `--timing-log-every` | `10` | Log inference progress after every N rows for each model. |
| `--no-flash-attention` | false | Force the non-FlashAttention path. |

Model identity and the default FlashAttention setting come from the run snapshot. The evaluation data directory remains explicit so training data cannot be selected accidentally.

Reported metrics include:

- Precision, recall, and F1 using CUAD matching semantics.
- JSON parse success rate.
- Missing and extra key counts.
- Verbatim grounding rate.
- Per-category metrics.
- SFT-minus-vanilla deltas for aggregate and per-category metrics.

Inference walltime is written to `inference_walltime.jsonl` beside the comparison report.
It contains start and periodic progress events for both vanilla and SFT, including completed rows, elapsed seconds, last-row seconds, and rows per second.

Plots are always written under `sft_plots/` beside the comparison report:

- `aggregate_metrics.png`
- `per_category_precision.png`
- `per_category_recall.png`
- `per_category_f1.png`
- `per_category_f1_delta.png`
- `format_metrics.png`

## Merge and serving

Merge an adapter into a standalone checkpoint:

```bash
./.venv/bin/python finetuning/merge_cuad_sft_adapter.py \
  --run-dir finetuning/artifacts/cuad-sft-runs/demo
```

The default adapter and merged paths are `{RUN}/adapter` and `{RUN}/merged`; `--adapter-dir` and `--output-dir` retain explicit overrides.

Serve the adapter directly through vLLM:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python finetuning/serve_cuad_sft.py \
  --run-dir finetuning/artifacts/cuad-sft-runs/demo
```

Serve the merged fallback:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python finetuning/serve_cuad_sft.py \
  --run-dir finetuning/artifacts/cuad-sft-runs/demo \
  --merged-dir finetuning/artifacts/cuad-sft-runs/demo/merged
```

Common serving options:

| Option | Default | Meaning |
|---|---|---|
| `--run-dir` | required | Run root containing `config_snapshot.yaml`. |
| `--adapter-dir` | run root/`adapter` | PEFT adapter path. Mutually exclusive with `--merged-dir`. |
| `--merged-dir` | unset | Standalone merged checkpoint path. |
| `--adapter-name` | derived from saved model | vLLM LoRA adapter name, e.g. `cuad-qwen35-9b`. |
| `--served-model-name` | derived from saved model | Model ID exposed by the OpenAI-compatible API. |
| `--host` | `0.0.0.0` | Server bind address. |
| `--port` | `8000` | Server port. |
| `--max-model-len` | `32768` | vLLM context limit. |
| `--max-lora-rank` | saved `training.lora_rank` | Maximum served adapter rank. |
| `--dry-run` | false | Print the vLLM command without launching it. |

Serving sets `enable_thinking=false` through vLLM's default chat-template arguments, matching training and HF evaluation. The context limit remains explicit and may need increasing for longer prompts plus generated tokens.

## Palimpzest benchmark integration

[cuad_qwen_benchmark.py](cuad_qwen_benchmark.py) retains the existing OpenRouter YAML format and also accepts an `endpoint` block for a local OpenAI-compatible server.

The new setting is:

```yaml
convert_mode: separate-converts
```

Allowed values:

- `separate-converts`: one requested category per conversion; this is the existing default.
- `one-convert`: all categories in the job are requested in one conversion.

The benchmark CLI remains:

```bash
./.venv/bin/python finetuning/cuad_qwen_benchmark.py \
  --config finetuning/config.yaml
```

Its command-line options are:

| Option | Meaning |
|---|---|
| `--config` | Required benchmark YAML path. |
| `--verbose` | Overrides YAML verbosity and enables verbose output. |

Important YAML settings include `dataset.mode`, `dataset.data_dir`, `split`, `num_contracts`, `seed`, `job_mode`, `convert_mode`, `models`, `categories`, `openrouter` or `endpoint`, and `execution`.

## Dependencies and compatibility

The project defines these optional extras in [pyproject.toml](../pyproject.toml):

```bash
UV_CACHE_DIR=/tmp/palimpzest-uv-cache uv sync --all-extras
```

The locked H200 stack uses Python 3.12, CUDA 13.0 Torch 2.13, Transformers 5.17, vLLM 0.30, sentence-transformers 6.1, FlashAttention-3 from the Hopper subdirectory of Dao-AILab/flash-attention, flash-linear-attention 0.5.2 with CUDA support, and causal-conv1d 1.7. FlashAttention-3 and causal-conv1d are built without isolation after Torch is installed.

## H200 Slurm workflow

The resource and storage choices use the repository `.venv` and durable NFS-backed paths.

[run_h200_cuad_sft.slurm](run_h200_cuad_sft.slurm) runs training and isolated test evaluation on one H200. It forwards training CLI arguments, then supplies `--model-id Qwen/Qwen3.8-27B` and its own `--output-dir` last. The script always selects the 27B model, and all artifacts stay under `finetuning/artifacts/cuad-sft-runs/{SLURM_JOB_ID}`.

- Account: `quanta`
- Partition: `quanta-h200`
- QoS: `quanta-main`
- Node: `quanta-h200-1`
- Constraint: `nvidia_h200`
- CPUs: 16
- RAM: 64 GB by default; choose resources appropriate to the model
- GPUs: 1
- Time limit: two days
- Environment: repository `.venv`
- Evaluation: the isolated test holdout, with output under `{RUN}/evaluation-test/`

Submit it from any Slurm node with access to the checkout. Evaluation reads the script's effective model from `{RUN}/config_snapshot.yaml`; no separate evaluation model ID is supplied.

From the repository root on any Slurm node:

```bash
sbatch finetuning/run_h200_cuad_sft.slurm
```
