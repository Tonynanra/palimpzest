# CUAD Qwen3.5 SFT

This document describes the current CUAD supervised fine-tuning workflow in this repository.
The implementation trains a Qwen3.5-4B LoRA/QLoRA adapter on the local CUAD chunk dataset and evaluates it through the existing Palimpzest JSON extraction pipeline.

## Workflow

```text
testdata/cuad-chunk + testdata/cuad-data
        │
        ▼
cuad_sft_data.py
        │  randomized one-convert views
        ▼
finetuning/artifacts/cuad-sft/{train,dev}.jsonl
        │
        ▼
train_cuad_sft.py
        │  Qwen3.5 + PEFT LoRA/QLoRA
        ▼
finetuning/artifacts/cuad-sft-run/adapter
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
| [cuad_sft_data.py](cuad_sft_data.py) | Builds deterministic train/dev JSONL examples. |
| [sft_config.yaml](sft_config.yaml) | Default Qwen3.5 training configuration. |
| [train_cuad_sft.py](train_cuad_sft.py) | Loads Qwen3.5, attaches LoRA/QLoRA, masks prompt tokens, and trains. |
| [evaluate_cuad_sft.py](evaluate_cuad_sft.py) | Generates predictions and reports CUAD extraction metrics. |
| [merge_cuad_sft_adapter.py](merge_cuad_sft_adapter.py) | Merges a PEFT adapter into a standalone checkpoint. |
| [serve_cuad_sft.py](serve_cuad_sft.py) | Launches adapter or merged-model vLLM serving. |
| [cuad_qwen_benchmark.py](cuad_qwen_benchmark.py) | Existing Palimpzest benchmark, extended with local endpoint and `convert_mode` support. |
| [generate_cuad_chunk_dataset.py](generate_cuad_chunk_dataset.py) | Existing generator for the `cuad-chunk` dataset. SFT consumes the generated files; it does not regenerate them automatically. |
| [../src/palimpzest/prompts/prompt_factory.py](../src/palimpzest/prompts/prompt_factory.py) | Existing prompt factory with an opt-in provided-field-order mode for SFT data. |

Generated data, checkpoints, and merged models are ignored under `finetuning/artifacts/`.

## Dataset construction

The builder uses the existing `cuad-chunk` records and aggregates labels through the existing CUAD label logic.
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
  --chunk-data-dir testdata/cuad-chunk \
  --raw-data-dir testdata/cuad-data \
  --split train \
  --output-dir finetuning/artifacts/cuad-sft \
  --seed 42 \
  --train-fraction 0.9
```

| Option | Default | Meaning |
|---|---|---|
| `--chunk-data-dir` | `testdata/cuad-chunk` | Chunk-native CUAD input directory. |
| `--raw-data-dir` | `testdata/cuad-data` | Raw CUAD directory used for source hashing. |
| `--split` | `train` | CUAD source split: `train` or `test`. |
| `--output-dir` | `finetuning/artifacts/cuad-sft` | JSONL and manifest output directory. |
| `--seed` | `42` | Split and view-sampling seed. |
| `--train-fraction` | `0.9` | Source-contract fraction assigned to training. |

Outputs:

- `train.jsonl`
- `dev.jsonl`
- `manifest.json`

## Training configuration

The checked-in [sft_config.yaml](sft_config.yaml) currently contains:

```yaml
model:
  id: Qwen/Qwen3.5-4B

training:
  max_seq_length: 32768
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

The validated single-L40S smoke path used `qlora: true`, FlashAttention2, and `CUDA_VISIBLE_DEVICES=0`. BF16 LoRA is still supported, but QLoRA is the safer memory setting for long CUAD rows.

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
  --config finetuning/sft_config.yaml
```

| Option | Default | Meaning |
|---|---|---|
| `--config` | `finetuning/sft_config.yaml` | YAML training configuration. |
| `--resume-from-checkpoint` | unset | Hugging Face Trainer checkpoint to resume. |
| `--log-level` | `INFO` | Python logging level. |

The trainer writes checkpoints, `adapter/`, tokenizer files, and `training_manifest.json` under the configured output directory.
It also writes `training_walltime.jsonl` with `train_start`, `checkpoint`, and `train_end` events.
Each checkpoint event records the checkpoint path, global step, epoch, UTC timestamp, and elapsed training seconds.

## Evaluation CLI

```bash
./.venv/bin/python finetuning/evaluate_cuad_sft.py \
  --checkpoint finetuning/artifacts/cuad-sft-run/adapter \
  --data-dir finetuning/artifacts/cuad-sft \
  --model-id Qwen/Qwen3.5-4B \
  --mode all \
  --output finetuning/artifacts/cuad-sft-run/comparison.json
```

Every invocation compares the vanilla base model against the supplied SFT checkpoint. There is no comparison flag and no plot-directory flag.
The output directory contains `vanilla.json`, `sft.json`, `comparison.json`, and a fixed `sft_plots/` subdirectory.

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | Adapter directory or merged model directory. |
| `--data-dir` | required | Directory containing `train.jsonl` and/or `dev.jsonl`. |
| `--model-id` | `Qwen/Qwen3.5-4B` | Base model used with an adapter. |
| `--mode` | `all` | `all`, `singleton`, `grouped`, `randomized`, or `canonical`. |
| `--max-new-tokens` | `8192` | Generation limit. |
| `--limit` | unset | Optional row limit for a smoke evaluation. |
| `--output` | checkpoint parent/`comparison.json` | Comparison JSON output path. The vanilla and SFT reports are written beside it. |
| `--timing-log-every` | `10` | Log inference progress after every N rows for each model. |
| `--no-flash-attention` | false | Force the non-FlashAttention path. |

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
  --model-id Qwen/Qwen3.5-4B \
  --adapter-dir finetuning/artifacts/cuad-sft-run/adapter \
  --output-dir finetuning/artifacts/cuad-sft-run/merged
```

Serve the adapter directly through vLLM:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python finetuning/serve_cuad_sft.py \
  --adapter-dir finetuning/artifacts/cuad-sft-run/adapter \
  --served-model-name cuad-qwen35-4b
```

Serve the merged fallback:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python finetuning/serve_cuad_sft.py \
  --merged-dir finetuning/artifacts/cuad-sft-run/merged \
  --served-model-name cuad-qwen35-4b
```

Common serving options:

| Option | Default | Meaning |
|---|---|---|
| `--model-id` | `Qwen/Qwen3.5-4B` | Base model for direct adapter serving. |
| `--adapter-dir` | unset | PEFT adapter path. Mutually exclusive with `--merged-dir`. |
| `--merged-dir` | unset | Standalone merged checkpoint path. |
| `--adapter-name` | `cuad-qwen35-4b` | vLLM LoRA adapter name. |
| `--served-model-name` | `cuad-qwen35-4b` | Model ID exposed by the OpenAI-compatible API. |
| `--host` | `0.0.0.0` | Server bind address. |
| `--port` | `8000` | Server port. |
| `--max-model-len` | `32768` | vLLM context limit. |
| `--max-lora-rank` | `32` | Maximum served adapter rank. |
| `--dry-run` | false | Print the vLLM command without launching it. |

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
./.venv/bin/pip install -e ".[sft,sft-serving,sft-kernels]"
```

Qwen3.5 may require a newer Transformers source checkout than the normal package constraint. The validated smoke environment used a current Transformers development build, FlashAttention2, and `flash-linear-attention`. `causal-conv1d` may require `nvcc` to build; if unavailable, Transformers uses its reference implementation, which is slower but does not prevent the tested 32k QLoRA step from running.

## Slurm job

[run_cuad_sft.slurm](run_cuad_sft.slurm) runs the complete workflow on one node:

- Partition: `pi_srmadden`
- CPUs: 16
- RAM: 32 GB
- GPUs: 1
- Time limit: 24 hours
- Environment: repository `.venv`
- Training: QLoRA, 3 epochs, 32k-token ceiling
- Evaluation: all-field vanilla-vs-SFT comparison with fixed `sft_plots/`

Submit it from the repository root:

```bash
sbatch finetuning/run_cuad_sft.slurm
```

Each job writes to `finetuning/artifacts/slurm/<job-id>/`:

- `data/` — generated train/dev SFT JSONL and manifest;
- `training/` — checkpoints, adapter, training manifest, and training walltime log;
- `comparison.json`, `vanilla.json`, and `sft.json`;
- `sft_plots/` — comparison figures;
- `inference_walltime.jsonl` — periodic vanilla/SFT inference timing events;
- `data_build.log`, `training.log`, and `evaluation.log`.
