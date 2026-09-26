# CUAD Qwen3.8-27B-FP8 SFT

This document describes the current CUAD supervised fine-tuning workflow in this repository.
The implementation trains a LoRA adapter on the Qwen3.8-27B-FP8 text backbone using the local CUAD chunk dataset and evaluates it through the existing Palimpzest JSON extraction pipeline.

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
        │  Qwen3.8-27B-FP8 + PEFT LoRA
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
| [sft_config.yaml](sft_config.yaml) | Default Qwen3.8-27B-FP8 training configuration. |
| [train_cuad_sft.py](train_cuad_sft.py) | Loads the Qwen3.8 text-only path, attaches LoRA, masks prompt tokens, and trains. |
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
  id: Qwen/Qwen3.8-27B-FP8

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
  lora_dropout: 0.0
  qlora: false
  gradient_checkpointing: true
  flash_attention: true
```

The training artifact directory is owned by `train_cuad_sft.py`, not this YAML file. It defaults to `finetuning/artifacts/cuad-sft-run` and can be changed with `--output-dir`.

The GH200 workflow uses `qlora: false`, FlashAttention2, and one visible CUDA device. It loads the FP8 checkpoint and dequantizes it to BF16 before attaching LoRA, because Transformers’ compressed FP8 matmul has no autograd formula for training. Only the LoRA adapter parameters are trainable.

The trainer:

- Uses Qwen3.8’s text-only `AutoModelForCausalLM` path from the composite checkpoint.
- Uses `enable_thinking=False` in the chat template.
- Masks all user/system prompt tokens with `-100`.
- Trains only the assistant JSON completion and delimiter.
- Uses Qwen3.8 attention, Gated DeltaNet, and MLP projections as LoRA targets.
- Uses Transformers’ native `add_adapter` path after BF16 dequantization, avoiding the inference-only FP8 backward path.
- Uses zero LoRA dropout because PyTorch does not implement `fused_dropout` for this checkpoint’s Float8 activations.
- Uses `logits_to_keep` so vocabulary logits are materialized only for completion tokens.
- Refuses to truncate an example that exceeds `max_seq_length`.
- Requires one visible CUDA device.

### Training CLI

```bash
./.venv/bin/python finetuning/train_cuad_sft.py \
  --config finetuning/sft_config.yaml
```

For a one-step functional smoke run on the same training path:

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
| `--resume-from-checkpoint` | unset | Hugging Face Trainer checkpoint to resume. |
| `--max-steps` | unset | Optional optimizer-step cap for a smoke run. |
| `--eval-limit` | unset | Optional limit on dev rows evaluated during training. |
| `--output-dir` | `finetuning/artifacts/cuad-sft-run` | Training artifact directory; owned by the CLI/script rather than YAML. |
| `--log-level` | `INFO` | Python logging level. |

The trainer writes checkpoints, `adapter/`, tokenizer files, and `training_manifest.json` under the selected output directory.
It also writes `training_walltime.jsonl` with `train_start`, per-optimizer-step,
`checkpoint`, and `train_end` events. Optimizer-step events track Triton cache
changes; steps that compile Triton kernels are included once as warmup but are
excluded from the steady-state mean and are never multiplied in the full-run
extrapolation. Use at least three smoke steps when you need a clean steady-state
estimate; a one-step functional smoke intentionally reports insufficient timing
evidence for extrapolation.
Each checkpoint event records the checkpoint path, global step, epoch, UTC timestamp, and elapsed training seconds.

## Evaluation CLI

```bash
./.venv/bin/python finetuning/evaluate_cuad_sft.py \
  --checkpoint finetuning/artifacts/cuad-sft-run/adapter \
  --data-dir finetuning/artifacts/cuad-sft \
  --model-id Qwen/Qwen3.8-27B-FP8 \
  --mode all \
  --output finetuning/artifacts/cuad-sft-run/comparison.json
```

Every invocation compares the vanilla base model against the supplied SFT checkpoint. There is no comparison flag and no plot-directory flag.
The output directory contains `vanilla.json`, `sft.json`, `comparison.json`, and a fixed `sft_plots/` subdirectory.

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | Adapter directory or merged model directory. |
| `--data-dir` | required | Directory containing `train.jsonl` and/or `dev.jsonl`. |
| `--model-id` | `Qwen/Qwen3.8-27B-FP8` | Base model used with an adapter. |
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
It contains model-load events and periodic progress events for both vanilla and SFT, including synchronized generation timing, compile/kernel-warmup diagnostics, completed rows, elapsed seconds, last-row seconds, and rows per second. The comparison also records a first-row-plus-steady-state extrapolation.

Plots are always written under `sft_plots/` beside the comparison report:

- `aggregate_metrics.png`
- `per_category_precision.png`
- `per_category_recall.png`
- `per_category_f1.png`
- `per_category_f1_delta.png`
- `format_metrics.png`

## Merge and serving

Merge an adapter into a standalone checkpoint:

For the FP8 model ID, the merge helper dequantizes the base to BF16 first so it matches the training representation.

```bash
./.venv/bin/python finetuning/merge_cuad_sft_adapter.py \
  --model-id Qwen/Qwen3.8-27B-FP8 \
  --adapter-dir finetuning/artifacts/cuad-sft-run/adapter \
  --output-dir finetuning/artifacts/cuad-sft-run/merged
```

Serve the adapter directly through vLLM:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python finetuning/serve_cuad_sft.py \
  --adapter-dir finetuning/artifacts/cuad-sft-run/adapter \
  --served-model-name cuad-qwen38-27b-fp8
```

Serve the merged fallback:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python finetuning/serve_cuad_sft.py \
  --merged-dir finetuning/artifacts/cuad-sft-run/merged \
  --served-model-name cuad-qwen38-27b-fp8
```

Common serving options:

| Option | Default | Meaning |
|---|---|---|
| `--model-id` | `Qwen/Qwen3.8-27B-FP8` | Base model for direct adapter serving. |
| `--adapter-dir` | unset | PEFT adapter path. Mutually exclusive with `--merged-dir`. |
| `--merged-dir` | unset | Standalone merged checkpoint path. |
| `--adapter-name` | `cuad-qwen38-27b-fp8` | vLLM LoRA adapter name. |
| `--served-model-name` | `cuad-qwen38-27b-fp8` | Model ID exposed by the OpenAI-compatible API. |
| `--host` | `0.0.0.0` | Server bind address. |
| `--port` | `8000` | Server port. |
| `--max-model-len` | `36864` | vLLM context limit. |
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

Qwen3.8 requires the current Transformers 5.x model implementation, PEFT native adapter integration, and the compatible fine-grained FP8 `kernels` package. The validated smoke environment used Transformers 5.18.0.dev0, PEFT 0.21.0, `kernels` 0.17.x, FlashAttention2, and `flash-linear-attention`. The GH200 job defaults `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` so it uses the fine-grained Triton FP8 path without an unauthenticated first-forward DeepGEMM download; set it to `0` only after validating a locally cached DeepGEMM build. `causal-conv1d` may require `nvcc` to build; if unavailable, Transformers uses its reference implementation, which is slower.

## Slurm job

[run_cuad_sft.slurm](run_cuad_sft.slurm) runs the complete workflow on one node:

- Account: `quanta`
- Partition: `quanta-gh200`
- QoS: `quanta-main`
- Node constraint: `nvidia_gh200_480gb`
- CPUs: 36
- RAM: 240 GB
- GPUs: 1
- Time limit: 24 hours
- Environment: repository `.venv`
- Training: FP8 checkpoint dequantized to BF16, LoRA, 3 epochs, 36,864-token ceiling
- Evaluation: all-field vanilla-vs-SFT comparison with fixed `sft_plots/`

The job requires the existing local inputs `testdata/cuad-data` and `testdata/cuad-chunk`; it does not download data on the compute node. The SFT JSONL files are rebuilt into the job-specific artifact directory before training.

Submit it from the repository root:

```bash
sbatch finetuning/run_cuad_sft.slurm
```

For a timing smoke run that still builds the complete CUAD train/dev dataset,
cap only the expensive model passes with exported Slurm variables:

```bash
sbatch --export=ALL,SMOKE_STEPS=8,SMOKE_EVAL_LIMIT=1,SMOKE_INFERENCE_LIMIT=8,SMOKE_TIMING_LOG_EVERY=1 finetuning/run_cuad_sft.slurm
```

The training manifest records synchronized first-step, Triton-warmup, and clean
steady-state timing, including a full-run extrapolation. Triton compilation is
counted as one-time overhead and is not treated as steady state. The comparison
JSON records a corresponding inference extrapolation; model-load time is logged
separately and excluded from that extrapolation.

Each job writes to `finetuning/artifacts/slurm/<job-id>/`:

- `data/` — generated train/dev SFT JSONL and manifest;
- `training/` — checkpoints, adapter, training manifest, and training walltime log;
- `comparison.json`, `vanilla.json`, and `sft.json`;
- `sft_plots/` — comparison figures;
- `inference_walltime.jsonl` — periodic vanilla/SFT inference timing events;
- `data_build.log`, `training.log`, and `evaluation.log`.
