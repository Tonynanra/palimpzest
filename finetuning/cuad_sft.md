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
| [evaluate_cuad_openrouter.py](evaluate_cuad_openrouter.py) | Runs standalone GPT-6 Luna inference through OpenRouter and records raw responses, usage, costs, and latency. |
| [evaluate_cuad_span_transitions.py](evaluate_cuad_span_transitions.py) | Scores paired span transitions from saved outputs and plots the four transition rates. |
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
| `--save-raw-outputs` | false | Save self-contained `vanilla.raw.jsonl` and `sft.raw.jsonl` beside the comparison report. |
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

To retain model outputs for downstream evaluation, add `--save-raw-outputs`:

```bash
./.venv/bin/python finetuning/evaluate_cuad_sft.py \
  --run-dir finetuning/artifacts/cuad-sft-runs/demo \
  --data-dir testdata/cuad/sft/test \
  --split test \
  --mode all \
  --save-raw-outputs \
  --output finetuning/artifacts/cuad-sft-runs/demo/evaluation-test/comparison.json
```

Each raw JSONL file contains one record per selected row, in evaluation order. The records include:

| Field | Meaning |
|---|---|
| `row_index` | 1-based position in the selected rows, shared between vanilla and SFT. IDs can repeat across chunks; retain this index and the chunk/view metadata when comparing outputs. |
| `model_role`, `model_id` | `vanilla` or `sft`, and the base model identity from the run snapshot. |
| `checkpoint` | SFT adapter or merged checkpoint path; null for vanilla. |
| `tokenizer_source` | The loaded tokenizer's source. Adapter comparisons share the checkpoint tokenizer when present, including for vanilla generation. |
| `mode` | Evaluation mode. |
| `generation_kwargs` | The `max_new_tokens`, `do_sample`, and `pad_token_id` controls passed to generation. |
| `enable_thinking` | The no-thinking chat-template setting (`false`). |
| `row` | The complete evaluated input, including contract text, categories, reference labels, messages, and source/chunk/view metadata. Canonical mode saves the updated categories and messages. |
| `prompt` | The decoded actual input tokens, including special-token markers, with tokenizer whitespace cleanup disabled. |
| `raw_completion` | Generated text excluding the prompt, including special-token markers and surrounding prose, with tokenizer whitespace cleanup disabled. |
| `completion` | The exact decoded text passed to the existing evaluator, which removes special tokens before parsing. |

JSON encoding escapes embedded newlines so each record occupies one physical line; `json.loads(line)` restores the saved text. Reference labels and contract text are included, so alternate parsers, matching rules, and grounding checks can operate without the original dataset files.

Records are written and flushed before parsing, including malformed responses. Both raw files are overwritten at the start of a capture-enabled invocation. Completed records remain available if a later generation or evaluation fails; file errors stop evaluation. Capture uses the same generation call as normal evaluation, and its additional decoding and writes are included in inference walltime. Existing metrics, report schemas, and plots retain their current behavior.

This option saves decoded text for new inference invocations. Existing aggregate reports cannot supply historical raw completions.

Plots are always written under `sft_plots/` beside the comparison report:

- `aggregate_metrics.png`
- `per_category_precision.png`
- `per_category_recall.png`
- `per_category_f1.png`
- `per_category_f1_delta.png`
- `format_metrics.png`

### Standalone GPT-6 Luna inference

Run one model against the selected CUAD rows without loading a local checkpoint or producing a comparison:

```bash
./.venv/bin/python finetuning/evaluate_cuad_openrouter.py \
  --data-dir testdata/cuad/sft/test \
  --split test \
  --mode all \
  --output finetuning/artifacts/cuad-openrouter-runs/luna
```

The script reads `OPENROUTER_API_KEY` directly from the repository-root `.env` file; a missing or empty key stops the run. It targets `openai/gpt-6-luna`, sends the saved CUAD messages as Chat Completions, and uses the configured reasoning effort (`medium`). OpenRouter uses its default routing and provider failover within the standard OpenAI/Azure allowlist; regional Azure variants are excluded. The Python SDK also uses its default HTTP retry policy, currently two retries, for both pricing lookup and inference. A terminal error after SDK retries are exhausted is recorded and stops the run; malformed completion JSON is retained and counted by the existing evaluator.

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | required | Directory containing the CUAD SFT JSONL rows. |
| `--split` | `test` | Read `train`, `dev`, `test`, or all three splits. |
| `--mode` | `all` | `all`, `singleton`, `grouped`, `randomized`, or `canonical`, using the same row selection as local inference. |
| `--max-new-tokens` | `8192` | Maximum completion tokens requested. |
| `--limit` | unset | Optional row limit for a small inference run. |
| `--output` | required | Run output directory. The runner creates `openrouter.json` and its raw, error, and walltime files inside it. Existing output files are not overwritten. |
| `--timing-log-every` | `10` | Write inference progress after every N rows. |

For a run directory such as `luna/`, the script writes `openrouter.json`, `openrouter.raw.jsonl`, `openrouter.errors.jsonl`, and `openrouter.inference_walltime.jsonl` inside it. Each raw record embeds the selected input row, exact request messages and settings, raw completion, full API response and usage, returned model/provider metadata, finish reason, request latency, observed SDK retries, and HTTP attempts. Completed raw records are flushed immediately. The summary is written initially and on completion/failure; the periodic walltime log supplies live progress.

Token usage is retained per response and aggregated in the summary: input, cached input, cache writes, and output tokens. The script snapshots standard OpenAI and Azure pricing before inference and records the retrieval time and source. Providers need not have matching prices or both be healthy during lookup. It calculates ordinary input, cache-read, cache-write, and output costs from the serving endpoint's rates, applying any prompt-length pricing override. OpenRouter's reported billed cost is retained separately in credits. Missing usage or pricing fields produce null costs with reasons and incomplete totals with missing-row counts. Usage and billing describe returned responses; charges for failed internal attempts are unavailable. The summary also contains the evaluator's precision, recall, F1, TP/FP/FN, parse, key, grounding, and per-category metrics.

The runner sends up to 64 row requests concurrently. Raw records are flushed as responses complete, so JSONL order can differ from evaluation order; use `row_index` when pairing records. A terminal request error stops submission of later batches, cancels work that has not started, and waits for already-running calls in the current batch so their responses and errors can be recorded.

The reported `end_to_end_seconds` covers dataset loading, pricing lookup, inference, and scoring. Request latency includes SDK retries and backoff. `requests_attempted` counts logical row requests started, while `inference_http_attempts` counts observed SDK HTTP attempts; neither measures OpenRouter's internal provider attempts. Unknown retry counts remain null. A failure summary preserves completed raw rows and their known usage and cost totals.

### Replay saved generation

Recreate the evaluator's reports and plots from a completed raw capture, without loading a model, tokenizer, or dataset:

```bash
./.venv/bin/python finetuning/evaluate_cuad_sft_from_raw.py \
  --evaluation-dir finetuning/artifacts/cuad-sft-runs/2376707 \
  --output-dir /tmp/cuad-sft-replay
```

The source directory must contain `comparison.json`, `inference_walltime.jsonl`, `vanilla.raw.jsonl`, and `sft.raw.jsonl`. For differently named captures such as `vanilla_raw.jsonl` and `sft_raw.jsonl`, supply their paths with `--vanilla-raw` and `--sft-raw`.

Replay uses the embedded input rows and the saved `completion` text through the original scoring and plotting code. It pairs records by `row_index` and rejects missing, duplicate, or mismatched rows and model/mode metadata. It writes the same three JSON reports, six plots, two raw captures, and timing log into the separate output directory.

All extraction, formatting, grounding, and delta metrics are recomputed. The three inference timing metrics and original timing log are preserved from the source evaluation because raw captures do not contain elapsed times. Those values describe the original generation, not replay performance. Only `inference_walltime_log` changes to point to the copied log in the output directory.

### Three-way saved inference comparison

Compare saved Vanilla, SFT, and OpenRouter API results without running inference:

```bash
./.venv/bin/python finetuning/compare_cuad_inference.py \
  finetuning/artifacts/cuad-sft-runs/2376707 \
  finetuning/artifacts/cuad-openrouter-runs/luna
```

Pass the Vanilla/SFT evaluation directory first and the API result directory second. The first directory must contain `comparison.json`, `vanilla.raw.jsonl`, and `sft.raw.jsonl`; the second must contain a complete `openrouter.json` run and `openrouter.raw.jsonl`. The script checks row counts, mode, split, complete row coverage, and identical embedded input rows, pairing by `row_index`.

All three saved completions are rescored with the shared evaluator. Source inference timings are retained. Reports and plots go to `finetuning/artifacts/cuad-three-way-comparison/`, leaving the input artifacts unchanged. The report includes only the API-minus-SFT delta. Its `sft_plots/` contains the aggregate, per-category precision/recall/F1, API-minus-SFT per-category F1, and formatting/grounding plots.

### Jaccard distribution from saved generation

Plot Vanilla and SFT Jaccard histograms with vertical threshold markers and x-axis ticks at `0.15` and `0.5`:

```bash
./.venv/bin/python finetuning/plot_cuad_jaccard_distribution.py \
  --evaluation-dir finetuning/artifacts/cuad-sft-runs/2376707
```

The script requires `comparison.json`, `vanilla.raw.jsonl`, and `sft.raw.jsonl`, with complete, matching captures. It reuses the evaluator's completion parsing, prediction normalization, and word-set Jaccard function without loading a model, tokenizer, or dataset. Output defaults to `sft_plots/jaccard_distribution.png` within the evaluation directory; use `--output` to choose another path.

Each sample is one gold annotation's highest Jaccard score against predictions for the same category and chunk. Both models use the same gold-span population, retaining repeated annotations. Missing predictions, missing keys, and malformed/non-object completions contribute zero scores; categories with no gold spans contribute no samples. The plot uses 20 shared bins over `[0, 1]`, shows counts, and measures pure Jaccard without the `Parties` substring exception. It describes gold-span coverage and match quality; false-positive predictions are not separate samples. The score distribution and both threshold markers are independent of the configured `IOU_THRESH`.

## Span transition evaluation

Score a completed `--mode all --save-raw-outputs` evaluation without loading model weights or generating new outputs:

```bash
./.venv/bin/python finetuning/evaluate_cuad_span_transitions.py \
  --evaluation-dir finetuning/artifacts/cuad-sft-runs/2607445/evaluation-test
```

The scorer requires `comparison.json`, `vanilla.raw.jsonl`, and `sft.raw.jsonl`, with all expected rows captured for both models. It pairs by `row_index`, verifies identical embedded inputs and matching model identity/mode, and rejects duplicate indices, incomplete captures, or malformed JSONL. Stored SFT IDs can repeat across chunks.

For each evaluated chunk, the candidate set is the union of distinct, original answer strings annotated under all 41 categories. Each candidate is positive for a category exactly when its string appears in that category's labels; otherwise it is negative. Identical strings are counted once per chunk/category, and a string annotated under multiple categories is positive for each. Unannotated text and hallucinated spans outside this candidate set do not add cases. Existing chunk boundaries are retained.

A model predicts positive when any of its extracted answers for the target category matches the candidate using the existing Jaccard threshold (`0.15`) or the `Parties` substring exception. Malformed/non-object completions and missing category values yield empty predictions, consistent with the existing evaluator. Parse failures are recorded in the report.

The paired cells use Vanilla as rows and SFT as columns:

| Gold label | Vanilla | SFT Predict 1 | SFT Predict 0 |
|---|---|---:|---:|
| Y = 1 | Predict 1 | a | b |
| Y = 1 | Predict 0 | c | d |
| Y = 0 | Predict 1 | h | g |
| Y = 0 | Predict 0 | f | e |

The command writes three additional artifacts beside the original reports:

- `span_transitions.json`: aggregate and per-category cells, supports, rates, numerators/denominators, and source/matching metadata. Overall rates use pooled counts.
- `span_transitions.md`: the two overall tables, four rates, and embedded plot.
- `span_transitions.png`: four bars, ordered FN recovery `c/(c+d)`, FP correction `g/(g+h)`, TP loss `b/(a+b)`, and New FP introduction `f/(e+f)`. The first two share blue (`#0072B2`, Improvements); the last two share orange (`#D55E00`, Regressions). The y-axis spans 0–100%, and each bar includes its percentage and numerator/denominator. The PNG is 10 × 5 inches at 180 DPI.

Zero-denominator rates are `null` in JSON and `N/A` in the Markdown and plot, without a numeric bar height. The existing evaluation reports and plots are unchanged.

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

For an inference-only job, submit [run_h200_cuad_inference.slurm](run_h200_cuad_inference.slurm) without script arguments:

```bash
sbatch finetuning/run_h200_cuad_inference.slurm
```

The script contains all inference arguments. It is configured for training run `2376707`, its `adapter/` checkpoint, the existing `testdata/cuad/sft/test/test.jsonl` holdout, `--mode all`, an 8,192-token generation limit, progress logging every 10 rows, and raw capture. The source run must contain `config_snapshot.yaml`. It performs standalone vanilla-vs-SFT evaluation without training or dataset generation.

Results go under the configured `INFERENCE_DIR`, currently `finetuning/artifacts/cuad-sft-runs/2376707/`. That directory contains `comparison.json`, `vanilla.json`, `sft.json`, both raw JSONL files, `inference_walltime.jsonl`, and `sft_plots/`. Repeated invocations overwrite those evaluation outputs. Scheduler stdout/stderr are written to `finetuning/cuad-inference-{SLURM_JOB_ID}.{out,err}`.

This job uses one H200 with the same Quanta account, partition, QoS, CPU count, and RAM as the training script, with a 12-hour limit and scheduler-selected node. Change resources in its `#SBATCH` directives, paths in its `readonly` assignments, and evaluator controls in the Python command inside the script. For a smoke run, add `--limit 8` and change `--max-new-tokens` to `128` there. A 128-token smoke run validates execution and may truncate responses; use the normal generation budget for accuracy evaluation.
