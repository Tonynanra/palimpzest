#!/usr/bin/env bash
#
# Complete CUAD Qwen3.8-27B-FP8 LoRA workflow on one Quanta GH200 node.
# Submit from the repository root (or pass this script by absolute path):
#
#   sbatch finetuning/run_cuad_sft.slurm
#
# The script intentionally uses the repository's already-prepared local CUAD
# data.  It creates all generated data and model artifacts in a job-specific
# directory so concurrent jobs do not overwrite one another.
#
#SBATCH --job-name=cuad-sft
#SBATCH --account=quanta
#SBATCH --partition=quanta-gh200
#SBATCH --qos=quanta-main
#SBATCH --constraint=nvidia_gh200_480gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=cuad_sft_%j.out
#SBATCH --error=cuad_sft_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
fi
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
JOB_ROOT="${REPO_ROOT}/finetuning/artifacts/slurm/${JOB_ID}"
RAW_DATA_DIR="${REPO_ROOT}/testdata/cuad-data"
CHUNK_DATA_DIR="${REPO_ROOT}/testdata/cuad-chunk"
DATA_DIR="${JOB_ROOT}/data"
TRAINING_DIR="${JOB_ROOT}/training"
TRAIN_CONFIG="${JOB_ROOT}/sft_config.yaml"

DATA_LOG="${JOB_ROOT}/data_build.log"
TRAINING_LOG="${JOB_ROOT}/training.log"
EVALUATION_LOG="${JOB_ROOT}/evaluation.log"

# Optional smoke controls. The data builder always creates the complete CUAD
# train/dev JSONL; these only cap the expensive model passes for a timing run.
SMOKE_STEPS="${SMOKE_STEPS:-}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-}"
SMOKE_INFERENCE_LIMIT="${SMOKE_INFERENCE_LIMIT:-}"
SMOKE_TIMING_LOG_EVERY="${SMOKE_TIMING_LOG_EVERY:-1}"

mkdir -p "${JOB_ROOT}"

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Required CUAD input is missing: ${path}" >&2
        echo "Prepare testdata/cuad-data and testdata/cuad-chunk before submitting this job." >&2
        exit 1
    fi
}

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON}" >&2
    exit 1
fi

require_file "${RAW_DATA_DIR}/train_separate_questions.json"
require_file "${RAW_DATA_DIR}/test.json"
require_file "${CHUNK_DATA_DIR}/train_separate_questions.json"
require_file "${CHUNK_DATA_DIR}/test.json"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-36}"
# The SFT data builder imports Palimpzest, whose LiteLLM dependency otherwise
# attempts a network fetch for model pricing during Python startup.
export LITELLM_LOCAL_MODEL_COST_MAP=true
# The GH200/Transformers FP8 dispatcher can otherwise fetch a large DeepGEMM
# bundle at first forward and reject this checkpoint's scale layout.  The
# fine-grained Triton FP8 fallback is stable for this workflow; callers may set
# TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=0 after validating a local DeepGEMM build.
export TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR="${TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR:-1}"
# Keep a job-local Triton cache when the caller has not supplied one so the
# timing callback can identify JIT compilation and exclude it from steady state.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${JOB_ROOT}/triton-cache}"

echo "CUAD SFT job ${JOB_ID}"
echo "Host: $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "Artifacts: ${JOB_ROOT}"
echo "Python: ${PYTHON}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR: ${TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR}"
echo "TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
"${PYTHON}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

run_logged() {
    local log_path="$1"
    shift
    echo "+ $*" | tee "${log_path}"
    "$@" 2>&1 | tee -a "${log_path}"
}

echo "== Building CUAD SFT data =="
run_logged "${DATA_LOG}" \
    "${PYTHON}" "${REPO_ROOT}/finetuning/cuad_sft_data.py" \
    --chunk-data-dir "${CHUNK_DATA_DIR}" \
    --raw-data-dir "${RAW_DATA_DIR}" \
    --split train \
    --output-dir "${DATA_DIR}" \
    --seed 42 \
    --train-fraction 0.9

echo "== Preparing job-local training configuration =="
sed "s|^  dir: .*|  dir: ${DATA_DIR}|" \
    "${REPO_ROOT}/finetuning/sft_config.yaml" > "${TRAIN_CONFIG}"
if ! grep -Eq '^[[:space:]]+qlora:[[:space:]]+false[[:space:]]*$' "${TRAIN_CONFIG}"; then
    echo "Expected FP8-checkpoint BF16 LoRA configuration with qlora: false" >&2
    exit 1
fi

echo "== Training Qwen3.8 FP8-checkpoint BF16 LoRA adapter =="
TRAIN_ARGS=(
    "${PYTHON}" "${REPO_ROOT}/finetuning/train_cuad_sft.py"
    --config "${TRAIN_CONFIG}"
    --output-dir "${TRAINING_DIR}"
)
if [[ -n "${SMOKE_STEPS}" ]]; then
    TRAIN_ARGS+=(--max-steps "${SMOKE_STEPS}")
fi
if [[ -n "${SMOKE_EVAL_LIMIT}" ]]; then
    TRAIN_ARGS+=(--eval-limit "${SMOKE_EVAL_LIMIT}")
fi
run_logged "${TRAINING_LOG}" "${TRAIN_ARGS[@]}"

if [[ ! -f "${TRAINING_DIR}/adapter/adapter_model.safetensors" ]]; then
    echo "Training completed without producing an adapter checkpoint." >&2
    exit 1
fi

echo "== Running vanilla-vs-SFT inference evaluation =="
EVAL_ARGS=(
    "${PYTHON}" "${REPO_ROOT}/finetuning/evaluate_cuad_sft.py"
    --checkpoint "${TRAINING_DIR}/adapter"
    --data-dir "${DATA_DIR}"
    --model-id Qwen/Qwen3.8-27B-FP8
    --mode all
    --max-new-tokens 8192
    --timing-log-every "${SMOKE_TIMING_LOG_EVERY}"
    --output "${JOB_ROOT}/comparison.json"
)
if [[ -n "${SMOKE_INFERENCE_LIMIT}" ]]; then
    EVAL_ARGS+=(--limit "${SMOKE_INFERENCE_LIMIT}")
fi
run_logged "${EVALUATION_LOG}" "${EVAL_ARGS[@]}"

for required_output in \
    "${JOB_ROOT}/comparison.json" \
    "${JOB_ROOT}/vanilla.json" \
    "${JOB_ROOT}/sft.json" \
    "${JOB_ROOT}/inference_walltime.jsonl" \
    "${JOB_ROOT}/sft_plots/aggregate_metrics.png"; do
    if [[ ! -e "${required_output}" ]]; then
        echo "Inference completed without producing ${required_output}." >&2
        exit 1
    fi
done

echo "CUAD SFT workflow completed successfully."
echo "Artifacts: ${JOB_ROOT}"
