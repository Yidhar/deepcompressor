#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)

MICROMAMBA_ENV="${MICROMAMBA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DEFAULT_MODEL_PATH="Qwen/Qwen-Image-Edit-2511"
if [[ -d "${WORKSPACE_ROOT}/models/Qwen-Image-Edit-2511" ]]; then
  DEFAULT_MODEL_PATH="${WORKSPACE_ROOT}/models/Qwen-Image-Edit-2511"
fi
DEFAULT_INT4_MODEL_ROOT="${WORKSPACE_ROOT}/models/Qwen-iImage-edit-2511-int4"
DEFAULT_NUNCHAKU_ROOT="${WORKSPACE_ROOT}/nunchaku"

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
INT4_MODEL_ROOT="${INT4_MODEL_ROOT:-${DEFAULT_INT4_MODEL_ROOT}}"
PROMPT_YAML="${PROMPT_YAML:-${REPO_ROOT}/examples/diffusion/prompts/qwen-image-edit-search-holdout.yaml}"
REF_SAMPLES_DIR="${REF_SAMPLES_DIR:-${REPO_ROOT}/references/torch.bfloat16/qwen-image-edit-2511/fmeuler50-cfg4.0/samples/YAML/qwen-image-edit-search-holdout-24}"
NUNCHAKU_ROOT="${NUNCHAKU_ROOT:-${DEFAULT_NUNCHAKU_ROOT}}"
NUNCHAKU_CKPT="${NUNCHAKU_CKPT:-${INT4_MODEL_ROOT}/nunchaku_qwen_image_edit_2511_current_best_quality_int4.safetensors}"
NUNCHAKU_RUN_TAG="${NUNCHAKU_RUN_TAG:-current-best-quality}"

ENV_PREFIX=()
if [[ -n "${MICROMAMBA_ENV}" ]]; then
  ENV_PREFIX=(micromamba run -p "${MICROMAMBA_ENV}")
fi

GPU="${GPU:-auto}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0,1,2,3}"
GPU_MAX_MEMORY_MIB="${GPU_MAX_MEMORY_MIB:-2048}"
GPU_MAX_UTIL_PCT="${GPU_MAX_UTIL_PCT:-10}"
DEVICE="${DEVICE:-cuda}"
NUM_STEPS="${NUM_STEPS:-50}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
LIMIT="${LIMIT:-0}"
BF16_OFFLOAD="${BF16_OFFLOAD:-none}"
NUNCHAKU_PRECISION="${NUNCHAKU_PRECISION:-int4}"
NUNCHAKU_BLOCKS_ON_GPU="${NUNCHAKU_BLOCKS_ON_GPU:-1}"

BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/runs/benchmarks/qwen-image-edit-2511}"
REPORT_TITLE="${REPORT_TITLE:-Qwen-Image-Edit-2511 BF16 vs Nunchaku}"
BF16_OUTPUT_DIR_OVERRIDE="${BF16_OUTPUT_DIR:-}"
NUNCHAKU_OUTPUT_DIR_OVERRIDE="${NUNCHAKU_OUTPUT_DIR:-}"
REPORT_OUTPUT_DIR_OVERRIDE="${REPORT_OUTPUT_DIR:-}"

SELECTED_GPU=""
SAMPLE_SCOPE_TAG=""
BF16_OUTPUT_DIR_RESOLVED=""
NUNCHAKU_OUTPUT_DIR_RESOLVED=""
REPORT_OUTPUT_DIR_RESOLVED=""

usage() {
  cat <<USAGE
Usage:
  bash examples/diffusion/scripts/qwen-image-edit-2511-benchmark.sh <stage>

Stages:
  bf16      Run the diffusers BF16 benchmark on an auto-selected free GPU.
  nunchaku  Run the exported Nunchaku int4 benchmark on an auto-selected free GPU.
  report    Export an HTML side-by-side report from the latest BF16 and Nunchaku runs.
  all       Run nunchaku -> bf16 -> report.

Defaults:
  REF_SAMPLES_DIR points at the existing 24-image holdout reference set.
  LIMIT=0 means replay the full reference set instead of a smoke subset.
  GPU=auto picks the first free GPU from GPU_CANDIDATES.

Environment:
  GPU                  Explicit GPU index, or auto. Default: ${GPU}
  GPU_CANDIDATES       Comma-separated candidate GPUs. Default: ${GPU_CANDIDATES}
  GPU_MAX_MEMORY_MIB   Free-GPU memory threshold. Default: ${GPU_MAX_MEMORY_MIB}
  GPU_MAX_UTIL_PCT     Free-GPU utilization threshold. Default: ${GPU_MAX_UTIL_PCT}
  MICROMAMBA_ENV       Optional micromamba env path used for Python calls.
  MODEL_PATH           Model root or HF id for Qwen-Image-Edit-2511.
  INT4_MODEL_ROOT      Unified root for exported Qwen Image int4 checkpoints.
  PROMPT_YAML          Prompt YAML for the benchmark set.
  REF_SAMPLES_DIR      BF16 reference sample directory used to align seeds.
  NUNCHAKU_CKPT        Merged Nunchaku int4 checkpoint. Default: ${INT4_MODEL_ROOT}/nunchaku_qwen_image_edit_2511_current_best_quality_int4.safetensors
  NUNCHAKU_RUN_TAG     Label used in default Nunchaku benchmark output directories. Default: ${NUNCHAKU_RUN_TAG}
  NUNCHAKU_ROOT        Path to the nunchaku repo.
  NUM_STEPS            Number of inference steps. Default: ${NUM_STEPS}
  TRUE_CFG_SCALE       CFG scale. Default: ${TRUE_CFG_SCALE}
  LIMIT                Number of reference samples to replay. Default: ${LIMIT}
  BF16_OFFLOAD         BF16 offload mode: none|model|sequential. Default: ${BF16_OFFLOAD}
  BENCH_ROOT           Output root for benchmark runs.
  BF16_OUTPUT_DIR      Override BF16 benchmark output directory.
  NUNCHAKU_OUTPUT_DIR  Override Nunchaku benchmark output directory.
  REPORT_OUTPUT_DIR    Override report output directory.
USAGE
}

count_reference_samples() {
  find "${REF_SAMPLES_DIR}" -maxdepth 1 -name '*.png' | wc -l | awk '{print $1}'
}

build_sample_scope_tag() {
  local ref_name count
  ref_name=$(basename "${REF_SAMPLES_DIR}")
  if [[ "${LIMIT}" =~ ^[0-9]+$ ]] && (( LIMIT > 0 )); then
    printf '%s-limit%s\n' "${ref_name}" "${LIMIT}"
  else
    count=$(count_reference_samples)
    printf '%s-all%s\n' "${ref_name}" "${count}"
  fi
}

resolve_gpu() {
  if [[ -n "${SELECTED_GPU}" ]]; then
    return
  fi
  if [[ "${GPU}" != "auto" ]]; then
    SELECTED_GPU="${GPU}"
    echo "[benchmark] using explicit GPU${SELECTED_GPU}"
    return
  fi

  local status cand idx mem util
  status=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  while IFS=',' read -r idx mem util; do
    idx=${idx// /}
    mem=${mem// /}
    util=${util// /}
    for cand in ${GPU_CANDIDATES//,/ }; do
      if [[ "${idx}" != "${cand}" ]]; then
        continue
      fi
      if (( mem <= GPU_MAX_MEMORY_MIB && util <= GPU_MAX_UTIL_PCT )); then
        SELECTED_GPU="${idx}"
        echo "[benchmark] auto-selected GPU${SELECTED_GPU} (memory.used=${mem} MiB, util=${util}%)"
        return
      fi
    done
  done <<< "${status}"

  echo "No free GPU found in [${GPU_CANDIDATES}] with thresholds memory<=${GPU_MAX_MEMORY_MIB} MiB and util<=${GPU_MAX_UTIL_PCT}%" >&2
  echo "Current status:" >&2
  echo "${status}" >&2
  exit 1
}

run_python() {
  local gpu="$1"
  shift
  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${ENV_PREFIX[@]}" "${PYTHON_BIN}" "$@"
  )
}

find_latest_matching_dir() {
  local prefix="$1"
  local latest
  latest=$(
    find "${BENCH_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${prefix}-gpu*" \
      | while IFS= read -r dir; do
          stat -c '%Y %n' "${dir}"
        done \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )
  if [[ -z "${latest}" ]]; then
    echo "" && return 1
  fi
  printf '%s\n' "${latest}"
}

resolve_bf16_output_dir() {
  if [[ -n "${BF16_OUTPUT_DIR_RESOLVED}" ]]; then
    printf '%s\n' "${BF16_OUTPUT_DIR_RESOLVED}"
    return
  fi
  if [[ -n "${BF16_OUTPUT_DIR_OVERRIDE}" ]]; then
    BF16_OUTPUT_DIR_RESOLVED="${BF16_OUTPUT_DIR_OVERRIDE}"
  else
    resolve_gpu
    BF16_OUTPUT_DIR_RESOLVED="${BENCH_ROOT}/bf16-fullgpu-${NUM_STEPS}steps-${SAMPLE_SCOPE_TAG}-gpu${SELECTED_GPU}"
  fi
  printf '%s\n' "${BF16_OUTPUT_DIR_RESOLVED}"
}

resolve_nunchaku_output_dir() {
  if [[ -n "${NUNCHAKU_OUTPUT_DIR_RESOLVED}" ]]; then
    printf '%s\n' "${NUNCHAKU_OUTPUT_DIR_RESOLVED}"
    return
  fi
  if [[ -n "${NUNCHAKU_OUTPUT_DIR_OVERRIDE}" ]]; then
    NUNCHAKU_OUTPUT_DIR_RESOLVED="${NUNCHAKU_OUTPUT_DIR_OVERRIDE}"
  else
    resolve_gpu
    NUNCHAKU_OUTPUT_DIR_RESOLVED="${BENCH_ROOT}/${NUNCHAKU_RUN_TAG}-nunchaku-${NUM_STEPS}steps-${SAMPLE_SCOPE_TAG}-gpu${SELECTED_GPU}"
  fi
  printf '%s\n' "${NUNCHAKU_OUTPUT_DIR_RESOLVED}"
}

resolve_report_output_dir() {
  if [[ -n "${REPORT_OUTPUT_DIR_RESOLVED}" ]]; then
    printf '%s\n' "${REPORT_OUTPUT_DIR_RESOLVED}"
    return
  fi
  if [[ -n "${REPORT_OUTPUT_DIR_OVERRIDE}" ]]; then
    REPORT_OUTPUT_DIR_RESOLVED="${REPORT_OUTPUT_DIR_OVERRIDE}"
  else
    REPORT_OUTPUT_DIR_RESOLVED="${BENCH_ROOT}/report-${NUM_STEPS}steps-${SAMPLE_SCOPE_TAG}"
  fi
  printf '%s\n' "${REPORT_OUTPUT_DIR_RESOLVED}"
}

resolve_existing_bf16_dir() {
  if [[ -n "${BF16_OUTPUT_DIR_OVERRIDE}" ]]; then
    printf '%s\n' "${BF16_OUTPUT_DIR_OVERRIDE}"
    return
  fi
  find_latest_matching_dir "bf16-fullgpu-${NUM_STEPS}steps-${SAMPLE_SCOPE_TAG}"
}

resolve_existing_nunchaku_dir() {
  if [[ -n "${NUNCHAKU_OUTPUT_DIR_OVERRIDE}" ]]; then
    printf '%s\n' "${NUNCHAKU_OUTPUT_DIR_OVERRIDE}"
    return
  fi
  find_latest_matching_dir "${NUNCHAKU_RUN_TAG}-nunchaku-${NUM_STEPS}steps-${SAMPLE_SCOPE_TAG}"
}

run_bf16() {
  SELECTED_GPU=""
  resolve_gpu
  resolve_bf16_output_dir >/dev/null
  echo "[benchmark] BF16 -> GPU${SELECTED_GPU} -> ${BF16_OUTPUT_DIR_RESOLVED}"
  run_python "${SELECTED_GPU}" "${REPO_ROOT}/examples/diffusion/scripts/qwen-image-edit-2511-compare.py" \
    --mode bf16 \
    --model-path "${MODEL_PATH}" \
    --prompt-yaml "${PROMPT_YAML}" \
    --ref-samples-dir "${REF_SAMPLES_DIR}" \
    --output-dir "${BF16_OUTPUT_DIR_RESOLVED}" \
    --num-steps "${NUM_STEPS}" \
    --true-cfg-scale "${TRUE_CFG_SCALE}" \
    --limit "${LIMIT}" \
    --device "${DEVICE}" \
    --offload "${BF16_OFFLOAD}"
}

run_nunchaku() {
  SELECTED_GPU=""
  resolve_gpu
  resolve_nunchaku_output_dir >/dev/null
  echo "[benchmark] Nunchaku -> GPU${SELECTED_GPU} -> ${NUNCHAKU_OUTPUT_DIR_RESOLVED}"
  run_python "${SELECTED_GPU}" "${REPO_ROOT}/examples/diffusion/scripts/qwen-image-edit-2511-compare.py" \
    --mode nunchaku \
    --model-path "${MODEL_PATH}" \
    --prompt-yaml "${PROMPT_YAML}" \
    --ref-samples-dir "${REF_SAMPLES_DIR}" \
    --output-dir "${NUNCHAKU_OUTPUT_DIR_RESOLVED}" \
    --num-steps "${NUM_STEPS}" \
    --true-cfg-scale "${TRUE_CFG_SCALE}" \
    --limit "${LIMIT}" \
    --device "${DEVICE}" \
    --nunchaku-ckpt "${NUNCHAKU_CKPT}" \
    --nunchaku-root "${NUNCHAKU_ROOT}" \
    --precision "${NUNCHAKU_PRECISION}" \
    --num-blocks-on-gpu "${NUNCHAKU_BLOCKS_ON_GPU}"
}

run_report() {
  local bf16_dir nunchaku_dir
  bf16_dir=${1:-$(resolve_existing_bf16_dir)}
  nunchaku_dir=${2:-$(resolve_existing_nunchaku_dir)}
  if [[ -z "${bf16_dir}" || -z "${nunchaku_dir}" ]]; then
    echo "Missing benchmark directories for report generation." >&2
    echo "BF16: ${bf16_dir:-missing}" >&2
    echo "Nunchaku: ${nunchaku_dir:-missing}" >&2
    exit 1
  fi
  resolve_report_output_dir >/dev/null
  echo "[benchmark] Report -> ${REPORT_OUTPUT_DIR_RESOLVED}"
  run_python 0 "${REPO_ROOT}/examples/diffusion/scripts/qwen-image-edit-2511-report.py" \
    --run-dirs "${bf16_dir}" "${nunchaku_dir}" \
    --labels bf16-fullgpu nunchaku-int4 \
    --output "${REPORT_OUTPUT_DIR_RESOLVED}" \
    --title "${REPORT_TITLE}"
}

SAMPLE_SCOPE_TAG=$(build_sample_scope_tag)
stage="${1:-}"

case "${stage}" in
  bf16)
    run_bf16
    ;;
  nunchaku)
    run_nunchaku
    ;;
  report)
    run_report
    ;;
  all)
    run_nunchaku
    run_bf16
    run_report "${BF16_OUTPUT_DIR_RESOLVED}" "${NUNCHAKU_OUTPUT_DIR_RESOLVED}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
