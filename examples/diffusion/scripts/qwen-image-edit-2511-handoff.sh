#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
RUNS_ROOT="${REPO_ROOT}/runs"
PYTHON_BIN="${PYTHON_BIN:-python}"
MICROMAMBA_ENV="${MICROMAMBA_ENV:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen-Image-Edit-2511}"
NUNCHAKU_ROOT="${NUNCHAKU_ROOT:-${WORKSPACE_ROOT}/nunchaku}"
EXPORT_ROOT="${EXPORT_ROOT:-${REPO_ROOT}/exports/nunchaku/search}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_ROOT}/models/Qwen-iImage-edit-2511-int4}"
REF_SAMPLES_DIR="${REF_SAMPLES_DIR:-${REPO_ROOT}/runs/benchmarks/qwen-image-edit-2511/bf16-fullgpu-50steps-qwen-image-edit-search-holdout-24-all24-gpu0-rerun/samples/YAML/qwen-image-edit-search-holdout-24}"
BENCH_GPU="${BENCH_GPU:-auto}"
BENCH_GPU_CANDIDATES="${BENCH_GPU_CANDIDATES:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-300}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/diffusion/scripts/qwen-image-edit-2511-handoff.sh <variant>

This script waits for a PTQ search run to finish, then:
1. converts the exported DeepCompressor checkpoint to Nunchaku split weights
2. merges them into a single int4 safetensors file
3. links the merged file into MODEL_ROOT (defaults to a sibling models directory)
4. runs the 24-image Nunchaku benchmark against the BF16 rerun reference
5. refreshes the unified catalog/report page

Environment:
  BENCH_GPU             Explicit benchmark GPU, or auto. Default: auto
  BENCH_GPU_CANDIDATES  Candidate GPUs used when BENCH_GPU=auto. Default: 0,1,2,3
  REF_SAMPLES_DIR       BF16 rerun reference sample directory.
  POLL_SECONDS          Wait interval while PTQ is still running. Default: 300
EOF
}

run_dc_python() {
  (
    cd "${REPO_ROOT}"
    if [[ -n "${MICROMAMBA_ENV}" ]]; then
      PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" micromamba run -p "${MICROMAMBA_ENV}" "${PYTHON_BIN}" "$@"
    else
      PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" "$@"
    fi
  )
}

run_nunchaku_python() {
  (
    cd "${REPO_ROOT}"
    if [[ -n "${MICROMAMBA_ENV}" ]]; then
      PYTHONPATH="${NUNCHAKU_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" micromamba run -p "${MICROMAMBA_ENV}" "${PYTHON_BIN}" "$@"
    else
      PYTHONPATH="${NUNCHAKU_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" "$@"
    fi
  )
}

variant="${1:-}"
if [[ -z "${variant}" ]]; then
  usage
  exit 1
fi

job_name="qwen-image-edit-2511-search-${variant}"
export_name="qwen-image-edit-2511-search-${variant}-gptq"
merged_name="${export_name}-int4.safetensors"
checkpoint_name="nunchaku_qwen_image_edit_2511_${variant//-/_}_int4.safetensors"

find_latest_named_run() {
  local latest
  latest=$(
    find "${RUNS_ROOT}" -type d \( -path "*/${job_name}/run-*" -o -path "*/${job_name}.RUNNING/run-*" \)       | rg -v '/run-[^/]+\.ERROR$'       | while IFS= read -r candidate; do
          stat -c '%Y %n' "${candidate}"
        done       | sort -nr       | head -n 1       | cut -d' ' -f2-
  )
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf '%s
' "${latest}"
}

ptq_still_running() {
  pgrep -af "deepcompressor.app.diffusion.ptq.*search\.${variant}\.yaml" >/dev/null
}

wait_for_model_dir() {
  local run_dir model_dir
  while true; do
    if run_dir=$(find_latest_named_run); then
      model_dir="${run_dir}/model"
      if [[ -f "${model_dir}/model.pt" && -f "${model_dir}/scale.pt" && -f "${model_dir}/wgts.pt" ]]; then
        if ptq_still_running; then
          echo "[handoff:${variant}] model files exist but PTQ is still alive; waiting ${POLL_SECONDS}s"
        else
          printf '%s
' "${run_dir}"
          return 0
        fi
      else
        echo "[handoff:${variant}] waiting for quantized model artifacts under ${run_dir}"
      fi
    else
      echo "[handoff:${variant}] no run directory yet; waiting ${POLL_SECONDS}s"
    fi
    sleep "${POLL_SECONDS}"
  done
}

run_dir=$(wait_for_model_dir)
quant_path="${run_dir}/model"
merged_path="${EXPORT_ROOT}/${merged_name}"
link_path="${MODEL_ROOT}/${checkpoint_name}"

mkdir -p "${EXPORT_ROOT}" "${MODEL_ROOT}"

echo "[handoff:${variant}] converting ${quant_path} -> ${EXPORT_ROOT}/${export_name}"
run_dc_python -m deepcompressor.backend.nunchaku.convert   --quant-path "${quant_path}"   --output-root "${EXPORT_ROOT}"   --model-name "${export_name}"   --model-path "${MODEL_ID}"

echo "[handoff:${variant}] merging split safetensors -> ${merged_path}"
run_nunchaku_python -m nunchaku.merge_safetensors   -i "${EXPORT_ROOT}/${export_name}"   -m NunchakuQwenImageTransformer2DModel   -o "${merged_path}"

ln -sfn "${merged_path}" "${link_path}"
echo "[handoff:${variant}] linked ${link_path} -> ${merged_path}"

echo "[handoff:${variant}] running 24-image Nunchaku benchmark"
GPU="${BENCH_GPU}" GPU_CANDIDATES="${BENCH_GPU_CANDIDATES}" NUNCHAKU_CKPT="${merged_path}" NUNCHAKU_RUN_TAG="${variant}" REF_SAMPLES_DIR="${REF_SAMPLES_DIR}"   bash "${REPO_ROOT}/examples/diffusion/scripts/qwen-image-edit-2511-benchmark.sh" nunchaku

echo "[handoff:${variant}] refreshing catalog/report"
run_dc_python "${REPO_ROOT}/examples/diffusion/scripts/qwen-image-edit-2511-refresh-catalog.py"

echo "[handoff:${variant}] completed"
