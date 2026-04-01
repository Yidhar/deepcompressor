#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
RUNS_ROOT="${REPO_ROOT}/runs"
PYTHON_BIN="${PYTHON_BIN:-python}"
MICROMAMBA_ENV="${MICROMAMBA_ENV:-}"
NUNCHAKU_ROOT="${NUNCHAKU_ROOT:-${WORKSPACE_ROOT}/nunchaku}"
CONFIG_HELPER="${REPO_ROOT}/examples/diffusion/scripts/qwen_image_edit_2511_configs.py"

MODEL_CFG="examples/diffusion/configs/model/qwen-image-edit-2511.yaml"
INT4_CFG="examples/diffusion/configs/svdquant/int4.yaml"
MODEL_ID="Qwen/Qwen-Image-Edit-2511"
NUNCHAKU_EXPORT_ROOT="${REPO_ROOT}/exports/nunchaku"
NUNCHAKU_EXPORT_NAME="qwen-image-edit-2511-gptq"
NUNCHAKU_MERGED_PATH="${NUNCHAKU_EXPORT_ROOT}/qwen-image-edit-2511-gptq-int4.safetensors"

usage() {
  cat <<'EOF'
Usage:
  bash examples/diffusion/scripts/qwen-image-edit-2511-baseline.sh <stage>

Stages:
  collect          Build the demo calibration dataset.
  quantize         Run the publishable W4A4 + SVDQuant + GPTQ baseline.
  export           Save the latest GPTQ cache as a DeepCompressor checkpoint.
  sample-bf16      Generate the 40-step BF16 demo samples.
  sample-bf16-50   Generate the 50-step BF16 demo samples.
  sample-gptq      Generate the 40-step GPTQ demo samples from the latest export.
  sample-gptq-50   Generate the 50-step GPTQ demo samples from the latest export.
  convert          Convert the latest exported checkpoint to split Nunchaku files.
  merge            Merge the split Nunchaku export into a single int4 safetensors file.
  all              Run collect -> quantize -> export -> convert.

Notes:
  Stage-specific Qwen Image overlays are generated on demand under
  .tmp/generated-configs/qwen-image-edit-2511/ and are not meant to be
  committed into the repo.

Environment:
  PYTHON_BIN       Python executable to use. Default: python
  MICROMAMBA_ENV   Optional micromamba environment prefix used for all Python calls.
  NUNCHAKU_ROOT    Path to the nunchaku repo for the merge step.
  CUDA_VISIBLE_DEVICES can be set outside the script, e.g. CUDA_VISIBLE_DEVICES=2,3
EOF
}

find_latest_named_run() {
  local job_name="$1"
  local run_dir

  run_dir=$(
    find "${RUNS_ROOT}" -type d \( -path "*/${job_name}/run-*" -o -path "*/${job_name}.RUNNING/run-*" \)       | rg -v '/run-[^/]+\.(RUNNING|ERROR)$'       | while IFS= read -r candidate; do
          stat -c '%Y %n' "${candidate}"
        done       | sort -nr       | head -n 1       | cut -d' ' -f2-
  )

  if [[ -z "${run_dir}" ]]; then
    echo "No run found for ${job_name} under ${RUNS_ROOT}" >&2
    exit 1
  fi

  printf '%s
' "${run_dir}"
}

run_python() {
  cd "${REPO_ROOT}"
  if [[ -n "${MICROMAMBA_ENV}" ]]; then
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" micromamba run -p "${MICROMAMBA_ENV}" "${PYTHON_BIN}" "$@"
  else
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" "$@"
  fi
}

materialize_configs() {
  local workflow="$1"
  local stage="$2"
  shift 2
  mapfile -t GENERATED_CFGS < <(run_python "${CONFIG_HELPER}" "${workflow}" "${stage}" "$@")
}

run_collect() {
  materialize_configs baseline collect
  run_python -m deepcompressor.app.diffusion.dataset.collect.calib     "${MODEL_CFG}"     "${GENERATED_CFGS[@]}"
}

run_quantize() {
  materialize_configs baseline quantize
  run_python -m deepcompressor.app.diffusion.ptq     "${MODEL_CFG}"     "${INT4_CFG}"     "${GENERATED_CFGS[@]}"
}

run_export() {
  local quant_run
  quant_run=$(find_latest_named_run "qwen-image-edit-demo-gptq-bf16")
  materialize_configs baseline export

  run_python -m deepcompressor.app.diffusion.ptq     "${MODEL_CFG}"     "${INT4_CFG}"     "${GENERATED_CFGS[@]}"     --load-from "${quant_run}/cache"
}

run_sample_bf16() {
  materialize_configs baseline sample-bf16
  run_python -m deepcompressor.app.diffusion.ptq     "${MODEL_CFG}"     "${GENERATED_CFGS[@]}"
}

run_sample_bf16_50() {
  materialize_configs baseline sample-bf16-50
  run_python -m deepcompressor.app.diffusion.ptq     "${MODEL_CFG}"     "${GENERATED_CFGS[@]}"
}

run_sample_gptq() {
  local export_run
  export_run=$(find_latest_named_run "qwen-image-edit-demo-gptq-bf16-export")
  materialize_configs baseline sample-gptq

  run_python -m deepcompressor.app.diffusion.ptq     "${MODEL_CFG}"     "${INT4_CFG}"     "${GENERATED_CFGS[@]}"     --load-from "${export_run}/model"
}

run_sample_gptq_50() {
  local export_run
  export_run=$(find_latest_named_run "qwen-image-edit-demo-gptq-bf16-export")
  materialize_configs baseline sample-gptq-50

  run_python -m deepcompressor.app.diffusion.ptq     "${MODEL_CFG}"     "${INT4_CFG}"     "${GENERATED_CFGS[@]}"     --load-from "${export_run}/model"
}

run_convert() {
  local export_run
  export_run=$(find_latest_named_run "qwen-image-edit-demo-gptq-bf16-export")

  run_python -m deepcompressor.backend.nunchaku.convert     --quant-path "${export_run}/model"     --output-root "${NUNCHAKU_EXPORT_ROOT}"     --model-name "${NUNCHAKU_EXPORT_NAME}"     --model-path "${MODEL_ID}"
}

run_merge() {
  if [[ ! -d "${NUNCHAKU_ROOT}" ]]; then
    echo "NUNCHAKU_ROOT does not exist: ${NUNCHAKU_ROOT}" >&2
    exit 1
  fi

  cd "${REPO_ROOT}"
  PYTHONPATH="${NUNCHAKU_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}"     -m nunchaku.merge_safetensors     -i "${NUNCHAKU_EXPORT_ROOT}/${NUNCHAKU_EXPORT_NAME}"     -m NunchakuQwenImageTransformer2DModel     -o "${NUNCHAKU_MERGED_PATH}"
}

stage="${1:-}"

case "${stage}" in
  collect)
    run_collect
    ;;
  quantize)
    run_quantize
    ;;
  export)
    run_export
    ;;
  sample-bf16)
    run_sample_bf16
    ;;
  sample-bf16-50)
    run_sample_bf16_50
    ;;
  sample-gptq)
    run_sample_gptq
    ;;
  sample-gptq-50)
    run_sample_gptq_50
    ;;
  convert)
    run_convert
    ;;
  merge)
    run_merge
    ;;
  all)
    run_collect
    run_quantize
    run_export
    run_convert
    ;;
  *)
    usage
    exit 1
    ;;
esac
