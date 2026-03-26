#!/usr/bin/env bash
set -euo pipefail

# REPO_DIR: path to your cosmos-predict2 clone
REPO_DIR="${REPO_DIR:-/path/to/cosmos-predict2}"

# Data roots — override via env vars or edit defaults here
EGO_ROOT="${EGO_ROOT:-/path/to/data/ego_condition}"
OPEN_DOMAIN_ROOT="${OPEN_DOMAIN_ROOT:-/path/to/data/open_domain}"
TEXT_SUBDIR="${TEXT_SUBDIR:-caption}"
IMAGE_SUBDIR="${IMAGE_SUBDIR:-imgs}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/output/cosmos_predict}"
OUTPUT_DIR="${OUTPUT_ROOT}/outputs"
MANIFEST_PATH="${OUTPUT_ROOT}/manifest.jsonl"

MODEL_NAME="${MODEL_NAME:-2B/post-trained}"
SEEDS_PER_SAMPLE="${SEEDS_PER_SAMPLE:-10}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-160}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-1}"
NUM_STEPS="${NUM_STEPS:-35}"
NUM_GPUS="${NUM_GPUS:-1}"
EGO_LIMIT="${EGO_LIMIT:-1}"
OPEN_LIMIT="${OPEN_LIMIT:-1}"

log() {
  printf '%s\n' "$*"
}

configure_python_cuda_runtime() {
  local site_packages
  site_packages="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
  if [[ -n "${site_packages}" ]]; then
    local runtime_lib="${site_packages}/nvidia/cuda_runtime/lib"
    local nvrtc_lib="${site_packages}/nvidia/cuda_nvrtc/lib"
    local cublas_lib="${site_packages}/nvidia/cublas/lib"
    local cudnn_lib="${site_packages}/nvidia/cudnn/lib"
    for libdir in "${runtime_lib}" "${nvrtc_lib}" "${cublas_lib}" "${cudnn_lib}"; do
      if [[ -d "${libdir}" ]]; then
        export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
      fi
    done
    # transformer_engine's _load_nvrtc searches CUDA_HOME recursively for libnvrtc.so*
    export CUDA_HOME="${site_packages}/nvidia"
    export CUDA_PATH="${site_packages}/nvidia"
  fi
}

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/prepare_cosmos_predict_batch.py" \
  --ego-root "${EGO_ROOT}" \
  --open-domain-root "${OPEN_DOMAIN_ROOT}" \
  --text-subdir "${TEXT_SUBDIR}" \
  --image-subdir "${IMAGE_SUBDIR}" \
  --output-jsonl "${MANIFEST_PATH}" \
  --seeds-per-sample "${SEEDS_PER_SAMPLE}" \
  --frames-per-video "${FRAMES_PER_VIDEO}" \
  --chunk-overlap "${CHUNK_OVERLAP}" \
  --num-steps "${NUM_STEPS}" \
  --ego-limit "${EGO_LIMIT}" \
  --open-limit "${OPEN_LIMIT}"

cd "${REPO_DIR}"
configure_python_cuda_runtime

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" examples/inference.py \
    -i "${MANIFEST_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL_NAME}" \
    --disable-guardrails
else
  python3 examples/inference.py \
    -i "${MANIFEST_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL_NAME}" \
    --disable-guardrails
fi
