#!/bin/bash
#SBATCH --job-name=cosmos-chunk
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=160G
#SBATCH --time=2:10:00
#SBATCH --output=/home/mila/s/stanict/scratch/logs/cosmos_chunk_%j.out
#SBATCH --error=/home/mila/s/stanict/scratch/logs/cosmos_chunk_%j.err

# CHUNK must be passed via --export=CHUNK=<n>  (0-indexed, 10 videos per chunk)
: "${CHUNK:?Must set CHUNK=<n>}"

MANIFEST_ALL=/home/mila/s/stanict/scratch/cosmos_predict_run/manifest_all.jsonl
CHUNK_MANIFEST=/home/mila/s/stanict/scratch/cosmos_predict_run/chunk_${CHUNK}.jsonl
OUTPUT_DIR=/home/mila/s/stanict/scratch/cosmos_predict_run/outputs
VENV=/home/mila/s/stanict/cosmos-predict2.5/.venv

export HF_HOME=/home/mila/s/stanict/scratch/huggingface
export HF_TOKEN=$(cat /home/mila/s/stanict/scratch/huggingface/token)

mkdir -p "${OUTPUT_DIR}"

echo "=== Cosmos Predict — Chunk ${CHUNK} (entries $((CHUNK*10)) to $((CHUNK*10+9))) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
date

# Extract this chunk's 10 entries from master manifest
${VENV}/bin/python3 - << PY
import json, sys
from pathlib import Path

manifest_all = Path("${MANIFEST_ALL}")
chunk_manifest = Path("${CHUNK_MANIFEST}")
chunk = int("${CHUNK}")
start = chunk * 10
end   = start + 10

lines = manifest_all.read_text().splitlines()
chunk_lines = lines[start:end]

if not chunk_lines:
    print(f"ERROR: chunk {chunk} is out of range (manifest has {len(lines)} entries)")
    sys.exit(1)

chunk_manifest.write_text('\n'.join(chunk_lines) + '\n')
print(f"Chunk {chunk}: {len(chunk_lines)} entries (lines {start}-{start+len(chunk_lines)-1})")
for l in chunk_lines:
    print(f"  {json.loads(l)['name']}")
PY

# cuda runtime fix
site_packages="$(${VENV}/bin/python3 -c 'import site; print(site.getsitepackages()[0])')"
for libdir in "${site_packages}/nvidia/cuda_runtime/lib" \
              "${site_packages}/nvidia/cuda_nvrtc/lib" \
              "${site_packages}/nvidia/cublas/lib" \
              "${site_packages}/nvidia/cudnn/lib" \
              "${site_packages}/nvidia/nccl/lib"; do
  [[ -d "${libdir}" ]] && export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
done
export CUDA_HOME="${site_packages}/nvidia"
export CUDA_PATH="${site_packages}/nvidia"

cd /home/mila/s/stanict/cosmos-predict2.5

echo ""
echo "=== Starting inference ==="
date

torchrun --nproc_per_node=4 examples/inference.py \
    -i "${CHUNK_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "2B/post-trained" \
    --disable-guardrails

echo ""
echo "=== Done ==="
date

# Report what was saved
echo "Videos in output dir:"
ls "${OUTPUT_DIR}"/*.mp4 2>/dev/null | wc -l
