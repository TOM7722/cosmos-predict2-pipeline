#!/bin/bash
# Monitor progress of cosmos predict run
# Usage: bash monitor_cosmos_predict.sh

MANIFEST=/home/mila/s/stanict/scratch/cosmos_predict_run/manifest_all.jsonl
OUTPUT_DIR=/home/mila/s/stanict/scratch/cosmos_predict_run/outputs
LOG_DIR=/home/mila/s/stanict/scratch/logs

TOTAL=$(wc -l < "${MANIFEST}")
DONE=$(ls "${OUTPUT_DIR}"/*.mp4 2>/dev/null | wc -l)
PENDING=$((TOTAL - DONE))

echo "=============================="
echo " Cosmos Predict Progress"
echo "=============================="
printf " Done:    %3d / %d\n" "${DONE}" "${TOTAL}"
printf " Pending: %3d / %d\n" "${PENDING}" "${TOTAL}"
echo ""

# Running jobs
RUNNING=$(squeue -u "$USER" -n cosmos-chunk --noheader 2>/dev/null)
if [[ -n "${RUNNING}" ]]; then
    echo "Running jobs:"
    squeue -u "$USER" -n cosmos-chunk -o "  %i  state=%-8T  node=%-12N  time=%M/%l" --noheader
else
    echo "No cosmos-chunk jobs currently running."
fi
echo ""

# Which chunks are done (have all 10 mp4s)
echo "Chunk status:"
python3 - << 'PY'
import json, re
from pathlib import Path

manifest = Path("/home/mila/s/stanict/scratch/cosmos_predict_run/manifest_all.jsonl")
output_dir = Path("/home/mila/s/stanict/scratch/cosmos_predict_run/outputs")

lines = manifest.read_text().splitlines()
total_chunks = (len(lines) + 9) // 10

done_names = {p.stem for p in output_dir.glob("*.mp4")}

for chunk in range(total_chunks):
    start = chunk * 10
    chunk_lines = lines[start:start+10]
    names = [json.loads(l)["name"] for l in chunk_lines]
    done = sum(1 for n in names if n in done_names)
    status = "DONE" if done == len(names) else ("partial" if done > 0 else "pending")
    print(f"  chunk {chunk:02d} (entries {start:3d}-{start+len(names)-1:3d}): {done}/{len(names)}  [{status}]")
PY
