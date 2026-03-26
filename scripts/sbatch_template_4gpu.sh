#!/bin/bash
#SBATCH --job-name=cosmos-predict
#SBATCH --partition=YOUR_PARTITION   # adjust to your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4            # a100 40GB works for 2B; for 14B use 80GB GPUs
#SBATCH --mem=160G
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/scratch/cosmos_output/slurm-%j.out
#SBATCH --error=/path/to/scratch/cosmos_output/slurm-%j.err

module load miniconda/3

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cosmos-predict25      # change to your env name

export HF_HOME=/path/to/scratch/huggingface
export HF_TOKEN=$(cat /path/to/scratch/huggingface/token)

export OUTPUT_ROOT=/path/to/scratch/cosmos_output

# ---- inference settings ----
export MODEL_NAME="2B/post-trained"  # or "14B/post-trained"
export EGO_LIMIT=1                   # ego-view samples to run (0 = skip)
export OPEN_LIMIT=5                  # open-domain samples to run (0 = skip)
export SEEDS_PER_SAMPLE=1
export FRAMES_PER_VIDEO=146          # 77 + (77-8) = 146 frames = ~9 s at 16 fps
export CHUNK_OVERLAP=8               # conditioning frames carried over between chunks
export NUM_STEPS=20                  # denoising steps (35 default; 20 is ~37% faster)
export NUM_GPUS=4                    # context-parallel inference across N GPUs

mkdir -p "${OUTPUT_ROOT}"
bash "$(dirname "$0")/run_inference.sh"
