# Cosmos Predict 2.5 — Inference Pipeline

Batch inference pipeline for [NVIDIA Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2) on SLURM clusters.
Supports image-conditioned video generation (image2world) with autoregressive multi-chunk generation and multi-GPU context parallelism.

---

## What this repo contains

```
scripts/
  prepare_cosmos_predict_batch.py   # builds the manifest JSONL from image+caption folders
  run_inference.sh                  # core run script (single or multi-GPU)
  sbatch_template_4gpu.sh           # SLURM submission template (4×GPU, edit before use)

data/open_domain/
  imgs/       # 5 sample driving images (PNG) from diverse cities
  caption/    # matching text prompts (.txt, one per image)

patches/
  mila_cluster_fix.patch            # fix for OSError crashes in distributed.py (see below)
```

---

## Prerequisites

### 1. Clone Cosmos Predict 2.5

```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict2
cd cosmos-predict2
pip install -e ".[all]"
```

### 2. Download model weights

The 14B model (~28 GB) will be fetched automatically on first run if you set:

```bash
export HF_HOME=/path/to/scratch/huggingface
export HF_TOKEN=your_huggingface_token
```

Or pre-download with:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos-Predict2-14B-Video2World')"
```

### 3. Prepare your data

Your input data must follow this layout:

```
data_root/
  caption/   *.txt   (one text file per sample, UTF-8)
  imgs/      *.png / *.jpg / *.jpeg / *.webp  (one image per sample, same stem as .txt)
```

The 5 sample scenes in `data/open_domain/` are ready to use out of the box.

---

## Quick start

### Single GPU (test)

```bash
export OUTPUT_ROOT=/path/to/output
export MODEL_NAME="14B/post-trained"
export EGO_LIMIT=0
export OPEN_LIMIT=2
export SEEDS_PER_SAMPLE=1
export FRAMES_PER_VIDEO=77       # 1 chunk = 4.8 s at 16 fps
export CHUNK_OVERLAP=8
export NUM_STEPS=20
export NUM_GPUS=1

bash scripts/run_inference.sh
```

### 4-GPU context-parallel (recommended)

Edit `scripts/sbatch_template_4gpu.sh` (partition, paths, conda env), then:

```bash
sbatch scripts/sbatch_template_4gpu.sh
```

**Performance on A100 80 GB (measured):**
| Setting | GPUs | VRAM/GPU | Time/chunk |
|---------|------|----------|------------|
| 2B model, 20 steps, 720p | 1 | ~21 GB | ~7 min |
| 2B model, 20 steps, 720p | 4 | ~11 GB | ~2 min |

4×A100 40 GB works fine for the 2B model (~11 GB/GPU). For the 14B model use 4×A100 80 GB (~34 GB/GPU).

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `2B/post-trained` | `2B/post-trained` or `14B/post-trained` |
| `FRAMES_PER_VIDEO` | 146 | Total output frames. chunk_size=77, so 146 = 2 chunks |
| `CHUNK_OVERLAP` | 8 | Frames of conditioning context carried over between chunks |
| `NUM_STEPS` | 20 | Diffusion denoising steps (35 = full quality; 20 = ~37% faster) |
| `NUM_GPUS` | 4 | Context-parallel GPUs (splits activations, replicates weights) |
| `EGO_LIMIT` | 1 | Max ego-view samples (set 0 to skip) |
| `OPEN_LIMIT` | 5 | Max open-domain samples (set 0 to skip) |
| `SEEDS_PER_SAMPLE` | 1 | Random seeds per input image |

**Frame math:**
- 1 chunk = 77 frames = 4.8 s at 16 fps
- 2 chunks with overlap=8: 77 + (77−8) = **146 frames ≈ 9.1 s**
- 3 chunks: 77 + 2×(77−8) = **215 frames ≈ 13.4 s**

---

## Cluster patch (distributed.py)

On some clusters, `torchrun` crashes with `OSError` during distributed init because:
1. `os.sched_setaffinity` is blocked by the SLURM cgroup
2. `ctypes.CDLL("libcudart.so")` fails if `LD_LIBRARY_PATH` is not set early enough

Apply `patches/mila_cluster_fix.patch` to make both errors non-fatal:

```bash
cd cosmos-predict2
git apply /path/to/patches/mila_cluster_fix.patch
```

The patch wraps both calls in try/except and logs a warning instead of crashing.

---

## Data layout for your own images

```
my_data/
  ego_condition/
    caption/   scene_01.txt  scene_02.txt ...
    imgs/      scene_01.png  scene_02.png ...
  open_domain/
    caption/   ...
    imgs/      ...
```

Then set `EGO_ROOT` and `OPEN_DOMAIN_ROOT` in `run_inference.sh` (or via env vars):

```bash
export EGO_ROOT=/path/to/my_data/ego_condition
export OPEN_DOMAIN_ROOT=/path/to/my_data/open_domain
```
