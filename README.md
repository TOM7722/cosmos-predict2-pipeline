# Cosmos Predict 2.5 — Inference Pipeline

Batch inference pipeline for [NVIDIA Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2) on SLURM clusters.
Supports image-conditioned video generation (image2world) with autoregressive multi-chunk generation and multi-GPU context parallelism.

> **This repo is a thin wrapper around [cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2).**
> It does not include model code — you need to clone that repo separately and point `REPO_DIR` to it.

---

## What this repo contains

```
scripts/
  prepare_cosmos_predict_batch.py   # builds the manifest JSONL from image+caption folders
  run_inference.sh                  # core run script (single or multi-GPU via torchrun)
  sbatch_template_4gpu.sh           # SLURM submission template (4×GPU, edit before use)

data/open_domain/
  imgs/       # 5 sample driving images (PNG) — Abu Dhabi, Amsterdam, Anchorage, Auckland, Austin
  caption/    # matching text prompts (.txt, one per image)

patches/
  mila_cluster_fix.patch            # optional: fix OSError crashes in distributed.py (see below)
```

The actual model inference is done by `examples/inference.py` inside the **cosmos-predict2** repo.
This pipeline just prepares the manifest and calls it.

---

## Step 1 — Clone cosmos-predict2 and install

```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict2
cd cosmos-predict2

conda create -n cosmos-predict25 python=3.10 -y
conda activate cosmos-predict25

pip install -e ".[all]"

# flash_attn must be exactly in [2.7.0, 2.7.4] — 2.7.3 confirmed working
# Build takes 20–40 min (compiles CUDA kernels for your GPU architecture)
pip install flash-attn==2.7.3 --no-build-isolation
```

> **Confirmed working:** PyTorch 2.7.0 + CUDA 12.6 + flash-attn 2.7.3 on A100 80 GB.

> **Do not rsync the env between clusters.** flash_attn compiles GPU-arch-specific CUDA
> kernels (A100 = sm_80, H100 = sm_90). They are not interchangeable — reinstall on each cluster.

---

## Step 2 — Download model weights

Set your HuggingFace credentials before running:

```bash
export HF_HOME=/path/to/scratch/huggingface
export HF_TOKEN=your_huggingface_token
```

Weights are downloaded automatically on first run. To pre-download:

```bash
# 2B model
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos-Predict2-2B-Video2World')"
# 14B model (~56 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos-Predict2-14B-Video2World')"
```

---

## Step 3 — Run inference

### Option A: Single GPU (quick test with sample images)

```bash
export REPO_DIR=/path/to/cosmos-predict2      # ← required: path to the cosmos-predict2 clone
export OUTPUT_ROOT=/path/to/output

export OPEN_DOMAIN_ROOT="$(pwd)/data/open_domain"   # use the 5 sample images from this repo
export EGO_LIMIT=0          # skip ego-view data (we have none)
export OPEN_LIMIT=2         # run 2 of the 5 samples

export MODEL_NAME="14B/post-trained"   # or "2B/post-trained"
export SEEDS_PER_SAMPLE=1
export FRAMES_PER_VIDEO=77             # 1 chunk = 4.8 s at 16 fps
export CHUNK_OVERLAP=8
export NUM_STEPS=20
export NUM_GPUS=1

bash scripts/run_inference.sh
```

### Option B: 4-GPU context-parallel via SLURM (recommended)

Edit `scripts/sbatch_template_4gpu.sh` and fill in:
- `--partition` — your cluster's partition name
- `REPO_DIR` — path to your cosmos-predict2 clone
- `HF_HOME` and `HF_TOKEN` — HuggingFace credentials
- `OUTPUT_ROOT` — where to write videos
- `OPEN_DOMAIN_ROOT` or `EGO_ROOT` — your image data

Then submit:

```bash
sbatch scripts/sbatch_template_4gpu.sh
```

---

## Key parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `REPO_DIR` | *(required)* | Path to your cosmos-predict2 clone |
| `MODEL_NAME` | `2B/post-trained` | `2B/post-trained` or `14B/post-trained` |
| `FRAMES_PER_VIDEO` | `146` | Total output frames (chunk_size=77 is fixed) |
| `CHUNK_OVERLAP` | `8` | Frames of context carried over between chunks |
| `NUM_STEPS` | `20` | Diffusion denoising steps (35 = full quality; 20 = ~37% faster) |
| `NUM_GPUS` | `4` | Context-parallel GPUs (splits activations, replicates weights) |
| `EGO_LIMIT` | `0` | Max ego-view samples to run (0 or omit `EGO_ROOT` to skip) |
| `OPEN_LIMIT` | `5` | Max open-domain samples to run (0 or omit `OPEN_DOMAIN_ROOT` to skip) |
| `SEEDS_PER_SAMPLE` | `1` | Random seeds per input image |

**Frame math:**
- 1 chunk = 77 frames = 4.8 s at 16 fps
- 2 chunks with overlap=8: 77 + (77−8) = **146 frames ≈ 9.1 s**
- 3 chunks: 77 + 2×(77−8) = **215 frames ≈ 13.4 s**

**VRAM requirements (4-GPU context parallel):**
| Model | GPUs | VRAM/GPU | Time/chunk |
|-------|------|----------|------------|
| 2B, 20 steps, 720p | 1× A100 80 GB | ~21 GB | ~7 min |
| 2B, 20 steps, 720p | 4× A100 40 GB | ~11 GB | ~2 min |
| 14B, 20 steps, 720p | 4× A100 80 GB | ~34 GB | TBD |

---

## Data layout for your own images

```
my_data/
  caption/   scene_01.txt  scene_02.txt ...   (UTF-8, one prompt per file)
  imgs/      scene_01.png  scene_02.png ...   (same stem as .txt)
```

Then:
```bash
export OPEN_DOMAIN_ROOT=/path/to/my_data
export EGO_LIMIT=0
```

---

## Cluster patch (distributed.py) — optional

On some clusters, `torchrun` crashes during distributed init with `OSError` because:
1. `os.sched_setaffinity` is blocked by the SLURM cgroup
2. `ctypes.CDLL("libcudart.so")` fails before `LD_LIBRARY_PATH` is set

Apply the patch to make both errors non-fatal warnings:

```bash
cd /path/to/cosmos-predict2
git apply /path/to/cosmos-predict2-pipeline/patches/mila_cluster_fix.patch
```
