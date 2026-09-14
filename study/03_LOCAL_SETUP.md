# Local setup — running Qwen-Drive-1.0 on this machine

This machine: **AMD Ryzen 9 3900X** (12 cores / 24 threads, AVX2, **no AVX-512, no bf16**),
**78 GB RAM**, one **RTX 3090 (24 GB)** that also drives the desktop.

The constraint that shaped every decision below: **the GPU is owned by the Alpamayo
perception sweep** (`alpamayo1.5/perception/run_round1.sh` — four `train.py` jobs in
parallel, 110 epochs each at ~310 s/epoch, ~11.5 GB VRAM, 99–100 % utilisation). Everything
here therefore runs **CPU-only**, and every command is written so it cannot touch the GPU.

---

## 1. Environment

```bash
cd /home/albert/Desktop/Qwen-Drive-1.0
uv venv --python 3.12 .venv

# CUDA build, so the same venv works on the GPU later
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

uv pip install --python .venv/bin/python \
    transformers==5.14.1 accelerate==1.12.0 safetensors==0.8.0 numpy==2.2.6 \
    pillow==12.0.0 opencv-python matplotlib==3.10.7 tqdm==4.67.1 pyarrow huggingface_hub
```

### What was deliberately **not** installed

`requirements.txt` pins three CUDA-only packages. None of them are needed:

| package | why it was skipped | what happens instead |
|---|---|---|
| `flash-attn` | needs nvcc + a GPU to build; ~30 min compile | pass `attn_implementation="sdpa"` — the expert's `_attend()` already branches on it, and `QwenDriveForPlanning._supports_sdpa = True` |
| `causal-conv1d` | CUDA kernel for the linear-attention conv | `transformers` falls back to `torch_causal_conv1d_update` |
| `flash-linear-attention` | Triton kernels for Gated DeltaNet | `transformers` falls back to `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` |

`transformers` 5.14.1 supports `qwen3_5` natively and logs one warning
("The fast path is not available…") before using the torch path. **Install all three when
you move to the GPU** — they are what the released numbers were verified with.

### Weights

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen-Drive-1.0-4B", local_dir="weights/Qwen-Drive-1.0-4B",
                  max_workers=6, ignore_patterns=["assets/*", ".DS_Store", ".ms_upload_cache"])
EOF
```

13.78 GB, ~7 minutes at 60 MB/s. Layout:

```
weights/Qwen-Drive-1.0-4B/
├── model.safetensors        9.079 GB  bf16   the shared VLM
├── planner-rl/              2.080 GB  bf16   reasoning-mode only
├── planner-sft/             2.080 GB  bf16   both planning modes
└── perception/              0.500 GB  fp32   the BEV head
```

---

## 2. The one source change

`voxel_pool_depth` — the fused depth-aware voxel-pooling kernel of the view transform — had
**no CPU path**. The other kernel already had one: `attention.py` dispatches to
`multi_scale_deformable_attn_pytorch` when `not value.is_cuda`.

Added `_voxel_pool_depth_torch` in
[`src/qwen_drive_perception/ops/__init__.py`](../src/qwen_drive_perception/ops/__init__.py),
dispatched the same way. It is an `index_add_` — the CUDA kernel is a weighted scatter-add
and `ranks` is already the flattened output index — chunked so the `[points, 256]`
intermediate stays bounded, accumulating in fp32 and casting back to
`promote_types(feats, depth)` to match the CUDA return dtype.

**That is the only modification to the repository.** Everything else lives in `local/`.

### Verifying it

There is no GPU available to diff against, so the fallback is checked against a **naive
transcription of the CUDA kernel** — a literal nested loop over
`voxel_pool_depth_forward_all_kernel`:

```bash
PYTHONPATH=src python local/test_voxel_pool_cpu.py
```

```
valid points: 39 / 216
max abs diff: 0.000e+00   nonzero voxels: 38
all-collide accumulation: max abs diff 0.000e+00  (216 points into 1 voxel/cam)
PASS - CPU fallback matches the kernel semantics
```

The second case matters most: many frustum points legitimately land in the same voxel, so
the op must **accumulate** rather than overwrite. An `index_add_` does; a naive
`out[idx] = value` would not.

---

## 3. The dtype trap (measured, not guessed)

`local/bench_dtype.py`, 12 threads, VLM prefill:

| | prefill 256 | prefill 1024 | decode 1 token |
|---|---|---|---|
| **bfloat16** | 6.0 tok/s | 7.2 tok/s | 2.30 s |
| **float32** | **22.0 tok/s** | **22.5 tok/s** | 2.64 s |

Two things to read off this:

- **fp32 is 3.1× faster at prefill.** Zen 2 has no `avx512_bf16` and no AMX, so every bf16
  op is emulated through fp32 with conversion overhead. Prefill is compute-bound, so the
  emulation dominates.
- **bf16 is 1.15× faster at decode.** Single-token decoding is *bandwidth*-bound — you
  stream 4.54 B weights per token — and bf16 weights are half the bytes. The crossover is
  real and goes the other way.

### But do not just switch the planner to fp32

The planning expert **deliberately reproduces bf16 rounding**, and the source says so:

- `WaypointRotaryEmbedding` computes rotary phases in the module dtype, not fp32, because
  "training held the inverse-frequency table in bfloat16 and cast positions to it, which
  rounds positions above 256 to a coarser grid."
- `FourierFeatureEncoder` rebuilds its frequency table in the module dtype every call for
  the same reason.

The waypoint rotary anchor is **522** for a demo scene — measured, not the 3385 token count,
because mRoPE gives each image a 2-D grid of positions rather than one position per token, so
the maximum position is far below the sequence length. (I first wrote 3385 here; it was wrong.)

At 522, bf16's 8 significant bits give a spacing of 4, so the 50 waypoint positions
`523 … 572` collapse onto **13 distinct values**:

```
positions 523,524,525,526,527,528, …   ->  bf16  524,524,524,528,528,528, …
```

Two consequences:

- **Groups of ~4 consecutive waypoints share identical rotary phases.** That is what training
  saw, so it is what the weights expect.
- **Running the planner in fp32 hands the expert 50 distinct phases instead of 13** — phases it
  was never fitted against. Not a free precision upgrade; a distribution shift.

Had the anchor really been 3385 the collapse would be to 4 distinct values stepping by 16, so
the effect is about 3x milder than that — but it is still 50 positions becoming 13.

Hence the split used here:

| | dtype | why |
|---|---|---|
| **planning / VQA** | **bfloat16** | fidelity — the bf16 rounding is part of the model |
| **perception** | **float32** | the head *ships* as fp32; nothing in that path reproduces bf16 rounding, and it is 3× faster |

Quantifying the planner's fp32 delta is a one-command experiment once the GPU is free:
run `local/run_planning_demo.py` twice and diff the ADE. It is listed as an open question in
[`00_START_HERE.md`](00_START_HERE.md).

---

## 4. Running it (GPU by default)

Everything defaults to the GPU now: `--device cuda`, `--dtype bfloat16`. `.vscode/launch.json`
has 16 configurations and **passes no device flags at all** - they inherit the script
defaults. F5 and pick one.

```bash
export PYTHONPATH=src
export PATH="$PWD/.venv/bin:$PATH"      # torch shells out to `ninja` to build the kernels
export CUDA_HOME=/usr                   # for those same kernels
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python local/run_planning_demo.py   --modes direct,reasoning --output outputs/planning_demo
python local/run_perception_demo.py --output outputs/perception_demo
python local/run_vqa_probe.py --max-new-tokens 400 --repetition-penalty 1.05
python local/make_video.py --output outputs/qwen_drive_demo.mp4

bash local/run_all.sh                   # all of the above; DEVICE=cpu still works
```

### Three things that had to be fixed to get onto the GPU

1. **`ninja` was missing.** torch JIT-compiles the two perception kernels by shelling out to
   `ninja`, and without it you get
   `RuntimeError: Ninja is required to load C++ extensions`. `uv pip install ninja`.
2. **`ninja` was then not on `PATH`.** Installing it into the venv is not enough: invoking
   `.venv/bin/python` by absolute path does **not** put `.venv/bin` on `PATH`, and torch looks
   the binary up there. Same error message, different cause. Every GPU launch config now sets
   `PATH=${workspaceFolder}/.venv/bin:${env:PATH}`.
3. **nvcc 12.0 vs torch cu128.** The system toolkit is a minor version behind the one torch
   was built with. It compiles anyway (same major version); the pip `nvidia-cuda-nvcc-cu12`
   package does **not** help, as it ships only `ptxas`, not the `nvcc` driver.

### Measured speed-up

| stage | CPU | GPU |
|---|---|---|
| `DIRECT_PLANNING`, 6 samples | 1348 s | **1.9-3.3 s** |
| `REASONING_PLANNING`, 6 samples | 1152 s | **2.7 s** |
| 16 planning runs (2 checkpoints x 4 scenes x 2 modes) | ~11 h | **95 s** |
| perception frame | 225-308 s | **1.6-2.1 s** (+75 s once, to compile the kernels) |
| VQA answer | 739 s | **1.5-10 s** |

### The CPU path still exists, and is worth keeping

`--device cpu` still works on every script, and `DEVICE=cpu bash local/run_all.sh` still runs
the whole pipeline without a GPU. It is no longer in the launch configs, but it is what made
this repository usable while the card was busy, and the two torch fallbacks it needs
(`multi_scale_deformable_attn_pytorch` upstream, `_voxel_pool_depth_torch` added here) are
still in place and still verified by `local/test_voxel_pool_cpu.py`.

**It is not numerically identical to the GPU path**, and the difference is not small - see
[`04_RESULTS.md`](04_RESULTS.md) for the side by side. CPU ran fp32 with torch fallbacks; GPU
runs bf16 with the authors' CUDA kernels, which is what the model was trained against.

## 5. Sharing the card with a training job

The GPU is free now, but this is what worked while it was not, and it is worth keeping
because the failure modes are not obvious.

**One inference job at a time.** Running two 4.5 B jobs concurrently pushed a training sweep
from ~300 s/epoch to **356 s (+19 %)** and dropped its GPU utilisation from 99 % to 74 % - the
trainers were starved of *CPU* for data loading, not of GPU. Each of their processes ran 8
dataloader workers.

**Renice the process AND torch's threads.** `renice` on the pid does not cover torch's
intra-op pool; each of its ~65 threads needs its own. Two traps: `pgrep -f <script>` matches
the **bash wrapper** as well as python, and the threads do not exist until torch has spun
them up, so renice a few seconds *after* launch.

```bash
PID=$(pgrep -f "python -u local/run_planning_demo" | head -1)
renice -n 19 -p $PID
for t in $(ls /proc/$PID/task); do renice -n 19 -p $t; done
taskset -apc 4-11,16-23 $PID          # leave physical cores 0-3 for the trainer
```

On this box cpu `i` and cpu `i+12` are the two SMT threads of physical core `i`, so
`4-11,16-23` reserves cores 0-3 entirely. After renice + taskset the sweep returned to its
baseline epoch time and stayed there for 30 consecutive epochs.

[`local/launch_protected.sh`](../local/launch_protected.sh) does all of this for any command.

**Watch memory too.** bf16 planning peaks at ~11 GB RSS, fp32 perception at ~19 GB. Either
alone is fine against 78 GB; both at once took free RAM down to 11 GB. A watchdog that kills
only *our* job (matched by script name, so it can never target the trainer) is cheap
insurance.

## 6. Files added by this setup

```
local/
├── anatomy.py              parameter tree from the safetensors headers, no GPU
├── token_layout.py         measured prompt composition for both input types
├── bench_dtype.py          the bf16/fp32 table in §3
├── run_planning_demo.py    device-flexible scripts/demo.py
├── run_perception_demo.py  device-flexible scripts/run_perception.py
└── make_video.py           composes outputs/ into one mp4

study/
├── 00_START_HERE.md        the course
├── 01_MODEL_STRUCTURE.md   the concrete dump
├── 02_VS_ALPAMAYO_1_5.md   the comparison
└── 03_LOCAL_SETUP.md       this file

src/qwen_drive_perception/ops/__init__.py   ← the only upstream file touched
```
