# Reproducing this repo on another machine

**The short version:** everything added here is **1.2 MB of code**. Weights,
environment and all outputs are downloadable or regenerable. The only thing you
must actively carry across is that 1.2 MB — and right now it is **not committed**.

---

## 0. First: the work is untracked

`git ls-files` on this clone returns only upstream directories. Everything added
lives in untracked paths, so **a fresh `git clone` of upstream gets none of it**:

```
training/  export_onnx/  study/  local/  .vscode/
README.md  TRAINING.md  ONNX_EXPORT.md  SETUP.md  setup.sh
assets/nuscenes_session.gif  assets/two_lifts.png
src/qwen_drive_perception/ops/__init__.py   (the one modified upstream file)
```

Pick one:

```bash
# A. commit it (recommended)
git checkout -b study
git add -A && git commit -m "Training pipeline, ONNX export, study docs"
git push origin study          # then clone that branch on the other machine

# B. or just tar the additions - 1.2 MB of code, 8.4 MB with the assets
tar czf qwen-drive-work.tgz training export_onnx study local .vscode \
    *.md setup.sh .gitignore assets/nuscenes_session.gif assets/two_lifts.png \
    src/qwen_drive_perception/ops/__init__.py
```

---

## 1. What has to be acquired, and what does not

| | size | how |
|---|---|---|
| upstream code + `data/demo` | 47 MB | `git clone` — **already tracked**, comes for free |
| **this work** | **1.2 MB** | commit or tar, see above |
| model weights | 13 GB | `hf download`, step 3 |
| `.venv` | 7.5 GB | rebuilt by `setup.sh` |
| `outputs/` | 55 GB | **do not copy** — every byte is regenerable |
| `data/nuscenes` | 9 GB | **optional**, only for the session videos |

`data/demo` ships in the upstream repo: 6 perception frames with full ground
truth and 4 planning scenes. That is enough to run **everything** in
`TRAINING.md` and `ONNX_EXPORT.md`. nuScenes is only needed to re-render the
README GIF.

---

## 2. Bootstrap

```bash
git clone <your-fork-or-branch> Qwen-Drive-1.0 && cd Qwen-Drive-1.0
bash setup.sh
```

`setup.sh` checks prerequisites, builds the venv, installs dependencies,
downloads the weights, and finishes with the gradient smoke test. Roughly 25-40
minutes, mostly the 13 GB download.

### The one decision that matters

`requirements.txt` pins **flash-attn**, **causal-conv1d** and
**flash-linear-attention**. This machine has **none of them installed**, and that
is not an accident:

```
is_causal_conv1d_available()          False
is_flash_linear_attention_available() False
```

With `flash-linear-attention` present, the 24 Gated-DeltaNet layers run fused
Triton kernels. Those are faster, and they **cannot be traced** — the ONNX export
of the language model would fail. The export in this repo works precisely because
the model falls back to `torch_chunk_gated_delta_rule`.

* `bash setup.sh` — skips the three, ONNX export works ← default
* `bash setup.sh --with-fla` — installs them, faster inference, **no ONNX export**

If you reproduce with `--with-fla` and the export breaks, that is why.

---

## 3. Verify, in order

```bash
export PYTHONPATH=src:. CUDA_HOME=/usr
```

### Environment is sane (~3 min)

The gradient test reads a cached VLM tap, so **cache first** or it errors on a
missing file:

```bash
.venv/bin/python training/cache_features.py                                 # ~2 min, needs a GPU
.venv/bin/python training/test_gradients.py --skip-patch --dtype bfloat16   # must FAIL
.venv/bin/python training/test_gradients.py --dtype bfloat16                # must PASS
```

The first is **supposed** to fail with `NotImplementedError` — that is the shipped
repo's missing backward. The second must print `682 / 682`.

### Training (~15 min)

```bash
.venv/bin/python training/cache_features.py
.venv/bin/python training/cache_planner_features.py
.venv/bin/python training/run_all_stages.py        # expect 5/5 stages passed
```

### ONNX (~45 min export, then ~20 min validate)

```bash
# export - see ONNX_EXPORT.md section 2 for all of them
.venv/bin/python export_onnx/export_perception.py --no-fold --opset 20 \
    --out outputs/onnx/perception/perception.onnx
.venv/bin/python export_onnx/export_vlm.py --part vision
.venv/bin/python export_onnx/export_vlm_layers.py --task perception

# validate
.venv/bin/python export_onnx/run_onnx_pipeline.py --phase run
.venv/bin/python export_onnx/run_onnx_pipeline.py --phase compare
```

Compare everything against
[study/10_EXPECTED_RESULTS.md](study/10_EXPECTED_RESULTS.md) /
`outputs/expected_results.json`.

---

## 4. Hardware, and what changes without it

Built on: 2 x RTX 3090 (24 GiB each, **no NVLink**), 78 GiB RAM, CUDA 12.8,
Python 3.12, torch 2.8.0.

| you have | what still works |
|---|---|
| **1 x 24 GiB GPU** | everything except the 2-GPU configs. Stage 1 peaks at 20.7 GiB |
| **1 x 16 GiB GPU** | stage 1 will not fit as-is (20.7 GiB peak). Lower `occ_max_points` / `map_max_points` in `training/config.py` - they are config fields, **not** CLI flags - or isolate losses with `--no-occ` / `--no-map`. Inference and ONNX are fine |
| **2 GPUs** | add `8a`-`8c`. DDP helps stage 1 (2x) and **hurts** stage 3 (1.7x slower) |
| **no GPU** | ONNX export and validation run entirely on CPU. Training does not |
| **low RAM** | the ONNX `--phase compare` step holds a 17 GiB fp32 model and peaks near 30 GiB resident. It was OOM-killed here (78 GiB) only when run alongside another heavy job - so run it alone. I have not measured the true floor |

nvcc is needed only for the two perception CUDA kernels, which compile on first
use. Training and ONNX export both call `enable_training_ops()` first, which
routes around those kernels entirely (verified: both `test_gradients.py` and
`export_perception.py` do this), so a mismatched nvcc blocks the *inference*
demos - `scripts/run_perception.py`, the study figures - not the training or
export work.

---

## 5. If something breaks

| symptom | cause |
|---|---|
| ONNX export of the VLM fails | you installed `flash-linear-attention`. See §2 |
| `NotImplementedError ... backward` in training | the patch is not active - `enable_training_ops()` must run first |
| process dies with no traceback | system OOM killer. Never run two heavy jobs at once |
| `ninja: not found` | prepend `.venv/bin` to PATH; torch shells out to it |
| NCCL hangs on 2 GPUs | consumer cards have no P2P - use `training/run_ddp.sh` |
| CUDA kernel build fails | nvcc must match the torch CUDA version (12.8 here) |

Full command reference: [TRAINING.md](TRAINING.md), [ONNX_EXPORT.md](ONNX_EXPORT.md).
