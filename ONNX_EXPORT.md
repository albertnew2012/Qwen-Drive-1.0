# Exporting Qwen-Drive-1.0 to ONNX

A runbook. Every command was run on this machine; the numbers are what it
printed. For *why* each obstacle exists, see
[study/09_ONNX_EXPORT.md](study/09_ONNX_EXPORT.md).

**Status: the whole model is exported and numerically validated against
PyTorch**, in 73 graphs totalling 54 GiB. Both driving pipelines - perception and
planning - run end to end from ONNX and match PyTorch.

**There is no single .onnx file, and that is a runtime limit, not a choice.** See
section 6.

---

## 0. Before anything

```bash
cd /home/albert/Desktop/Qwen-Drive-1.0
export PYTHONPATH=src:.
export PATH="$PWD/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""      # export and validation run on CPU in fp32
export OMP_NUM_THREADS=8
```

Dependencies (already installed): `onnx`, `onnxruntime`, `onnxscript`.

---

## 1. What exists, and how well it matches

| graph | params | nodes | rel. diff vs PyTorch |
|---|---|---|---|
| VLM vision tower | 0.3335 B | 3,747 | **7.1e-05** |
| VLM text prefill, 33 graphs (perception shape) | 4.2058 B | 344,127 | **1.4e-06** |
| VLM text prefill, 33 graphs (planning shape) | 4.2058 B | 366,983 | ~1e-06 |
| VLM decode step (64 KV states) | 4.2058 B | 10,233 | **2.0e-06** |
| BEV perception head | 0.1251 B | 12,436 | **5.3e-04** |
| Planning expert (one denoise step) | 1.0398 B | 9,315 | **4.2e-07** |

### End to end

**Perception** - `vision -> 32 layers -> head`:

```
VLM hidden_states      3.146e-04   ok
VLM vit tap            7.162e-05   ok
VLM llm tap            5.099e-04   ok
planner KV keys  (x8)  5.183e-04   ok
planner KV values(x8)  2.248e-04   ok
all_cls_scores         1.032e-03   ok
all_bbox_preds         1.050e-03   ok
occ_pred               1.307e-06   ok
seg_preds              3.654e-06   ok
PIPELINE PASS  (worst 1.05e-03, tolerance 5e-03)
```

**Planning** - `vision -> 32 layers -> KV caches -> planner x10 Euler steps`:

```
ADE 0.00002 m     FDE 0.00004 m
PLANNER PASS  (ADE 1.64e-05 m, tolerance 5e-02)
```

Error scales differ by three orders of magnitude and that is expected: the
planner is a transformer evaluated once (4e-07), while the perception head
accumulates ~668,000 scattered contributions per frame in a different order than
PyTorch, and fp32 addition is not associative (5e-04).

---

## 2. Export - all commands

Order matters only in that the pipelines need their graphs to exist first.

### 2.1 Perception path

```bash
# BEV perception head                                              # 6a  ~4 min
.venv/bin/python export_onnx/export_perception.py --no-fold --opset 20 \
    --out outputs/onnx/perception/perception.onnx

# VLM vision tower                                                 # 6e  ~1 min
.venv/bin/python export_onnx/export_vlm.py --part vision

# VLM text, as 33 per-layer graphs                                 # 6g  ~12 min
.venv/bin/python export_onnx/export_vlm_layers.py --task perception
```

### 2.2 Planning path

```bash
# planner denoise step                                             # 6c  ~1 min
.venv/bin/python export_onnx/export_planner.py --opset 20 \
    --out outputs/onnx/planner/planner_step.onnx

# vision + layers at the PLANNING token count                      # 6i, 6h  ~13 min
.venv/bin/python export_onnx/export_vlm.py --part vision --task planning
.venv/bin/python export_onnx/export_vlm_layers.py --task planning
```

### 2.3 Text generation (decode)

```bash
.venv/bin/python export_onnx/export_vlm_decode.py --prefill 64      # 6n  ~6 min
```

### 2.4 Diagnostics

```bash
.venv/bin/python export_onnx/export_submodules.py                   # 6d
.venv/bin/python export_onnx/inspect_onnx.py \
    --model outputs/onnx/perception/perception.onnx                 # 6b
```

`export_submodules.py` exports `vit_neck`, `adaptor` and `depth_net`
individually. It exists as a bisect tool: a piece that passes there but fails
inside the whole head tells you the fault is in the composition.

---

## 3. Run and validate - all commands

Both runners are split into two phases **and must stay split**: holding the
17 GiB fp32 model and a 20 GiB perception forward at once gets the process
OOM-killed on a 78 GiB machine. That failure is silent - no traceback.

### Perception

```bash
.venv/bin/python export_onnx/run_onnx_pipeline.py --phase run       # 7a  ~5 min
.venv/bin/python export_onnx/run_onnx_pipeline.py --phase compare   # 7b  ~3 min
```

### Planning

```bash
.venv/bin/python export_onnx/run_onnx_planner.py --phase run        # 7c  ~6 min
.venv/bin/python export_onnx/run_onnx_planner.py --phase compare    # 7d  ~4 min
```

`--phase run` executes the graphs and saves outputs to `outputs/onnx_run*/`;
`--phase compare` loads those and checks them against PyTorch. Useful flags:
`--frame` / `--scene` to target another sample, `--tol` to change the threshold.

### Runtime, so you know what normal looks like

| stage | time |
|---|---|
| vision tower | ~30 s |
| 32 decoder layers (33 ORT sessions) | ~190 s |
| perception head | ~80 s |
| planner, 10 Euler steps | ~26 s |
| **ONNX total** | **~300 s** |
| PyTorch reference (the compare half) | ~150 s |

---

## 4. How the graphs fit together

```
pixel_values ─► vision.onnx ──┬─► vit_tap ──────────────────────────┐
                              └─► merged_tokens                     │
                                      │ (host: embed gather,        │
                                      │  vision scatter, mRoPE)     │
                                      ▼                             ▼
                     layer_00..31.onnx + final_norm.onnx ─► hidden ─► perception.onnx
                                      └─► 8 x (keys, values) ─► planner_step.onnx x10
```

Four things stay on the host. Getting any wrong gives plausible but wrong output:

1. **Image preprocessing** - 896x512, snapped to the patch grid, pixels scaled to
   [0,1] and normalised with mean/std 0.5.
2. **The embedding scatter** - merged vision tokens replace image-token positions.
3. **mRoPE position ids** `[3, B, S]`. Images get a 2-D position grid, so this is
   **not** an arange - use the repo's `_rope_positions`.
4. **`_premerge_grids`' un-permutation** of the merger's 2x2 block ordering. Skip
   it and you get a spatially scrambled patch grid that still looks plausible.

The 10-step Euler loop also stays on the host, so the step count can change
without re-exporting - matching how `sample()` is written.

---

## 5. The obstacles, and what each really was

Short version; full detail in [study/09](study/09_ONNX_EXPORT.md).

| # | obstacle | fix |
|---|---|---|
| 1 | two custom CUDA kernels have no ONNX equivalent | reuse `training/differentiable.py` - what a tracer needs is what a gradient needs |
| 2 | `torch.inverse`, numpy `img_metas`, data-dependent masks | freeze the calibration geometry (verified **exact**, 0.000e+00) |
| 3 | a single 3.93 GiB tensor over protobuf's 2 GiB per-tensor cap | scatter one camera at a time (655 MiB each) |
| 4 | **`index_add` with duplicate indices** | `scatter_add` -> `ScatterElements(reduction='add')` |
| 5 | 5-D `GridSample` unsupported at opset 17 | opset 20 |
| 6 | planner: int64 inputs, GQA in SDPA | pass nav as float one-hot; expand KV heads by hand |

**Obstacle 4 is the one to remember.** Voxel pooling is *nothing but* duplicate
indices - summing many frustum points per voxel is the whole operation. The
export silently dropped the accumulation: it produced a graph that passed
`onnx.checker`, ran in onnxruntime, and returned **relative error 0.96**. One
substitution took it to 5.29e-04. It looked like a success.

---

## 6. Why there is no single .onnx file

The monolithic text prefill **does export** - `outputs/onnx/vlm_text/`, 341,977
nodes, 13.4 GiB. onnxruntime cannot build a session from it: over an hour, twice.

That is a scaling law, not a size limit. Merging per-layer graphs with
`onnx.compose` and timing session creation:

| layers merged | nodes | ORT load | **s per 1k nodes** |
|---|---|---|---|
| 1 | 14,161 | 3.8 s | 0.27 |
| 2 | 28,322 | 16.0 s | 0.57 |
| 4 | 56,644 | 84.3 s | **1.49** |

The per-node cost **doubles every time the graph doubles** - session creation is
quadratic. Extrapolated to 32 layers (453k nodes): 3.8 x 32^2 = **~65 min**,
exactly what the real monolith did. Merging also fails on its own at 6 layers,
inside `check_model`, on protobuf's 2 GiB limit.

So: **33 graphs load in ~2 minutes; one graph of the same size does not load at
all.** The quadratic term lives in the graph, not the model - it comes from the
Gated-DeltaNet chunk rule unrolling into 341,977 nodes. A custom ONNX operator
for that rule would shrink the graph ~100x and likely make a single file
practical. That is the one change that would flip this answer.

---

## 7. Limits worth knowing

* **Graphs are shape-frozen.** Perception (2744 tokens) and planning (3383-3386)
  need separate exports. A different token count will not load.
* **Graphs are calibration-frozen.** The perception head bakes in `lidar2img`.
  Running it on a frame from a differently-calibrated scene yields **NaN** - it
  fails loudly rather than silently, which is the good outcome. Re-export per rig.
* **The decode step is one step, not a generator.** It is exported and verified at
  a fixed past length of 64. A real loop needs the KV cache to grow each step,
  which a shape-frozen graph cannot do; that needs a max-length KV buffer with a
  position index and masking. Not built here.
* **RL (stage 4) is out of scope** by request.

---

## 8. Expected results

[study/10_EXPECTED_RESULTS.md](study/10_EXPECTED_RESULTS.md) and
`outputs/expected_results.json`. Regenerate with:

```bash
.venv/bin/python study/scripts/12_collect_results.py                 # 4c
```
