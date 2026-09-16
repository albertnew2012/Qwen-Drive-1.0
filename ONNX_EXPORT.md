# Exporting Qwen-Drive-1.0 to ONNX

A runbook. Every command was run on this machine; the numbers are what it
printed. For *why* each obstacle exists, see
[study/09_ONNX_EXPORT.md](study/09_ONNX_EXPORT.md).

**Status: the whole model is exported and numerically validated against
PyTorch**, in 73 graphs totalling 54 GiB. Both driving pipelines - perception and
planning - run end to end from ONNX and match PyTorch.

Last verified end to end on **2026-09-16** by `bash export_onnx/run_export.sh`,
56 minutes, both pipelines PASS. The full printout is in section 9.1.

**There is no single .onnx file, and that is a runtime limit, not a choice.** See
section 6.

**One command for all of it:**

```bash
bash export_onnx/run_export.sh              # export + optimize + validate, ~55 min
bash export_onnx/run_export_parallel.sh     # same artifacts, ~30 min, needs ~130 GB RAM
STAGE=validate bash export_onnx/run_export.sh   # just the equivalence check
STAGE=bench    bash export_onnx/run_export.sh   # GPU timing, needs .venv-ortgpu
```

The parallel variant runs the six independent export steps concurrently in
RAM-sized waves (each exporter loads the 4.5 B VLM at ~31 GB, so RAM is the
limit, not cores). It deliberately does **not** shard the 33 layer graphs and
does **not** parallelise validation - the reasons are in its header.

Section 9 is the summary: what is done, what it costs, what is left.

---

## 0. Before anything

```bash
cd /path/to/Qwen-Drive-1.0
export PYTHONPATH=src:.
export PATH="$PWD/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""      # export and validation run on CPU in fp32
export OMP_NUM_THREADS=8
```

Dependencies (already installed): `onnx`, `onnxruntime`, `onnxscript`.

[export_onnx/run_export.sh](export_onnx/run_export.sh) sets all of this itself,
and caches the VLM feature taps first if they are missing.

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

Numbers below are from an earlier run and are kept as a second, independent
sample; the latest verified run is in section 9.1. Both agree to well within the
tolerance, which is itself the useful signal - the residual moves a little
between runs because ORT's threaded `ScatterElements` is nondeterministic.

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
[export_onnx/run_export.sh](export_onnx/run_export.sh) runs everything below,
plus the section 2.5 rewrites and the section 3 validation, in one go.

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

### 2.5 GPU-placement rewrites (run these after exporting)

Neither changes what a graph computes - both change **where** ORT runs it. ORT
silently places a node on the CPU when it has no CUDA kernel for that opset or
rank, and the device copies around it cost more than the node.

```bash
# 5-D GridSample has no CUDA kernel at any opset -> the BEV view transform ran
# on the CPU. Rewrite it as an 8-corner gather + trilinear weights.
.venv/bin/python export_onnx/gridsample5d_to_gather.py \
    outputs/onnx/perception/perception.onnx

# Pad and Resize have CUDA kernels only up to opset 18; declaring 20 stranded
# them on the CPU. --verify checks every rewritten graph is bit-identical.
.venv/bin/python export_onnx/retarget_opset.py \
    outputs/onnx/perception/perception.onnx \
    outputs/onnx/vlm_layers_v2 \
    outputs/onnx/vlm_layers_plan_v2 \
    --opset 18 --verify
```

Perception head **13,111 -> 1,106 ms**, decoder **3,944 -> 2,605 ms**. Both
scripts take `--revert` and keep backups.

**Re-exporting a graph undoes these.** If you rerun `export_perception.py`, the
5-D `GridSample` comes back and the head returns to the CPU - rerun 2.5 after.

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

---

## 9. Status: what is done, what it costs, what is left

### 9.1 Is ONNX equivalent to PyTorch? Yes.

Verbatim from a clean `bash export_onnx/run_export.sh` on **2026-09-16** - every
graph re-exported from the checkpoint, rewritten, and validated in one 56-minute
run. This is the output to compare against if you ever doubt an export.

```
  tensor                  relative diff   verdict
  VLM hidden_states           2.055e-04   ok
  VLM vit tap                 2.819e-05   ok
  VLM llm tap                 3.331e-04   ok
  planner KV keys (x8)        2.187e-04   ok
  planner KV values (x8)      8.779e-05   ok
  all_cls_scores              9.272e-04   ok
  all_bbox_preds              8.646e-04   ok
  occ_pred                    1.787e-06   ok
  seg_preds                   3.337e-06   ok
  PIPELINE PASS  (worst 9.27e-04, tolerance 5e-03)
```

```
  trajectory (50, 3)   PyTorch endpoint [ 6.907 -0.814 -0.323]
                       ONNX    endpoint [ 6.907 -0.814 -0.323]
  max |diff| per axis (metres): x 0.0000  y 0.0000  heading 0.0000
  ADE(onnx, pytorch) 0.00002 m      FDE 0.00005 m
  PLANNER PASS  (ADE 1.81e-05 m, tolerance 5e-02)
```

The planner endpoint agrees to every printed decimal and the per-axis maximum
difference rounds to zero at 0.1 mm. Both pipelines pass with roughly 5x and
2700x margin against their tolerances.

Per graph, against its PyTorch module:

| graph | params | nodes | rel. diff |
|---|---|---|---|
| VLM vision tower | 0.3335 B | 3,747 | 7.1e-05 |
| VLM text prefill x33 (perception) | 4.2058 B | 344,127 | 1.4e-06 |
| VLM text prefill x33 (planning) | 4.2058 B | 366,983 | ~1e-06 |
| VLM decode step | 4.2058 B | 10,233 | 2.0e-06 |
| BEV perception head | 0.1251 B | 12,436 | 5.3e-04 |
| planning expert (one step) | 1.0398 B | 9,315 | 4.2e-07 |

The head is three orders worse than the planner, and that is expected rather than
a defect: the planner is a transformer evaluated once, while the head accumulates
~668,000 scattered contributions per frame in a different order than PyTorch, and
fp32 addition is not associative.

### 9.1b What the run itself cost

| stage | measured |
|---|---|
| export, 6 steps | ~31 min |
| optimize, GridSample + opset retarget with `--verify` | ~5 min |
| perception `--phase run` | 408 s |
| perception `--phase compare` | 216 s |
| planning `--phase run` | 451 s |
| planning `--phase compare` | 120 s |
| **total wall clock** | **~56 min** |

Inside the perception run, on CPU:

```
  frame 90162f90eceb4ada  6 cameras  2744 tokens
         20.9s   vit_tap (10752, 1024)
         8/32    67.8s
        16/32   139.0s
        24/32   213.3s
        32/32   287.1s
        hidden_states (1, 2744, 2560)   287.2s
         77.6s   cls (6, 1, 900, 7)
  ONNX pipeline total 388s
```

The PyTorch halves of the comparison were 79 s (VLM) + 37 s (head) and 92 s
(planner prefill + 10 Euler steps).

### 9.2 What it costs, against PyTorch

**CPU, fp32 - the configuration the equivalence check runs in:**

| | time |
|---|---|
| ONNX, whole perception pipeline | ~300 s |
| PyTorch reference, same frame | ~150 s |

**GPU (RTX 3090, 24 GiB), after the optimization pass in
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md):**

| path | first export | optimized | PyTorch |
|---|---|---|---|
| perception, frame wall time | 73,808 ms | **22,159 ms** | - |
| trajectory, frame wall time | ~75,500 ms | **11,695 ms** | - |
| both | ~149 s | **33.9 s** | **3.83 s** |

GPU compute alone, excluding session construction, is **7,308 ms** of that 33.9 s.

**PyTorch is still 8.8x faster end to end, and that is the honest headline.**
ORT cannot keep a 17 GB fp32 decoder resident on a 24 GiB card, so it rebuilds
layer sessions every frame; that rebuild, not arithmetic, is the dominant cost.

### 9.3 What is NOT done

* **`vlm_decode` is not exported by `run_export.sh`.** It is the autoregressive
  chain-of-thought path - ~15 GB of artifacts that perception and trajectory
  never read. The exporter still works: `export_vlm_decode.py --prefill 64`.
* **`planner-rl` is not exported.** `planner-sft` is what both runners load.
* **RL (stage 4) is out of scope** and is not implemented anywhere in the repo.
* **No single `.onnx` file.** Section 6 - ORT session creation is quadratic in
  node count, so the 341,977-node monolith never loads.
* **Shapes are frozen.** The planner is exported at one scene's KV length (3385
  tokens for the bundled demo scene). A different camera rig re-exports: a
  nuScenes frame is 4364 tokens and will be rejected with "invalid dimensions
  for scene_v_7".
* **fp16 is not usable for the decoder.** It converts and loads, but the
  Gated-DeltaNet recurrence overflows: perception hidden 8.74e-01 relative,
  trajectory **NaN**. fp16 is used only on the trajectory path where it was
  measured against the fp32 reference (ADE 0.0236 m).

### 9.4 What to optimize next

In priority order, from [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) section 7:

1. **Chunk the deformable attention over the camera axis.** Of perception's
   22.2 s, only 3.9 s is GPU compute; the other 18.3 s is rebuilding 32 layer
   sessions because the head cannot coexist with a resident decoder. One
   allocation drives it - sampled values `[48, 32, 10046, 32]` = 1.975 GB.
   Chunking caps it at ~330 MB and would let the decoder stay resident.
2. **A fused Gated-DeltaNet operator.** The chunk rule unrolls into 4,720 nodes
   per layer even after the blocked triangular inverse. An ONNX `Loop`, or a
   custom op with a CUDA kernel, would shrink the graph ~40x and cut session
   build time with it. Note this is *not* a kernel-launch win - CUDA Graphs
   measured 1.00x, so the decoder is not launch-bound.
3. **More VRAM.** At ~21 GiB usable, a 17 GB fp32 decoder plus a ~13 GiB head
   cannot coexist. A second card or a larger one removes the whole problem.

Section 5 of the optimization report lists eleven approaches that were tried and
rejected **with numbers**, including fp16 for the head, view-transform fusion,
Einsum fusion and CUDA Graphs. Read it before re-attempting any of them.

### 9.5 Exporting faster: `run_export_parallel.sh`

`run_export.sh` is the reference path - sequential, and the one whose output
produced the 9.1 printout. `run_export_parallel.sh` produces the same artifacts
with two levels of parallelism.

**Across steps.** The six export steps read nothing from each other, so they run
in RAM-sized waves. RAM is the limit, not cores: each exporter loads the 4.5 B
VLM at ~31 GB.

**Inside a layer export.** `export_vlm_layers.py --jobs N` forks N workers that
each trace a disjoint subset of the 32 decoder layers. Forking rather than a
process pool is deliberate: the per-layer closure is not picklable, and children
inherit the weights copy-on-write, so N workers cost ~31 GB in total instead of
~31 GB each. Measured: four children showed 22.9 GB RSS apiece against a 31.7 GB
parent - a naive sum of 123 GB - while `MemAvailable` fell by only 19 GB.

Measured on four layers, four workers, back to back on an otherwise idle machine:

| | `--jobs 1` | `--jobs 4` |
|---|---|---|
| layer export phase | 92 s | **27 s** (3.4x) |
| total wall clock | 314 s | **222 s** (1.41x) |

The layer phase is what scales; the rest is a fixed ~200 s prologue - model load,
the forward pass that threads the hidden state through all 32 layers to give each
export its real tracing input, and the 2.37 GiB `embed_tokens.npy` write. Over
the real 32 layers that prologue is paid once against 8x more work, so the whole
step goes ~15.6 min -> ~6.9 min, **2.25x**.

**It is byte-identical.** Exporting the same layers both ways and comparing every
file: all five graphs identical by MD5, and `manifest.json` differed in exactly
three fields - `rel` values, at the 1e-7 level. Those come from the verification
run, where onnxruntime's reduction order depends on thread count. Node counts and
every structural key matched. That the graphs are identical at all is the same
property that makes sharding sound: ONNX topology depends on input *shape*, not
values, which is why all 24 linear layers trace to 4,720 nodes and all 8 full
layers to 538.

**The hazard, if you touch this code.** `fork()` duplicates only the calling
thread. A child that inherits a futex held by an OpenMP worker which does not
exist on its side blocks forever. With `OMP_NUM_THREADS=8` this reproduced every
time: four children at 0% CPU and `00:00:00` cpu time, all in
`futex_wait_queue_me`, indefinitely - and Python warns about it
(`DeprecationWarning: ... multi-threaded, use of fork() may lead to deadlocks`).
Setting `OMP_NUM_THREADS=1` avoids it but makes the whole prologue
single-threaded. The fix in the code is narrower: `torch.set_num_threads(1)`
immediately before the fork loop and restore after, so the prologue keeps all
eight threads and only the fork point is quiesced.

**Benchmark honestly.** The first measurement of this said parallel was 2x
*slower*. It was comparing against a baseline taken earlier under a cleaner page
cache; re-running `--jobs 1` under the same conditions as `--jobs 4` moved it
from 119 s to 314 s. Repeated 4 GiB writes to the same scratch directory shift
the result more than the change being measured. Always run the A and the B back
to back.

