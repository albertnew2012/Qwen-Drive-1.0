# Qwen-Drive-1.0 — ONNX GPU runtime optimization report

**Goal.** Perception (3D detection / occupancy / segmentation) and trajectory only, in ONNX, on
GPU, as fast as possible. Chain-of-thought and everything else not needed for those two outputs was
dropped.

**Hardware.** 1× RTX 3090 (24 GiB, ~3 GiB held by the desktop → ~21 GiB usable),
onnxruntime-gpu 1.22.0, CUDA 12.8.

---

## 1. Headline result

| path | before | after | speedup |
|---|---|---|---|
| perception (frame wall time) | 73,808 ms | **22,159 ms** | **3.33×** |
| trajectory (frame wall time) | ~75,500 ms (est.) | **11,695 ms** | **~6.5×** |
| both | ~149 s | **33.9 s** | **~4.4×** |

(First run of a configuration is ~25.5 s for perception because it populates the graph cache;
22.2 s is the steady state.)

GPU compute alone, excluding the session-construction overhead discussed in §2:

| stage | original export | now |
|---|---|---|
| perception vision | 622.5 | 550.0 |
| perception host glue | 7.2 | 6.7 |
| perception decoder | 3,943.9 | 2,203.2 |
| perception head | 13,111.5 | **1,109.4** |
| trajectory vision | 602.6 | 293.2 |
| trajectory host glue | 7.7 | 7.7 |
| trajectory decoder | 4,710.6 | 2,690.3 |
| trajectory planner | 466.4 | 448.2 |
| **total** | **23,472 ms** | **7,308 ms (3.21×)** |

**Accuracy is unchanged.** Perception hidden state `7.04e-02` relative vs the fp32 CPU reference —
identical to the pre-optimization baseline. Trajectory ADE **0.0236 m** vs the fp32 CPU reference on
a ~6.9 m trajectory.

**Honest caveat: PyTorch is still faster.** The same scene in PyTorch is 3,831 ms (0.26 FPS). ONNX
Runtime cannot keep a 17 GB fp32 decoder resident on a 24 GiB card, and rebuilding sessions per
frame is the dominant cost (§2). You asked for ONNX specifically, so that is what was optimized, but
if raw FPS is the only criterion, the PyTorch path wins today.

---

## 2. The finding that mattered most

Every timing previously reported was a **sum of instrumented stages**, and it excluded roughly
**70 seconds per frame**. From `outputs/onnx_bench/latency_fp32_cuda.json`:

```
perception: ms=73808.6   stages={vision 557.7, host 6.6, decoder 2373.7, head 1020.1}  (sum 3958)
```

The 69,850 ms gap is `onnxruntime.InferenceSession` construction inside the decode loop, which was
never timed. The decoder is 32 separate graphs totalling 17 GB in fp32; they do not fit alongside
the perception head, so they were being **rebuilt from scratch on every frame**.

This reframed the whole problem: the bottleneck was never the kernels, it was graph loading.

---

## 3. What was done

### 3.1 Pre-optimized graph caching — the single biggest win
`SessionOptions.optimized_model_filepath` dumps the graph *after* ORT's optimization passes.
Reloading that with `ORT_DISABLE_ALL` produces identical kernels:

```
build from original, ENABLE_ALL   3.93 s   infer 84.5 ms
reload pre-optimized, DISABLE_ALL 0.69 s   infer 84.5 ms
```

Every per-frame layer rebuild now goes through this cache
(`make_streamed()` in [export_onnx/run_onnx_drive.py](export_onnx/run_onnx_drive.py)).
Perception: **73,808 → 22,159 ms**.

The cache key includes the parent directory — the fp32 and fp16 exports share the basename
`vlm_layers_v2`, and keying on the basename alone silently loaded one precision's weights into the
other. That bug was caught and fixed before it produced any reported number.

### 3.2 Partial residency (`--resident N` / `--resident-plan N`)
Keep as many layers permanently loaded as fit next to the head, stream the rest. The trajectory path
has no perception head competing for VRAM, so it holds 20 of 32 layers resident:
**27,524 → 11,695 ms**.

### 3.3 Perception head: 5-D GridSample → gather-based trilinear interpolation
`GridSample` has no 5-D CUDA kernel at any opset, so the BEV view transform was running on the CPU
and dragging `MemcpyFromHost` with it. Replaced with explicit 8-corner gather + trilinear weights.
Head: **13,111 → 1,106 ms, 100% CUDA.** Bit-identical on CPU-vs-CPU; on GPU the difference is within
the measured run-to-run noise floor (ORT's threaded `ScatterElements` is nondeterministic and swings
~9e-3 by itself).

### 3.4 Opset-18 retarget
`Pad` and `Resize` have CUDA kernels registered only up to opset 18; declaring opset 20 silently
moved them to the CPU. Retargeted to 18 (with a `Gelu` downgrade), verified **66/66 layer graphs
bit-identical**. Decoder: 3,944 → 2,605 ms.

### 3.5 Blocked triangular inverse in the Gated-DeltaNet chunk rule
The export unrolled a 63-step sequential triangular inverse. Replaced with a blocked inverse.
Decoder: 2,605 → 2,384 ms, 1.5e-4 relative.

### 3.6 4-D GridSample → `com.microsoft` domain
Opset 20 renamed the mode `bilinear` → `linear`; ORT's CUDA kernel only accepts the old spelling, so
it fell back to CPU. Pinned to `domain="com.microsoft"` with `mode=b"bilinear"`.

### 3.7 Dropped everything CoT-related
`vlm_decode` (the autoregressive text path) is not exported at all — about **15.4 GB** of ONNX
artifacts and the entire token-generation loop removed.

---

## 4. Precision: what is and is not safe

Weights-only fp16 (cast only initializer inputs of MatMul/Gemm) plus
`optimization.disable_specified_optimizers=ConstantFolding` gets a layer from 549 to 237 MiB:

```
fp32                      549.0 MiB/layer => 17.16 GiB
fp16 full                 234.5           =>  7.33 GiB   accuracy broken
fp16 weights (CF on)      499.7           => 15.61 GiB   ORT refolds fp32 weights
fp16 weights, CF off      236.8           =>  7.40 GiB   6e-4 per layer
```

This makes the decoder resident and perception runs in **10,667 ms** — the fastest number measured
in this effort. **It is not the recommended configuration**: the error compounds over 32 layers to
`2.59e-01` on the hidden state, and the trajectory comes out **NaN**. It is reported here for
completeness, not as a deliverable.

Isolating that NaN one component at a time showed the fp16 **flow-matching planner** was the actual
culprit, not the decoder — worth knowing if fp16 is revisited.

Full-graph fp16 on the decoder fails for an intrinsic reason: the Gated-DeltaNet recurrence computes
`exp(cumsum(log a))`, which underflows in fp16 (rel err 0.43–0.74 per layer). Blocking individual
ops from conversion changes nothing — it is a dynamic-range failure, not a per-op precision one. A
control run proved this is inherent to the model, not to my rewrites: the *original* export shows
the identical `4.964e-01` error.

---

## 5. Tried and rejected (with numbers, so they are not retried)

- **fp16 perception head** — 12,442 ms, 90% CPU. `com.microsoft.GridSample` has no fp16 CUDA kernel;
  blocking it from conversion did not restore CUDA placement.
- **View-transform camera-sum fusion** — mathematically correct (within noise) and cut peak memory
  by 1.4 GB, but made the head *fail to allocate*. The BFC arena had been reusing the freed 3.93 GB
  block to serve a later 1.975 GB request; removing the big allocation removed the big free chunk.
  Lower peak memory is not the same as more allocatable memory. Kept in
  [export_onnx/fuse_view_transform.py](export_onnx/fuse_view_transform.py), not enabled.
- **Einsum for the deformable-attention weighted sum** — correct and slightly faster, but ORT's CUDA
  Einsum transposes internally and used ~2 GB *more*. Behind `--fuse-deformable`, off by default.
- **CUDA Graphs** — 1.00×, not launch-bound.
- **Shared VLM prefill between the two paths** — different token layouts.
- **Doubling-identity triangular inverse** — NaN.
- **chunk 128/256/512** — slower than 8.
- **Depthwise conv1d as shift+Mul+Add** — 7× slower than cuDNN.
- **`cudnn_conv_algo_search=EXHAUSTIVE`** — identical to HEURISTIC.
- **Private arena for the head / reverting the head rewrite** — no effect; the original head uses
  *more* memory (15,207 vs 13,540 MiB).

Two reported "hotspots" were **profiler artifacts**. ORT's per-node CUDA `kernel_time` claimed
`Gelu` = 173.6 ms and a depthwise `Conv` = 16.7 ms. Measured device-resident with `io_binding`:
0.12 ms and 0.46 ms, both at the memory-bandwidth floor. ORT also emits
`"OP Conv() running in Fallback mode. May be extremely slow"` for that conv, which is misleading.
An early version of my own microbenchmark passed numpy arrays to `run()` and was measuring PCIe
transfer, not the kernel — caught and corrected.

---

## 6. How to reproduce the best numbers

```bash
cd /home/zhengzhiliu/Desktop/Qwen-Drive-1.0
unset CUDA_VISIBLE_DEVICES
NV=$PWD/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusparse/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=src:. TOKENIZERS_PARALLELISM=false

# perception — 22,159 ms warm, hidden rel 7.04e-02
.venv-ortgpu/bin/python -u export_onnx/run_onnx_drive.py --phase run \
  --precision fp32 --layers outputs/onnx/vlm_layers_v2 \
  --head-dir outputs/onnx/perception --chunk 8 --skip-planning --reps 3

# trajectory — 11,695 ms, ADE 0.0236 m
.venv-ortgpu/bin/python -u export_onnx/run_onnx_drive.py --phase run \
  --precision fp16 --layers-plan outputs/onnx/vlm_layers_plan_v2 \
  --planner-dir outputs/onnx/planner --chunk 0 --resident-plan 20 \
  --skip-perception --reps 3
```

The first run of each populates `outputs/onnx_preopt/`; later runs are the fast ones.

---

## 7. Where the remaining time goes, and what I would do next

Of perception's 22.2 s, only 3.9 s is GPU compute — the other 18.3 s is still rebuilding 32 layer
sessions at ~0.57 s each. The fix is to stop rebuilding, which requires the perception head to fit
alongside a resident decoder. The head's peak is driven by one allocation pattern in the BEV
encoder: sampled values `[48, 32, 10046, 32]` = 1.975 GB, multiplied by the attention weights and
immediately reduced. Chunking the deformable attention over the camera axis would cap that at ~330
MB and is, in my judgement, the highest-value remaining change. Two cheaper approaches (Einsum
fusion, view-transform fusion) were tried and both backfired for the arena reasons above, so this
one needs to be done properly rather than as a graph-level peephole.

Failing that, the structural answer is a second GPU or a card with more VRAM — at ~21 GiB usable,
a 17 GB fp32 decoder plus a ~13 GiB head simply cannot coexist.
