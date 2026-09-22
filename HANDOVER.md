# Handover: making Qwen-Drive fast in ONNX

State as of 2026-09-22, on 2x RTX 3090. Two separate efforts are in here:

1. **Optimising the export of the existing model** — finished, documented in
   [ONNX_EXPORT_V2.md](ONNX_EXPORT_V2.md). 29,761 ms -> 907 ms, then a hard floor.
2. **Distilling a compact replacement** — started, running, not finished. This is
   where the remaining speed has to come from, and it is what to pick up.

## Where the numbers stand

| | frame ms | Hz | notes |
|---|---|---|---|
| shipped ONNX export (perception only) | 29,761 | 0.03 | before any of this |
| v2 export, six cameras | 2,434 + planning | 0.30 | export defects fixed |
| v2 + front camera + pruned, planning side 0.5 | **907** | **1.10** | trajectory preserved |
| same, planning side 0.25 | 713 | 1.40 | endpoint moves 1.5 m -- not usable |
| PyTorch perception, front camera + pruned | **322** | **3.10** | validated over 6 frames |
| **student model, both heads, one graph** | **39.1** | **25.6** | untrained; shape only |

The target is 10 Hz for the whole thing -- detection *and* planning -- and on one card,
because that is what an Orin has. Pruning reached 1.10 Hz and stopped. The student
clears the speed target with 2.6x margin; whether it can match the teacher is open.

## Part 1: the export work (complete)

Read [ONNX_EXPORT_V2.md](ONNX_EXPORT_V2.md) for the detail. The short version of what
mattered, because several of these are non-obvious and cost days to find:

* **ONNX Runtime 1.22 has no CUDA `Resize` for opset 20.** Six nodes in the BEV head ran
  on the host, each dragging tensors off the device and back. 1.25.1 has the kernel:
  head 1,527 -> 892 -> 744 ms. 1.30 needs CUDA 13 and will not load against torch's
  CUDA 12; 1.23 still lacks it. **1.25.1 is the version to use.**
* **`GridSample` is the head's whole cost if it lands on the CPU.** 39 nodes, 5,418 ms,
  92% of the head. Both fixes already existed in this repo and had never been applied to
  the shipped artifacts: `export_perception.py::_gridsample_to_cuda_contrib` for the 4-D
  nodes, `gridsample5d_to_gather.py` for the two 5-D ones. Together 7.7x.
* **The Gated-DeltaNet chunk rule traces into 14,161 nodes**, and the node count was never
  the cost. What mattered was five `Pad` nodes with no CUDA kernel (39% of a layer), a
  depthwise `Conv1d` ORT hands to cuDNN at 15.5 ms for 0.18 GFLOP, and in-place indexed
  assignment inside the chunk loop. `gdn_onnx_v2.py` rewrites it: 67.3 -> 19.0 ms.
* **A bit-exact voxel crop.** With one camera the 16x200x200 voxel volume is 8.7%
  occupied, and the writable box is fixed by calibration. Cropping the 3-D convs to it is
  *exact* (verified 0.000e+00) and saves 64 ms. `local/prune/voxel_crop_v1.py`.
* **Only the projection GEMMs can go fp16.** The chunk decay is `exp(cumsum)` and
  overflows -- PyTorch in float16 returns a NaN trajectory too. `fp16_weights_v3.py`
  converts one GEMM at a time with explicit casts, which is what the two existing
  converters fail to do.

### Measured dead ends -- do not spend time re-trying these

fp16 on the head or vision tower (0.98-1.00x, and vision moves an output by 2.3e-01) ·
CUDA Graphs (0.99x, on both decoder and head; capture is refused outright while any CPU
node remains) · opset-18 retarget (changes the answer by 3.7e-01 and self-reverts) ·
coarsening the lift-splat depth grid (4.7 ms of 420) · dropping any voxel conv (destroys
detection, 0/36) · narrowing them (saturates at -72 ms, memory bound) · truncating the
DETR decoder (4 ms, even though module-path profiling attributes 234 ms to it -- that
attribution is misleading) · requesting a subset of ONNX outputs at run time (ORT
partitions the whole graph regardless: 696 vs 704 ms).

### Why the export still trails PyTorch

Stage for stage, ONNX against PyTorch bf16: vision 2.64x, decoder **1.09x**, head 2.18x.
The decoder -- the part that was rewritten -- is at parity. The rest is **precision**:
ONNX runs fp32 where PyTorch runs bfloat16. It is *not* the un-representable fused
kernels; PyTorch using the portable twins costs 1.24x in time and zero extra memory,
while the fp32 ONNX head costs 2.18x time and 2.07x memory, which is the element-size
factor of two. That is also why the export wants more than one card when PyTorch fits on
one. Blocker: ORT's CUDA EP has no fp16 `GridSample` and the head uses 39 of them.

## Part 2: the distillation (in progress -- pick up here)

### The idea

The student emits **the teacher's exact output format**: 900 queries x (7 class logits +
10 box params), plus a 50x3 trajectory. Query *i* of the student trains against query *i*
of the teacher, so distillation is a plain per-query regression -- no Hungarian matching,
no label assignment, **no ground truth needed for detection**. Every downstream consumer,
`get_bboxes` included, works on the student unchanged.

One network, two heads, shared backbone and BEV. Two separate students would each have
had to fit the whole frame budget alone; sharing gets planning for the cost of a 50-query
decoder, which is why the *whole model* clears 10 Hz rather than just perception.

Planning is supervised by **recorded nuScenes ego futures**, not by the teacher: for
trajectory the ground truth is the objective, and the teacher's own ADE against it is
0.335 m. This also avoids converting nuScenes into the 40-field WOD_E2E scene format the
planner consumes.

### Shape, and why it is this shape

`local/distill/student.py`, 81.8 M parameters, 39.1 ms, 25.6 Hz. Sized by measuring:

```
r34  d384 bev256 L6 blocks3    47.0 M   19.0 ms   52.7 Hz
r50  d384 bev256 L6 blocks3    49.8 M   20.2 ms   49.6 Hz
r50  d512 bev384 L8 blocks3    77.8 M   29.4 ms   34.0 Hz
r50  d512 bev384 L8 blocks6    81.8 M   39.1 ms   25.6 Hz   <- current default
r50  d512 bev512 L8 blocks8    92.3 M   63.6 ms   15.7 Hz
r101 d512 bev512 L8 blocks8   111.3 M   68.1 ms   14.7 Hz
```

The first shape tried was 3.56 M and 5.23 ms, which is far too small to absorb a 4 B
teacher -- BEVFormer-tiny is 33 M and BEVDet with a ResNet-50 is ~50 M. Running faster
than needed is lost accuracy, so the default sits near the top of the range. There is
room to go to ~111 M and still clear 14 Hz if the agreement numbers demand it.

Export-friendly by construction, which the teacher never was:
* attention written out by hand -- `nn.MultiheadAttention` dispatches to
  `aten::_native_multi_head_attention`, which has no ONNX export at any opset
* one `scatter_add` for the view transform (`index_add` does not trace)
* no 5-D ops, so no GridSample and no opset-20 lock-in
* ImageNet-pretrained ResNet backbone: with a few thousand frames, pretrained init
  matters more than parameter count
* BEV grid 200x200 at 0.512 m, matching the teacher, so localisation is not capped
* extra capacity goes into the BEV stack, not input resolution -- the teacher only ever
  saw 896x512, so a student given more pixels cannot use them to match it better

### Two bugs already found and fixed -- both would have quietly wasted a day

1. **Box targets are not on comparable scales.** Over asserted queries, x and y have std
   24.6 and 16.9 m while the other eight dims are 0.34-2.1. An unweighted L1 is 94%
   position and size/rotation/velocity go effectively unsupervised. The student now
   regresses a standardised target with a fixed affine restoring teacher units at the
   output (`box_stats.py`, `set_box_stats`), so the ONNX signature is unchanged.
2. **Ego extraction rebuilt each scene's pose track per sample** -- 34k samples x ~2k
   poses, about 68M quaternion conversions for data identical within a scene. Hoisted:
   that stage went from minutes-and-climbing to 41 s.

### How to run it

```bash
source export_onnx/env_gpu.sh                     # cuDNN/cuBLAS from torch's wheels
setsid nohup .venv/bin/python -u local/distill/orchestrate_10hz.py \
    > outputs/distill.log 2>&1 < /dev/null &
```

Cycles forever, every stage resumable and skipping finished work. Per cycle: index new
nuScenes samples -> extract ego futures -> refresh box stats -> cache teacher output
(**spawned on GPU 1, overlapping everything else**) -> train (GPU 0) -> export and time
-> evaluate against the teacher. State in `outputs/distill/state.json`; live logs in
`outputs/distill_train.log` and `outputs/distill_cache.log`.

Individual stages:

```bash
python local/distill/nusc_frames.py  --root data/nuscenes_trainval --version v1.0-trainval
python local/distill/nusc_ego.py     --root data/nuscenes_trainval --version v1.0-trainval
python local/distill/cache_teacher.py                     # ~0.51 fps, GPU
python local/distill/box_stats.py
python local/distill/train_student.py --steps 50000       # ~3.1 it/s at batch 4
python local/distill/export_student.py                    # exports and times
python local/distill/eval_student.py                      # agreement with the teacher
```

### Data

nuScenes trainval. On the machine this was built on: metadata complete, image blobs
downloading at ~67 MB/s via `scripts/download_nuscenes_trainval.sh`. Counts at handover:

| | |
|---|---|
| trainval keyframes | 34,149 |
| ego records (future fits in scene) | 25,599 |
| frames indexed (images present) | 7,068 and climbing with the download |
| teacher output cached | ~950 |

**25,599 is the ceiling** on training frames -- 8,550 keyframes sit too close to the end
of their scene to have a 5 s future. Caching all of them is ~14 h at 0.51 fps.

`data/` and `outputs/` are gitignored. On a new machine everything regenerates from
nuScenes plus `weights/`; nothing in the pipeline depends on the artifacts left behind
here. A symlink is fine: `ln -s /path/to/nuscenes data/nuscenes_trainval`.

### What is not done

* **No trained checkpoint.** Training restarted from scratch after the box-standardisation
  fix and had reached ~1,500 steps of a throwaway run. There are no agreement numbers yet,
  so **whether 81.8 M can follow a 4 B teacher is still unmeasured.** That is the first
  thing to find out, and `eval_student.py` reports it: detection recall, precision, centre
  distance against the teacher, and trajectory ADE against ground truth with the teacher's
  0.335 m printed beside it.
* **Student not yet compared against the real teacher pipeline end to end** -- only
  per-query agreement is wired up, not a run through `get_bboxes` and the nuScenes metrics.
* **No planning-side distillation from the teacher**, only ground-truth supervision. If the
  teacher's trajectories turn out better than the student can reach from GT alone, caching
  them means building the WOD_E2E scene conversion that was deliberately skipped.
* **The 1.10 Hz pruned export is the fallback** if distillation does not converge. It is
  reproducible today and its parity is verified.

### If the student does not converge

In order of what I would try: raise capacity (measured room to 111 M at 14.7 Hz) ·
train on more of the 25,599 frames before judging · add the teacher's soft class
distribution rather than only its asserted queries · supervise the BEV feature map
against the teacher's, not just the outputs · and only then reconsider the architecture.

## Repository map

| | |
|---|---|
| `ONNX_EXPORT_V2.md` | the export work, with every measurement and dead end |
| `export_onnx/gdn_onnx_v2.py` | the chunk rule, rewritten to trace into few large nodes |
| `export_onnx/gdn_fast_v2.py` | conditional padding, shifted depthwise conv, `pad_to` |
| `export_onnx/export_vlm_layers_v2.py` | per-layer export, `--skip-layers`, `--cams`, `--side` |
| `export_onnx/fp16_weights_v3.py` | fp16 projection weights, one GEMM at a time |
| `export_onnx/run_onnx_drive_v2.py` | resident sessions, device-to-device chaining, GPU placement |
| `export_onnx/orchestrate_fast_onnx.py` | the twelve-stage export pipeline, resumable |
| `export_onnx/check_parity_v2.py`, `compare_parity_v2.py` | parity against PyTorch |
| `local/prune/voxel_crop_v1.py` | the bit-exact lift-splat crop |
| `local/prune/*.py` | every pruning measurement behind the numbers above |
| `local/distill/student.py` | the compact joint model |
| `local/distill/orchestrate_10hz.py` | the distillation loop |
| `scripts/download_nuscenes_trainval.sh` | resumable trainval download |

## Environment

```
onnxruntime-gpu==1.25.1     # NOT 1.22 (no CUDA Resize at opset 20), NOT 1.30 (needs CUDA 13)
torch 2.x + CUDA 12, torchvision 0.23
source export_onnx/env_gpu.sh    # puts torch's bundled cuDNN/cuBLAS on LD_LIBRARY_PATH
```

`env_gpu.sh` also sets `$SP`, which collides with a shell variable of that name -- if a
script sources it and then uses `$SP`, rename the script's variable.
