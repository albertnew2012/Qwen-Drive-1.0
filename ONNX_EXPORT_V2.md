# ONNX export, second pass

`ONNX_EXPORT.md` describes the first export: the model reaches ONNX, runs, and is
correct. This is what happened when it was profiled on this machine (2x RTX 3090,
onnxruntime-gpu 1.22, CUDA EP) and made fast.

The short version: almost none of the time was where it was assumed to be. It was
not launch overhead, not node count, and not ONNX. It was ops with no CUDA kernel
silently running on the host, and one recurrence written for eager PyTorch that
traced into thousands of tiny serialised kernels.

## Result

| | shipped | v2 | |
|---|---|---|---|
| perception, end to end | 29,761 ms | **2,434 ms** | 12.2x |
| &nbsp;&nbsp;vision tower | | 481 ms | |
| &nbsp;&nbsp;decoder, 32 layers | ~3,420 ms | **757 ms** | 4.5x |
| &nbsp;&nbsp;BEV head | 8,702 ms | **1,185 ms** | 7.3x |
| planning, end to end | not measured | **1,822 ms** | |
| &nbsp;&nbsp;decoder, 32 layers | | 880 ms | |
| &nbsp;&nbsp;Planning Expert, 10 Euler steps | | 392 ms | |

Against PyTorch bfloat16 on the same machine, stage for stage on the same frame
(`local/bench_drive_torch.py` for the whole-pipeline figure, and a per-stage
timing of the same modules for the breakdown):

| perception stage | PyTorch bf16 | ONNX v2 | ONNX/torch |
|---|---|---|---|
| vision tower | 181.9 ms | 480.4 ms | 2.64x |
| decoder, 32 layers | 692.6 ms | 755.4 ms | **1.09x** |
| BEV head | 428.6 ms | 1,188.5 ms | 2.77x |
| total | 1,303.2 ms | 2,424.3 ms | 1.86x |

Whole frame, both tasks: PyTorch 3,196 ms against ONNX 4,256 ms.

**ONNX has gone from ~12x slower than PyTorch to ~1.3x slower overall. It has not
overtaken PyTorch.** The breakdown says exactly where the remainder sits:

*The decoder, which is what this pass rewrote, is at parity* -- 1.09x, from 4.5x
slower. There is nothing material left there.

*The whole remaining gap is precision.* PyTorch runs bfloat16 throughout. The
export runs fp32 everywhere except the decoder's projection weights -- and the
decoder is exactly the stage at parity. Measured on the head, which is the largest:

| BEV head | time | peak activations |
|---|---|---|
| PyTorch bf16, fused CUDA kernels | 437.6 ms | 5.60 GiB |
| PyTorch bf16, portable twins (what the export traces) | 544.3 ms | 5.59 GiB |
| ONNX, fp32 | 1,188.5 ms | ~11.6 GiB |

Two things to read off this.

**It is not the missing fused kernels.** PyTorch's head uses two hand-written
kernels -- `ops._VoxelPoolDepthCuda` and `ms_deform_attn_bf16_forward` -- that
cannot be represented in ONNX, so `export_perception.py` calls
`enable_training_ops()` and the export traces the portable twins (`index_add_`
becomes ScatterElements, deformable attention becomes `grid_sample`). It is
tempting to blame that, and `training/differentiable.py` does warn that "the torch
twins are slower and use more memory than the kernels". Measured, the twins cost
1.24x in time and *nothing* in memory. They are not the explanation.

**It is fp32 against bfloat16.** 2.18x the time and 2.07x the memory is the factor
of two you would predict from the element size alone, and the vision tower's 2.64x
is the same story on a graph with no CPU fallback left in it.

This is also the whole answer to why the export needs more than one card when
PyTorch does not. PyTorch's entire pipeline peaks at 18.1 GiB in bfloat16. The ONNX
head alone holds ~11.6 GiB of fp32 activations where PyTorch holds 5.6 GiB of
bf16 ones, and that surplus is what will not fit alongside a resident decoder.

The head cannot simply be converted: ORT's CUDA EP registers no fp16 GridSample
(see `to_fp16.py`), and the head is built on 39 of them. `fp16_weights_v3.py` does
not help here either -- it converts weight tensors, and the head's cost is
activations. The tractable version is a mixed conversion that runs the head in
fp16 with fp32 islands around each GridSample, which would recover most of both
the time and the memory; the alternative is the TensorRT EP, which does have fp16
GridSample, and which is on the path to Orin anyway.

## The decoder

The language model is Qwen3.5-text: 32 layers, 24 of them Gated-DeltaNet linear
attention, 8 full attention every fourth. A single linear-attention layer at 2752
tokens, fp32, measured with IOBinding and warm:

| | ms | nodes |
|---|---|---|
| shipped export | 125.96 | 14,161 |
| + blocked triangular inverse | 78.97 | 3,952 |
| + `gdn_onnx_v2` | 30.59 | 2,121 |
| + fp16 projection weights | 23.57 | 2,121 |

For comparison the full-attention layer, of near-identical FLOP count, is 27.4 ms
and 538 nodes. The linear-attention layer had been 4.6x more expensive than the
attention layer it exists to be cheaper than.

### What the node count was, and was not

`torch_chunk_gated_delta_rule` inverts a unit lower-triangular matrix by forward
substitution — `for i in range(1, chunk_size)`, 63 sliced in-place assignments.
Traced, that is ~12,600 of the layer's 14,161 nodes.
`export_vlm_layers.patch_chunk_rule_blocked` already replaced it with a blocked
inversion and took the layer to 3,952 nodes.

It did not get faster. 1,972 nodes after ORT's folding ran in 130.8 ms; 445 nodes
(at chunk 256) ran in 133.3 ms. A captured CUDA graph replayed in 1.01x the time,
so the GPU really was busy that long. **Node count was never the cost.** That
matches `ONNX_EXPORT.md` §9.4 and contradicts `local/bench_drive_torch.py`'s
docstring, which attributes the slowness to launch overhead.

### What the cost actually was

Profiling the built session, per layer:

```
op                ms/run     %    n   provider
Mul                18.56  15.3   60   CUDA
MatMul             17.27  14.3   58   CUDA     <- the only real compute
Pad                16.42  13.6    5   CPU  (!)
Conv               15.46  12.8    1   CUDA
MemcpyFromHost     14.85  12.3    5
MemcpyToHost       14.46  11.9    5
Split              14.07  11.6    7   CUDA
```

Three things, and only the third is about the recurrence's mathematics:

**1. Five `Pad` nodes per layer on the CPU.** ORT has no CUDA kernel for the pad
configuration the chunk rule emits, so each one drags a 45 MiB activation to the
host and back: 17.4 ms of pad plus 29.3 ms of memcpy, 39% of the layer. The pads
exist only to round the sequence up to a multiple of the chunk size, so the fix is
to hand the graph a sequence that is already a multiple. Both the delta rule and
the full-attention layers are causal, so appended rows cannot affect real ones;
the host appends zeros and slices them off. `manifest.json` records `sequence` and
`padded`, and the export checks the padded result against the unpadded stock model
before writing anything (measured 3.7e-07).

**2. A depthwise `Conv1d` with `groups == in_channels == 8192`,** 0.18 GFLOP, that
ORT hands to a cuDNN grouped convolution costing 15.5 ms. Written out as its four
shifted multiply-accumulates it is ordinary elementwise work. Layout matters: a
`Mul` broadcasting over the *last* axis hits ORT's vectorised path, while
broadcasting over the middle axis fell back to a strided kernel at 8.7 ms for a
90 MiB tensor, so the replacement transposes to `(B, T, C)` first.

**3. The chunk loop.** Splitting the layer in two settled where the time was:

```
mlp            12.1 ms   390 GFLOP   32   TFLOPS   <- optimal, nothing to fix
linear_attn    67.3 ms   257 GFLOP    3.9 TFLOPS   <- 3,911 nodes
```

The recurrence spent ~55 ms on 25 GFLOP. `core_attn_out[:, :, i] = ...` emits a
ScatterND per chunk; `query[:, :, i]` and friends emit a Gather per tensor per
chunk, 43 x 7 of them; `g.exp()` is recomputed inside the loop.
`export_onnx/gdn_onnx_v2.py` collects the chunks in a list and stacks once, uses
`unbind(2)` for one Split per tensor, and hoists the exponentials. 67.3 -> 19.0 ms.

Chunk size was swept: 64 (3,952 nodes, 79.0 ms), 128 (2,234, 82.7), 256 (1,338,
89.0). 64 wins — the blocked inverse costs `2*T*C^2*log2(C)`, which grows faster
than the node count falls.

### Precision

A whole-graph fp16 conversion is not usable: the chunk decay is an `exp` of a
cumulative sum and fp16 has no range for it — 0.41 relative on a layer output.
PyTorch agrees; running the model in float16 returns a NaN trajectory.

`mixed_precision.py` has the right idea, converting only the heavy GEMMs, but does
not shrink anything: torch exports a projection as `initializer -> Transpose ->
MatMul`, the Transpose is not a GEMM so it lands in the block list, the weight
stays fp32, and the pass buys a per-frame Cast of 451 MiB. Measured 78.97 ms
before, 78.10 ms after. Moving the Transpose into the conversion set instead makes
the converter emit a MatMul with one fp16 and one fp32 input, which will not load.

`export_onnx/fp16_weights_v3.py` does the rewrite directly: for each MatMul/Gemm
with a weight of at least 2^20 elements, the weight becomes fp16, the activation
gets a Cast in and the result a Cast back. The casts are on 28 MiB activations,
not 451 MiB weights. 1.34x, and the decoder drops from 13.7 to 7.0 GiB — which is
also what makes the stack fit at all.

The same pass was tried on the vision tower (1.16x but moves an output by 2.3e-01)
and the BEV head (1.00x). Both rejected; they stay fp32.

## The BEV head

39 `GridSample` nodes, 5,418 ms, **every one on the CPU**, plus 1,915 ms of the
host round-trips they force: 92% of the head. ORT registers a CUDA GridSample only
in the `com.microsoft` domain and only for the opset-16 spelling `bilinear`; opset
20 renamed that mode to `linear`, so a stock export runs them all on the host.

Both fixes already existed in this repo and had simply never been applied to the
shipped artifact:

* `export_perception.py::_gridsample_to_cuda_contrib` moves the 37 four-dimensional
  nodes to `com.microsoft`. 8,535 -> 1,699 ms.
* `gridsample5d_to_gather.py` rewrites the 2 five-dimensional nodes, which have no
  CUDA kernel in any domain, as gather-based trilinear sampling. 1,699 -> 1,136 ms.

Together 7.7x. What is left is one `Resize` still on the CPU (73 ms);
`retarget_opset.py --opset 18` would move it but changes the answer by 6.6e-01 on
this graph and correctly reverts itself.

## Numerical equivalence

Per layer, each exported graph is checked against stock PyTorch at export time:
3.5e-07 to 4.0e-07 relative, and the padded sequence reproduces the unpadded
result to the same figure.

End to end, the decoder was run in float32 PyTorch on exactly the input the ONNX
runner uses, so the comparison isolates the export from everything around it:

| decoder output, vs float32 PyTorch | rel_max | rel_rms |
|---|---|---|
| ONNX v2, fp32 graphs | 1.11e-03 | 1.00e-04 |
| ONNX v2, fp16 projection weights | 2.11e-03 | 1.40e-04 |
| PyTorch bfloat16, the shipped path | 7.40e-01 | 1.48e-02 |

**The export is two orders of magnitude nearer exact arithmetic than the bfloat16
path the model ships.** This matters for reading the other direction: compared
against bfloat16 PyTorch the head's per-query argmax label agrees on only 79% of
900 queries, which looks alarming until one notices that the fp32 and fp16-weight
exports disagree with bfloat16 by *identical* amounts (1.483e-02 and 1.482e-02).
That number measures how far bfloat16 moves this head's logits, not export error.
Against each other the two exports agree on 99.44% of queries.

The planned trajectory, against bfloat16 PyTorch: **ADE 0.043 m, FDE 0.062 m**,
worst waypoint 0.088 m, heading within 0.021 rad over 50 waypoints.

## Why not 3 Hz

The workload is two full prefills of a 3.2B-parameter (non-embedding) decoder:
2,744 tokens for perception over 6 cameras, 3,385 for planning over 3 views and
4 timesteps. 98% of both prompts are image tokens, so there is no prompt slack —
truncating everything after the last image token saves 0.4%.

That is ~47 TFLOP per frame. 3 Hz is 333 ms, so 141 TFLOPS sustained. One RTX 3090
peaks at 71 TFLOPS fp16 dense; two peak at 142. **3 Hz requires ~100% of both
cards' theoretical peak**, which no real kernel reaches. The measured ceiling for
an excellent implementation here is roughly 1 Hz per card.

Reaching 3 Hz needs less work per frame, not a better export. In rough order of
cost to accuracy: shorten the planning history from 4 timesteps to 2 (~35% off
planning); drop from 6 cameras to 3 for perception; reduce the vision token budget
per camera. Each changes what the model sees, so each is a product decision rather
than an engineering one.

## Using both cards

There is no placement that keeps everything resident on 2x24 GiB, because the BEV
head needs **15.4 GiB on its own** (measured: 3.4 GiB of session, then a 3.9 GiB
and a 625 MiB buffer in the view transform). With the decoder at 7.0 GiB and a
vision tower at ~3.3 GiB in use, perception plus its head already fills both cards
between them. Putting both decoders on one card and both heads on the other fails
in the planning decoder's `BiasSoftmax` at 736 MiB.

So the measured configuration is perception decoder on GPU 0, head on GPU 1
(2,434 ms), and planning on either card alone (1,822 ms). Running them as one
frame is 4,256 ms sequential.

The head's 15.4 GiB is the thing to attack next: it is a pre-existing graph that
was improved here but not rewritten, and it is now both the largest single stage
(1,185 ms) and the reason the second card cannot be used properly.

## Files

| | |
|---|---|
| `export_onnx/gdn_onnx_v2.py` | the chunk rule, written to trace into few large nodes |
| `export_onnx/gdn_fast_v2.py` | conditional padding, `CausalDepthwiseShift`, `pad_to` |
| `export_onnx/export_vlm_layers_v2.py` | per-layer export with the above, aligned sequence, in-graph KV transpose and trim |
| `export_onnx/fp16_weights_v3.py` | fp16 projection weights, fp32 everything else |
| `export_onnx/run_onnx_drive_v2.py` | resident sessions, device-to-device layer chaining, per-stage GPU placement |
| `export_onnx/check_parity_v2.py` | the PyTorch side, one stage per process |
| `export_onnx/compare_parity_v2.py` | the comparison above |

```bash
source export_onnx/env_gpu.sh
python export_onnx/export_vlm_layers_v2.py --task perception
python export_onnx/export_vlm_layers_v2.py --task planning
python export_onnx/fp16_weights_v3.py outputs/onnx/vlm_layers_v2_perception \
                                      outputs/onnx/vlm_layers_v2f16_perception
python export_onnx/run_onnx_drive_v2.py --reps 3 --skip-planning --gpu-head 1 \
    --layers outputs/onnx/vlm_layers_v2f16_perception \
    --head-dir outputs/onnx/perception_v2
```

The head copy is made once with
`_gridsample_to_cuda_contrib` followed by `gridsample5d_to_gather.py`; the shipped
`outputs/onnx/perception` is left untouched.
