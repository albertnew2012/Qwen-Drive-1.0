# Exporting Qwen-Drive-1.0 to ONNX

> **The one thing to take away:** the dangerous failure was not a crash. The
> perception head exported cleanly, passed `onnx.checker`, ran in onnxruntime —
> and returned **numerically wrong answers**, with a relative error of 0.96
> against PyTorch. It looked like a success. Only comparing against PyTorch
> caught it. Every export script here verifies, and none of them are finished
> until they do.

---

## 1. What exports, and how well

**The whole model exports, and both end-to-end paths match PyTorch.**

| graph | params | nodes | vs PyTorch |
|---|---|---|---|
| VLM vision tower | 0.3335 B | 3,747 | 7.08e-05 (the tap the heads read) |
| VLM text, 32 layers + norm (perception shape, 2744 tok) | 4.2058 B | 344,127 in 33 graphs | 1.36e-06 |
| VLM text, 32 layers + norm (planning shape, 3385 tok) | 4.2058 B | 366,983 in 33 graphs | ~1e-06 |
| BEV perception head | 0.1251 B | 12,436 | 5.29e-04 |
| Planning Expert, one denoise step | 1.0398 B | 9,315 | 4.17e-07 |
| VLM **decode** step (token generation) | 4.2058 B | **10,233** | 1.99e-06 over all 65 outputs |

### End-to-end, against the real PyTorch model

**Perception path** - `vision -> 32 layers -> perception head`:

| tensor | relative diff |
|---|---|
| VLM hidden_states | 3.15e-04 |
| VLM vit tap | 7.16e-05 |
| VLM llm tap | 5.10e-04 |
| planner KV keys (x8) | 5.18e-04 |
| planner KV values (x8) | 2.25e-04 |
| all_cls_scores | 1.03e-03 |
| all_bbox_preds | 1.05e-03 |
| occ_pred | 1.31e-06 |
| seg_preds | 3.65e-06 |
| **verdict** | **PASS** (worst 1.05e-03) |

**Planning path** - `vision -> 32 layers -> 8 KV caches -> planner x10 Euler`:

```
PyTorch endpoint [ 6.907 -0.814 -0.323]
ONNX    endpoint [ 6.907 -0.814 -0.323]
ADE 0.00002 m     FDE 0.00004 m        PASS
```

The planning path agrees to **16 micrometres** over a 50-waypoint trajectory.

### Reading the error scale

It spans three orders of magnitude, and that is expected rather than alarming.
The planner is a plain transformer evaluated once (4e-07). The perception head
accumulates ~668,000 scattered contributions per frame in a different order than
PyTorch, and fp32 addition is not associative (5e-04). The end-to-end numbers are
set by the vision tower, whose error is then carried through 32 more layers -
measured amplification from the vision tower to `hidden_states` is only **3x**,
which is why the chain stays near 1e-04 instead of diverging.

### The decode STEP is covered; a generation LOOP is not

The decode step exports too, and it is **33x smaller than prefill** - 10,233
nodes against 341,977 - for one reason: at decode the sequence length is 1, so
``torch_recurrent_gated_delta_rule`` loops once and none of the forward-
substitution machinery that dominates prefill is reached.

**Be precise about what this does and does not give you.** ONE decode step is
exported and verified, at a FIXED past length (64 tokens here). A real generation
loop needs the key/value cache to grow by one position every step, and a
shape-frozen graph cannot do that. Making it a usable generator needs a
max-length KV buffer written at a position index, with masking - a standard
technique, but it is not built here. So: the decode *mechanism* is proven
exportable and numerically correct; **autoregressive VQA generation is not
runnable from these graphs.**

It carries **64 state tensors** in and out: 24 conv states (kernel 4), 24
Gated-DeltaNet recurrent states, and 8 key/value pairs. Getting it to export took
removing three in-place writes, because ONNX cannot write into a graph input:

| in-place write | why it exists | replacement |
|---|---|---|
| `conv_states[i].copy_(...)` | "keep the static address" for CUDA-graph capture | assign the slice |
| `recurrent_states[i].copy_(...)` | same | assign |
| `torch_causal_conv1d_update` mutating `conv_state` | the caller reads the state back through the cache | made **functional**: compute the new state and record it, then return it as a graph output |

The third is the interesting one. The tracer says exactly what is wrong -
``aten::copy_ on block input: 'conv_state'. This changes graph semantics.`` - and
the fix is not a substitution but a change of contract: the function has to
*return* the new state instead of writing it somewhere.

**Shapes are frozen.** Each graph is traced at one token count, so the perception
and planning paths need separate exports (2744 vs 3385 tokens), and the
perception head additionally bakes in the camera calibration. Feeding a
different-calibration frame to it produces NaN - it fails loudly, which is the
good outcome; see 3b.

## 2. The obstacles, and what each one really was

### 2.1 The custom kernels cannot be represented at all

Same blocker as training: `voxel_pool_depth` and `ms_deform_attn_bf16_forward`
are CUDA kernels with no ONNX equivalent. `training.differentiable`'s
`enable_training_ops()` already routes both to pure-PyTorch twins, and a tracer
needs exactly what a gradient needs. One switch serves both purposes.

### 2.2 Geometry the tracer cannot follow

Three things resist tracing: `torch.inverse` (no standard ONNX operator),
`img_metas` (a list of dicts holding numpy arrays), and the boolean mask
selection in the voxel pool (data-dependent shapes).

All three depend **only on the camera calibration**, never on image content. For
a fixed rig they are constants. `geometry_freeze.py` runs the real functions once
and replaces them with their captured output.

This is what deployed BEVFormer-family models do; it is not a shortcut. It is
also **verified exact** — freezing changes the answer by `0.000e+00`:

```
frozen geometry only     worst relative diff 0.000e+00  OK
frozen voxel pool only   worst relative diff 0.000e+00  OK
both                     worst relative diff 0.000e+00  OK
```

Re-export only if the rig changes.

### 2.3 A single tensor over 2 GiB

The scatter buffer is `[B*N_cam*X*Y*Z, C]` = `[3.84 M, 256]` = **3.93 GiB** in
fp32. External data lets a *model* exceed protobuf's 2 GiB ceiling — the 4.16 GB
planner proves that — but **no single tensor may**.

Fixed by scattering one camera at a time: `[640 k, 256]` = 655 MiB per slab,
stacked afterwards. `ranks` already encodes the camera in the row index, so the
arithmetic is unchanged.

### 2.4 `index_add` with duplicate indices — the silent one

This is the bug that produced correct-looking, wrong output. PyTorch says it
plainly, if you ever trigger it in isolation:

```
ONNX export does not support exporting 'index_add_()' function with
duplicated values in 'index' parameter yet.
```

**Voxel pooling is nothing but duplicate indices.** Many frustum points land in
the same voxel, and summing them is the entire operation. Inside the large graph
the check did not fire, and the export silently dropped the accumulation.

`scatter_add` lowers to `ScatterElements(axis=0, reduction='add')` and is
bit-exact through onnxruntime (`0.000e+00` on an isolated test). That one
substitution moved the head from **9.65e-01** to **5.29e-04**.

### 2.5 5D GridSample needs opset 20

The occupancy path samples a 3-D volume. `GridSample` only gained 5-D support in
**opset 20**; at 17 the exporter refuses outright. Harmless once known — and both
4-D and 5-D `grid_sample` were checked against onnxruntime independently
(9.3e-08 and 1.5e-07).

### 2.6 Planner-specific: integer inputs and grouped-query attention

Three smaller ones, all in `export_planner.py`:

- `_one_hot`'s range guard, `clamp` on an integer tensor, and `F.one_hot` itself
  lower to `prims.ge`, `prims.ne(x, x)` and `prims.iota`, none of which the
  exporter can dispatch. Solved by passing the navigation command **already
  one-hot and float**, so the graph has no integer input at all.
- `_attend` calls SDPA with `enable_gqa=True`, which the torchscript exporter
  rejects outright. Repeating the key/value heads by hand is the same function
  in ops it can lower.

---

### 2.7 The missing causal mask - self-consistently wrong

Splitting the text model into per-layer graphs means each wrapper must supply
what the whole model used to build for it. `create_causal_mask` is the one that
is easy to forget, because a wrapper that passes `attention_mask=None` still
runs, still exports, and still **verifies per layer** - both sides of that
comparison share the same mistake.

It only shows up when the layers are chained and compared against the whole
model:

```
chained layers, attention_mask=None (what I exported first)  rel 6.47e-01
chained layers, explicit causal mask                         rel 0.00e+00
```

Bit-exact with the mask, 65 % wrong without it. Per-layer verification cannot
catch this class of bug; only an end-to-end comparison can.

### 2.8 The final norm is applied TWICE, and that is the model

The one that took longest. After fixing the mask, `hidden_states` still differed by
4.7e-01. The cause is in the repo, not the export:

```
len(hidden_states) = 33  for 32 layers
norm(hidden_states[-1]) vs hidden_states[-1]: rel 6.19e-01
```

transformers appends the **post-norm** output as the last entry of
`hidden_states`, and `modeling_perception.py` then calls
`language_model.norm()` on it again. The released perception head was fitted
against that double-normed tensor, so reproducing the model means reproducing
the second norm - the pipeline runs `final_norm.onnx` twice, deliberately.

Whether that was intended by the authors is beside the point. The exported graph
has to match the shipped behaviour, and the only way to know which behaviour is
shipped is to compare numbers.

### 2.9 Every graph is frozen to one input shape

The Gated-DeltaNet chunk loop unrolls at trace time, so the graphs are locked to
the sequence length they were traced at. The two tasks differ:

| task | cameras | tokens | layer graphs |
|---|---|---|---|
| perception | 6, one timestamp, 896x512 | 2744 | 344,127 nodes |
| planning | 3 x 4 timestamps, mixed | 3385 | 366,983 nodes |

so each needs its own vision graph and its own 33 layer graphs. The planning
scenes even differ from each other (3383 / 3385 / 3386 tokens), which is why
`export_planner.py` takes `--record`: the step graph must be traced on the same
scene the pipeline runs.

---

## 3. Which exporter, and why it matters

Both were needed, for opposite reasons:

| | perception | planner |
|---|---|---|
| **torchscript** (`dynamo=False`) | **works** | works |
| **dynamo** (`dynamo=True`) | fails: `prims.signbit` dtype mismatch in decomposition | fails: unregistered `prims.*` chain |

On torch 2.8 the dynamo ONNX path repeatedly bottomed out in `prims`-level ops
with no registered ONNX function. Downgrading `onnxscript` 0.7.2 -> 0.4.0 changed
nothing, so it is not a version mismatch. The mature torchscript exporter handled
both models once the structural problems above were fixed.

---

## 3a. Why there is no single .onnx file

The obvious question. The monolithic text prefill **does export** - it is on disk
at `outputs/onnx/vlm_text/`, 341,977 nodes and 13.4 GiB - but onnxruntime cannot
build a session from it. Twice, over an hour, without finishing.

That is not a size limit, it is a scaling law. Merging per-layer graphs with
`onnx.compose` and timing session creation:

| layers merged | nodes | ORT session load | **seconds per 1k nodes** |
|---|---|---|---|
| 1 | 14,161 | 3.8 s | 0.27 |
| 2 | 28,322 | 16.0 s | 0.57 |
| 4 | 56,644 | 84.3 s | **1.49** |

**The per-node cost doubles every time the graph doubles**, so session creation is
quadratic in node count. Extrapolating to all 32 layers (453k nodes):
3.8 x 32^2 = **~65 minutes**, which is exactly the behaviour observed on the real
monolith. Merging further also fails on its own: at 6 layers
`compose.merge_models` dies inside `check_model` with protobuf's 2 GiB
serialisation limit.

So the split is forced by the runtime, not chosen for convenience. Two
consequences worth stating plainly:

* **33 graphs load in ~2 minutes total**; one graph of the same size does not
  load at all. Splitting is what makes the model runnable.
* The quadratic term is in the *graph*, not the model. It is driven by the
  341,977 nodes, and those come from the Gated-DeltaNet chunk rule unrolling
  (section 3b). A custom ONNX operator for that rule would shrink the graph by
  two orders of magnitude and very likely make a single file practical.

The host-side glue between graphs is small and explicit - an embedding gather, a
vision-token scatter, mRoPE positions, and the Euler loop - and
`run_onnx_pipeline.py` / `run_onnx_planner.py` implement all of it.

---

## 3b. The VLM: why it had to be split into per-layer graphs

The monolithic text-prefill graph **exports** (341,977 nodes, 13.4 GiB) and then
onnxruntime cannot build a session from it - over an hour without finishing.

It is not the parameter count; the 1.04 B planner loads fine. It is the node
count, from the Gated-DeltaNet chunk rule unrolling at trace time. Measured:

| sequence length | nodes |
|---|---|
| 2744 tokens | 341,977 |
| 256 tokens | 253,057 |

A 10.7x shorter prompt gives only a 26 % smaller graph, so **~244k nodes are
fixed cost** - the 63-step forward substitution inside
`torch_chunk_gated_delta_rule`, which runs once per layer, not per chunk.

### Two fixes that did not work

**Shrink the chunk.** `chunk_size` is a blocking parameter and the output is
invariant to it (3.6e-07 across 64/32/16/8). But shrinking the inner loop
inflates the outer one. Fitting the two measurements above gives

```
nodes ~ 24 x [ 161 x (chunk-1) + 95 x (seq/chunk) ]
```

which predicts 341k at chunk 64 (it measures 341,977) and **809k at chunk 8**.
The optimum is near chunk 40 at ~307k. There is nothing to win.

**Use the closed form.** The matrix being inverted is strictly lower triangular,
hence nilpotent, so `(I-A)^-1 = (I+A)(I+A^2)(I+A^4)...` - 5 squarings instead of
63 steps, exact to 1e-15 in fp64. On the real model it returns **NaN**: `A` comes
from `k_beta @ key.T` with entries O(head_dim) = O(100), and `A^32` overflows
fp32. The shipped loop is stable precisely *because* it never forms a high power.
`patch_chunk_rule_for_export` is kept, unwired, as a record.

### The fix that did work: split at layer boundaries

One graph per decoder layer, run in order by the host.

| | nodes | ORT session load |
|---|---|---|
| one linear_attention layer | 14,161 | **5.1 s** |
| one full_attention layer | 538 | 0.5 s |
| all 33 graphs | 344,127 | **~2 min** |

Same total node count, but each piece loads. This is a normal deployment pattern
for very deep models, not a workaround - and it is what made the end-to-end
validation in section 1 possible.

---

## 4. Wiring it together

Everything below the dashed line is ONNX. The host supplies only glue:

```
              pixel_values
                    |
              vision.onnx  ──►  vit_tap ─────────────────────────┐
                    └────────►  merged_tokens                    │
                                     |                           |
  host: embed gather, vision scatter, mRoPE position_ids         |
                                     |                           |
                 layer_00 .. layer_31.onnx   (33 graphs, in order)
                     |                                   |       |
        8 x (keys, values)                       hidden_states   |
                     |                                   |       |
                     |                      final_norm.onnx x2   |
                     |                                   |       |
        planner_step.onnx x10 Euler            perception.onnx ◄─┘
                     |                                   |
             host: denormalize                  boxes / occupancy / map
                     |
            trajectory [50, 3]
```

Five things stay on the host, and each one is a place to be silently wrong:

1. **Image preprocessing** - 896x512 snapped to the patch grid, pixels in [0, 1]
   normalised with mean and std 0.5.
2. **Embedding gather** - a lookup, kept off the graph so the 2.4 GiB table is
   not duplicated into every export.
3. **Vision scatter** - merged tokens replace the image-token rows.
4. **mRoPE position ids**, `[3, B, S]`. Images get a 2-D position grid, so this
   is not an arange; use the repo's `_rope_positions`.
5. **The Euler loop** - 10 steps around `planner_step.onnx`, so the step count
   can change without re-exporting.

Plus two details that are inside the graphs but easy to get wrong:
`_premerge_grids`' un-permutation of the merger's 2x2 block ordering, and the
**double** final norm (section 2.8).

Every one of these was verified against PyTorch individually: the host
reconstruction of `inputs_embeds` matches the model's own to **0.000e+00**.

**Not covered: text generation.** VQA needs a decode graph carrying the 24
Gated-DeltaNet recurrent states across calls. Perception and planning never
generate tokens, so neither path needs it.

---

## 5. Run it

| config | what |
|---|---|
| **6a** / **6b** | export + inspect the perception head |
| **6c** | export + verify the planning-expert step |
| **6d** | export submodules (the bisect that located 2.4) |
| **6e** / **6i** | export the vision tower (perception / planning rig) |
| **6f** | the monolithic text prefill - exports, will not load (see 3b) |
| **6g** / **6h** | export the text model as per-layer graphs |
| **6j** / **6k** | RUN and VALIDATE the full perception pipeline |
| **6l** / **6m** | RUN and VALIDATE the full planning pipeline |

Order from scratch: `6e`, `6g`, `6a` then `6j`, `6k` for perception;
`6i`, `6h`, `6c` then `6l`, `6m` for planning.

Both runners split into `--phase run` and `--phase compare`. That is not
cosmetic: holding the 17 GiB fp32 model and the 20 GiB perception forward at
once gets the process OOM-killed on a 78 GiB machine.
