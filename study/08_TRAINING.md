# Training Qwen-Drive-1.0 — a pipeline the release does not ship

> **The finding that governs everything here:** the released repository is not
> merely missing training *code*. It is missing a training *capability*. Two of
> its operators have no backward pass, so nothing in `src/` can be
> differentiated as shipped. That is fixed first, in
> [`training/differentiable.py`](../training/differentiable.py); everything else
> follows from it.

---

## 1. Why the repo cannot train as shipped

Two custom CUDA kernels sit directly in the gradient path:

| operator | what is missing |
|---|---|
| `ops._VoxelPoolDepthCuda` | a `torch.autograd.Function` with `forward` and **no `backward`** |
| `ms_deform_attn_bf16_forward` | a bare function, never wrapped in an `autograd.Function`; also bf16-only |

Calling `.backward()` through either raises:

```
NotImplementedError: You must implement either the backward or vjp method for
your custom autograd.Function to use it with backward mode AD.
```

Reproduce it: **`5z`**, or `python training/test_gradients.py --skip-patch`.

**The fix.** Both operators already have pure-PyTorch twins that ship for CPU
fallback, and both of those *are* differentiable — `index_add_` and
`grid_sample`. `enable_training_ops()` redirects the two call sites to them. It
monkey-patches rather than editing `src/`, so the released package stays
byte-identical and inference keeps the fast kernels.

```
parameters receiving a NON-ZERO gradient: 682 / 682
  depth_net (push)           42/42
  view_trans (voxel pool)    12/12
  bev_embedding (pull seed)   1/1
```

The price is memory and speed: the torch twins peak at **20.7 GiB** for a single
sample where the kernels need a fraction of that. That is what a backward pass
costs, and it is only paid while training.

---

## 2. The recipe, from the technical report

arXiv:2609.00111 gives the stages and every loss coefficient. It does **not**
give an optimiser table; those values are ours and are marked (*).

### Stages

| stage | trainable | frozen | data |
|---|---|---|---|
| **1** perception pretrain | BEV head | vision encoder, VLM | nuScenes, OpenScene |
| **2** joint perception + VQA | head, vision encoder, VLM | — | perception + **1.54 M** vision-language samples |
| **3** planning pretrain | Planning Expert | vision encoder, VLM | **~2.83 M** samples: NAVSIM, OpenScene, WOD-E2E, PAI-AV |
| **4** RL | Planning Expert | vision encoder, VLM | 15 K NAVSIM, 15 K PAI-AV, 479 WOD-E2E |

Stage 2 is the one that matters for the model's character: mixing general
vision-language data in with driving data is what stops it forgetting how to
talk. The report also pins one hyperparameter — **the BEV head trains at 20x the
VLM's learning rate**.

### Losses — quoted, not inferred

```
L_perc = L_det + L_occ + L_map
L_det  = sum_{l=1..6} ( 2 * L_focal^(l) + 0.75 * L_l1^(l) )      <- deep supervision
L_occ  = 100 * L_focal + L_geo + L_sem + L_lov
L_map  = 100 * L_focal + L_lov
L_plan = L_fm + 2e-4 * L_d1 + 2e-5 * L_d2
L_ntp  = -sum_t log p(y_t | x, y_<t)
```

Two readings worth stating, because the report names the symbols but not the
formulas. `L_geo`/`L_sem` are MonoScene's geometric and semantic scaling losses
and `L_lov` is Lovasz-softmax — the standard pairing for semantic occupancy, and
the only one that fits a 10-class voxel grid. `L_d1`/`L_d2` penalise the first
and second differences along the trajectory; at 2e-4 and 2e-5 they are
regularisers, not objectives.

### What the report does not say, and we chose (*)

| | value | why |
|---|---|---|
| optimiser | AdamW, wd 0.01 | BEVFormer default |
| head lr | 2e-4 | BEVFormer default — and **20 x 1e-5 = 2e-4**, consistent with the paper's ratio |
| schedule | linear warmup 500 it, then cosine | BEVFormer |
| grad clip | 35.0 (norm) | BEVFormer |
| matcher cost | focal 2.0, L1 0.25 | DETR3D |

The learning-rate coincidence is worth noticing: the paper's "20x" and
BEVFormer's 2e-4 agree exactly if the VLM trains at 1e-5.

---

## 3. Box encoding — get this wrong and nothing converges

The head predicts, and the matcher must target, a 10-dim encoding that is *not*
the ground-truth layout:

```
GT        [cx, cy, cz,   w,      l,    h,    rot,     vx, vy]      (9)
predicted [cx, cy, log w, log l, cz, log h, sin rot, cos rot, vx, vy]  (10)
```

`normalize_bbox` is the exact inverse of the repo's `denormalize_bbox`
(round-trip error **3e-8**). Velocity is down-weighted to 0.2 in the code
weights and excluded from the matching cost, as in BEVFormer.

---

## 4. What runs, and what it proves

Stage 1 and stage 3 freeze the VLM, so its outputs are **cached once** and the
4.5 B model is never loaded during training. That is what makes this fit on one
24 GB card.

| | cache | cost |
|---|---|---|
| perception taps | `img_vit_feats`, `img_llm_feats` | ~45 MiB/frame |
| planner scene cache | 8 KV groups (the 8 full-attention layers) | ~106 MiB/scene |

### The tests that matter

An overfit test is the only honest smoke test: if a model cannot memorise one
sample, the pipeline is broken regardless of what the loss curve looks like on
real data.

| stage | test | result |
|---|---|---|
| 1 | 120 steps, ONE frame | **15.48 -> 3.36 (-78.3 %)**, `det_cls` -> 0.000 |
| 1 | 180 steps, ALL 6 frames | **15.50 -> 3.82 (-75.3 %)**, 4.6 s/step, peak 20.7 GiB |
| 2 | 8 steps, LoRA + checkpointing | **15.37 -> 9.45 (-38.5 %)**, gradient reaches the VLM, 18.5 GiB |
| 3 | 200 steps, random init | **0.00786 -> 0.00006 (-99.3 %)**, 0.22 s/step |

The all-frames run matters as much as the overfit: a single frame only proves the
optimiser can memorise, whereas six frames with different camera counts and very
different object counts (3 to 40 boxes) exercise the Hungarian matcher on varied
targets and still converge.

**Stage 3 needs `--scratch`, and that is a finding.** From the released
`planner-sft` weights the loss starts at **4e-5** — the shipped planner already
fits the demo scenes almost exactly, so an overfit test from there measures
nothing. Only random initialisation makes it informative.

---

## 4b. Two GPUs: which parallelism, and when it helps

Two RTX 3090s, 48 GiB total, **no NVLink** - `can_device_access_peer` is False,
so NCCL needs `NCCL_P2P_DISABLE=1` or its collectives can hang.
`training/run_ddp.sh` sets it.

### The choice is not DDP-vs-FSDP in the abstract, it is per stage

What actually occupies memory, measured (AdamW keeps its state in the PARAM
dtype - bf16 here - not fp32):

| stage | params+grads+Adam | peak | of which ACTIVATIONS |
|---|---|---|---|
| 1 perception head (125 M) | 1.9 GiB | 20.7 GiB | **18.8 GiB** |
| 2 VLM + head, full fine-tune | ~37 GiB | - | - |

**FSDP shards the first column and not the second.** For stage 1 that is 1.9 GiB
out of a 20.7 GiB peak - sharding it saves 4 %, which buys nothing. For a full
stage-2 fine-tune the persistent state is ~37 GiB; halved across two cards that
is 18.7 GiB each, and the head's activations alone are ~19 GiB, so it still does
not fit. FSDP is the wrong tool on both.

### Stage 1: DDP, and it scales

| | s/step | samples/s |
|---|---|---|
| 1 GPU | 4.60 | 0.217 |
| **2 GPU DDP** | 4.60 | **0.435** |

Exactly 2x: the all-reduce is 0.25 GiB against 4.6 s of compute, so
communication is free. `training/train_perception_ddp.py`, launched with
`run_ddp.sh`.

### Stage 3: DDP makes it SLOWER

| | s/step | samples/s | |
|---|---|---|---|
| 1 GPU | 0.210 | **4.76** | |
| 2 GPU DDP | 0.727 | 2.75 | **1.7x slower** |

Same code, opposite result. The planning expert is 1.04 B, so every step
all-reduces ~2.1 GiB while doing only 0.21 s of compute - it would need ~10 GB/s
of interconnect to break even, and PCIe without P2P is nowhere near that. **Run
stage 3 on one card.**

The rule that falls out: DDP pays when *compute per step / bytes reduced* is
large. Stage 1 is 0.05 GB/s of demand, stage 3 is 10 GB/s.

### Stage 2: MODULE parallel, which is a capacity win

The VLM and the head exchange exactly two tensors - the ViT tap and the LLM tap,
~34 MiB. Cutting there splits parameters AND activations, and autograd crosses
the device boundary by itself because `.to(device)` is differentiable.

| | 1 GPU | **2 GPU module-parallel** |
|---|---|---|
| vision encoder | frozen | **trainable, 333.5 M** |
| LM | LoRA r=16 | LoRA r=16 (or 64) |
| head | 125 M | 125 M |
| peak | 18.5 GiB | **13.8 GiB (cuda:0) / 8.5 GiB (cuda:1)** |

More parameters trained on less memory per card, and closer to the report, which
trains "the vision encoder, VLM and perception head" jointly. `8a` / `8b`.

**A latent bug this exposed.** `Uni3DVoxelPoolDepth.frustum` does
`device = "cuda"` with no index and caches on `.device.type`. On one card that is
invisible; across two it allocates the frustum on cuda:0 regardless of where the
module lives, and the cache never rebuilds because "cuda" == "cuda". The voxel
indices then index features on the other card and torch refuses.
`patch_frustum_device()` in `training/differentiable.py` binds it to
`next(self.parameters()).device`.

---

## 5. Where we deviate from the paper, and why

| | paper | here |
|---|---|---|
| stage 2 VLM | full fine-tune | **LoRA r=16** on attention projections (~0.3 % of params) |
| batch size | unstated, certainly large | 1 |
| data | 1.54 M VL + 2.83 M planning | the 6 bundled perception frames, 4 planning scenes |

The stage-2 deviation is forced arithmetic, not preference: 4.5 B parameters with
gradients and two Adam moments is **~72 GB** before activations. `--full`
selects the faithful path for anyone with the hardware; LoRA plus gradient
checkpointing is what fits on one card.

Stage 4 (RL over PDMS/RFS rewards with G=8 rollouts) is **not implemented** — it
needs a closed-loop simulator (NAVSIM/PDMS) that is not part of this repo.

---

## 6. Run it

| config | what |
|---|---|
| **5a** | cache VLM features (do this first) |
| **5b** | gradient test — proves the patch is what enables training |
| **5z** | the same WITHOUT the patch: shows the shipped failure |
| **5c** | stage 1 overfit — the headline test |
| **5d** | stage 1 on all 6 frames |
| **5e** | cache planner scene caches |
| **5f** | stage 3 overfit from scratch |
| **5g** | stage 2 joint, LoRA + checkpointing |
