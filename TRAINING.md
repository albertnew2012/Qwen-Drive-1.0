# Training Qwen-Drive-1.0

A runbook. Every command below was run on this machine; the numbers are what it
printed. For *why* any of it is built this way, see
[study/08_TRAINING.md](study/08_TRAINING.md).

**One-line summary:** the released repo cannot back-propagate at all - two CUDA
kernels ship without a backward pass - so the first thing this pipeline does is
make the model differentiable. Everything else follows from that.

---

## 0. Before anything

```bash
cd /home/albert/Desktop/Qwen-Drive-1.0
export PYTHONPATH=src:.
export PATH="$PWD/.venv/bin:$PATH"
export CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Every command assumes those. Prefix with `.venv/bin/python` as shown.

---

## 1. The fastest check: does it all still work?

```bash
.venv/bin/python training/run_all_stages.py          # VS Code: 5s
```

Runs every stage in sequence and prints PASS/FAIL. Takes ~13 minutes. Expected:

```
PASS  gradient test WITHOUT the patch (failing IS the pass)      7s
PASS  gradient test WITH the patch                              10s
PASS  stage 1 - perception head, overfit one frame             527s
PASS  stage 3 - planning expert, overfit from scratch           68s
PASS  stage 2 - joint perception + VLM (LoRA)                  152s
5/5 stages passed
```

Stages run one at a time on purpose: two of them peak near 20 GiB and the
machine OOM-kills concurrent runs.

---

## 2. The recipe (arXiv:2609.00111)

### Stages

| stage | trainable | frozen | data |
|---|---|---|---|
| **1** perception pretrain | BEV head | vision encoder, VLM | nuScenes, OpenScene |
| **2** joint perception + VQA | head, vision encoder, VLM | - | perception + 1.54 M VL samples |
| **3** planning pretrain | Planning Expert | vision encoder, VLM | ~2.83 M samples |
| **4** RL | Planning Expert | vision encoder, VLM | 15 K NAVSIM, 15 K PAI-AV, 479 WOD-E2E |

**Stage 4 is not implemented.** It needs closed-loop PDMS/RFS scoring with G=8
rollouts - a simulator that is not in this repo and not public. Stages 1-3 are
implemented from the report; stage 4 is absent, not stubbed.

### Losses - quoted from the paper, not inferred

```
L_perc = L_det + L_occ + L_map
L_det  = sum_{l=1..6} ( 2 * L_focal^(l) + 0.75 * L_l1^(l) )      <- deep supervision
L_occ  = 100 * L_focal + L_geo + L_sem + L_lov
L_map  = 100 * L_focal + L_lov
L_plan = L_fm + 2e-4 * L_d1 + 2e-5 * L_d2
L_ntp  = -sum_t log p(y_t | x, y_<t)
```

`L_geo`/`L_sem` are MonoScene's geometric and semantic scaling losses, `L_lov` is
Lovasz-softmax. The report names the symbols but not the formulas; this is the
only reading that fits a 10-class voxel grid. `L_d1`/`L_d2` penalise the first and
second differences along the trajectory - at 2e-4 and 2e-5 they are regularisers.

Implemented in [`training/losses.py`](training/losses.py).

### Hyperparameters

The report gives **no optimiser table**. These are ours, from BEVFormer/DETR3D,
in [`training/config.py`](training/config.py):

| | value | source |
|---|---|---|
| optimiser | AdamW, wd 0.01 | BEVFormer |
| head lr | 2e-4 | BEVFormer - and **20 x 1e-5**, matching the paper's "20x the VLM" |
| schedule | linear warmup 500 it, then cosine | BEVFormer |
| grad clip | 35.0 norm | BEVFormer |
| matcher cost | focal 2.0, L1 0.25 | DETR3D |

The one real anchor: the paper states the BEV head trains at 20x the VLM's rate,
and 20 x 1e-5 = 2e-4 is exactly the BEVFormer default. The two agree.

---

## 3. Commands, in order

### 3.1 Cache the VLM features (do this first)

```bash
.venv/bin/python training/cache_features.py          # 5a   ~2 min
.venv/bin/python training/cache_planner_features.py  # 5e   ~2 min
```

Stages 1 and 3 freeze the VLM, so its outputs are constant. Caching them means
the 4.5 B model is never loaded during training - that is what makes this fit on
one card. ~45 MiB per perception frame, ~106 MiB per planning scene.

### 3.2 Prove the model is differentiable

```bash
.venv/bin/python training/test_gradients.py --skip-patch --dtype bfloat16   # 5z
```

**This is supposed to FAIL**, with:

```
NotImplementedError: You must implement either the backward or vjp method
for your custom autograd.Function to use it with backward mode AD.
```

That is the shipped repo. Now with the patch:

```bash
.venv/bin/python training/test_gradients.py --dtype bfloat16                # 5b
```

```
parameters receiving a NON-ZERO gradient: 682 / 682
  depth_net (push)           42/42
  view_trans (voxel pool)    12/12
  bev_embedding (pull seed)   1/1
PASS: the perception head is differentiable end to end.
```

### 3.3 Stage 1 - perception head

```bash
# overfit one frame - the test that matters
.venv/bin/python training/train_perception.py --steps 150 --overfit \
    --out outputs/train_perception_overfit                                  # 5c

# all six frames
.venv/bin/python training/train_perception.py --steps 180 --out outputs/train_all6   # 5d
```

Expected: **15.5 -> 3.4** overfit, **15.5 -> 3.8** on all six. ~4.6 s/step, peak
20.7 GiB. An overfit run must at least halve; if it plateaus above ~8 the
differentiable-ops patch probably is not active - run `5b` first.

Useful flags: `--scratch` re-initialises the head, `--no-det/--no-occ/--no-map`
isolate one loss, `--lr` overrides.

### 3.4 Stage 3 - planning expert

```bash
.venv/bin/python training/train_planner.py --steps 250 --overfit --scratch   # 5f
```

Expected **0.0079 -> 0.0001**. ~0.21 s/step.

**`--scratch` is required for this test to mean anything.** From the released
`planner-sft` weights the loss starts at 4e-5 - the model already fits the demo
scenes, so an overfit run from there measures nothing and passes vacuously.

### 3.5 Stage 2 - joint perception + VLM

```bash
.venv/bin/python training/train_joint.py --steps 10 --overfit                # 5g
```

Expected **15.4 -> 9.5**, and critically:

```
gradient reached the VLM: True
```

A falling loss alone is **not** sufficient - the head can improve while the VLM
is effectively frozen. Judge over >=8 steps: L_perc spikes hard on step 1
(15 -> 39 observed) before falling, so a 5-step run reports a spurious FAIL.

`--full` attempts real full fine-tuning instead of LoRA and needs far more than
one card. `--no-checkpoint` disables gradient checkpointing (and will OOM).

---

## 4. Two GPUs

No NVLink on consumer 3090s, so NCCL must be told: `training/run_ddp.sh` sets
`NCCL_P2P_DISABLE=1`. **The right parallelism differs per stage.**

### Stage 1 - DDP, exactly 2x

```bash
bash training/run_ddp.sh training/train_perception_ddp.py --steps 60
```

| | s/step | samples/s |
|---|---|---|
| 1 GPU | 4.60 | 0.217 |
| 2 GPU DDP | 4.60 | **0.435** |

### Stage 3 - DDP makes it SLOWER, use one card

```bash
.venv/bin/python training/train_planner.py --steps 200 --scratch \
    --out outputs/train_planner_1gpu                                         # 8c
```

| | s/step | samples/s | |
|---|---|---|---|
| 1 GPU | 0.210 | **4.76** | |
| 2 GPU DDP | 0.727 | 2.75 | **1.7x slower** |

1.04 B parameters means ~2.1 GiB all-reduced per step against only 0.21 s of
compute - it needs ~10 GB/s of interconnect and PCIe without P2P is nowhere near.
The rule: **DDP pays when compute-per-step / bytes-reduced is large.** Stage 1
demands 0.05 GB/s, stage 3 demands 10 GB/s.

### Stage 2 - MODULE parallel, a capacity win

```bash
.venv/bin/python training/train_joint_2gpu.py --steps 8 --overfit            # 8a
.venv/bin/python training/train_joint_2gpu.py --steps 14 --overfit --lora-rank 64  # 8b
```

The VLM and head exchange only the two taps (~34 MiB), so cutting there splits
parameters **and** activations:

| | 1 GPU | 2 GPU |
|---|---|---|
| vision encoder | frozen | **trainable, 333.5 M** |
| LM | LoRA r=16 | LoRA r=16 or 64 |
| peak | 18.5 GiB | **13.8 GiB (cuda:0) / 8.5 GiB (cuda:1)** |
| L_perc (14 steps, r=64) | - | 15.43 -> **7.19** |

More parameters trained on *less* memory per card, and closer to the report,
which trains the vision encoder jointly.

**Why not FSDP?** It shards parameters/gradients/optimizer state and not
activations. Stage 1's peak is 20.7 GiB of which **18.8 GiB is activations** -
sharding the remaining 1.9 GiB saves 4%. For a full stage-2 fine-tune the
persistent state is ~37 GiB; halved that is 18.7 GiB/card plus ~19 GiB of head
activations, so it still does not fit. Wrong tool on both.

---

## 4b. Stage 3b - grounding the plan in perception

The shipped recipe never makes the trajectory answer to what the model sees: no
loss reads both heads, so the planner may drive through a car its own perception
head has correctly boxed. Stage 3b adds two terms that charge a plan for
disagreeing with perception. The reasoning - and the two ways of writing the cost
that do **not** work - is in
[study/11_GROUNDING_PLAN_IN_PERCEPTION.md](study/11_GROUNDING_PLAN_IN_PERCEPTION.md).

Needs nuScenes trainval: it is the only source here where a frame has both a full
camera ring and a 5 s ego future.

```bash
# 1. cache (~1.7 h, ~114 GB for 850 scenes)
.venv/bin/python -u training/cache_grounded_features.py \
    --dataroot /path/to/nuscenes --version v1.0-trainval --max-scenes 850

# 2. control - identical fine-tune, grounded terms off (~47 min)
.venv/bin/python -u training/train_planner_grounded.py \
    --collision-weight 0 --offroad-weight 0 --seed 0 --out outputs/plan_control

# 3. grounded (~1.7 h)
.venv/bin/python -u training/train_planner_grounded.py \
    --collision-weight 0.007 --offroad-weight 0.004 \
    --occ-sigma 2.0 --map-sigma 2.0 --seed 0 --out outputs/plan_grounded

# 4. score all three on the same held-out frames, with paired CIs
.venv/bin/python -u training/eval_grounded.py \
    --ckpt sft=pretrained \
    --ckpt control=outputs/plan_control/planning_expert.pt \
    --ckpt grounded=outputs/plan_grounded/planning_expert.pt \
    --paired-against control
```

The VLM and perception head are frozen and enter as cached constants, so only the
1.04 B expert is on the optimiser - it fits one 3090 with ~7 GB to spare.

**Read `excess_collision` / `excess_offroad`, not the raw rates.** The raw rates
charge a plan for occupancy that has simply gone stale, and rank the human-driven
path *below* the released model.

**Read the paired CI, not the means.** With 80 held-out frames the per-frame
variance swamps the effect; `--paired-against` reports a bootstrap interval on the
difference, which is the only number that supports a claim.

### Result as measured

On 120 **training** frames the grounded terms cut excess collision 43 %
(0.0058 -> 0.0033, 95 % CI [-0.00478, -0.00068]) and *improve* ADE by 0.05 m
(1.264 -> 1.214, CI [-0.077, -0.024]) - safety and imitation are not in tension.
On 80 **held-out** frames the same comparison is -0.00024 with CI
[-0.00119, +0.00050]: not resolved. Tripling the weights changes nothing.

The loss works where it is applied and does not generalise from 770 records. See
[study/11](study/11_GROUNDING_PLAN_IN_PERCEPTION.md) section 5; the missing
experiment is `--frames-per-scene 8`.

---

## 5. What the pipeline is made of

| file | role |
|---|---|
| [`differentiable.py`](training/differentiable.py) | **the enabler** - routes the two forward-only CUDA kernels to differentiable torch twins |
| [`losses.py`](training/losses.py) | all five losses + the Hungarian matcher, plus the stage-3b grounded terms |
| [`config.py`](training/config.py) | hyperparameters, with paper-vs-ours marked |
| [`cache_features.py`](training/cache_features.py) | pre-extract the frozen VLM's two taps |
| [`train_perception.py`](training/train_perception.py) | stage 1 |
| [`train_joint.py`](training/train_joint.py) | stage 2 (LoRA + checkpointing) |
| [`train_planner.py`](training/train_planner.py) | stage 3 (flow matching) |
| [`cache_grounded_features.py`](training/cache_grounded_features.py) | stage 3b - nuScenes frames carrying both a camera ring and a 5 s future |
| [`train_planner_grounded.py`](training/train_planner_grounded.py) | stage 3b (flow matching + perception agreement) |
| [`eval_grounded.py`](training/eval_grounded.py) | scores several planner checkpoints on identical frames |
| [`lora.py`](training/lora.py) | ~70-line LoRA, no extra dependency |
| [`checkpointing.py`](training/checkpointing.py) | gradient checkpointing, incl. the view transform |
| [`run_all_stages.py`](training/run_all_stages.py) | the PASS/FAIL sweep |
| `*_ddp.py`, `train_joint_2gpu.py` | multi-GPU variants |

### The box encoding - get this wrong and nothing converges

```
GT        [cx, cy, cz,    w,     l,     h,   rot,      vx, vy]        (9)
predicted [cx, cy, log w, log l, cz, log h, sin rot, cos rot, vx, vy] (10)
```

`normalize_bbox` is the exact inverse of the repo's `denormalize_bbox`
(round-trip error 3e-8). Velocity is down-weighted to 0.2 and excluded from the
matching cost, as in BEVFormer.

---

## 6. Troubleshooting

| symptom | cause |
|---|---|
| `NotImplementedError ... backward` | the patch is not active; `enable_training_ops()` must run before the forward |
| loss plateaus above ~8 on an overfit run | same - run `5b` |
| process dies silently, no traceback | system OOM killer. Do not run two heavy jobs at once |
| CUDA OOM at ~20 GiB | expected without checkpointing; `train_joint.py` enables it by default |
| stage 2 "FAIL" after <8 steps | run-length artifact, see 3.5 |
| stage 3 "passes" instantly from released weights | you forgot `--scratch` |
| stage 3b grounded terms print 0.0000 forever | either you truncated the horizon (the planner never collides inside 1.2 s) or you scored `predict_endpoint` at a random `t` instead of a replayed rollout |
| stage 3b runs but changes nothing / ADE gets worse | the loss and the metric are scoring different tensors - the terms must see the sampler's output, see study 11 §2.5 |
| NCCL hangs on 2 GPUs | no P2P on consumer cards - use `training/run_ddp.sh` |
| cross-device index error | the frustum bug; `patch_frustum_device()` fixes it |

---

## 7. Expected results

[study/10_EXPECTED_RESULTS.md](study/10_EXPECTED_RESULTS.md) and the
machine-readable `outputs/expected_results.json`. Regenerate with:

```bash
.venv/bin/python study/scripts/12_collect_results.py                          # 4c
```
