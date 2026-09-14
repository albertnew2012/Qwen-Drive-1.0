# Measured results — Qwen-Drive-1.0 on this machine

**Headline numbers are from the RTX 3090** (bf16, the authors' CUDA kernels). Reproduce with:

```bash
bash local/run_all.sh          # GPU by default
```

Everything was *originally* produced on CPU (AMD Ryzen 9 3900X, fp32, torch fallbacks) while
a training job owned the card, then re-run on GPU once it freed. Both sets are kept:
GPU results in `outputs/`, CPU results archived in `outputs/cpu_reference/`.

## The two runs do not agree, and that is informative

| | CPU (fp32, torch fallbacks) | GPU (bf16, CUDA kernels) |
|---|---|---|
| `planner-rl` mean ADE, direct | 0.396 | **0.389** |
| `planner-rl` mean ADE, reasoning | 0.402 | **0.398** |
| `planner-sft` mean ADE, direct | 0.364 | **0.344** |
| `planner-sft` mean ADE, reasoning | 0.340 | **0.328** |
| perception, mean occupancy class-agreement | 0.830 | **0.847** |
| perception, mean map pixel accuracy | 0.881 | **0.931** |

Per-scene ADE moves by up to 20 %. The unrolled pipeline verifies **bit-exact against the
library on both devices**, so this is not an implementation error — it is bf16 arithmetic and
reduction order differing between the CPU emulation and the GPU tensor cores, plus fp32 vs
bf16, plus torch fallbacks vs the authors' kernels.

**The GPU path is the better one**, and not only faster: it is the configuration the model was
trained and released against. Map accuracy alone improves 5 points. So the CPU numbers
slightly *understated* the model.

**Every qualitative conclusion in this document survived the change of backend.** That is
worth more than either set of numbers alone — see §1.1.

---

## 0. The four demo scenes

Characterised from `data/demo/planning_scenes.jsonl` before looking at any prediction, so
the results below can be read against what each scene actually asks for:

| # | token | nav | v₀ | 5 s displacement | final heading | character |
|---|---|---|---|---|---|---|
| 0 | `a53176b0…` | straight | **0.00 m/s** | 7.5 m | −32° | standing start at a green light |
| 1 | `74cf0a3e…` | **left** | 0.06 m/s | 18.2 m | **+71°** | left turn from near-rest |
| 2 | `3c243f1b…` | **right** | 2.67 m/s | 20.6 m | −24° | right turn while rolling |
| 3 | `2438d38a…` | straight | 6.99 m/s | **34.6 m** | 0° | cruise, the fastest scene |

They span the interesting axes: two starts from rest and two while moving, a left turn, a
right turn, and a 34 m cruise. Scene 0 is the hardest kind of prediction for a
displacement metric — from a standstill the model has to guess *how hard* the driver
accelerates, which is barely observable from images.

> **The README's description of scene 3 does not match its trajectory.** It is described as
> "a slow-down past a parked truck", but the ground truth holds **6.98 → 6.85 m/s** over the
> full 5 s — a 1.9 % speed reduction — with 0.20 m of lateral excursion. There is no
> slow-down and no nudge in the predicted window. A parked truck may well be visible in the
> images, but the ego does not react to it within the horizon. That makes scene 3 the
> *easiest* of the four for a planner (constant speed, straight ahead), which is worth
> knowing before reading its ADE either way.

Note scene 0's ground truth ends at −32° heading despite a `GO STRAIGHT` command: the
vehicle drifts right through the intersection. The nav command is a route hint, not a
description of the manoeuvre.

---

## 1. Planning

Numbers below are 6-sample runs at 10 Euler steps with `planner-rl`. Scene 0 is discussed
first because it was measured first; §1.3 aggregates across scenes.

**Scene 0** (`a53176b0…`) is a WOD-E2E night scene: the ego vehicle is **stopped** at a
signalised intersection (all 16 history poses are `(0,0,0)`) and the light ahead is green.

### 1.1 Does reasoning conditioning help? On this evidence, no.

| mode | ADE | FDE | wall-clock |
|---|---|---|---|
| `DIRECT_PLANNING`, 6 samples | 0.366 m | 1.329 m | 1348 s |
| `REASONING_PLANNING`, 6 samples | 0.341 m | 1.228 m | 1152 s |
| best-of-6 (direct, sample 2) | 0.350 m | 1.214 m | — |

Paired, all four scenes, same seeds, `planner-rl`, **on GPU** (CPU in brackets):

| scene | manoeuvre | direct ADE | reasoning ADE | change |
|---|---|---|---|---|
| 0 `a53176b0…` | standing start, straight | 0.351 (0.366) | **0.334** (0.341) | −5.0 % |
| 1 `74cf0a3e…` | left turn | **0.233** (0.242) | 0.253 (0.263) | +8.9 % |
| 2 `3c243f1b…` | right turn | **0.500** (0.523) | 0.573 (0.592) | +14.6 % |
| 3 `2438d38a…` | cruise | 0.471 (0.454) | **0.432** (0.413) | −8.3 % |
| | | mean **0.389** (0.396) | mean **0.398** (0.402) | **+0.009 m** |

**Identical conclusion on both backends:** reasoning better on exactly the same 2 of 4 scenes,
mean change +0.009 m on GPU and +0.006 m on CPU. A null result that survives a complete change
of arithmetic backend is a much stronger null result.

**Reasoning conditioning is a wash: better on 2 of 4, mean change +0.006 m, and best-of-6
identical to three decimal places.** Each intermediate n told a different story — at n = 1 it
"helped by 6.8 %", at n = 3 it "hurt by 5.8 %". Neither was real.

#### But "reasoning conditioning hurts" is probably the wrong reading

The right reading is **"ADE is the wrong metric for `planner-rl`"**, and the README's own
planning table says so:

| | SFT | RL |
|---|---|---|
| NAVSIM v1.1 navtest, PDMS | 88.2 | **90.7** |
| Waymo E2E test, RFS | 7.78 | **7.91** |
| NVIDIA PhysicalAI open-loop, **minADE 3 s** | **0.34 m** | 0.38 m |

Reward optimization **improves** the driving-quality scores (PDMS, RFS) and **degrades**
displacement error — the authors bold `0.34` for SFT themselves. PDMS scores collision
avoidance, drivable-area compliance and comfort; it does not reward matching the human
driver's exact path, and an RL stage that chases it will happily move away from that path.

Two facts then line up with everything measured above:

1. `planner-rl` was reward-optimized **only on reasoning-conditioned rollouts**
   ([`docs/cookbook.md`](../docs/cookbook.md)).
2. This section scores it with **ADE**.

So the reasoning mode is exactly where the RL tuning lives, and ADE is exactly the metric
that tuning degraded. Direct mode is off-distribution for that RL stage and reverts toward
pre-RL behaviour — which is why it looks *better* under ADE. The measurement is not
evidence against the architecture's premise; it is evidence that **the checkpoint and the
metric are mismatched**.

#### A clean control, by construction

When the same scenes are run with `planner-sft`, the **generated rationale is byte-identical**
to the `planner-rl` run:

> *"Accelerate through the intersection as the traffic light is green."*

That is expected and worth noticing: the reasoning is produced by the **shared VLM**, which
is the same file in both configurations — only the 1.04 B expert differs. So swapping the
planner gives a controlled comparison for free: same images, same prompt, same generated
text, same KV cache, same noise seeds. The only thing that changes is the network that
decodes a trajectory from that cache.

This is a direct consequence of the architecture in [`01_MODEL_STRUCTURE.md`](01_MODEL_STRUCTURE.md):
the VLM is shared and untouched, and the heads are interchangeable readers of it.

#### The experiment that settles it — run, and it confirms the mechanism

`planner-sft` was **not** reward-optimized. Running the same four scenes, same seeds, same
(shared-VLM, therefore identical) rationales gives a complete 2x2 of 16 planning runs:

| mean ADE over 4 scenes | direct | reasoning | effect of reasoning |
|---|---|---|---|
| **`planner-sft`** GPU | 0.344 | **0.328** | **−4.7 %**, better on **3/4** scenes |
| **`planner-rl`** GPU | 0.389 | 0.398 | +2.3 %, better on 2/4 scenes |
| `planner-sft` CPU | 0.364 | 0.340 | −6.5 %, better on 3/4 |
| `planner-rl` CPU | 0.396 | 0.402 | +1.5 %, better on 2/4 |

Both backends agree on direction, magnitude and which scenes flip. `planner-sft` beats
`planner-rl` on **6 of 8** (scene, mode) pairs, mean 0.057 m on GPU (0.048 m on CPU).

**Reasoning conditioning helps the imitation-trained expert and is a wash for the
reward-optimized one.** That is the predicted interaction: RL traded displacement accuracy
for its reward objective *inside the reasoning regime it was optimized in*, so measuring
`planner-rl` with ADE in reasoning mode scores exactly what RL gave away.

And `planner-sft` beats `planner-rl` on ADE across **6 of 8** (scene, mode) pairs, mean
0.048 m — independently reproducing the direction of the README's own
`SFT 0.34 / RL 0.38` minADE on entirely different data.

Per scene, `planner-sft`:

| scene | direct | reasoning | change |
|---|---|---|---|
| 0 `a53176b0…` standing start | 0.275 | **0.233** | −15.3 % |
| 1 `74cf0a3e…` left turn | 0.488 | **0.479** | −1.8 % |
| 2 `3c243f1b…` right turn | **0.408** | 0.440 | +7.8 % |
| 3 `2438d38a…` cruise | 0.283 | **0.207** | −26.9 % |

Two things the means hide:

- **The left turn is the only scene where `planner-rl` wins**, and it wins decisively
  (−45 % to −50 % vs SFT). Every other scene favours SFT by 28-100 %. The checkpoints are
  not ordered; they have specialised differently.
- **The best single run of all sixteen is SFT + reasoning on the cruise scene, ADE 0.207 m**,
  with the correct rationale *"Slow down to safely pass the parked pickup truck on the
  right."*

> **Still n = 4 scenes.** What makes this stronger than the earlier readings in this document
> is that it is a *paired* design testing a prediction made **in advance** from the README's
> SFT/RL table — not a pattern found after looking. The earlier claims were post-hoc reads of
> whichever scenes had finished.

**Practical rule: do not evaluate `planner-rl` with displacement error.** Use `planner-sft`
for ADE/FDE work, and `planner-rl` only with the benchmark scores it was optimized for.

What the rationales *do* demonstrate is correct scene understanding:

> scene 0: **"Accelerate through the intersection as the traffic light is green."**
> scene 1: **"Turn left at the clear intersection and accelerate to the target speed."**

Both correct, generated unprompted beyond the mode switch. So the reasoning is semantically
sound — it simply is not measurably improving the trajectory on these scenes.

> Worth noting what this *is* a test of. `planner-rl` was reward-optimized **only** on
> reasoning-conditioned rollouts, so reasoning mode is the regime it was tuned for; the
> docs say to run it there. A null result in its own best regime is more informative than a
> null result would be with `planner-sft`.
>
> **n = 2.** Two more scenes are needed, and on GPU that is seconds.

### 1.2 The six samples of scene 0

Ground truth ends at `(7.32, -1.65)`. Direct planning, per sample:

| sample | endpoint (x, y, heading) | ADE | FDE |
|---|---|---|---|
| 0 | (6.45, -0.66, -0.344) | 0.366 | 1.329 |
| 1 | (6.24, -0.59, -0.322) | 0.384 | 1.516 |
| **2** | (6.57, -0.70, -0.363) | **0.350** | **1.214** |
| 3 | (6.69, -0.76, -0.376) | 0.353 | 1.099 |
| 4 | (6.36, -0.71, -0.371) | 0.375 | 1.344 |
| 5 | (6.69, -0.70, -0.342) | 0.393 | 1.147 |

**The samples are tightly clustered** — 0.45 m of spread in the endpoint across six
independent noise draws. For a "pull away from a stop, go straight through a green light"
scenario that is the right behaviour: there is only one reasonable manoeuvre, and the
flow-matching sampler is not manufacturing diversity where none exists. It also means
best-of-6 buys little here (0.366 → 0.350) — the gains reported for best-of-N in the paper
must come from genuinely ambiguous scenes, not this kind.

Note the model under-shoots longitudinally (6.45 vs 7.32 m) and under-turns laterally
(-0.66 vs -1.65 m) — it is slightly conservative about accelerating from rest, which is a
sensible failure direction for a planner.

### 1.3 Error decomposition, and a trap in reading it

`local/analyze_planning.py` breaks the error into longitudinal, lateral and heading
components. Doing that surfaced a methodological trap worth recording, because it produced a
wrong conclusion before it produced a right one.

| scene | manoeuvre | Δlon (endpoint) | Δlon (traj mean) | Δheading (endpoint) | Δheading (traj mean) |
|---|---|---|---|---|---|
| 0 `a53176b0…` | right drift | −0.88 m | **+0.03 m** | +12.5° | **+2.7°** |
| 1 `74cf0a3e…` | left turn | −0.83 m | **−0.12 m** | +7.0° | **+3.7°** |

Read only the **endpoint** columns and there appear to be two systematic biases: the model
under-shoots distance by ~0.85 m, and its heading is biased left. Read the **trajectory-mean**
columns and the first one evaporates — mean longitudinal error is ≈ 0 and its sign flips
between scenes.

**The distance bias was an artifact of where I looked.** On a 5 s rollout the final waypoint
carries the accumulated error of the whole trajectory, so an endpoint-only statistic will
manufacture an apparent bias out of ordinary error growth. FDE is a perfectly good *metric*;
it is bad *evidence of direction*.

That left an apparent **leftward heading bias**. At n = 4 it resolves into something with a
mechanism, and the variable is **not** turn direction — it is **initial speed**:

| scene | manoeuvre | v₀ | Δheading (traj mean) |
|---|---|---|---|
| 0 `a53176b0…` | standing start | **0.00 m/s** | **+2.7°** |
| 1 `74cf0a3e…` | left turn from near-rest | **0.06 m/s** | **+3.7°** |
| 2 `3c243f1b…` | right turn, rolling | 2.67 m/s | +0.1° |
| 3 `2438d38a…` | cruise | 6.99 m/s | +0.3° |

An order-of-magnitude separation, split exactly by whether the ego is moving.

**The plausible mechanism: from a standstill, heading is barely observable.** There is no
velocity vector to read it from, and the 16 history poses are all `(0,0,0)`, so the expert
has only the images and the nav command — and it falls back on a prior. Once the vehicle is
moving, heading is well constrained by the direction of travel and the error collapses to
~0.2°.

This also explains the earlier confusion: scenes 0 and 1 (both from rest) drove the apparent
"leftward bias" and scene 2 (rolling) appeared to refute it. The variable was never the
direction of the turn.

> **2 scenes vs 2.** A clean separation with a plausible mechanism, not an established
> result. Testing it properly needs a scene set stratified by initial speed — cheap on GPU,
> and a better experiment than adding more scenes at random.

Note also that the right turn has the **worst ADE (0.523) with the best heading**. Its
endpoint sits 1.1 m further right than ground truth with the heading essentially correct — a
lateral offset, not a rotational one. So heading error is not what drives position error here.

### 1.4 You cannot differentiate a bf16 trajectory twice

Scene 3 (the constant-speed cruise) produced an inversion worth chasing: **ADE 0.454 m but
FDE 0.040 m.** The endpoint lands within 4 cm while the path wanders — the opposite of the
usual pattern, where error grows with horizon.

Differencing the waypoints suggested something alarming. Ground truth holds 6.85-6.98 m/s;
the prediction's implied speed oscillates between **4.83 and 8.86 m/s**, and the implied
accelerations look impossible:

| scene | GT peak accel | GT→bf16 peak accel | prediction peak accel |
|---|---|---|---|
| `2438d38a…` cruise | 0.17 | **16.11** | 32.23 |
| `3c243f1b…` right turn | 1.10 | 9.42 | 16.11 |
| `74cf0a3e…` left turn | 3.31 | 9.42 | 24.66 |
| `a53176b0…` standing start | 2.49 | 4.71 | 8.30 |

The middle column is the control that kills the story: it is the **ground truth** normalised
by `trajectory_scale`, rounded to bfloat16, and denormalised — nothing else. A perfectly
smooth human trajectory acquires 16 m/s² of apparent acceleration just by passing through
the output representation.

The arithmetic is exact. The expert's `out_proj` is a bf16 `Linear`, so waypoints arrive
quantised at bf16 before `.float()`. At x = 34.6 m the normalised value is 0.2097, whose
bf16 ulp is 9.77e-4 — **0.161 m** after multiplying by the 165 m scale. Differencing that
twice at 10 Hz:

```
0.161 m x 10 Hz x 10 Hz = 16.1 m/s^2     (measured: 16.11)
```

| x | normalised | bf16 ulp | metres |
|---|---|---|---|
| 7.5 m | 0.0455 | 2.44e-4 | 0.040 m |
| 18.0 m | 0.1091 | 4.88e-4 | 0.081 m |
| 34.6 m | 0.2097 | 9.77e-4 | **0.161 m** |

Position quantisation grows with distance, so the artifact is worst on exactly the scenes
where the vehicle travels furthest.

**What follows from this:**

- **Acceleration, jerk and comfort metrics computed from these outputs are meaningless.**
  Anyone auditing this model for kinematic feasibility needs fp32 output.
- The predictions do sit roughly 2x above the quantisation floor, so there may be genuine
  excess jitter — but this measurement cannot separate model from representation, and it
  would be wrong to claim otherwise.
- ADE/FDE are unaffected: 0.16 m of quantisation is well below the 0.24-0.52 m errors
  measured, and it does not accumulate.

> This is a good argument for the `x1` (clean-endpoint) parameterization the model uses: it
> predicts *positions*, so quantisation stays bounded per waypoint. A velocity-field
> parameterization integrates its own quantisation, and the error would compound along the
> horizon.

### 1.5 Two things that are clear across three scenes

**Per-scene difficulty varies 2.2×** — ADE ranges 0.242 to 0.523. Any single-scene number
from this four-scene demo set says more about scene selection than about the model. That
applies to every figure quoted above.

**Best-of-6 buys ~3 %** — mean ADE 0.377 against mean minADE(6) 0.366, with sample spread of
only 0.17-0.27 m at the endpoint. The sampler does not manufacture diversity on
unambiguous manoeuvres. This is consistent in magnitude with the paper's own best-of-6 gain
(NAVSIM PDMS 88.2 → 89.3, ~1.2 %), and it means best-of-N is worth its cost only where the
manoeuvre is genuinely ambiguous.

---

### 1.6 A config trap: `num_inference_steps` above 10 silently breaks the sampler

Found by sweeping the Euler step count on one prefill
(`study/scripts/07_qwen_drive_pipeline.py --sweep-steps 1,2,3,5,10,20`):

| steps | endpoint (x, y, heading) | ADE | FDE |
|---|---|---|---|
| 1 | (6.41, −0.72, −0.308) | 0.357 | 1.309 |
| 2 | (6.73, −0.78, −0.319) | 0.309 | 1.059 |
| 3 | (6.77, −0.80, −0.319) | 0.291 | 1.019 |
| **5** | (6.85, −0.80, −0.322) | **0.275** | 0.977 |
| 10 (default) | (6.69, −0.78, −0.319) | **0.275** | 1.080 |
| **20** | (7.37, −1.81, −0.289) | **3.529** | 0.164 |

**20 steps is 13x worse than 10.** The cause is arithmetic, not noise. The sampler is

```
x += (x1_hat - x) / max(1 - t, min_one_minus_t) * dt
```

and to finish *on* the predicted endpoint the final step needs `dt / remaining == 1`. The
floor `min_one_minus_t = 0.1` only equals `dt` while `1/n >= 0.1`:

| steps | dt | remaining (last) | coefficient | |
|---|---|---|---|---|
| ≤ 10 | ≥ 0.100 | = dt | **1.000** | lands exactly on the prediction |
| 12 | 0.083 | 0.100 | 0.833 | under-shoots 17 %, residual noise left in |
| 20 | 0.050 | 0.100 | 0.500 | under-shoots 50 % |
| 50 | 0.020 | 0.100 | 0.200 | under-shoots 80 % |

So **`num_inference_steps` and `min_one_minus_t` are coupled, and the released defaults sit
exactly on the boundary.** Raising the step count — the normal instinct with a diffusion
sampler, where more steps means better quality — leaves the trajectory partly un-denoised.
The tell is in the numbers: at 20 steps the **FDE is the best of any setting (0.164)** while
the ADE is the worst, because the endpoint is roughly right and the *path* is still noisy.

To use more steps you must lower the floor too: `min_one_minus_t <= 1 / num_inference_steps`.

**Going the other way is free.** 5 steps reproduces the 10-step ADE exactly and 3 steps is
within 6 %, which halves or thirds the expert's cost (10 passes of a 1.04 B network).

> Two claims of different strength here. The under-shoot is **arithmetic** — certain, and
> independent of the scene. "5 steps is as good as 10" is **n = 1** and needs the other
> scenes before it is a recommendation.

---

## 2. Timing — what CPU-only inference actually costs

bf16, 12 threads at nice 19, confined to 8 of 12 physical cores:

| stage | measured |
|---|---|
| model load (5.58 B, mmap) | 24 s |
| `DIRECT_PLANNING`, 6 samples | **1348 s** |
| `REASONING_PLANNING`, 6 samples | **1152 s** |
| VQA, single frame, <=200 tokens | **739 s** |
| perception frame, fp32 (6-8 cam) | **225-308 s** |

Reasoning mode was **faster than direct** despite doing strictly more work (a generation
pass, a turn-closing forward, then the expert). Both pay the same ~3385-token prefill, and
the difference is CPU contention noise. The lesson: **the prefill dominates so completely
that everything else is rounding error.** On GPU the same passes are seconds.

### The dtype asymmetry

`local/bench_dtype.py`, VLM prefill:

| | prefill 256 | prefill 1024 | decode 1 token |
|---|---|---|---|
| bfloat16 | 6.0 tok/s | 7.2 tok/s | 2.30 s |
| float32 | **22.0 tok/s** | **22.5 tok/s** | 2.64 s |

fp32 is **3.1× faster at prefill** (compute-bound, and Zen 2 emulates bf16) but **1.15×
slower at decode** (bandwidth-bound, and fp32 weights are twice the bytes). Planning still
runs in bf16 — see [`03_LOCAL_SETUP.md`](03_LOCAL_SETUP.md#3-the-dtype-trap-measured-not-guessed)
for why fp32 is not a free upgrade for the expert.

---

## 3. Perception

Three demo frames, fp32, CPU, single-frame inference. The two CUDA kernels are replaced by
the torch fallbacks (`multi_scale_deformable_attn_pytorch`, shipped; `_voxel_pool_depth_torch`,
added here and verified exact — see [`03_LOCAL_SETUP.md`](03_LOCAL_SETUP.md#verifying-it)).

**It is far cheaper than planning.** 225-308 s per frame, against 1348 s for one planning
pass, because a perception frame is a single prefill of 2744-3654 tokens with no
autoregressive generation and no 10-step sampler.

All **six** bundled frames, both camera rigs:

| frame | rig | pred/GT boxes >0.3 | per class (pred/GT) | occ class-agree | map acc |
|---|---|---|---|---|---|
| `4d0d1ccbb1035a90` | nuPlan 8cam | **3/3** | ped 2/2 · veh 1/1 | 0.743 | 0.851 |
| `6e1c5b330852568d` | nuPlan 8cam | 11/9 | **cone 8/6** · veh 3/3 | 0.843 | 0.877 |
| `7124051adb6d5d5d` | nuPlan 8cam | 7/5 | ped 6/4 · veh 1/1 | 0.658 | 0.912 |
| `90162f90…` | nuScenes 6cam | 34/31 | veh 23/20 · ped 10/9 · bike 1/2 | **0.950** | 0.890 |
| `957810c6e4ff5648` | nuPlan 8cam | 46/40 | veh 24/19 · ped 21/20 · obj 1/1 | 0.909 | 0.868 |
| `f41862469…` | nuScenes 6cam | 37/33 | ped 32/28 · veh 5/5 | 0.875 | 0.885 |
| | | | | **mean 0.830** | **mean 0.881** |

**Predictions exceed ground truth in 6 of 6 frames**, by ~17 % on average (ratios 1.00-1.40).
That is a consistent over-detection tendency **at the 0.3 threshold used here** — which is my
choice, not the model's. Raising it to 0.5 brings counts closer to ground truth but costs a
third of the pedestrians on `f41862469…`, so the operating point genuinely trades precision
against recall and 0.3 sits on the recall side.

Class handling is scene-appropriate across the set: 8 traffic cones on a construction scene,
24 vehicles + 21 pedestrians on a busy one, and an exact 3/3 on a quiet campus road.

Timing, per frame:

| frame | rig | tokens | time |
|---|---|---|---|
| `4d0d1ccbb1035a90` | nuPlan, 8 cam | 3654 | 308 s |
| `90162f90…adb71b5f` | nuScenes, 6 cam | 2744 | 253 s |
| `f41862469…392c4f4` | nuScenes, 6 cam | 2744 | 225 s |

### Detection — predicted vs ground-truth class counts

| frame | thr | pred | GT | per class (pred/GT) |
|---|---|---|---|---|
| `4d0d1ccbb1035a90` | 0.3 | 3 | 3 | pedestrian 2/2 · vehicle 1/1 |
| | 0.5 | 2 | 3 | pedestrian 2/2 · vehicle 0/1 |
| `90162f90…` | 0.3 | 34 | 31 | vehicle 23/20 · pedestrian 10/9 · bicycle 1/2 |
| | 0.5 | 28 | 31 | vehicle 19/20 · pedestrian 8/9 · bicycle 1/2 |
| `f41862469…` | 0.3 | 37 | 33 | pedestrian 32/28 · vehicle 5/5 |
| | 0.5 | 19 | 33 | pedestrian 14/28 · vehicle 5/5 |

On the quiet nuPlan campus road it finds **exactly** the right three objects with the right
classes. On the busy nuScenes streets it is within a handful of the ground-truth count in
every class. Visually (see the video) the boxes are tightly fitted to the taxi, the parked
cars, the SUVs and the pedestrians on the sidewalk, and the one bicycle is correctly
classified as `bicycle` rather than `vehicle`.

> **What this is and is not.** These are *class-count agreements*, not mAP or NDS. They say
> the head finds about the right number of about the right things; they do **not** measure
> localisation quality, and a count can be right for the wrong reasons. The official
> detection metrics need the nuScenes evaluation toolkit and the full validation split.
> Note also that the 0.5 threshold costs a third of the pedestrians on the last frame, so
> pedestrian confidence is the soft spot.

### Occupancy and map

| frame | occupancy IoU (occupied vs occupied) | class agreement where both occupied | map pixel accuracy |
|---|---|---|---|
| `4d0d1ccbb1035a90` | 0.284 | 0.743 | 0.851 |
| `90162f90…` | 0.379 | **0.950** | 0.890 |
| `f41862469…` | 0.445 | 0.875 | 0.885 |

The pattern is consistent across all three: **modest occupancy IoU, high class agreement.**
The prediction is *sparser* than the ground truth — it does not hallucinate structure, it
just declines to fill in distant voxels — but wherever it does commit, it usually commits to
the right class (95 % on the busy nuScenes frame). This matches the rendered voxel panels,
where the predicted volume is a thinner version of the same road layout.

Map segmentation lands at **0.85-0.89 pixel accuracy**, and the rendered rasters track the
ground-truth lane lines, road edges and crosswalks closely.

---

---

## 4. Visual question answering

Single-frame questions (~868 image tokens, vs 3054 for a full planning scene), VLM only —
the planning expert is not involved. `local/run_vqa_probe.py`.

### Driving question, and an OCR result worth checking

**Q** (Front camera, current frame): *"What is the state of the traffic light ahead, and what
should the ego vehicle do?"*  — **739 s on CPU**

> The traffic light ahead is **green**.
>
> ### What this means:
> - A green light indicates that vehicles in the direction of travel are **permitted to
>   proceed** through the intersection, assuming it is safe to do so.
> - The ego vehicle should continue forward, maintaining awareness of surrounding traffic,
>   pedestrians, and any potential hazards […]
>
> ### Additional context from image:
> - There's a **"Central"** street sign above the lights — likely indicating the cross street.
> - A **"ONE WAY"** sign with an arrow points right […]
> - A pedestrian crossing sign is also visible […]
>
> ✅ **Action**: Proceed cautiously through the intersection while obeying all other traffic
> rules and being mindful of surroundings.

**I checked the two sign claims against the raw frame rather than taking them on trust.**
Brightening and 2× upscaling the current front frame shows a green street sign reading
**"Central"** and, below the left traffic head, a **"ONE WAY"** sign. Both are real. They are
small, unlit and sitting in heavy low-light sensor noise — this is genuine OCR under bad
conditions, not a lucky guess.

The third claim, a pedestrian-crossing sign, I could **not** confirm in the region I
inspected. Treat that one as unverified.

**Why this matters for the Alpamayo comparison.** This is exactly the capability the
benchmark table says Alpamayo has lost — `OCRBench 3.2`, `MMBench 7.5`, both below chance.
Rather than trust the Qwen team's evaluation of their own competitor, here is Qwen-Drive's
half of that claim reproduced locally: correct scene understanding, correct small-text OCR
at night, and a structured, well-formatted answer — from the same weights that produced the
trajectory above, with no head attached.

Note also the *style*: markdown headings, bullets, a bolded action line. That is retained
instruction-following from the Qwen3.5-4B base, not something a driving-specialised model
would produce.

### A real failure mode: greedy decoding degenerates on enumeration

**Q** (Front Right camera): *"Read every piece of text visible in this image, including street
signs and road markings."* — **741 s on CPU**

```
Portland
1100 N
ONLY
FOUNDRE
ONLY
ONLY
ONLY          <- and so on, to the 200-token cap
```

Checked against the raw frame (brightened, 3x upscaled), the content before the loop is
**very accurate**:

| output | in the image? |
|---|---|
| `Portland` | yes — the green street sign |
| `1100 N` | **yes, exactly**, including the `N` block-number suffix |
| `ONLY` | yes — on the right-turn-only sign |
| `FOUNDRE` | **yes** — a vertical illuminated sign on the building at the left edge |
| `ONLY` x22 | **no** — a repetition loop |

**Every non-repeated token it produced is real text in the frame.** Reading `1100 N` off a
small, noisy, night-time sign — and picking up `FOUNDRE` running vertically down an
illuminated building sign at the edge of frame — is a strong result. The failure is
**not perception, it is decoding.**

(I first wrote `FOUNDRE` off as garbled; rendering the frame at video resolution showed the
sign plainly. Worth the second look.)

`VQA_DECODE_DEFAULTS` is:

| parameter | value |
|---|---|
| `do_sample` | True |
| `temperature` | 0.01 |
| `top_k` | **1** |
| `top_p` | 0.001 |
| `repetition_penalty` | **1.0** |

`top_k=1` with a near-zero temperature is **greedy decoding with no repetition penalty**.
That is deliberate — it is how the released benchmark numbers were produced, and it keeps
scoring deterministic. But greedy decoding is exactly what loops on open-ended enumeration
("list everything you see"), because once `ONLY` is the argmax it stays the argmax.

**Practical guidance:** the released defaults are right for *reproducing benchmarks* and
wrong for *open-ended generation*. For enumeration-style prompts pass
`repetition_penalty=1.05` (or `top_k=20, temperature=0.6`, the model's own
`generation_config.json` values) — both are per-call overrides on `generate_text`. The
benchmark tables in the README are all short-answer or multiple-choice formats, where this
never bites; it is a property you only meet off the evaluation path.

> This also puts a caveat on the README's own comparison table. If a scoring harness meets
> a degenerate loop like this, it records a low score, and the recorded number reflects the
> decoding configuration rather than the model. That is worth remembering when reading
> Alpamayo's below-chance rows in [`02_VS_ALPAMAYO_1_5.md`](02_VS_ALPAMAYO_1_5.md) — the
> failure mode I just reproduced on Qwen-Drive is the same *kind* of thing that produces
> them.

---

## 5. What has not been measured

- **`planner-sft` in either mode.** Queued: the same four scenes with the non-RL checkpoint,
  which is what decides whether §1.1's result is a checkpoint/metric mismatch or a real
  problem with reasoning conditioning. Run it with
  `local/analyze_planning.py --compare-with outputs/planning_sft`.
- **Any head-to-head number against Alpamayo 1.5.** Still the highest-value open experiment;
  see Session 6 of [`00_START_HERE.md`](00_START_HERE.md).
- **The fp32-vs-bf16 planning delta.** One command once the GPU is free, and worth doing
  because the expert deliberately reproduces bf16 rotary rounding.
- **PDMS / RFS / official detection metrics.** Everything here is ADE/FDE and class-count
  agreement, because the benchmark harnesses (NAVSIM package + nuPlan maps + metric cache;
  the Waymo metrics; the nuScenes toolkit) are not installed and the scene files are not
  shipped. **This matters more than it sounds** — §1.1 shows ADE and PDMS disagreeing in
  sign on the very checkpoint used here.
- **More than one sample of anything.** Three scenes, one seed set, one machine. Every
  number above is an observation, not a benchmark result.

## 6. Corrections made while producing this document

Recorded because each one was wrong in a way that looked convincing first:

| claim | why it was wrong |
|---|---|
| "reasoning conditioning improves ADE by 6.8 %" | n = 1. At n = 3 it looked 5.8 % *worse*; at n = 4 it is a wash (2/4, mean +0.006 m). Three different stories from the same experiment as n grew. |
| "the model under-shoots distance by ~0.85 m" | endpoint-only statistic; the trajectory mean is ≈ 0 with flipping sign. |
| "a systematic +3° leftward heading bias" | held on two scenes, vanished on the third. At n = 4 the variable turns out to be **initial speed**, not turn direction: ~+3° from rest, ~+0.2° while moving. |
| "`FOUNDRE` is a garbled read" | it is a real vertical sign in the frame; visible once rendered at video resolution. |
| "12 threads does not slow the sweep" | two concurrent jobs cost it 19 % and dropped GPU utilisation to 74 %. |
| "perception will be the expensive stage" | it is the cheapest — 4-5 min vs 22 min for planning. |
| "the session video plays at 0.75x real speed" | arithmetically true (36 frames over 24.0 s of video vs 18.1 s of real driving) but it looked wrong, because 2 Hz keyframes give only **1.5 distinct images per second**. The defect was frame **rate**, not speed. Fixed by rendering at the 10 Hz sweep cadence — 148 frames per scene instead of 36. I had hardcoded `-framerate 1.5` rather than deriving it from the data. |
| "DepthNet compresses distance badly past 25 m" | measured -12 m at 30-40 m and -22 m at 40-60 m, and I wrote it up as a property of the network. It is mostly a property of the **grid**: the BEV volume has a z ceiling, so an upward-looking ray exits it at `d_exit` and the model is not permitted to place anything beyond. **85 % of lidar cells past 40 m sit outside the volume.** Scoring only representable cells: corr +0.788 -> **+0.879**, and the 40-60 m bias goes -22.7 m -> **-5.3 m**. The lesson repeats the one below: I compared against a ground truth the system was never allowed to reproduce. |
| "DepthNet does not predict metric depth" | **the worst one in this table.** Its expectation reads 56 m for road 5 m ahead, so I checked alignment (occlusion test: correct), mode (worse), a bin flip (sky breaks), and concluded the head was non-metric. It is metric. The distribution is **bimodal** - a small correct peak plus a spike at the 59.5 m far clip, with the 15-40 m band empty - so an expectation averages two modes and describes neither. The far bin is a *discard channel*: 95 % of it falls outside the +-50 m BEV grid and is masked away - and it is used most heavily for bare road surface (94 % reject), not for distant objects (32 %). Weighted by what survives, road ahead reads **4.9 m** against a lidar truth of **5.2 m** (corr +0.788, median error 1.4 m). I ruled out four hypotheses and published the fifth without testing it, when `lidar.npy` was sitting in the frame directory the whole time. |
| "the model emits 3.3 g accelerations" | bf16 output quantisation; pushing the *ground truth* through the same rounding produces 16 m/s² on its own. |

**The pattern in this table is the lesson.** Eight of these 10 are cases where a real measurement
supported a wrong conclusion, and the fix was always a control: a second scene, a
trajectory-mean instead of an endpoint, the ground truth pushed through the same
quantisation, the frame rendered at full resolution. The DepthNet row is the exception that proves it: I ran controls, but stopped one short and published a negative conclusion instead of saying "unresolved". None needed more compute — only asking
"what else would produce this number?" before writing it down.

The video frame-rate row is a different failure, and worth naming separately: a number that
was **correct but answered the wrong question.** 0.75x was what was asked for and what was delivered; it
simply was not the property making the video unwatchable. Verifying the arithmetic confirmed
the wrong thing. What caught it was asking what the number was supposed to *achieve*.
