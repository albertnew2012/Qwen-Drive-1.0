# Plan: get the distilled ONNX student to <10% degradation

Living document. The orchestrator appends results to `outputs/distill/lab_notebook.json`
and this file records the reasoning, the ladder, and the decision rules.

Hardware: **4x A100, all usable.**

---

## 1. The gates, corrected

| gate | bar | why this bar |
|---|---|---|
| speed | **>= 10 Hz** single GPU, whole graph | the requirement. Currently 20.6 Hz, so ~48 ms of headroom is available to SPEND |
| detection | student F1 vs GT **>= 67.6%** | teacher scores F1 75.1% vs ground truth; -10% of that |
| trajectory | ADE **<= 1.532 m** AND **< 1.469 m** | teacher 1.393 m +10%; the second clause is the ego-only linear probe |
| occupancy | mIoU vs teacher, best effort | no occupancy ground truth exists here, so only replication is measurable |
| map | mIoU vs teacher, best effort | same |

**The trajectory second clause matters.** A least-squares probe on ego state alone scores
1.469 m; the teacher scores 1.393 m. nuScenes open-loop ADE is nearly solved by ego
kinematics ("Ego Status Is All You Need?"). A student that does not beat the probe has
learned nothing about planning, whatever its ADE. The 78k run scored 1.483 m — inside the
ADE gate but WORSE than the probe. That is not a pass.

## 2. Where we are (measured, 78k steps, full 25,599-frame cache)

```
speed        20.6 Hz   PASS (2x the bar)
detection    F1 34.5% vs GT   (teacher 75.1%)   -> 54% degradation   FAIL
trajectory   1.483 m          (probe 1.469 m)   -> no real planning  FAIL on clause 2
occupancy    6.0% mIoU                                               FAIL
map          57.7% mIoU                                              FAIL
```

## 3. Root cause of the detection failure

Diagnostic over 60 frames, student detections at score >= 0.3:

```
54.7 detections/frame          (teacher ~22)
  51.2% within 2 m of a real object    -> duplicates piled on true objects
  48.8% isolated, nothing within 2 m   -> hallucinations
  1.40 other detections within 2 m of each detection
```

Both halves trace to one decision. `HANDOVER.md` chose per-query distillation *specifically*
to avoid Hungarian matching and ground truth: "query i of the student trains against query i
of the teacher... no matching, no label assignment, no ground truth needed."

1. **No one-to-one assignment.** DETR-family models get duplicate suppression for free from
   Hungarian matching. Without it, nothing stops three queries firing on one car, and there
   is no NMS either.
2. **No hard background supervision.** The target is the teacher's *soft* sigmoid, so
   background queries are trained toward small-but-nonzero scores; `pos_weight` then pushes
   them up. Hence 49% hallucinations.
3. **A hard ceiling at the teacher's 75.1%.** Every teacher mistake is a training target.

Point 3 is no longer forced: `gt_boxes.npz` now exists for every frame.

## 4. The ladder

Screening runs are **20k steps** (~1.2 h on 4 GPUs) so interventions can be compared
cheaply; only the winning recipe gets a long run. Every experiment writes to
`outputs/distill/exp/<id>/` so nothing collides with the other session's
`outputs/distill/student/`.

| id | change | hypothesis | kill criterion |
|---|---|---|---|
| **E0** | NMS + threshold sweep, no retraining | duplicates are ~half the FPs; recoverable for free | none, it is an hour |
| **E1** | Hungarian matching + focal loss + GT boxes; teacher kept as auxiliary soft target at low weight | one-to-one assignment removes duplicates, hard background removes hallucinations | if F1 does not clear ~45% at 20k steps, the objective is not the problem and the architecture is |
| **E2** | E1 + iterative box refinement across decoder layers | one-shot box prediction is weak; DETR3D refines reference points per layer | < +5 F1 over E1 |
| **E3** | E2 + finer BEV memory (less pooling than 200->40, i.e. 2.56 m/cell) | centre error is 0.83 m against a 2.56 m memory grid; the decoder cannot see finely enough | < +3 F1 over E2 |
| **E4** | E3 + capacity, funded by the speed budget | HANDOVER measured room to ~111 M at 14.7 Hz, and we are at 20.6 Hz | speed < 10 Hz |
| **E5** | occupancy: replace the 1x1 conv head with a real decoder | 6% mIoU against the teacher's dedicated decoder is an architecture gap, not a training one | speed < 10 Hz |
| **EF** | best recipe, long run (>= 78k steps) | — | — |

**Speed is re-measured after every architecture change.** It is the one gate currently
passing and the budget is what funds E3–E5; spending it accidentally is the main risk.

## 5. Self-improvement loop

After each experiment the orchestrator runs the full diagnostic suite and appends one
record to `outputs/distill/lab_notebook.json`:

- every gate metric, plus speed and node count
- detection: duplicate/hallucination split, precision-recall across thresholds, centre error
- occupancy and map: predicted vs teacher class histograms (this is what caught the
  0.75-exponent over-correction)
- trajectory: ADE against the ego-only probe and constant velocity, and the magnitude of
  the learned residual (this is what caught the dead head)

Decision rule: **an intervention is kept only if it beats the previous best on its target
metric by more than the run-to-run noise.** Where a hypothesis is refuted, the notebook
records the refutation so the ladder is not re-tried. Hypotheses that fail their kill
criterion stop that branch rather than escalating steps.

## 6. Reflection: what went wrong so far, so it is not repeated

Five defects were found by interrogating results rather than trusting them. Each cost hours.

1. **`set_box_stats` NameError** — the planning head was built inside a setter, so it never
   existed. Caught by a parameter count that did not match the docs (68.6 M vs 81.8 M).
2. **Front-camera student vs six-camera teacher** — only 28% of teacher detections were
   reachable; recall was capped at ~28% by geometry, not capacity.
3. **Map head at 3.4x wrong scale** — resized the whole +/-51.2 m BEV onto a 30x60 m window.
   Caught because its loss was flat while every other term moved.
4. **Trajectory regressing raw metres** — started at the all-zeros trajectory (12.8 m) and
   crawled. Then, after the anchor was added, the head converged to *exactly* zero, because
   L1's optimal constant is the median of the residual and that median is ~0.
5. **A gate transplanted from another benchmark** — 0.335 m was the teacher's ADE on the
   WOD_E2E demo scenes, not nuScenes. The real figure is 1.393 m. The gate demanded the
   student beat the teacher by 4x.

The pattern: **every one was a number that looked plausible and was never checked against a
baseline.** So the rule for the rest of this work is that no metric is reported without the
trivial baseline beside it — zeros, constant velocity, a linear probe, or the teacher
measured on the same data.


---

## 7. Reflection after the E-series (appended 2026-09-23)

### What worked

- **NMS at inference** — +4.8 F1 for nothing (32.9 -> 37.7%). Confirmed half the false
  positives were duplicates, which is what a missing one-to-one assignment produces.
- **Warm-starting Hungarian from the distilled checkpoint (E3).** cls loss 0.33 -> 0.20
  where three from-scratch runs sat frozen at 0.49.
- **Refusing "needs more training" as an explanation.** Every real defect this session
  was found by a cheap targeted probe, never by escalating steps.

### What failed, and the actual cause each time

| run | symptom | true cause |
|---|---|---|
| E1 | 0 detections | focal loss without its prior-probability bias init |
| E1b | 0 detections | all 900 queries predicted ONE point (0.90 m spread) |
| E2 | 1.1% F1 | symmetry fixed, but the climb out of silence is uphill in 12 epochs |

### The mistake worth remembering

The matching cost compared a class term bounded by 2 against a box term in **raw metres**
(5 x tens of metres). Assignment became purely geometric, confident queries were never
selected, and their confidence was charged as a false positive. Measured, the loss was
INVERTED: F1 34.5% scored 1.293 while F1 1.1% scored 0.537.

**And then I misdiagnosed my own fix.** Re-measuring after normalising still looked
inverted (0.782 vs 0.537) -- but that run had reference points ON against a checkpoint
whose box head predicts absolute coordinates, so the good model's boxes were corrupted by
the measurement itself. Measured correctly the objective is ORDERED (0.374 vs 0.521).
A verification harness needs to be as carefully controlled as the thing it verifies.

Both lessons are now enforced by `verify_objective.py`, which must return ORDERED before
any new objective gets GPU time. It distinguishes INVERTED (loss is wrong) from FLAT (loss
cannot see the difference) because they need different fixes.

### Standing rule

No metric is reported without its trivial baseline beside it -- zeros, constant velocity,
a linear probe, or the teacher measured on the same data. Five defects this project got
through review because a plausible-looking number was never compared to anything.

## 8. Correction (2026-09-24 13:30): E3 and E4 were misreported by ~30 F1 points

`diagnose.py` built `StudentConfig()` from defaults, so `ref_points=True` was applied to
checkpoints trained with it off, adding a reference grid to already-absolute boxes.
Re-evaluated with the config read from the checkpoint:

| run | reported | actual | recall | precision |
|---|---|---|---|---|
| E3 Hungarian warm-start | 2.2% | **28.3%** | 22.4% | 38.4% |
| E4 hybrid warm-start | 1.7% | **33.0%** | 26.4% | **44.0%** |

Neither was a failure; both trade recall for precision and sit near the E0 baseline (37.7%).
The "boxes drift on unmatched queries" theory in section 7 was built on the bad numbers and
is withdrawn. What survives: the linear probe (AUC 0.844) still says the BEV features carry
the objects and the query head is the bottleneck, so E5 (dense center head) proceeds — but its
motivation is DETR's slow convergence and the 40x40 pooling, not drift.

The tell that caught it: an `e5` row whose occupancy, map and trajectory were byte-identical
to `e0` while F1 differed by 30 points. Same weights cannot score differently unless the
harness is wrong. Rule added to section 6: **when two evaluations of the same weights
disagree, suspect the evaluator first.**

## 9. Reflection (2026-09-24 14:00): E5 took five launches to start

E5 (dense center head) is the first experiment whose detection losses descend from step
one -- cls 1.90 -> 1.17, box 0.98 -> 0.70 in 200 steps -- where six DETR-style runs sat
flat. But it crashed four times before it ran, and every crash was mine:

| launch | failure | cause |
|---|---|---|
| 1 | DDP "parameters not used in producing loss" | the 900-query decoder was still built and run, contributing nothing |
| 2 | same error | my "removal" block sat BEFORE the real assignments and was overwritten; param count never moved from 83.17 M |
| 3 | same error | `find_unused_parameters` patch matched `DistributedDataParallel(` -- the file spells it `DDP(` -- so nothing was patched |
| 4 | `SyntaxError: keyword argument repeated` | the kwarg was already there on a continuation line my one-line grep hid; `ast.parse` passed because repeated kwargs fail at `compile()`, not parse |

Three process failures underneath:

1. **I smoke-tested on one GPU and launched on four.** Single-GPU training tolerates unused
   parameters; DDP does not. Smoke tests must use the launch configuration.
2. **I asserted an edit happened without checking its effect.** "Patched" was printed while
   the parameter count stayed at 83.17 M. An edit that changes architecture must be
   verified by the number it is supposed to change, not by the string replacement.
3. **I reported "E5 is running" from the launch, not from the log.** It had died within
   seconds. The two-minute log check after every launch is now mandatory, and the loop
   prompt's "check whether you have reached your goal" starts with `pgrep`, not memory.

And the one that cost the most: the evaluator itself was wrong (section 8). Two of the
day's three "catastrophic" results were the harness. **Verify the instrument before
trusting the reading** -- the same weights must score the same under any evaluation path.

## 10. E5 result (2026-09-24 18:30): degenerate, my target bug

`cls 0.0000` for the final ~10k steps was not convergence. `center_targets` centred each
Gaussian on the fractional `(cx, cy)`, so no cell ever equalled exactly 1.0 and
`pos = hm_tgt.eq(1.0)` was empty: no positive term, heatmap pushed to zero everywhere,
**0 detections at every threshold** at step 114k. Box loss stayed nonzero (its mask is
set separately), which hid it. Fixed: Gaussian on the integer cell, offset in reg[0:2],
verified `positives == mask cells == in-range GT` on 20 real frames and the init loss
matches the analytic 4.5. Occupancy (9.8%) and map (64.6%) did improve in this run from
the extra 36k steps, so E5c warm-starts from its checkpoint with the center head re-initialised.

Rule added: **a loss printing exactly 0.0000 is a target bug until proven otherwise.** It
was in the log for hours.

## 11. Launch hygiene (2026-09-24 19:00)

E5c also needed three launches. Both crashes were `KeyError: 'opt'` (I dropped the
optimiser state from the warm-start checkpoint; the resume only caught `ValueError`), and
the second happened because my patch's `assert` failed **but the relaunch on the next
line ran anyway** -- a heredoc that exits non-zero does not stop the commands after it
unless they are chained with `&&`. Third time today a failed edit was followed by a launch
of the unfixed code (the `sed` delimiter failure and the overwritten decoder block were the
others). Rule: **an edit and the launch that depends on it go in one `&&` chain, and the
edit ends with a check of the number or string it was supposed to change.**

## 12. E5c early result (2026-09-24 19:40): first run to beat the baseline

At step 120k, 5.5k steps into the fixed center head (100 frames, NMS 2 m):

| | E0+NMS | E4 | **E5c @120k** |
|---|---|---|---|
| F1 | 37.7% | 33.0% | **45.5%** |
| precision | 37.4% | 44.0% | **65.8%** |
| recall | 38.1% | 26.4% | 34.8% |
| hallucinated | 54.8% | -- | **14.0%** |

The linear-probe bisection (AUC 0.844 -> "features carry the objects, the head is the
bottleneck") was right. The failure mode has inverted: the dense head is precise and
under-recalls. Recall is the lever now. Best F1 sat at the sweep floor of 0.2, so the
sweep is extended to 0.05 before the end-of-run evaluation. 34k steps remain.

Recall levers, in the order to try if the final number is short: threshold (free);
Gaussian radius / positive mass; class-balanced heatmap weights (rare classes are almost
certainly the missing recall -- check per-class); longer training; the depth supervision
already in the loss.

## 13. Where E5c's recall goes (2026-09-24 20:00, step 120k, 100 frames, thr 0.1)

Not class imbalance: every class misses in proportion to its GT share (class 0 is 56%
of GT and 52% of misses; rare classes 2/3 are ~29% recall but only 8% of misses).
It is RANGE:

    0-10 m 94.3%   10-20 m 75.6%   20-30 m 62.8%   30-40 m 47.3%   40-52 m 39.0%

Near-field is solved; recall falls ~linearly with distance. Monocular signature: a car
at 40 m is ~10 px wide at 896x512, one cell at stride 16. Class-balanced heatmap weights
are therefore a minor lever; the levers that matter are far-field evidence -- stride-8
features (FPN P3, 4x the lift-splat cells, affordable inside the speed budget), depth
bins (64 vs the teacher's 118, ~0.92 m/bin), and the depth supervision already in the
loss. Gate is relative to the teacher, so the teacher's own range profile (section 14)
decides how much of this gap is real.

## 14. Teacher's range profile, same 100 frames (2026-09-24 20:15)

    range    teacher  student@120k   gap
    0-10 m    97.0%     94.3%       -2.7
    10-20 m   91.1%     75.6%      -15.5   <- the gap
    20-30 m   74.2%     62.8%      -11.4   <- the gap
    30-40 m   56.3%     47.3%       -9.0
    40-52 m   33.3%     39.0%       +5.7   <- student ahead

Section 13's conclusion is withdrawn: the teacher collapses at range too and the student is
already ahead of it beyond 40 m, so stride-8 features / more depth bins would target the
band where there is no gap. The gap is 10-30 m: objects 20-40 px wide (not a resolution
problem), the densest band (1,338 of 2,640 GT), and class 4 over-represented in misses.
Suspect the DECODE first: a 2 m class-wise NMS radius suppresses true neighbours in
crowds, and thr 0.1 may sit above mid-range peaks. Sweep NMS {0.5,1,2} x thr {0.05,0.1,0.2}
on the existing checkpoint before any training. Lesson repeated: measure the reference on
the same data before naming the lever.

## 15. Decode sweep on E5c @120k (2026-09-24 20:30): the decoder is not the lever

    nms   thr    F1     recall  precision  ped(4)  10-30m
    0.5  0.05  17.3%   55.7%    10.2%     60.1%   77.1%
    2.0  0.20  45.6%   35.7%    63.4%     38.2%   61.8%

NMS radius 0.5 vs 2.0 moves F1 < 2 points: the crowd-suppression hypothesis from
section 14 is wrong. Threshold is a pure recall/precision trade -- at thr 0.05 recall is
55.7%, only 14 points behind the teacher's 70.1%, and 10-30 m recall reaches 77%, but
precision is 10%. So the mid-range evidence IS in the heatmap, at low confidence,
indistinguishable from noise. That is separation, i.e. training and head capacity.

E6 levers (both config-driven, defaults reproduce E5c so its evaluation still loads):
  * center_min_radius 1 -> 2: a pedestrian was a single positive cell in 40,000
  * heatmap head 64 x 2 blocks -> 128 x 3 blocks: ~0.1 ms of the ~46 ms budget
  * warm-start from E5c's final trunk, re-init the head, +40k steps
Class-balanced heatmap weights and stride-8 features are NOT taken: misses are
proportional to class share, and the far field is where the student already leads.

## 16. E6 queued (2026-09-24 20:50)

Config-driven center head (`center_hidden`, `center_blocks`, `center_min_radius`; defaults
reproduce E5c, verified 0 missing / 0 unexpected on its checkpoint). E6 = 128 x 3 blocks,
radius floor 2, warm-started from E5c's final trunk with the head re-initialised, +40k steps.
Speed bound for the E6 shape: 96.5 ms -> 10.4 Hz measured on a GPU shared with four-way
training -- clears the gate even pessimistically; clean number to be measured at E6's
export. `e6.sh` blocks on E5C_DONE, self-checks its own log at 3 minutes (step lines,
no traceback, cls not 0.0000) and aborts with E6_LAUNCH_FAILED otherwise.

## 17. Trajectory clause 2 was measured unfairly (2026-09-24 21:10)

The ego linear probe was fit and scored on its own 90/10 split of unrelated ego records
(1.469 m). Fit on non-eval records and scored on the SAME frames the student is scored on:

    constant velocity   2.006 m
    ego linear probe    1.566 m     <- the fair bar
    teacher             1.393 m     gate 1.532 m
    student E0          1.486 m     beats probe
    student E4          1.476 m     beats probe
    student E5c @120k   1.248 m     beats probe, inside the gate, ahead of the teacher

"Student loses to a linear probe", reported for two days, was the instrument. Trajectory
passes both clauses on matched frames; E5c's final 250-frame evaluation will confirm.
`diagnose.py` now fits the probe on non-eval records and scores it on the eval tokens.
Fourth instrument defect this project. Rule: **a baseline is only a baseline on the same
data as the thing it baselines.**

## 18. Strategic reset (2026-09-24 22:15) -> PLAN_V2.md

The user's objection stands: "< 10% of the teacher" made a ~68% F1 teacher the ceiling,
and even meeting it yields a model no one would ship. `PLAN_V2.md` restates the target in
the official nuScenes mAP/NDS against published camera-only R50 models (deliverable v1:
mAP >= 0.35, NDS >= 0.47 at >= 10 Hz), and lays out five phases: official metric harness,
training recipe (aug/CBGS/EMA/schedule), temporal fusion (the largest known gain, and the
fix for the measured 10-30 m recall gap), backbone/resolution funded by the 2x speed
budget, and occupancy/map on real ground truth. E5c/E6 continue only as the baseline v2
must beat. Nothing new starts until the four decisions in PLAN_V2 section 5 are made.

## 19. Target restated by the user; recipe implemented (2026-09-24 23:30)

The user's bar: **the student keeps >= 80% of the teacher**, i.e. F1 >= 0.8 x 67% =
**53.6%** on the same-frame metric (plus the 10 Hz gate). Best so far: E5c @120k at 68%
retained (45.5%). The v2 official-metric plan stands for "deliverable", but 80% retained
is the immediate objective.

Training recipe implemented in `train_student.py` behind flags, defaults off so nothing in
flight changes; each piece CPU-verified, then the whole set smoke-tested under 2-rank DDP:
  * `--cbgs`  class-balanced frame resampling (mmdet3d algorithm; 3.2x epoch, barrier 35->50%,
    generic 23->38% frame share). Measured: uniform sampling showed a barrier one frame in 3.
  * `--ema D` averaged weights incl. BatchNorm buffers; saved as `ema`, preferred by
    diagnose/export (`--no-ema` to compare). Verified: lags raw by exactly (1-D) per step.
  * `--cam-flip` mirror image + mirror `lidar2img` (u' = W - u) so scatter indices follow;
    verified 12/12 GT centres project to W - u. Requires feat_weight == 0.
E7 = E6 architecture + recipe, 20k steps on a fresh one-cycle schedule (step counter reset:
a continuation would have started deep in the annealing tail), warm from E6's trunk (E5c
fallback with head re-init), self-checked at 200 s. Queued behind E6.
Next lever if E7 falls short of 53.6%: temporal fusion (PLAN_V2 phase 2).

## 20. Temporal fusion built and verified; the prove-it chain is E6 -> E7 -> E8 (2026-09-24 23:00)

**Done this session.** `temporal.py` (ego-motion warp of the previous keyframe's BEV via one
`grid_sample`, `TemporalFuse` conv), `student.py` (`temporal` cfg flag, `bev_from()`, sixth
output `bev_state`, per-sample `bev_index`), `train_student.py` (`--temporal`, previous
frame loaded with the SAME camera flip, per-sample indices instead of sample 0's),
`diagnose.py` / `export_student.py` (temporal eval and stateful export). Verified: (a) per-
sample vs shared index gives a bit-identical BEV; (b) identity warp reconstructs the BEV to
1e-6; (c) stateful ONNX exports with inputs `image, bev_index, valid, ego, prev_bev,
warp_grid` and outputs `cls, box, occ, seg, trajectory, bev_state`, 602 nodes, GridSample
in-graph; (d) a 30-step 2-GPU DDP smoke trains, evaluates and exports end to end.

**Bug found by the smoke, not by reading.** The first 2-GPU smoke failed with DDP's
"Expected to have finished reduction": every conv weight in the backbone and lift-splat had
received no gradient. Cause: the no_grad `bev_from` pre-pass ran first inside the training
autocast region, so autocast's cached bf16 weights carried no autograd edge and the real
forward reused them. A single-process run had shown NO error -- it would have trained 20k
steps with a frozen backbone. Fix: the pre-pass gets its own autocast context, closed
before the main one opens. Rule added to memory: smoke-test new training paths under DDP
with `find_unused_parameters=False`; it is the only free check that every parameter gets a
gradient.

**Bug fixed on the way.** The training loop passed `b["bev_index"][0]` -- sample 0's
scatter index -- for the whole batch. Calibration differs per frame (and camera flip
changes it per sample), so every batch trained samples 1..3 against slightly wrong
projections. E5c/E6 trained with this; E7 and E8 do not.

**Chain and gates (unchanged).** E5c finishes ~23:10 -> E6 (wider head) ~01:40 -> E7 (E6 +
CBGS, EMA, flip) ~04:30 -> E8 (E7 recipe + temporal, warm from the better of E6/E7) ~08:30,
each followed by its evaluation. G1: E7 >= E6 + 4 F1. G2: E8 >= seed + 5 F1 and the 10-30 m
recall gap to the teacher < 8 points. Bar: F1 >= 53.6% (80% of the teacher's 67%). If both
gates fail, stop and report; if the bar is met, one long run with the winning recipe, export,
and a clean single-GPU timing (temporal fuse adds ~2.7 M params, ~1 ms).

## 21. The split leaks scenes; the deliverable moves to the official scene split (2026-09-24 23:00)

**Found.** `train_student.py` holds out the first 10% of *sorted tokens*, i.e. random
frames, not scenes. The cache holds 30 of the 40 keyframes of every one of the 850 scenes,
so of the 250 eval frames' 500 neighbouring keyframes (0.5 s away), 324 are in the training
set. Static objects (barriers, cones, parked cars) look the same 0.5 s later; the student
is partly scored on scenes it has memorised, while the teacher never trained on nuScenes.
Every "retained %" so far (E5c 68%) is optimistic by an unknown amount.

**Decided.** `data/distill/scene_split.json` holds the official split restricted to the
cache: 21,147 train frames (700 scenes) / 4,452 val frames (150 scenes), plus `eval250`,
250 frames spread over 143 val scenes. `train_student.py --split scene` and
`diagnose.py --tokens scene` use it; defaults are unchanged so the running chain is not
disturbed. The chain (E6/E7/E8) stays on the frame split: its gates are *deltas* between
runs that share the same leak, so they still tell direction. The **final run trains from
scratch on the train scenes only** (a warm start from any chain checkpoint would carry the
val scenes with it) and is reported on the val scenes; its eval on the held-out *frames*
of train scenes (the 6,983 uncached ones have images and GT) measures the inflation.

**Also decided.** The center objective uses GT heatmaps only (`det = l_cls + l_box`; the
teacher's 7-class logits never enter the loss), so the final run's head switches to the 10
nuScenes classes at no cost beyond the label file and the class-group mapping needed to
keep the teacher (7 classes) comparable in F1. Official mAP/NDS then apply cleanly.

## 22. Official yardstick in place; final-run design fixed (2026-09-24 23:25)

**Built and tested tonight (CPU side, while the chain trains).**
- `eval_official.py`: the nuScenes `DetectionEval` (mAP / NDS, per-class AP, TP errors) on
  the official val split, for the student (all 6,019 val keyframes -- uncached frames get
  their calibration from `calib.npz`, verified against the teacher's lidar2img) or the
  cached teacher (4,452). Test on 80 frames: teacher mAP 0.165, E5c 0.208 (its training
  frames; 7-class mapping gives truck/bus/trailer/motorcycle 0 AP by construction).
- `gt_boxes10.npz` for all 34,149 keyframes: the 10 official classes, `num_pts` (the
  evaluator drops GT with no lidar/radar return -- 19% of boxes on a 12-frame sample),
  visibility, ego-frame velocity, and the box yaw in the EGO frame. **The 7-class file
  stores the global yaw next to an ego-frame position**, so every chain model learned a
  meaningless yaw (E5c orient_err 1.05 rad vs teacher 0.82). Only sin/cos regression saw it;
  F1 and mAP (centre distance) are unaffected. Fixed in the 10-class file only.
- `train_student.py --classes nuscenes10 --split scene --velocity`, `diagnose.py`
  (auto-detects the label space; teacher and student compared in the 7 teacher groups so
  the F1 stays comparable), `temporal_index.json` for all keyframes (0.5 s links),
  `run_stateful.py` (deployment loop: one backbone pass per frame, `bev_state` fed back;
  proven equal to the two-frame training path on random weights; ORT timing on one GPU).
- `bev_state` fixed to the UNFUSED lift-splat BEV. As exported before tonight it was the
  fused BEV, i.e. a recurrent state over all history the model never trained on.

**Final run, when a gate passes:** from scratch (ImageNet init) on the 700 train scenes,
`--classes nuscenes10 --split scene --velocity` plus the winning recipe (E7 flags, plus
`--temporal` if G2 passes), ~80k steps (~9 h on 4 GPUs), reported as official mAP / NDS on
the 150 val scenes plus the grouped F1 against the teacher on the same frames, ONNX from
`export_student.py`, speed from `run_stateful.py` on an idle GPU.

## 23. E5c final: F1 55.2%, 80.9% of the teacher -- on the leaked split (2026-09-25 00:05)

E5c at its last step (154,500; raw weights, it predates EMA): recall 44.3%, precision
73.0%, F1 55.2% at thr 0.3 (NMS 2 m: 54.0%). Teacher 68.2% on the same 250 frames, so
80.9% retained -- the user's 80% bar, met by 0.6 points, **on the frame split that leaves
every eval frame's neighbours in training (section 21)**. It is a milestone for the
direction, not a deliverable number.

What moved it: 45.5% at 120k -> 55.2% at 154.5k. The last 34k steps, the low-LR tail of
the one-cycle schedule, were worth +10 F1. Lesson for the final run: one uninterrupted
schedule over the whole budget, no warm restarts, and a step count sized for it (~100k).

Where the gap is (recall, student vs teacher):
- by range: 0-10 m 83 vs 95, 10-20 m 70 vs 87, 20-30 m 54 vs 74, 30-40 m 38 vs 56,
  40+ m 14 vs 19. Not a mid-range dip but a ~12-19 point deficit everywhere inside 40 m.
- by class: vehicle 47 vs 56, bicycle 35 vs 55, generic object 26 vs 48, **pedestrian 29
  vs 59**, cone 57 vs 68, barrier 58 vs 64. Pedestrians are half the miss: ~0.7 m objects
  at stride 16 on 896x512 and 0.512 m BEV cells. E6/E7 already floor the heatmap radius at
  2 cells for them; if E7 does not close much of this, the final recipe should spend speed
  budget on resolution (the trunk at 1408x768 or a 0.4 m BEV), not on depth of the head.

Occupancy 9.4% mIoU (occupied classes) and map 64.5% are unchanged from mid-run;
trajectory 1.470 m beats the ego-probe 1.566 and the teacher's 1.393 is 5% away.

E6 launched 23:58 from this trunk (head re-initialised at 128x3). Chain and gates as in
section 20.

**00:20 addendum.** E6 as first launched continued E5c's step counter (154.5k of a 194.5k
one-cycle), i.e. a fresh 128x3 head trained at LR ~4e-5 decaying to zero for 40k steps
(5.3 h): slow, and a soft baseline for G1. Relaunched at 00:18 on the protocol E7/E8 use
-- step reset to 0, a fresh 20k one-cycle at 3e-4 -- so every chain delta is measured under
one schedule. Cost: 20 minutes. Chain ETA: E6 ~02:50, E7 ~05:40, E8 ~09:00.

**Teacher, official metric** (`eval_official.py --who teacher`, 4,452 cached val
keyframes, 7 classes mapped so truck/bus/trailer/construction/motorcycle score 0 by
construction): mAP 0.216, NDS 0.222; car 0.48, pedestrian 0.48, cone 0.58, barrier 0.50,
bicycle 0.12; ATE 0.78 m, AOE 0.73 rad. This is the 1 Hz teacher's own standing on the
public yardstick; the student is not bound by it.

**Final run queued (00:06), unattended policy.** `final.sh` waits for E8_DONE, then trains
from scratch on the 700 train scenes: 10 classes, velocity, E7 recipe, `--temporal` iff E8
beat its seed by >= 2 F1 (100k steps without, 80k with; ~13-14 h), then: grouped F1 vs the
teacher on 250 val-scene frames, the same model on held-out frames of its *train* scenes
(the size of the leak), official mAP/NDS on all 6,019 val keyframes and on the cached
4,452 (like-for-like with the teacher), ONNX export, deployment-loop equivalence check and
single-GPU timing. A from-scratch scene-split model is needed for any honest number, so
it runs even if both chain gates fail; the gates decide only what goes into it.

## 24. E6: 58.3% F1 (+3.1 over E5c) -- the wider head pays (2026-09-25 02:50)

E6 (128x3 head, min radius 2, fresh 20k one-cycle from the E5c trunk): recall 50.3%,
precision 69.4%, F1 58.3% at thr 0.3 (NMS 2 m 57.0%); 85.5% of the teacher on the leaked
split. Per class: vehicle 51.1 (E5c 47.1), bicycle 39.2 (35.2), generic 24.6 (26.2),
pedestrian 34.4 (29.2), cone 53.5 (56.9), barrier 54.4 (57.5). By range 84.6 / 73.3 /
61.4 / 49.4 / 18.8: the gain is at 20-40 m (+7, +12). Occupancy 10.4%, map 68.1%,
trajectory 1.458 m -- all slightly up. E7 (recipe) launched 02:47 from this checkpoint.

**03:00 addendum -- E7 relaunched with a fixed flip.** E7's depth loss opened at 3.65 where
E6 had opened at 1.28 from the same weights: `--cam-flip` mirrored the images and the
projection but not the per-pixel lidar depth target, so half of every batch trained the
depth head on the mirror of its target. Fixed in the loader (`depth[:, :, ::-1]`, verified
on a real frame), E7 killed at step ~400 and relaunched at 02:58 from the E6 checkpoint.
Cost ~12 minutes; the chain ETA moves to E7 ~05:50, E8 ~09:15.

## 25. E7: 59.2% F1 (+0.9 over E6) -- gate G1 not met; the instrument is biased against it (2026-09-25 05:50)

E7 (E6 + CBGS + EMA + camera flip with the depth target fixed, 20k fresh one-cycle, EMA
weights): recall 48.4%, precision 76.2%, F1 59.2% at thr 0.4; 86.8% of the teacher on the
leaked split. G1 asked for >= +4 over E6 (58.3%); it got +0.9, below even the +1.5 kill line
of PLAN_V2 Phase 1. Per class the recipe moved what CBGS targets: pedestrian 37.4 (E6 34.4),
cone 57.7 (53.5), barrier 61.2 (54.4), generic 29.6 (24.6); vehicle 50.2 (51.1) and bicycle
38.1 (39.2) gave a little back. Range profile unchanged (84.8 / 73.1 / 59.1 / 45.0 / 16.9).

Reading it honestly: regularisers (augmentation, class balancing, EMA) buy generalisation,
and the frame-split eval rewards the opposite -- memorising scenes whose other frames are
in training (section 21). On this instrument a recipe that helps on unseen scenes can look
flat, so G1's verdict is "not demonstrated here", not "useless"; the recipe stays in the
final run because it is cheap, standard, and the only eval that can show its value is the
scene-split one the final run gets. What G1 does establish: the *chain* models will not
reach the 80% bar by recipe alone on an honest split; the wider head (+3.1) was the real
gain so far, and temporal fusion (E8, launched 05:44 from E7) is the remaining lever
before the from-scratch final run.

**06:05 addendum -- E8 launch failed, final run held back.** E8 started at 05:44 and printed
nothing for 300 s (no CBGS line, no step), so its self-check killed it; four rank processes
survived the kill and held the GPUs until found with a /proc scan. `final.sh`, which was
waiting on E8's marker, would have launched without temporal on that failure -- killed
first. Cause: the first log line comes after CBGS reads 23,040 GT files from NFS; E7 got
them warm, E8 got them cold. Fix: `data/distill/gt_labels_index.json` (one read for all
frames, both GT files), a 600 s check window, and the stale marker log moved aside before
re-queuing the final run. E8 relaunches from E7 (59.2%) as soon as the index is built.

**06:20 addendum -- second E8 failure, same family.** With CBGS fast, the ranks then sat in
`DistillSet.__init__`, which stat()s every token's frame directory: 74k calls per rank for
the resampled list, x4 ranks, on an NFS mount that my own index build was hammering at the
time (7 ms per stat measured afterwards, 170 s for the 23k unique tokens). Fixed to one
stat per unique token; third launch at 06:12 reached the model print at 06:15. Rule for the
scripts: nothing else touches NFS during a launch window, and the window is 600 s.

**06:25 -- E8 running.** Third launch passed its check at 06:20 (cls 2.54 -> 1.00 over the
first 200 steps, 51.75 M params, 1.33 it/s: the two-frame loader costs ~35% throughput).
ETA ~10:50 with its evaluation. `final.sh` re-queued behind it (600 s launch window).

## 26. E8: temporal fusion is the lever -- 64.2% F1, +5.0, G2 passed (2026-09-25 10:30)

E8 (E7 recipe + previous-keyframe BEV warped by ego motion, fused before the BEV encoder;
20k fresh one-cycle from E7, EMA weights): recall 58.5%, precision 71.2%, **F1 64.2%** at
thr 0.3 (NMS 2 m: 63.0%) -- 94.1% of the teacher's 68.2% on the leaked frame split.
Gate G2 asked for >= +5 over the seed and a 10-30 m gap under 8 points: +5.0 and 5.5.

Where it came from, recall by range (E7 -> E8, teacher): 0-10 m 84.8 -> 88.7 (95.1),
10-20 m 73.1 -> 81.5 (87.1), 20-30 m 59.1 -> 70.7 (73.5), 30-40 m 45.0 -> 62.1 (56.4),
40+ m 16.9 -> 24.9 (18.9). Motion parallax bought +8 to +17 points from 10 m out, and
beyond 30 m the student now out-recalls the 4 B teacher. By class the student is at or
above the teacher on vehicle (57.1 vs 55.7), cone (68.8 vs 67.9) and barrier (74.9 vs
64.3); pedestrian 52.7 vs 58.6 and bicycle 50.0 vs 55.1 remain the deficits. Occupancy
10.9%, map 70.8%, trajectory 1.460 m (teacher 1.393).

Chain summary on one instrument (leaked split, teacher 68.2%): E5c 55.2 -> wider head
58.3 -> recipe 59.2 -> temporal 64.2. Two of three levers paid; the deliverable question
is how much of the 64.2 survives on scenes the model has never seen.

**Final run launched 10:27** (`final.sh`): from scratch on the 700 train scenes, 10 nuScenes
classes, velocity, E7 recipe, temporal (gain +0.050 >= +0.02), 80k steps at ~1.33 it/s
(~17 h, ETA ~03:30 + ~1 h of evaluation, export and timing). Everything after it is
automatic: grouped F1 vs the teacher on 250 val-scene frames, the same model on held-out
frames of its train scenes (the leak's size), official mAP/NDS on all 6,019 val keyframes
and on the cached 4,452, ONNX export, deployment-loop equivalence, single-GPU timing.

**10:45 addendum -- E8 artefacts.** `outputs/onnx/student_e8/student.onnx` (602 nodes,
212 MB, EMA weights, stateful interface) exported as the fallback deliverable; summed
kernel time ~52 ms (wall-clock 127 ms only because the final run saturates every GPU;
clean timing comes after it). `run_stateful.py --check` with E8's trained weights:
recurrent loop vs the two-frame training path over 12 keyframes of scene-0003 -- fed state
vs recomputed previous BEV <= 1.5e-4, heatmap logits <= 5.9e-3 (cuDNN algorithm noise).
The first version of the check compared decoded top-k lists elementwise and reported a
"mismatch" of 100 m: a last-bit difference reorders near-ties in the top-k. Compare dense
tensors, never sorted lists.

## 27. Final run, mid-training snapshot: 56.4% F1 on scenes it has never seen (2026-09-25 14:20)

Checkpoint at step 19,000 of 80,000 (24% of the one-cycle, LR still near its peak; EMA
weights), 100 frames from the 150 held-out val scenes, GT in the 10 official classes with
the no-return boxes dropped: student recall 46.8%, precision 70.9%, **F1 56.4%**; the
teacher on the same frames 63.1%, so **89.4% retained on an honest split**, above the
80% bar (50.5%) already. By range 80.8 / 72.6 / 56.2 / 32.9 / 5.1 vs teacher 90.0 / 82.7
/ 68.3 / 45.5 / 14.0; by class the deficits are pedestrian (32.8 vs 53.2), trailer,
motorcycle and far range -- the same shape as the chain at a comparable point, so the
remaining 61k steps and the schedule tail should close part of it (E5c gained +10 in its
tail). Occupancy 6.4%, map 55.2% (still learning from scratch), trajectory 1.313 m.
Not a result yet -- the run ends ~02:15 -- but the leak question has a first answer: the
honest number is not collapsing relative to the chain's leaked 64.2%.

## 28. Moved to the 8-GPU machine (8x H200) -- the final run stopped at step 36,000 (2026-09-25 17:35)

At the user's request the 4-GPU machine was vacated: `final.sh` was stopped right after its
step-36,000 checkpoint (model, EMA, optimiser, step; nothing lost), the FINAL watcher and
this session's 15-minute loop were cancelled, and all four GPUs are idle. The run resumes
on the 8-GPU machine with `local/distill/scripts/final_resume_8gpu.sh` (4 ranks keep the effective
batch of 16; 44k steps at H200 speed ~5-6 h; then the same evaluation, export and timing
as before). `local/distill/HANDOFF_8GPU.md` is the entry point for the 8-GPU session:
bring-up checklist, the resume command, shared-disk hygiene, and the next experiment
(input resolution for pedestrians and far range) for the four spare H200s.

## 28. The A100 run was killed at step 36k; resumed on the 8-GPU machine (2026-09-25 17:50)

The user moved to the 8-GPU machine (8x H200). When the A100 session was closed, the final run's launcher died at
step 36,050 (17:31; NCCL "TCPStore shut down" warnings, no TRAIN_RC line, GPUs on the 4-GPU machine
empty, no surviving process). Its checkpoint at step 36,000 (model, EMA, optimiser, step)
resumes cleanly: `train_student.py` fast-forwards the one-cycle, so the schedule is the
same 80k; only the EMA restarts from the raw weights (decay 0.999 forgets that in ~3k of
the remaining 44k steps). `local/distill/scripts/final_resume_8gpu.sh`: identical recipe,
4 ranks x batch 4 on GPUs 0-3 (same per-GPU BatchNorm batch), caches copied to the 8-GPU machine's
local disk (`/local/$USER/distill`) after NFS latency cost two launches
yesterday, 12 loader workers, then the same evaluation/export/timing steps as final.sh.
8-GPU machine bring-up: venv works as is (torch 2.8 cu128 sees 8x H200, ORT GPU), 2-GPU smoke
of the full recipe passed in 0.3 min. GPUs 4-7 are free for the next experiment.

## 29. 8-GPU machine: the resume is running; resolution experiment prepared (2026-09-25 18:05)

**Resume.** The 4-rank resume of the final run (GPUs 0-3, port 29641) has been training since
17:50 from the step-36k checkpoint, saving every 500 steps to `outputs/distill/exp/final`.
A second launch (8 ranks) collided with it on the port and truncated the shared log; both
driver scripts were killed so that neither could kill the training, which runs on under
torchrun pid 2689958. `scripts/final_post_8gpu.sh` blocks on that pid and then runs the
evaluation, official metrics, export, loop check and timing (log `final_post_8gpu.log`).
Rate 1.85 it/s (GPUs ~85% busy; images are symlinks into /perception_data), so training
ends ~00:30 and the numbers land ~01:30. If it dies, rerun `final_resume_8gpu.sh` (4 ranks;
the 8-rank variant needs a free port and an untouched log).

**Resolution experiment R1, on GPUs 4-7.** Same final recipe at 1152x640 (72x40 feature
grid instead of 56x32; 1.6x the trunk cost) -- the one lever aimed at the pedestrian and
30 m+ deficit that every run has shown. Built: `nusc_depth.py --out-name` (depth targets at
the new size, `depth_1152x640.npz` beside each frame, 8 CPU shards running); the loader
rescales the teacher's 896x512 lidar2img and picks the matching depth file when
`--image-size` differs from the default (default path verified bit-identical);
`export_student.py --image-size` times the untrained shape. Speed is measured on both an
idle A100 (the claim's GPU) and an H200 before anything is launched; real images are being
copied to `/local/$USER/distill/frames_real` so the heavier decode does not stall on NFS.

## 30. New target: 95% of the teacher, 10 Hz, 3 days on 8 H200 -- unattended orchestrator (2026-09-25 18:15)

The user raised the bar from 80% to **95% of the teacher's F1** (held-out val scenes) with the
ONNX above 10 Hz, gave 3 days on the 8-GPU machine (8x H200; the 4-GPU machine is no longer ours -- every
timing and speed gate now runs on H200), and asked for one unattended orchestrator.

**Instruments in place.** `scripts/lane.sh` runs one experiment end to end: H200 speed gate
on the untrained shape (skip below 10 Hz), 2-GPU smoke, launch with a 600 s self-check,
resume from its own checkpoint if relaunched, then grouped F1 on 250 val-scene frames,
official mAP/NDS, ONNX export with H200 timing, and the deployment-loop check.
`scripts/orchestrate_95.py` (nohup, log `outputs/logs/orchestrate_95.log`) runs rounds,
retries failures up to 3x, resumes runs that die early, and rewrites `local/distill/REPORT.md`
and `outputs/onnx/deliverable/` after every finished lane. `scripts/orchestrate_lanes.json`
is re-read every 5 min, so lanes not yet launched can be changed while it runs.

**Plan.** Round 0 (running): FINAL 896x512 on GPUs 0-3 (ends ~23:30) and R1 1152x640 on
GPUs 4-7 (~08:30 tomorrow). Resolution decision: R1 if it beats FINAL by >= 1 F1.
Round 1: `r2_long` (120k steps, starts on GPUs 0-3 as soon as FINAL is done) and
`r2_arch101` (ResNet-101 at the chosen size on GPUs 4-7 after R1; falls back to 896x512 if
the 1152 shape fails the 10 Hz gate). Round 2: one 8-GPU run combining whatever beat the
round-0 best by >= 1 F1, steps sized to the deadline. Candidates not yet coded, to add to
the config if ready in time: 4-keyframe temporal history, Occ3D occupancy labels (download
approved), feature distillation (the ViT cache covers only 10k frames at 256-d; not viable).

**Mistakes of the first hour, fixed.** (1) The orchestrator's liveness check looked for a
script name at the END of the command line; `final_post_8gpu.sh <pid>` ends with the pid, so
it declared the final run's driver dead and launched a duplicate resume on the same GPUs and
checkpoint directory -- caught within two minutes, duplicate killed, no checkpoint written by
it (mtime unchanged), training intact at step 38,950. (2) Two more self-kills of my own tool
shell through pattern matches on text present in the shell's heredoc. Rule now in memory:
exclude the scanner's ancestry and tool shells; match torchrun as a substring.

## 31. K-frame temporal history implemented; plan uses both GPU halves at all times (2026-09-25 18:40)

`--history K` fuses the K previous keyframes (0.5 s, 1.0 s, 1.5 s back), each warped by
its own ego motion, concatenated with the current BEV (fuse conv (K+1)C -> C; +2.7 M params
for K=3). Loader walks the index chain and pads a missing older frame by repeating the last
available one (no history: the current frame with an identity warp); `run_stateful.py`
keeps a deque of the last K unfused states and pads identically, so deployment reproduces
training; ONNX inputs become `prev_bev (1,K,C,H,W)`, `warp_grid (1,K,H,W,2)`. K=1 is
bit-identical to before (verified: item tensors and fuse channels). Training cost: K
no-grad backbone passes per step (K=3 ~ 1.7x the K=1 step).

Plan update (`orchestrate_lanes.json`): round 1 = `r2_long` (120k steps, GPUs 0-3 after
FINAL) and `r2_hist3` (K=3, 60k steps, GPUs 4-7 after R1); backlog `r2_arch101` runs on
whichever half frees up first; round 2 takes whatever half (or all 8) is free once round 1
is done. The orchestrator was restarted with these rules (lanes launched so far: none).

## 32. Occupancy gets ground truth: Occ3D-nuScenes via UniOcc (2026-09-25 19:30)

Occupancy has been the weakest output (~11% mIoU *agreement with the teacher*) and had no
ground truth. The Occ3D-nuScenes labels are behind a permission-gated Google Drive, but the
UniOcc benchmark repackages them on Hugging Face (`tasl-lab/uniocc`, NuScenes-via-Occ3D-2Hz):
per keyframe `occ_label` (200,200,16) at 0.4 m over [-40,40]^2 x [-1,5.4] m, camera mask,
sample token. The val split (6,019 frames, 9.3 GB) is down; the train split is fetching
slowly (per-IP rate limit; unauthenticated).

UniOcc's label ids are its own and undocumented. Established empirically: the array is
(x, y, z) -- under the transpose 3.1% of occupied voxels fall inside our GT boxes versus
<1.5% for every other orientation -- and ids 1/2/4/5 sit inside car/bicycle/pedestrian/
cone boxes at 96/89/88/83%; id 7 is the road (ground slices), 8 other flat ground, 6 and 9
tall structure/vegetation, 0 rare 'others', 10 free. `occ3d_labels.py` writes
`occ3d.npz` per frame in the student grid (0.512 m, +-51.2 m, 16 x 0.4 m pillars from
-1 m) in the teacher's 10 classes with a validity mask (inside range and camera-visible).
Val done. On the road voxels the *teacher* predicts 'driveable' for only 13% and 'empty'
for 74%: its occupancy is poor on this yardstick, so a student trained on Occ3D labels
should clear 95% of it easily -- the number to report is both models vs GT.

Built: `diagnose.py` now reports occupancy mIoU vs Occ3D GT for student and teacher (and
the retained %) whenever a frame has `occ3d.npz`; `train_student.py --occ-gt occ3d` trains
the occupancy head on the labels (masked cross-entropy, teacher target where a frame has
none). A lane with `--occ-gt occ3d` goes into the plan once the train-split labels land.

## 33. Equal-step read: 1152x640 is +0.8 F1 over 896x512 at step 19k (2026-09-25 22:00)

Same 100 unseen val-scene frames, same step (19k of 80k), same recipe: R1 (1152x640) F1
57.2% / 90.7% retained vs the final run (896x512) 56.4% / 89.4%. Recall is up at every
range (84.0/74.8/62.5/37.0/7.0 vs 80.8/72.6/56.2/32.9/5.1) and on pedestrians (37.0 vs
32.8), precision a little down (66.3 vs 70.9). A real but modest gain at a quarter of the
schedule; the 80k comparison (~09:00) decides the resolution for round 2. Cost: 1152 trains
at 1.48 it/s vs ~2.4 for 896 and runs at 16.9 Hz (H200) vs 21.3. Occupancy vs Occ3D GT for
this teacher-distilled head: 5.5% vs teacher 3.5% -- both useless; the Occ3D-trained lanes
start at midnight.

## 34. The honest number: 92.3% of the teacher on unseen scenes; the leak was ~10 points (2026-09-25 23:20)

FINAL (896x512, temporal K=1, 10 classes, velocity, recipe, 80k steps from scratch on the
700 train scenes), 250 frames from the 150 held-out val scenes, GT with the official
no-return filter, teacher and student matched in the 7 teacher groups:

| | recall | precision | F1 | retained |
|---|---|---|---|---|
| teacher | 54.7% | 71.5% | 62.0% | -- |
| student @thr 0.4 | 47.2% | 72.7% | **57.2%** | **92.3%** |
| student + NMS 2 m | 48.1% | 66.1% | 55.7% | 89.8% |

By range 84.9 / 76.0 / 59.0 / 34.7 / 9.6 vs teacher 92.1 / 84.0 / 67.9 / 44.7 / 14.0. By
class the deficit is concentrated: pedestrian 38.2 vs 51.0, bicycle 27.4 vs 41.7, car 53.7
vs 61.1, truck 38.3 vs 45.9, cone 55.7 vs 68.0; barrier already above the teacher.
Trajectory ADE 1.540 m (constant-velocity anchor 2.26 m on these frames -- a more dynamic
set than the 250 sorted frames where the teacher measured 1.393). Occupancy vs Occ3D GT:
student 5.0% mIoU, teacher 3.2% -- both useless; the first Occ3D-trained lanes start now.

**The leak, measured.** The same model on 250 held-out *frames of its train scenes*: F1
79.2% vs teacher 71.9% (110% retained). The teacher's own F1 is 10 points higher there
(easier frames), the student's 22 points: ~12 points of scene memorisation. Every chain
number in sections 19-26 (E8 "94%") carried that inflation; the honest bar of 95% needs
58.9% F1 on the val scenes, 1.7 points above where a single 80k run at 896x512 lands.

**Levers now running for the last 1.7 points**: R1 1152x640 (+0.8 at equal steps, +3-6
recall at every range, finishes ~09:00), `r2_long` (1152, 120k steps, launching at
midnight with Occ3D occupancy labels), `r2_hist3` (3-keyframe history, after R1), late
checkpoint averaging on every new lane, ResNet-101 and 1408x768 in the backlog, and a final
8-GPU run combining what helped. 

**TODO (trajectory reference).** The teacher's ADE of 1.393 m was measured on the 250
sorted (frame-split) frames; on the 250 val-scene frames the student reads 1.540 m against a
constant-velocity anchor of 2.26 m (vs 2.006 there), i.e. a more dynamic set. For a
like-for-like trajectory "retained %" the teacher's planner (`QwenDriveForPlanning`,
`local/run_planning_demo.py`) must be run on the same 250 frames from the `data/distill/ego`
state + images; ~1 h of wiring, one GPU for ~10 min. Do it in the next idle window.

**23:35 addendum -- official metric and speed for FINAL.** nuScenes DetectionEval on all
6,019 val keyframes: **mAP 0.3045, NDS 0.3941** (cached 4,452 subset: 0.3084 / 0.3947); per
class car 0.52, pedestrian 0.35, cone 0.51, barrier 0.55, bus 0.27, motorcycle 0.25, truck
0.23, bicycle 0.23, trailer 0.08, construction 0.06; ATE 0.73 m, ASE 0.29, AOE 0.72 rad, AVE
0.59 m/s, AAE 0.25. That sits between BEVDet-R50 (~0.30 / 0.38) and BEVDet4D-R50 (~0.32 /
0.46) in the public table -- with the same backbone, one previous frame, and a 10 Hz budget.
The teacher, mapped onto the 10-class metric, scored 0.216 / 0.222 on the cached subset: on
the official yardstick the student is well past it (the teacher cannot score five classes).
Speed on one H200: 44.2 ms -> **22.6 Hz** for the graph (608 nodes, all five outputs); the
deployment loop measured 66.9 ms -> 14.9 Hz because the 1x384x200x200 state (61 MB) is copied
host->device and device->host every frame -- fixable with ORT IO binding (keep the state on
the device); the loop reproduces the training path to 5e-3.

**23:40 -- deployment loop at 20.1 Hz.** `run_stateful.py` now binds the ONNX inputs and
outputs on the device (ORT IO binding) and feeds the previous frame's `bev_state` OrtValue
straight back as `prev_bev`, so the 61 MB state never crosses PCIe: FINAL's loop went from
66.9 ms (14.9 Hz) to 49.8 ms (**20.1 Hz**) on one H200 with all five outputs and the host-side
warp included. Round 1 has begun: `r2_long` (1152x640, 120k steps, Occ3D occupancy labels,
snapshot averaging) launched on GPUs 0-3 at 23:34; R1 continues on 4-7 (step 28k).

## 35. Trajectory, like for like: student 1.54 m vs teacher 1.63 m on unseen scenes (2026-09-26 00:10)

`teacher_traj.py` runs the teacher's SFT planner (direct mode, scenes built exactly as the
session demo builds them) on the student's 250 val-scene frames. On the frames that have a
full 5 s future -- the only ones the student's trajectory metric counts -- the teacher's ADE
is **1.629 m** (161 frames, constant-velocity anchor 2.105 m there); the student's is
**1.540 m** (184 frames, anchor 2.26 m). The student's trajectory is at least the teacher's
on scenes it never saw: the trajectory half of "95% of the teacher" is met with margin.
(Frames whose future is clamped at a scene end inflate everything -- the first pass read
3.25 m for the teacher over 227 frames for that reason; the ego cache's 5 s rule is the
right filter. A cross-check of the same script on the old 250 sorted frames against the
recorded 1.393 m is running.)

**00:20 -- trajectory cross-check.** The same script on the old 250 sorted frames (5 s
future only, 175 frames) reads 1.511 m for the teacher against the 1.393 m recorded in
September's orchestrator (that figure's frame filter / sampling mode is not recorded). Within
one consistent protocol: teacher 1.511 (sorted) / 1.629 (val scenes); student 1.470 (E5c,
sorted, leaked) / 1.540 (FINAL, val scenes). Either way the student's trajectory is within
5% of the teacher's or better; the report uses the val-scene pair 1.540 vs 1.629.

## 36. Bug: the 1152x640 depth targets were binned in 896x512 pixel space (2026-09-26 02:05)

`nusc_depth.py` projected lidar with the teacher's `lidar2img` (which has the 896x512 resize
folded in) and binned `u / stride` into a grid sized by `--image-size` without rescaling
the pixels. For 1152x640 every depth target sat at 0.78x its true image position -- R1
(since 18:20) and `r2_long` (since 23:34) have been learning depth against a target that
is right only near the top-left corner. R1 still led 896x512 by +0.8 F1 at equal steps, so
the true resolution gain is larger than measured. The 896 targets were never wrong.

Fix: rows 0-1 of lidar2img scale with the image; frames without a teacher cache get the
matrix from `calib.npz` (so depth supervision now covers all 34,149 keyframes, not 25,599);
writes are atomic (temp + rename) so files can be replaced under a running loader. The
1152 and 1408 targets are being regenerated and synced into the local copies in place: the
loaders read depth per sample, so R1 (42k/80k) and r2_long (11k/120k) switch to correct
targets on their next pass over each frame without a restart; their first steps carry the
misaligned signal. Lesson (memory): a resolution flag has to be traced through every
consumer of the geometry -- I checked the loader and the exporter, not the extractor.

## 37. More data and the corrected depth targets in place (2026-09-26 02:40)

- The corrected 1152x640 / 1408x768 depth targets (and 896 for the teacher-less frames) are
  regenerated for all 34,149 keyframes and synced into the local copies (02:20). R1 and
  `r2_long` now read them per sample; their first 14 h / 2.5 h were on the misaligned ones.
- The 8,550 "uncached" keyframes are exactly each scene's last ~10 keyframes: the teacher
  cache (and the ego cache) were built for frames with a full 5 s future, so these have no
  trajectory target but do have images, GT boxes, Occ3D occupancy and now depth. 6,983 of
  them are in train scenes. `--split scene-all` adds them (28,130 train frames, +33%); the
  map loss is masked where there is no teacher target and the occupancy loss where there
  is neither Occ3D nor teacher (`has_seg` / `has_occ`, weighted-CE normalisation proven equal
  to the unmasked loss when every sample is real). Smoke-tested; part of the lane recipe.
- Per-class score thresholds calibrated on held-out train-scene frames
  (`diagnose.py --write-calib / --calib-from`): for FINAL they coincide with the global 0.4
  (57.2% either way) -- no free points there, but the machinery is in for later models.
- Trajectory: the val-scene set has 66 frames near scene ends without a 5 s future; the
  metric excludes them for both models (student 184 frames, teacher 161).

**02:15 -- r2_long restarted under the corrected recipe.** Killed at step ~14k (12%): its
first 2.5 h were on the misaligned 1152 depth targets and it lacked the extra 6,983 frames.
Checkpoint dir cleared; the orchestrator relaunches it (try 2) with correct depth from step
0, `--split scene-all`, Occ3D occupancy and snapshot averaging. Cost 2.7 GPU-hours on one
half; round 2 still waits on the history lane, so nothing downstream moves. R1 keeps
running (45k/80k): it is the resolution ablation, and a fresh 1152 run with every fix is
exactly what r2_long now is.

**02:35 -- 1152x640 pinned.** R1 led at equal steps and spent 14 of its 15 hours on
misaligned depth targets, so its final number is not allowed to decide the resolution:
`force_res` in the lanes config pins 1152x640 for the history lane, the backlog and round 2
(the 1408 lane stays as a separate candidate). `r2_long` try 2 passed its launch check at
1.44 it/s (ends ~01:30 Sep 27).

## 38. Corrected recipe at step 19k: 95.2% of the teacher on unseen scenes (2026-09-26 06:00)

`r2_long` try 2 (1152x640, correct depth targets, all 28,130 train-scene frames, Occ3D
occupancy, snapshot averaging; 19k of 120k steps, LR near its peak) on the same 100
val-scene frames used for every equal-step read:

| model @ step 19k | F1 | retained | occupancy mIoU vs Occ3D GT |
|---|---|---|---|
| FINAL 896x512 | 56.4% | 89.4% | (teacher-distilled) 5% |
| R1 1152x640, misaligned depth | 57.2% | 90.7% | 5% |
| **r2_long 1152x640, corrected recipe** | **60.0%** | **95.2%** | **29.5%** (teacher 3.5%) |

Recall by range 83.5 / 75.4 / 63.1 / 42.8 / 8.9 (teacher 90.0 / 82.7 / 68.3 / 45.5 / 14.0);
mid-range gap 7.3 points; bicycle and barrier already above the teacher; pedestrians
36.3 vs 53.2 remain the largest deficit. What produced the +2.8 over R1 at equal steps:
depth targets that are actually where the lidar returns are, and a third more frames.
The remaining 100k steps include the entire low-LR tail (worth +10 F1 on E5c); the
250-frame confirmation of this snapshot follows, the lane's own full evaluation lands
~01:30 Sep 27.

**06:10 -- 250-frame confirmation of the 19k snapshot:** F1 58.7% vs teacher 62.0% =
**94.7% retained** (0.2 points under the 95% bar of 58.9%) at 16% of the schedule; recall
by range 86.6 / 78.8 / 62.7 / 39.0 / 8.7, mid-range gap 5.2 points; occupancy **28.2% mIoU
vs Occ3D GT** (teacher 3.2%); trajectory 1.536 m (teacher 1.629). Deficits left: pedestrian
37.3 vs 51.0, truck 38.3 vs 45.9, cone 60.2 vs 68.0, and the 40 m+ band. The snapshot's
ONNX is exported as an interim fallback (`outputs/onnx/r2_long_snap19k/`).

## 39. R1 final: 56.6% (91.4%) -- a run poisoned by its own depth targets (2026-09-26 09:35)

R1 (1152x640, 80k steps) ends at F1 56.6% / 91.4% retained, 0.6 below the 896x512 run it led
by +0.8 at step 19k. It trained on the misaligned depth targets (section 36) from step 0 to
~44k -- the whole high-LR phase -- and on correct ones only for its last 36k steps; a
lift-splat whose depth head was pulled toward the wrong image cells for that long does not
recover in the anneal. Its number says nothing about resolution; the corrected lane's 19k
snapshot (94.7% on 250 frames) does, and the pin on 1152x640 stands. Occupancy vs Occ3D GT
4.9% (teacher-distilled head, as in FINAL). R1's remaining post steps (official metric,
export, timing) run now; then GPUs 4-7 go to the history lane.

## 40. Target crossed at step 40k: 97.3% of the teacher on unseen scenes (2026-09-26 10:05)

`r2_long` snapshot at step 40,000 of 120,000, the full 250 val-scene frames, official GT
filter, teacher scored on the same frames:

| | recall | precision | F1 | retained |
|---|---|---|---|---|
| teacher | 54.7% | 71.5% | 62.0% | -- |
| **student @ thr 0.3** | **54.5%** | 67.4% | **60.3%** | **97.3%** |

Recall by range 88.3 / 80.0 / 69.2 / 49.1 / 14.2 vs teacher 92.1 / 84.0 / 67.9 / 44.7 /
14.0 -- level with or above the teacher from 20 m out; by class car 61.2 vs 61.1,
motorcycle 51.8 vs 45.9, bicycle 44.0 vs 41.7, barrier 62.0 vs 53.2 above the teacher,
pedestrian 44.7 vs 51.0 and cone 64.8 vs 68.0 below. Occupancy 30.2% mIoU vs Occ3D GT
(teacher 3.2%). Two thirds of the schedule, including the whole anneal, remain.

What made the difference against the 92.3% of the 896x512 run, in order of evidence:
correctly placed depth targets at 1152x640 (+2.8 over R1 at equal steps), the 6,983 extra
training frames, the resolution itself (+0.8), Occ3D supervision (occupancy only). What
failed: R1 -- 15 hours on misaligned depth targets ended 0.6 below the 896 run.

Standing: detection 97.3% (bar 95%), trajectory 1.54 vs 1.63 m, speed for this shape
16.9 Hz on an idle H200 (graph) -- the clean deployment-loop timing and the official mAP
for the snapshot are being taken now; the lane's own final numbers land ~01:30 Sep 27.

**10:15 addendum.** Scoreboard rule: a lane's H200 Hz is the best timing of its graph shape
on an idle GPU (launch gate or export); timings taken under another lane's load (the 40k
snapshot read 9.2 Hz graph / 6.1 Hz loop on a GPU at 100% from r2_long) are lower bounds
and excluded. `run_stateful.py` in lane.sh now uses device index 0 (the process sees one
GPU; R1's loop timing crashed on index 4 -- harmless, LANE_DONE still printed). The
orchestrator declares TARGET MET by r2_long (97.3%, 14.3 Hz gate) and the deliverable copy
is the 40k snapshot; `REPORT_FINAL.md` is the narrative for the user. r2_hist3 launched
09:53 (K=3, 1152, 60k steps, ~0.75 it/s -> ~08:00 Sep 27). Official mAP of the snapshot is
being computed on GPU 1.

**10:50 -- official metric of the 40k snapshot:** mAP **0.3500**, NDS **0.4248** on all 6,019
val keyframes (car 0.57, barrier 0.57, cone 0.56, pedestrian 0.38, bus 0.34, motorcycle
0.33, bicycle 0.28, truck 0.27, trailer 0.12, construction 0.09; ATE 0.67, ASE 0.29, AOE
0.71, AVE 0.59, AAE 0.25). PLAN_V2's "deliverable v1" bar was mAP >= 0.35 / NDS >= 0.47: mAP
met at a third of the schedule; NDS is held back by velocity (0.59 m/s) and orientation
(0.71 rad), both of which the remaining 80k steps (and the anneal) should tighten. Running
that evaluation on a lane GPU halved r2_long's speed for 30 min; extra evaluations wait for
idle GPUs from now on.

**14:05 -- r2_long at 60k:** F1 60.2% (97.2%), flat against 40k (60.3%) with the operating
point moved to higher precision (thr 0.4: recall 50.5 / precision 74.6 vs 54.5 / 67.4);
occupancy up to 31.2% mIoU. A plateau at peak LR is expected; the anneal (last third of the
one-cycle) decides the final number. Pedestrians (41.6 vs 51.0) and cones (57.6 vs 68.0)
remain the deficit -- small objects, which the 1408x768 backlog lane targets.

## 41. Three keyframes of history: +1.4 F1 at equal steps (2026-09-26 16:20)

`r2_hist3` (r2_long's recipe with `--history 3`) at step 19k on the 100 comparison frames:
F1 **61.4%** / 97.3% retained vs r2_long's 60.0% at the same step (R1 57.2, FINAL 56.4).
Recall 84.3 / 77.6 / 67.7 / 49.9 / 12.2 by range -- the extra second of parallax pays at
20-40 m (+4.6, +7.1 over r2_long@19k); car 64.7 vs teacher 63.3, trailer 32.5 vs 27.3,
construction 23.2 vs 19.6 above the teacher; pedestrians 41.7 vs 53.2 unchanged. Occupancy
30.8% mIoU. Cost: 0.87 it/s vs 1.45 (three no-grad backbone passes per step), the same
ONNX interface with K=3 states, ~16 Hz for the graph on an idle H200 (speed gate 19.2 Hz
for the untrained shape).

**Plan change.** The final combining run no longer waits for both lanes: `r3_final`
(1152x640, history 3, 100k steps, all 28,130 frames, Occ3D, averaging) is queued on GPUs
0-3 right behind r2_long (~01:30 Sep 27 -> ~08:30 Sep 28 + evaluation, inside the
deadline), and the 1408x768 candidate with history 3 takes GPUs 4-7 behind r2_hist3
(~06:30 Sep 27 -> ~03:00 Sep 28); a 1408 history-1 fallback follows if the speed gate
rejects it. The orchestrator's built-in round 2 is disabled in favour of these lanes.

**17:50 -- r2_long at 80k: 61.0% F1, 98.4% of the teacher** (250 unseen frames; thr 0.4,
recall 51.3 / precision 75.1). The anneal has started to pay: 60.3 (40k) -> 60.2 (60k) ->
61.0 (80k). Occupancy 32.2% mIoU vs Occ3D GT. Recall by range 85.3 / 76.3 / 63.9 / 45.5 /
13.1 -- now above the teacher at 30-40 m, pedestrians 42.7 vs 51.0. 40k steps of low LR
remain (~01:30 Sep 27), then averaging of the 90k-120k snapshots.

## 42. Demo video of the student on an unseen scene (2026-09-26 20:55)

`local/distill/student_video.py` renders the student through the session-demo layout
(camera ring with 3-D boxes and both ego paths, lidar BEV, occupancy, map) on any scene;
`outputs/student_video/scene-0276/student_scene-0276.mp4` is 40 keyframes of scene-0276
(a busy intersection from the held-out val split: pedestrians on crosswalks, scooters,
parked bicycles) at real time, 20 s, from the 80k checkpoint. Two conventions had to be
matched to the renderer, which was written for the teacher head's raw output: boxes in the
lidar frame with bottom-centre z (the student's are ego-frame, centre z), and the extent
order -- the head/renderer put the along-heading extent first, nuScenes GT and the student
put the width first. A frame whose predicted path was flat turned out to be an ego at a
standstill about to pull away (history ~0), not a defect. 45 boxes/frame at thr 0.4 vs 31
annotations/frame in the renderer's GT (which includes boxes without lidar returns).

**21:25 -- video fixed; r2_long at 100k.** The ego-state cache now covers the scene-end
keyframes (history/velocity/acceleration real, future clamped, `has_future=0`; the loader
masks the trajectory loss there), so the demo's planner has kinematics on every frame; the
recorded path and ADE are drawn only where a full 5 s future exists. r2_long snapshot at
100k on the 250 unseen frames: F1 59.9% (96.6%) at thr 0.3 -- a dip from 61.0 at 80k
(snapshot-to-snapshot noise of ~1 point is expected; averaging the 90k-120k snapshots at the
end is meant to smooth exactly this); occupancy 32.9% mIoU.
