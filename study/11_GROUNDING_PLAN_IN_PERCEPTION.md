# Grounding the plan in perception — a stage the release does not have

## 1. The gap

Qwen-Drive-1.0 has two heads hanging off one frozen VLM trunk:

```
12 camera images ──► VLM (4.54 B, shared) ──┬──► perception head (0.125 B) ──► BEV occupancy + map
                                            └──► planning expert (1.04 B)   ──► 50 waypoints / 5 s
```

Read [training/losses.py](training/losses.py) as it ships and the arrangement is
plainer than the diagram suggests. `detection_loss`, `occupancy_loss` and
`map_loss` score the perception head against perception labels.  `planning_loss`
scores the planner against the driven path. **No term reads both heads.** The
recipe's four stages train perception, then perception+VLM, then the planner,
then RL — and at no point does a gradient flow from a planning mistake into a
perception feature, or from a perception detection into a waypoint.

So the two heads are *separable*. The trajectory is decoded from the chain of
thought and the scene cache; it is never checked against what the model itself
claims to see. The planner can drive through a car the perception head has
correctly boxed, and nothing in the objective notices.

That is the gap this stage attacks. Section 5 reports how far it gets: the loss
demonstrably works where it is applied, and does not yet generalise — for reasons
that turn out to be about data volume, not formulation.

---

## 2. What "grounded" has to mean

The obvious move — penalise the planned waypoints wherever the predicted
occupancy is high — is wrong, and the failure is instructive.

### 2.1 The metric must score the human at zero

Before training anything, score the *driven path* under the proposed cost. Over
40 held-out nuScenes frames:

| | collision | offroad |
|---|---|---|
| human (the actual driven path) | **0.0335** | 0.0143 |
| released planner-sft | 0.0274 | 0.0080 |

The ground truth is *worse than the model*. A cost that ranks the demonstration
below the thing you are trying to improve cannot be a training signal — minimising
it moves away from human driving by construction.

### 2.2 Why: a snapshot of now, a plan for later

The per-waypoint profile of the human path pins the cause exactly:

```
  i   t(s)   x_ahead   collision   offroad
  0   0.1     0.61 m    0.0000     0.0002
 11   1.2     7.31 m    0.0000     0.0026
 12   1.3     7.92 m    0.0249     0.0023   <- collision breaks here
 14   1.5     9.14 m    0.0998     0.0016
 21   2.2    13.47 m    0.1063     0.0002
 49   5.0    31.1  m    0.0002     0.0607   <- map head degrades at range
```

The driven path is collision-free for exactly 1.2 s — the car-following gap — and
then drives straight through the lead vehicle. Of course it does: the occupancy
grid is a snapshot at *t = 0*, and by the time the ego arrives the lead vehicle
has moved on. Charging that cost trains **timidity, not safety**. The offroad
term fails at the other end: past ~3 s the plan reaches beyond where the map head
is accurate, so late waypoints are charged for the head's error, not the plan's.

### 2.3 Truncation is well-posed but empty

Restricting each term to the span where the snapshot is still evidence —
collision over 1.2 s, offroad over 3.0 s — makes the metric well-posed: the human
scores 0.0000 / 0.0027. But a 30-step probe at weight 1.0 printed
`collision 0.0000  offroad 0.0000` throughout, with gradient norms identical to
the imitation-only control.

The planner never collides within 1.2 s. It cannot: the near field is fixed by
the ego's own inertia. A term that is zero on every sample teaches nothing.

### 2.4 The hinge

Charge only the risk the plan incurs **over** what the driven path incurs:

$$\mathcal{L}_{\text{col}} = \frac{1}{P}\sum_p \operatorname{relu}\big(r(\hat{\tau}_p) - r(\tau^{\text{gt}}_p)\big)$$

with $r(\cdot)$ the bilinearly-sampled occupancy risk. This is the whole idea:

- the stale-snapshot artefact is present in **both** terms and cancels;
- the driven path scores exactly 0 by construction, so the metric is well-posed;
- the full 5 s horizon becomes usable, because map-head error at range also cancels;
- what survives is *disagreement the human did not commit* — real signal.

Measured for the released planner over the same 40 frames:

```
   i  t(s)  risk_pred risk_gt  EXCESS | off_pred off_gt  EXCESS
  16   1.7   0.0753  0.0750  0.0003 |  0.0008  0.0004  0.0004
  20   2.1   0.1249  0.1001  0.0249 |  0.0003  0.0002  0.0000
  24   2.5   0.0747  0.0500  0.0247 |  0.0020  0.0042  0.0000
  44   4.5   0.0003  0.0001  0.0001 |  0.0621  0.0385  0.0235
  48   4.9   0.0001  0.0001  0.0000 |  0.0865  0.0540  0.0325

  H=12: excess collision 0.0000   excess offroad 0.0003
  H=30: excess collision 0.0056   excess offroad 0.0003
  H=50: excess collision 0.0034   excess offroad 0.0049
```

Excess collision peaks at 2.1–2.5 s — the planner cuts corners the human does not.
Excess offroad grows past 4.5 s — the planner drifts off the drivable surface at
the far end. Both are genuine, and neither is visible in the absolute metric.

### 2.5 The loss and the metric must score the same object

With a well-posed cost, the first grounded run still came out a **null**: ADE
1.848 → 1.796 (the imitation-only control reached 1.652) and excess collision
*rose*, 0.0046 → 0.0050. The training log explains itself — `collision 0.0000` at
almost every step.

The cost was being applied to the wrong tensor. Flow matching trains
`predict_endpoint(x_t, t)` at a random `t`, and that is what the loss was scoring:

- at high `t`, `x_t` is already mostly the target, so the prediction matches it
  trivially and the grounded terms read zero;
- at low `t`, the prediction is a guess from noise — genuinely ambiguous, and
  never something the car drives.

Neither is the deployed trajectory. Meanwhile the *metric* used the real 10-step
sampler. Loss and metric were looking at different objects, so optimising one did
nothing for the other, and the leftover gradient was noise — which is exactly what
cost 0.14 m of ADE.

The fix falls out of the sampler itself:

```python
remaining = max(1.0 - index * step, min_one_minus_t)
waypoints  = waypoints + (endpoint - waypoints) / remaining * step
```

At `num_steps = 10` the divisor is floored at `min_one_minus_t = 0.1`, which equals
`step`, so the last update lands exactly on the last endpoint prediction. Replay
the sampler under `no_grad` and keep the graph on **only that final call**: the
grounded terms then score the trajectory that will actually be driven, for one
extra backward path instead of ten. Cost: 1.25 → 3.9 s/step.

The gradient behaves completely differently afterwards:

| | endpoint at random `t` | replayed rollout |
|---|---|---|
| `\|g\|` at weight 1.0 | 0.06 … **30.9** | steady **0.17** |
| grounded term | 0.0000 at nearly every step | nonzero on **30/40** records |

### 2.6 A near-binary raster is not a cost field

Weighting the terms so each *loss value* matched imitation gave
$\lambda_c = 0.1$, $\lambda_o = 1.0$ — and a run that was worse on everything:
ADE 1.848 → **1.998**, excess collision 0.0046 → **0.0061**. Splitting train from
val explained it:

| model | train ADE | train ex-col | val ADE | val ex-col |
|---|---|---|---|---|
| sft | 1.874 | 0.0062 | 1.937 | 0.0039 |
| control | 1.555 | 0.0063 | 1.714 | 0.0031 |
| grounded (sharp) | 1.808 | **0.0006** | 2.237 | 0.0058 |

The loss worked — it cut train excess collision 10× — and generalised not at all.
The per-term gradients say why:

| | mean | median | p90 | max |
|---|---|---|---|---|
| `\|g\|` imitation | 0.0796 | 0.0548 | 0.1707 | 0.6142 |
| `\|g\|` collision | 7.0713 | **0.0011** | 2.5981 | **110.01** |
| `\|g\|` offroad | 4.2688 | **0.0073** | 0.4424 | **123.40** |

Five orders of magnitude between median and max. The perception head emits a
near-binary raster, so bilinear sampling has **zero** derivative inside the free
space and an enormous one across a single cell at an object edge. The collision
term was also silent on 19 of 30 records. No scalar weight fixes that: small
enough to survive the spikes is too small to teach anything, and the few records
that do fire get memorised.

The fix is to blur the cost field before differentiating it — a safety margin
expressed in metres, `smooth_field()`, applied to the loss only while the metric
keeps scoring the sharp raster:

| blur (occ, map) m | term | median `\|g\|` | max `\|g\|` | nonzero |
|---|---|---|---|---|
| 0.0 / 0.0 | collision | 0.0011 | 45.8 | 11/30 |
| 0.0 / 0.0 | offroad | 0.0073 | 136.6 | 15/30 |
| 2.0 / 1.0 | collision | 2.8001 | 20.8 | **26/30** |
| 2.0 / 1.0 | offroad | 0.2986 | 113.5 | 20/30 |
| 4.0 / 2.0 | collision | 1.4709 | 21.7 | 27/30 |
| 4.0 / 2.0 | offroad | 4.8712 | 61.9 | 25/30 |

Blurring does two things at once: it bounds the spatial derivative (median-to-max
spread collapses 41,600× → 7.4×) and it makes the cost nonzero *near* obstacles
rather than only on them, so most records contribute instead of a memorisable few.

Then set the weights by **gradient** parity, not loss-value parity, at ~40 % of
imitation's median: $\lambda_c = 0.007$, $\lambda_o = 0.004$ with 2.0 m of blur on
both fields. (Loss-value parity is what produced the 20× ratio above; it is the
wrong currency, because the map raster's 0.15 m cells yield far more gradient per
unit loss than occupancy's 0.4 m cells.)

---

## 3. The data problem

Grounding needs, per frame, **both** a full camera ring (so the perception head is
meaningful) and a 5 s ego future (so there is something to plan). Neither bundled
cache has both:

| cache | frames | has perception | has 5 s future |
|---|---|---|---|
| `data/train_cache` | nuPlan | yes | no |
| `data/train_cache_plan` | WOD-E2E | 3 cameras only | yes |

Their token sets are disjoint — `perc & plan` is empty. So the cache is rebuilt
from nuScenes trainval, where every keyframe has a 6-camera ring *and* a
continuing ego trajectory:

```bash
.venv/bin/python -u training/cache_grounded_features.py \
    --dataroot /path/to/nuscenes --version v1.0-trainval \
    --out data/train_cache_grounded --frames-per-scene 1 --max-scenes 850
```

850 scenes → 850 records, ~114 GB, ~1.7 h on one 3090. Each record carries the
frozen VLM scene cache, the ego history/status, the normalised 5 s target, and the
perception head's own `occ_risk` and `map_probs` for that frame.

Two implementation notes that cost real time:

- **`infer()` is `@torch.no_grad()` and returns a byte argmax.** The raw logits
  are needed. A forward hook on `head.bev_modeling` captures `outs` without
  touching `src/`.
- **Axis order is not symmetric.** Occupancy is `[B, 1, X, Y]`; the map raster is
  `[B, C, 200, 400]` = `[y, x]`. Settled empirically by indexing 121 ground-truth
  boxes and checking the class at the cell: `occ[x_idx, y_idx]` matched 119, the
  transpose matched 9.
- **"Drivable" must be calibrated, not assumed.** Scoring the driven path over 25
  records: class `[1]` → 0.0803, `[1,2]` → 0.0097, `[1,2,4]` → **0.0002**,
  `[1,2,3,4]` → 0.0000. `[1,2,4]` (driveable surface, road line, crosswalk) is
  the tightest set that does not punish correct driving; adding `road_edge`
  scores zero only because it swallows everything.

---

## 4. The stage

VLM and perception head are frozen and enter as cached constants; only the 1.04 B
planning expert trains. The loss is the shipped flow-matching objective plus the
two hinged terms:

$$\mathcal{L} = \mathcal{L}_{\text{fm}} + \lambda_c \mathcal{L}_{\text{col}} + \lambda_o \mathcal{L}_{\text{off}}$$

### Weight calibration

See §2.6 — measured, not guessed: $\lambda_c = 0.007$, $\lambda_o = 0.004$, with the
cost fields blurred by 2.0 m (`--occ-sigma 2.0 --map-sigma 2.0`) for the loss only.

The grounded terms score a replayed 10-step rollout (§2.5); the imitation term is
untouched, so the only difference from the control is the two extra terms.

### Control

The only honest comparison is against the *same* fine-tune without the two terms.
Same seed, same 1500 steps, same lr, same data, same split — only the loss differs.

---

## 5. Results

See [study/04_RESULTS.md](study/04_RESULTS.md) for the released-model baselines.

All rows below are the same 1.04 B expert initialised from `planner-sft`, trained
1,500 steps with the same seed, sampler and 10-step schedule; the only difference
between *control* and *grounded* is $\lambda_c, \lambda_o \ne 0$. Means over 80
frames are not enough to support a claim at this effect size, so every comparison
is also given as a paired bootstrap 95 % CI on the per-frame difference.

### Held out: 80 frames from 80 unseen scenes

| | ADE (m) | FDE (m) | excess-col | excess-off |
|---|---|---|---|---|
| human (driven path) | 0.000 | 0.000 | 0.0000 | 0.0000 |
| planner-sft (released) | 2.181 | 5.844 | 0.0102 | 0.0015 |
| control (imitation only) | 2.002 | 5.499 | 0.0088 | 0.0018 |
| grounded ($\lambda_c$ 0.007, $\lambda_o$ 0.004) | 1.998 | 5.454 | 0.0085 | 0.0019 |
| grounded, 3× weights | 1.999 | 5.492 | 0.0087 | 0.0018 |

Paired bootstrap, 95 % CI on (model − control), n = 80:

| | Δ ADE (m) | Δ excess-col |
|---|---|---|
| grounded | −0.004 [−0.026, +0.017] | −0.00024 [−0.00119, **+0.00050**] |
| grounded 3× | −0.003 [−0.042, +0.034] | −0.00006 [−0.00128, **+0.00112**] |

**Both intervals straddle zero.** On held-out scenes the grounded terms are not
distinguishable from the control. Tripling the weights changes nothing, so this is
not a weight that needs tuning. Reporting the 0.0088 → 0.0085 difference as an
improvement would be reading noise.

### On the frames it was fitted on: 120 training frames

| | ADE (m) | FDE (m) | excess-col | excess-off |
|---|---|---|---|---|
| human (driven path) | 0.000 | 0.000 | 0.0000 | 0.0000 |
| planner-sft | 1.853 | 4.709 | 0.0050 | 0.0034 |
| control | 1.264 | 3.238 | 0.0058 | 0.0016 |
| grounded | 1.214 | 3.051 | **0.0033** | **0.0008** |

| | Δ ADE (m) | Δ excess-col |
|---|---|---|
| grounded | **−0.050 [−0.077, −0.024]** | **−0.00248 [−0.00478, −0.00068]** |

Both intervals exclude zero. Where the loss was applied it cuts excess collision
**43 %** and halves excess offroad — *while also improving ADE by 5 cm*. That last
part is the important one: agreement with perception and agreement with the
demonstration are not in tension here. The grounded terms are not dragging the
plan away from the human; they are resolving cases the imitation loss alone leaves
ambiguous.

The caveat that belongs next to it: this is the *training* split, so some of that
ADE gain is simply fitting these frames harder. Control and grounded saw the same
frames for the same number of steps from the same seed, so the comparison is fair,
but "no trade-off against imitation" is demonstrated here only where supervised.

### What this means

The loss works and does not generalise. That is a **data** verdict, not a method
verdict:

- the effect is real and correctly signed where supervised (CI excludes zero);
- it is the right *kind* of effect (safety up, displacement also up);
- it does not survive the jump to unseen scenes, on **770 training records** for a
  1.04 B expert.

770 frames is roughly 2 % of nuScenes' 34,149 keyframes — one frame per scene,
chosen because the cache costs 134 MB/frame. The honest next step is not a new
loss; it is `--frames-per-scene 8` and a 20× larger cache. The gradient
measurements in §2.6 say the optimisation is healthy; nothing about the §5 val
table suggests the formulation is wrong, only that it is under-supervised.

Note also what the control row already shows: plain imitation fine-tuning on these
770 frames moves ADE 2.181 → 2.002 (CI on sft − control: +0.179 [+0.081, +0.280],
resolved). So the cache is large enough to teach displacement and not large enough
to teach the safety signal. That is a question of signal *magnitude*, not
frequency: after blurring the collision term is nonzero on 26 of 30 records, but
its mean is 0.0037 and its median gradient 2.8 against a maximum of 20.8 — most
frames contribute a whisper, and the frames that actually contain a near-miss are
rare. Displacement error, by contrast, is large and informative on every single
frame.

---

## 6. Run it

```bash
export PYTHONPATH=src:. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. build the cache (~1.7 h, ~114 GB)
.venv/bin/python -u training/cache_grounded_features.py --max-scenes 850

# 2. control: imitation only (~47 min)
.venv/bin/python -u training/train_planner_grounded.py \
    --collision-weight 0 --offroad-weight 0 --seed 0 --out outputs/plan_control

# 3. grounded (~1.7 h)
.venv/bin/python -u training/train_planner_grounded.py \
    --collision-weight 0.007 --offroad-weight 0.004 \
    --occ-sigma 2.0 --map-sigma 2.0 --seed 0 --out outputs/plan_grounded

# 4. score all three on identical frames, with paired CIs against the control
.venv/bin/python -u training/eval_grounded.py \
    --ckpt sft=pretrained \
    --ckpt control=outputs/plan_control/planning_expert.pt \
    --ckpt grounded=outputs/plan_grounded/planning_expert.pt \
    --paired-against control

# 5. the diagnostic that separates "no effect" from "no generalisation"
.venv/bin/python -u training/eval_grounded.py --split train --limit 120 \
    --ckpt control=outputs/plan_control/planning_expert.pt \
    --ckpt grounded=outputs/plan_grounded/planning_expert.pt \
    --paired-against control --out outputs/plan_eval_train.json
```

Always read the paired CI, not the means. The val means suggest a 3 % improvement
that the CI shows is noise; the train means suggest 43 % and the CI confirms it.

---

## 7. What this does not do

- **It does not yet generalise.** This is the headline limitation and §5 measures
  it: the effect is resolved on training frames and not on held-out ones. One
  frame per scene gives 770 training records, which teaches displacement but not a
  sparse geometric safety signal. `--frames-per-scene 8` over 850 scenes gives
  6,800 records (~900 GB, ~12 h on one 3090) and is the experiment this write-up
  is missing.
- **Perception is frozen.** Gradients flow planner ← perception, never the
  reverse. Making the trajectory able to *correct* the BEV features needs the
  perception head in the graph, which does not fit alongside the expert on 24 GB.
- **The occupancy snapshot is still static.** The hinge cancels the artefact; it
  does not model that other agents move. A predicted occupancy *flow* would.
- **80 held-out frames is a coarse ruler.** Resolving a difference the size of the
  one observed (−0.00024) needs roughly 1,000 frames. `--scenes-file` densifies
  the held-out scenes without touching the training split.
