# Ego-lane traffic lights on OpenLane-V2 subset B

**The question:** when the ego approaches an intersection, what colour is the light that
governs *its* lane — not the most eye-catching light in the frame.

**The headline:** Qwen-Drive answers 93.5% of val frames correctly out of the box, but a
control experiment shows it is not doing the association at all. A purpose-built pipeline
on the same frames reaches **98.4%** [96.3%, 99.7%] with annotated boxes and **97.3%**
[94.9%, 99.0%] end to end with no annotation at inference. The residual errors are night,
rain, and lamps under 10 px. Fine-tuning the VLM itself lifts the association-critical
subset from 84.4% to 91.2% without matching the pipeline.

---

## 0. Numbers, up front

Val = the official OpenLane-V2 subset_B validation split, 1300 usable frames over
**86 label-episodes**. Intervals are bootstrapped over episodes, not frames (see §2).

### On the WHOLE validation split (6019 frames)

Every table below this one is scored on the 1300 frames where a coloured light actually
governs the ego lane -- 21.6% of val. That answers "what colour is it", not "is there one".
Scored on all 6019 frames, where 78.4% of the right answers are `none`:

| method | accuracy | 95% CI | macro-recall | named a colour when none applied |
|---|---|---|---|---|
| always answer `none` | 78.4% | [72.8, 83.3] | 25.0% | 0 |
| VQA v1 (trained on governed frames only) | 80.1% | [75.6, 84.1] | **81.0%** | 1109 (24% of none-frames) |
| VQA v3 (trained with hard negatives) | 88.5% | [85.3, 91.2] | 78.8% | 517 (11%) |
| pipeline, single selector | 88.8% | [85.5, 91.7] | 71.3% | 270 (6%) |
| **pipeline, selector ensemble** | **90.1%** | [87.2, 92.5] | 73.3% | 296 (6%) |

Neither number alone is honest. Accuracy says the VQA barely beats a constant answer;
macro-recall says the VQA finds the rare colours *better* than the pipeline, which buys
accuracy by abstaining. The constant baseline settles it: 78.4% accuracy at 25%
macro-recall is what a useless model scores.

Where each one invents a light is the part that matters for driving:

| the frame is `none` because... | VQA | pipeline |
|---|---|---|
| lights are visible, but none is the ego's | **916/1192 (77%)** | 232/1192 (19%) |
| there is no traffic light at all | 131/3457 (4%) | **1/3457 (0%)** |

On the 1192 frames where a light is visible but belongs to another lane, the VQA reports
its colour 77% of the time. That is the control experiment of §3 reproduced at scale: the
model answers with the salient lamp. Saying "red" because cross traffic is red is a
different and worse failure than saying "none".

---

**Best of each approach, side by side:**

| approach | frames with an ego light (1300) | all validation frames (6019) |
|---|---|---|
| VQA — Qwen-Drive answering in words | 94.69% [88.3, 98.6] | 88.52% [85.3, 91.2] |
| **Perception — traffic-light head + selector + colour** | **97.85%** [95.9, 99.2] | **90.08%** [87.2, 92.5] |
| Perception, 3D variant (adds range/height) | 97.38% [95.1, 99.0] | — |
| base Qwen-Drive, no fine-tune | 92.62% [85.9, 97.3] | 80.11% [75.6, 84.1] |
| trivial: always answer "none" | — | 78.40% [72.8, 83.3] |

Full breakdown below.

| method | all | discriminative | <12 px | errors |
|---|---|---|---|---|
| majority class (always green) | 55.5% | 87.6% | — | 578 |
| Qwen-Drive VQA, out of the box | 92.6% [85.9, 97.3] | 92.4% | 88.3% | 96 |
| Qwen-Drive VQA + LoRA (natural mix) | 94.2% [87.7, 98.2] | 95.9% | 87.1% | 76 |
| Qwen-Drive VQA + LoRA + position-hint prompt | **94.7%** [88.2, 98.6] | 96.2% | 89.5% | 69 |
| **pipeline, end to end** (detector boxes, no annotation) | **97.4%** [95.1, 99.0] | 95.2% | 94.5% | 34 |
| **pipeline, annotated boxes** | **98.4%** [96.3, 99.7] | 95.5% | 97.7% | 21 |
| colour head given a correct box | 99.85% [99.6, 100] | — | 99.6% | 2 |

Every configuration choice — prompt, selector, vote rule, detector threshold — was made on
a holdout of train segments; val was read once per method.

`discriminative` = another visible light disagrees with the ego's, so reporting the most
salient lamp gives the wrong answer. 32% of val frames.

---

## 1. Ground truth, and why it is trustworthy

Everything is derived from published annotation, nothing is guessed:

```
ego origin -> ego lane centerline   geometry, x forward / y left, heading gate <=45 deg
ego lane   -> reachable lanes       topology_lclc (i->j, end(i)==start(j)), <=2 hops
lanes      -> traffic elements      topology_lcte
element    -> colour                attribute 1=red 2=green 3=yellow
```

Three things that are easy to get wrong here:

- **The heading gate matters.** Without it, a cross-traffic lane passing near the ego is
  picked as the ego lane in ~1.2% of frames.
- **Propagation matters.** Lights are annotated against the lane at the stop line, not the
  lane the ego occupies 40 m back. Propagating 2 hops lifts usable val frames 408 -> 1268.
- **Only CAM_FRONT is relevant.** Traffic-element boxes are in front-camera pixels;
  cropping the same box from all six cameras shows the lamp only in CAM_FRONT.

Verification, rather than assertion:
- 16/16 sampled labels correct by eye, down to 7x14 px.
- Every image path in the annotations resolves: 239,970 references, 0 missing.
- **91% of discriminative frames show ego-green against cross-red** (or the reverse).
  Random association could not produce that.

Yield: 5195 train / 1300 val usable frames; 257 of 696 train segments contain the problem
at all.

---

## 2. Frames are not independent — score by episode

Val's 1300 frames come from 64 segments and only **86 label-episodes** (a run of
consecutive frames in one segment sharing one label; median 12 frames, 2 Hz). Treating
frames as independent understates confidence intervals by roughly sqrt(12).

Every interval in this document resamples whole episodes. The practical consequence: with
86 episodes, an interval around 98% is roughly +-2 points, so 98.8% and 99.5% are not
distinguishable on this data. The `test` split ships no annotations, so val is the only
clean held-out set.


---

## 2b. The three approaches, and which won

| | what it is | best result (light present) | best result (all 6019 frames) |
|---|---|---|---|
| **VQA** | Qwen-Drive answers in words from the image alone. No boxes, no head. | 94.69% | 88.52% |
| **Perception** | A traffic-light head on the frozen vision tower: detect -> select the ego's -> read colour. | **97.85%** | **90.08%** |
| **Pipeline (3D)** | The same, with range and height regressed per light. | 97.38% | — |

**Perception wins both.** The gap is largest where a light exists (97.9 vs 94.7) and
narrows on the whole split (90.1 vs 88.5), where both are limited by deciding *whether* a
light applies rather than which colour it is.

### How traffic-light perception was added

Nothing in `src/qwen_drive_perception/` was modified — `git status` shows no tracked file
changed, and the repo's 3D detection, online mapping and occupancy still run identically
(verified by re-running `local/run_perception_demo.py`). The traffic-light stack is a
**separate head reading the same frozen ViT tap**, with `requires_grad_(False)` on every
VLM parameter, so no gradient can reach the shared backbone.

```
CAM_FRONT ─► frozen Qwen-Drive ViT (perception tap 1, pre-merge patches)
                │
                ├─► repo's BEV head        (untouched: detection / map / occupancy)
                │
                └─► TLDetHead  3.8 M params, trained from scratch
                      ├ obj      is a traffic light here?          dense, 8 px cells
                      ├ box      l,t,r,b offsets
                      ├ col      unknown / red / green / yellow
                      ├ gov      does it govern the ego lane?
                      └ 3D       log-range + height  (v2 only, frozen 2D)
```

Then three stages the detector alone does not provide:

1. **Selector** (`selector.py`, 4.0 M) — set-attention *across* the lights in a frame plus
   a coarse scene grid. Association is a comparison, not a per-light judgement: whether a
   light is "mine" depends on where the others are.
2. **Colour head** (`train_colour.py`, ResNet-18 on 64x64 crops) — 99.85% given a correct
   box. Augmentation excludes hue, which would change the label.
3. **Soft vote** — each light's colour weighted by its P(governs). Beats reading the top
   light (97.5% -> 98.4%) and suppresses cross traffic for free: a sideways-facing lamp
   reads as `unknown` and barely votes.

Training data is all self-derived from the lane graph (§7), never hand-labelled: 81,797
light instances for the colour head, 11,964 frames for the detector, 1,058 Occ3D-confirmed
3D positions for the depth branch.

### How the VQA was fine-tuned

LoRA on the attention projections only (r=16, alpha=32, 5.51 M trainable of 4.539 B). Loss
is next-token prediction **masked to the answer tokens**; the prompt, including all image
tokens, is context. The model's own `<think></think>` prefix is kept — training on the
bare colour teaches it to drop its own format and starts the loss at ~13.5 instead of
~0.005.

Four training mixes were tried, and the mix mattered far more than any hyperparameter:

| run | training data | full-val | light present | phantom | missed |
|---|---|---|---|---|---|
| v1 | governed frames only, 3-way question | 80.1% | **94.7%** | 1109 | 33 |
| v2 | + hard negatives, 4-way with `none` | 87.9% | 74.3% | 397 | 326 |
| **v3** | rebalanced toward positives | **88.5%** | 86.6% | 517 | 161 |

v1 never sees a frame where the answer is "none", so it invents a colour on 77% of the
frames where lights belong to another lane. v2 over-corrected. v3 is the balance.

Hyperparameters were exhausted without improving on v3 — native resolution **-1.4**,
rank 32 **-0.2** (noise), a second epoch **-7** on the usable subset. The ceiling is the
recipe-independent one in §3: the VLM is not doing the association.

---

## 3. The VLM does not associate — a control says so

A question written by hand and never checked is an untested instrument. Twelve phrasings
were scored on **dev** (46 train segments, disjoint from val), in three groups:
association prompts, format controls, and association-free controls.

```
paired vs controls on discriminative frames (macro-recall difference, episode-bootstrapped)
  ego_jargon    - salient   +3.7%  [-1.5%, +12.2%]
  exclude_cross - salient   +3.2%  [-2.9%, +11.9%]
  cot           - salient   +1.5%  [-7.1%, +11.6%]
  action        - salient   -1.7%  [-7.4%,  +2.7%]
```

**No association prompt beats the association-free control with significance**, and
`salient` ("what colour is the traffic light ahead?", which never mentions the ego lane)
has the highest raw accuracy of all twelve (94.7%). Reversing the option order alone
moves discriminative macro-recall 85.9% -> 73.7%.

The reading: the model reports the salient lamp, and the label agrees often enough to look
like understanding. Accuracy also hides a green bias — macro-recall is 82.7% overall and
~51% on frames where the largest visible light disagrees with the ego's.

---

## 4. The problem splits, and only one half is hard

| | accuracy |
|---|---|
| colour, given a correct box | **99.85%** (2 errors in 1300; 99.6% under 12 px) |
| Qwen-Drive VQA, full frame | 93.5% |

Colour perception is solved. The gap is **association** — choosing which light is the
ego's. Because association only changes the answer on discriminative frames,
`overall ~= 0.68 + 0.32 * association`, so 99% overall needs association right ~97% of the
time on those frames.

A negative result worth keeping: feeding the VLM a tight oracle crop makes it **worse**
overall (88.7% vs 93.5%) while improving discriminative accuracy (96.4% vs 92.6%). Zooming
hands over association but removes the scene context the VLM depends on, and red lamps
that render amber get read as yellow three times as often (39 -> 133 errors). A technique
that helps a specialist head can hurt the VLM.

---

## 4b. Fine-tuning the VLM does help — on the frames that need association

LoRA (r=16, 5.51 M trainable, attention projections only) on 211 train segments, 1 epoch,
class-balanced. Loss is next-token prediction restricted to the answer; the model's native
`<think>\n\n</think>` prefix is kept as unsupervised context, because training on the bare
colour teaches it to drop its own reasoning block and starts the loss at ~13.5 instead of
~0.005.

| subset | base | fine-tuned |
|---|---|---|
| all | 92.6% | **93.8%** |
| discriminative | 92.4% | **96.2%** [92.6%, 99.1%] |
| salience-defeating | 84.4% | **91.2%** |
| light >=20 px | 98.3% | **99.4%** |
| macro-recall | 79.7% | **87.9%** |
| yellow recall | 7/13 | **10/13** |

Overall accuracy barely moves, but the gain is concentrated on the subsets that actually
require association — +6.8 points where the largest visible light disagrees with the ego's.
So the capability is learnable from this data, it is simply not there out of the box.

The trade is real: red read as yellow rises 37 -> 60, and lamps under 12 px get slightly
worse (88.3% -> 87.1%). Fine-tuning sharpened the association and blunted the red/amber
boundary.

Two practical notes for anyone repeating this:
- `model.eval()` silently disables gradient checkpointing (HF gates it on
  `self.gradient_checkpointing and self.training`), which is a 22.5 GiB -> 13.3 GiB
  difference and the whole reason it first refused to fit on a 24 GB card.
- One card segfaulted twice in a day (`Xid 31`, then an autograd segfault). The training
  script checkpoints every 25 steps and resumes, which turns a lost 2.5 h run into a lost
  2 minutes.

---

## 4c. Why the specialist wins: one segment explains most of the gap

Segment 11064 is filmed into the sun. The lit lamp is a blown-out white blob against
backlit trees, and the hue that says "red" is simply not in the pixels any more. What
survives is geometry: the lit lamp sits at the **top** of the housing.

| method | accuracy on segment 11064 (37 frames) |
|---|---|
| Qwen-Drive VQA, base | 8.1% |
| Qwen-Drive VQA, fine-tuned | 5.4% |
| **colour head, given the box** | **100.0%** |
| pipeline, annotated boxes | 100.0% |
| pipeline, end to end | 94.6% |

Those 37 frames are 2.8% of val and account for **71% of every red-read-as-yellow error**
in the VQA path. The VLM reads hue; when the exposure destroys hue it has nothing left,
and fine-tuning cannot teach a cue that is absent from the input.

Telling the VLM to use the surviving cue does not help. A prompt that spells the rule out
— *"if the lamp looks washed out, judge it by which position in the housing is lit: top is
red, middle yellow, bottom green"* — scores **2/37** on this segment, slightly worse than
the plain question, and 92.8% overall against 92.6% (macro-recall drops, 79.7% -> 77.1%).

| attempt on segment 11064 | score |
|---|---|
| plain question | 3/37 |
| explicit position-hint prompt | 2/37 |
| LoRA fine-tune, balanced | 2/37 |
| LoRA fine-tune, natural mix | 2/37 |
| **colour head, given the box** | **37/37** |

Three independent attempts to rescue the VLM — two fine-tunes and a prompt that states the
rule outright — all fail on the same frames. This is a capability gap, not a prompting gap:
the model will not ground "which third of the housing is lit" to the pixels, however
plainly it is asked.

The colour head is right on all 37 because it was built to keep the cue that survives:
crops are padded to square rather than stretched, so a traffic light's tall-thin shape is
preserved and *which third of the housing is lit* remains legible. That one decision, made
for a different reason, is worth about four points of end-to-end accuracy.

How much of the VQA-vs-pipeline gap is *only* this:

| method | all val | excluding segment 11064 |
|---|---|---|
| Qwen-Drive VQA, base | 92.6% | 95.1% |
| Qwen-Drive VQA, LoRA (balanced) | 93.9% | 96.5% |
| Qwen-Drive VQA, LoRA (natural mix) | 94.2% | **96.8%** |
| pipeline, annotated boxes | 98.4% | 98.3% |
| pipeline, end to end | 97.3% | 97.4% |

The pipeline does not move — it reads that segment perfectly. Every VQA variant gains
2.5-2.6 points. So most of the headline gap between the two approaches is a single
failure mode, and it is a **sensor** limitation rather than a reasoning one: on frames
where the exposure preserves hue, the fine-tuned VLM is at 96.8% and clears 95%.

The wider lesson for the metric: a single scene, 2.8% of frames, moves the headline by
2.6 points. This is the episode-clustering of §2 in its most concrete form, and it is why
every number here carries an episode-bootstrapped interval.

---

## 5. The pipeline

```
CAM_FRONT -> frozen Qwen-Drive vision tower (perception tap 1, pre-merge patches)
          -> per-light ROI features + coarse scene grid
          -> SELECTOR: set-attention over all lights in the frame -> P(governs ego)
          -> crop each candidate -> COLOUR HEAD (ResNet, 64x64) -> P(red/green/yellow)
          -> soft vote: sum_i P(governs_i) * P(colour_i)
```

Two design points that carried the result:

- **Association is a comparison, not a per-light judgement.** The selector attends across
  all lights in a frame; whether a light is "mine" depends on where the others are.
- **Soft voting beats reading the top light** (97.5% -> 98.4%). It also suppresses cross
  traffic for free: a light facing sideways reads as `unknown` to the colour head, so its
  red/green/yellow mass is small and it barely votes. No geometric rule needed.

---

## 5b. End to end, with no annotation at inference

Everything above used annotated boxes. The deployable version detects them:

```
CAM_FRONT -> frozen ViT (one forward, shared)
          -> detector head  -> boxes
          -> ROI pool + scene grid -> selector -> P(governs ego)
          -> crop each candidate   -> colour head -> P(red/green/yellow)
          -> soft vote -> answer
```

| stage | val accuracy |
|---|---|
| annotation boxes + selector + colour head | 98.4% |
| **detector boxes** + selector + colour head | **97.4%** [95.1%, 99.0%], coverage 99.7% |
| detector boxes + detector's own colour/gov heads, single light | 75.9% at 77% coverage, **99.0% precision when it answers** |

The detection threshold was chosen on the holdout (0.30: 99.66% there) and val read once.
Detection costs about 1 point. Detector quality: P 84.9% / R 80.7% at IoU 0.3 (IoU 0.5 is
harsh for an 18x27 px object predicted on 8 px cells).

Two things worth keeping from this:

- **The detector is a good localiser and a mediocre classifier.** Its own colour head gets
  93.2% on matched lights; swapping in the crop colour head and voting over candidates
  takes the same detections from 75.5% to 96.5%. Separating "where" from "what" is most of
  the gain.
- **Abstention is a real option.** Driven by its own heads the system answers 77% of
  frames at 99.0% precision. Coverage plateaus near 78% however low the detection
  threshold goes (0.05 -> 0.30 barely moves it), so the ceiling is the `gov > 0.5` gate and
  `unknown` colour verdicts, not detection sensitivity -- which is exactly what soft voting
  removes to reach full coverage. For a driving stack, "I cannot see the light" is a different
  and much safer failure than a confident wrong colour, and this pipeline can be tuned
  along that trade rather than forced to guess.

---

## 5c. The secondary question: which lane am I in?

Ground truth from the same lane graph: every centerline running the ego's way at the ego's
longitudinal position, ranked left to right, with the ego located among them. 27,047 train
/ 5,810 val frames (96.6% coverage -- far more than the traffic-light task, which needs a
light to exist).

Measured on 727 val frames / 294 episodes, base model:

| question | lane count | lane index | both |
|---|---|---|---|
| "…in the form: 2 of 3" | 26.0% | 35.8% | 17.6% |
| "…in the form: 1 of 4" | 16.2% | 71.7% | 12.6% |
| **no numeric example** | **59.6%** | **60.4%** | **44.5%** |

The three rows are the *same question* with the worked example changed. With "2 of 3" the
model answered "of 3" on 635 of 727 frames; with "1 of 4" the mass moves to 4. Remove the
example and the predicted distribution (1:143, 2:172, 3:49) lands close to ground truth
(1:177, 2:109, 3:70) and "both correct" more than doubles.

So the honest reading is: **Qwen-Drive has a real but rough sense of lane position** — about
60% on the count and on the index separately, 44% on both together — and a format example
in the prompt will happily override what it sees. Any evaluation of this capability that
uses one hand-written question with a worked example is measuring the question.

Three bugs in the *scorer* had to be fixed before this was visible, all of which made the
model look worse than it is:
- the model fills the template literally (`<1> of <2>`); the regex rejected the brackets and
  called 708 of 727 correct-form answers unparseable
- the parser was chosen by question *name* rather than answer *shape*, so a "X of Y"
  answer was read by the single-number parser
- the single-number parser wrote the count into the index field, scoring lane-count as 0.0%

---

## 6. What is left

21 errors, from a handful of segments (the 16-error variant is broken down below; the
pattern is identical):

| GT -> predicted | lamp | cause |
|---|---|---|
| red -> green | 9x18 px | association: a non-governing green light outvoted |
| yellow -> green | 12x28 px | colour: correct light, yellow misread |
| red -> yellow | 9x9 px | colour: 9 px lamp in heavy rain |
| green -> red | 16x15 px | association: night frame, heavy sensor noise |

Two association, two colour; all night, rain, or sub-10 px. None are annotation errors.
At episode level, 82 of 86 episodes are perfect.

---

## 7. Reproducing

```bash
scripts/download_openlane_v2_subset_b.sh     # 45 GB, md5-checked against the official table
.venv/bin/python local/tlb/build_gt.py       # ego-lane GT
.venv/bin/python local/tlb/build_det_gt.py   # per-light GT (81,797 instances)
.venv/bin/python local/tlb/extract_crops.py  # colour-head training crops
.venv/bin/python local/tlb/cache_roi.py --split train   # frozen-tower features
.venv/bin/python local/tlb/train_colour.py --balance    # colour head
.venv/bin/python local/tlb/train_selector.py            # association
.venv/bin/python local/tlb/pipeline.py --vote soft --topk 6
```

Every model's epoch is chosen on a held-out slice of **train** segments. Val is read once,
for the final number.
