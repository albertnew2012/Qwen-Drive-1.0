# Qwen-Drive-1.0 vs Alpamayo 1.5 — is it better, and better at what?

> Short answer: **as a driving model, the evidence is thin. As a platform for the thing you
> are actually building, it is clearly ahead — and it already ships the head you are
> hand-rolling.** The rest of this document separates those two claims, because the
> headline benchmark table conflates them.

---

## 0. The one-paragraph verdict

Qwen-Drive-1.0 is **half the size** (5.70 B vs 11.08 B), **fits your GPU with room to
spare** where Alpamayo does not, **retains general vision-language ability** where Alpamayo
has lost it, and **ships an official BEV perception head** that does 3D detection +
occupancy + map segmentation — a superset of the detection probe you built by hand in
`alpamayo1.5/perception/`. Against that: it plans a **shorter horizon** (5.0 s vs 6.4 s),
its expert reads **only 8 of 32 VLM layers** where Alpamayo's reads all 36, its action space
carries **no kinematic feasibility guarantee**, and — the one that will actually cost you —
**it is an inference-only release. There is no training code.**

---

## 1. Side by side, structurally

| | **Qwen-Drive-1.0-4B** | **Alpamayo 1.5-10B** |
|---|---|---|
| total params | **5.70 B** (VLM 4.54 + expert 1.04 + perception 0.13) | **11.08 B** (VLM 8.80 + expert 2.28) |
| bf16 resident | **11.66 GB** | 22.16 GB |
| fits a 24 GB 3090? | yes, with 12 GB spare | only just; falls back to CPU offload in practice |
| VLM backbone | Qwen3.5-4B, **hybrid** 24 linear-attn + 8 full-attn | Cosmos-Reason2-8B (Qwen3-VL-8B), **dense** 36 layers |
| vocab | 248 320, `tie_word_embeddings=True` | 155 697, **untied** (2 × 637.73 M) + 4000 trajectory bins |
| cameras | 3 (front, front-left, front-right) × 4 timesteps | 4 (3 × 120° + 1 × 30° tele) × 4 timesteps |
| expert | 32 layers × width 1024, **1.04 B** | 36 layers × width 2048, **2.28 B** |
| expert reads | **8 KV caches** (the full-attn layers only), 4 expert layers each | **36 KV caches**, one per VLM layer |
| conditioning | adaLN from (flow time + nav + ego status), 19.7 % of the expert | none — the expert *is* a Qwen3VLTextModel |
| flow matching | **clean-endpoint (`x1`)** parameterization, 10 Euler steps | **velocity field (`v`)**, 10 Euler steps |
| action space | `(x, y, heading)` × 50 @ 10 Hz = **5.0 s** | `(accel, curvature)` × 64 → **unicycle integration** = **6.4 s** |
| feasibility | not guaranteed | **guaranteed by construction** |
| perception | **official BEV head, 125.06 M**: 3D det (7 cls, 300 boxes) + occupancy (200×200×16, 10 cls) + map seg (200×400, 6 cls) | **none released** — yours is a 28.44 M bolt-on via a hook on layer 24 |
| reasoning | optional; VLM writes a rationale, expert reads that cache | always; chain-of-causation is the point |
| training code | **none — inference only** | recipes published (`NVlabs/alpamayo-recipes`) |

---

## 2. Where the published numbers actually support "better"

The Qwen-Drive README puts Alpamayo-1.5-10B in three tables. Read them in this order,
because they are not equally meaningful.

### 2.1 Driving VQA and spatial understanding — a real, large win

| | Alpamayo-1.5-10B | Qwen-Drive-1.0-SFT |
|---|---|---|
| LingoQA | 64.0 | **77.8** |
| Ego3D RMSE ↓ | 25.31 | **7.78** |
| VLAD | 9.1 | **66.5** |
| SURDS | 3.1 | **66.1** |
| WaymoQA safety | 42.6 | **70.7** |
| WaymoQA all | 44.4 | **74.5** |
| PAI Chain-of-Causation, all | 3.4 | **41.3** |
| EmbSpatial | 20.6 | **78.9** |
| ERQA | 27.5 | **48.5** |

Ego3D RMSE — metric 3D spatial reasoning from images — improving 25.31 → 7.78 is the single
most relevant number on this page for you, because it is a *direct proxy for how much 3D
structure is recoverable from the shared representation*. That is precisely the quantity your
detection probe is measuring on Alpamayo. Qwen-Drive's representation carries roughly **3×
less 3D positional error**.

The CoC row deserves a flag: **3.4 for Alpamayo on chain-of-causation** is its own home turf.
A score that low on the task the model was built for is a strong hint that this is a
**response-format / parsing failure**, not a reasoning failure — see §3.

### 2.2 General VQA — a win, but a nearly tautological one

| | Alpamayo-1.5-10B | Qwen3.5-4B (base) | Qwen-Drive-1.0-SFT |
|---|---|---|---|
| MMBench | 7.5 | 87.1 | 85.5 |
| MMStar | 26.1 | 75.3 | **75.9** |
| MMMU | 27.4 | 73.4 | 72.7 |
| CharXiv | 1.5 | 65.1 | 64.4 |
| OCRBench | 3.2 | 86.9 | 86.4 |
| RealWorldQA | 46.9 | 76.3 | **79.0** |

**Do not read this as "Qwen-Drive is 11× better at MMBench."** Read it as two separate facts:

1. **Alpamayo has catastrophically forgotten.** MMBench 7.5 and OCRBench 3.2 are *below
   chance* for multiple choice. A model does not "understand images 7.5 % as well"; it fails
   to emit a parseable answer. Alpamayo was post-trained into a specialist that produces
   chain-of-causation prose and trajectory tokens, and the README's own footnote says
   `–` marks "an invalid or unparsable response". This is a **format** collapse.
2. **Qwen-Drive did not forget.** Compare the last two columns, not the first two. Against
   its own base model, Qwen-Drive-1.0-SFT gives up **1.6** points on MMBench (87.1 → 85.5)
   and **0.5** on OCRBench (86.9 → 86.4), while *gaining* on MMStar (75.3 → 75.9) and
   RealWorldQA (76.3 → **79.0**). Driving post-training cost it almost nothing on general
   vision-language, and that retention — not the gap to Alpamayo — is the actual
   achievement.

   (All figures in §2 were re-checked against the README table programmatically rather
   than transcribed by eye.)

That property is what makes the "unified framework" claim more than marketing: the LLM
decoder is genuinely unchanged and still answers general questions.

### 2.3 Planning — **there is no comparison at all**

This is the gap nobody points at. The planning table is:

| | SFT | RL |
|---|---|---|
| NAVSIM v1.1 navtest, PDMS | 88.2 (89.3 best-of-6) | **90.7** (91.4 best-of-6) |
| Waymo Open Dataset E2E test, RFS | 7.78 | **7.91** |
| NVIDIA PhysicalAI open-loop, minADE 3 s | **0.34 m** | 0.38 m |

**Alpamayo is not in it.** Qwen-Drive vs Qwen-Drive, that is the whole table. So on the one
task both models were actually built to do — produce a trajectory — the release gives you
**zero** head-to-head evidence.

And the datasets differ: Alpamayo is trained and evaluated primarily on NVIDIA PhysicalAI
data; Qwen-Drive reports NAVSIM, Waymo E2E and PhysicalAI. The only shared surface is
PhysicalAI minADE, and Alpamayo's number is not quoted here.

> Your own measurement — `minADE 0.373 m at n=1` from
> [`alpamayo1.5/study/00_START_HERE.md`](/home/albert/Desktop/alpamayo1.5/study/00_START_HERE.md) —
> is **not** comparable to Qwen-Drive's 0.34 m: different clip subset, different horizon
> (Alpamayo 6.4 s vs a 3 s metric), different sample count. Making that comparison real is
> the single highest-value experiment available to you, and it is written up as Session 6
> in [`00_START_HERE.md`](00_START_HERE.md).

---

## 3. Three reasons to discount the table before you act on it

1. **The authors evaluated their own competitor.** Every Alpamayo number here was produced
   by the Qwen team's harness, not NVIDIA's. Reproducing a specialist model's expected
   output format is exactly the kind of thing that goes wrong silently, and the sub-chance
   scores are the signature of it.
2. **Prompt-format sensitivity is doing real work.** A model with `MMBench 7.5` and
   `CoC 3.4` is not being measured on capability. Some of the driving-VQA gap is real; the
   size of it is not trustworthy.
3. **`Qwen3.5-4B`, the base model, already beats Alpamayo on nearly every row** — including
   LingoQA (70.4 vs 64.0) and Ego3D (13.17 vs 25.31), *before any driving training at all*.
   That tells you most of the delta is "Alpamayo lost its VLM", not "Qwen-Drive gained
   something extraordinary". The genuinely new part is the last step: 13.17 → 7.78 on Ego3D,
   and 70.4 → 77.8 on LingoQA, which is what driving post-training bought.

---

## 4. Where Alpamayo is still ahead

Not everything favours the newer model.

| | why it matters |
|---|---|
| **6.4 s horizon vs 5.0 s** | 28 % further into the future. For highway merges and long-range yielding this is not a rounding difference. |
| **Unicycle action space** | Alpamayo predicts `(accel, curvature)` with bounds `±9.8 m/s²` and `±0.33 1/m`, then integrates. Every output is kinematically drivable **by construction**. Qwen-Drive predicts `(x, y, heading)` directly and can in principle emit a physically impossible path. **This is a theoretical advantage, not one I measured** — differencing Qwen-Drive's bf16 waypoints *looks* like it shows 32 m/s² accelerations, but pushing the ground truth through the same bf16 output quantisation produces 16 m/s² on its own, so the measurement cannot separate model from representation. See [`04_RESULTS.md`](04_RESULTS.md) §1.4. |
| **The expert sees the whole VLM** | 36 caches, one per layer, vs 8 of 32. Qwen-Drive's linear-attention layers leave no KV to read, so 75 % of its VLM is invisible to the planner — it only sees layers 3, 7, 11, …, 31. |
| **2.28 B expert vs 1.04 B** | more than twice the trajectory-decoding capacity. |
| **A 30° tele camera** | Alpamayo carries a narrow tele view alongside its three wide ones, which is what you want for distant traffic lights and lead vehicles. Qwen-Drive uses three forward views only (their FOVs are not stated in the release). |
| **Training code exists** | `NVlabs/alpamayo-recipes` + the PhysicalAI dataset. Qwen-Drive released weights and inference only. **You cannot fine-tune Qwen-Drive with what is in this repository.** |
| **Reasoning is architecturally mandatory** | In Alpamayo the trajectory is always decoded from the activations that wrote the rationale, so they cannot silently disagree. In Qwen-Drive that coupling is one of three optional modes. |

---

## 4b. How the two get 3D, mechanically

This is the part worth understanding properly, because it is where the designs genuinely
diverge — [`06_HOW_3D_PERCEPTION_WORKS.md`](06_HOW_3D_PERCEPTION_WORKS.md) has the full
walkthrough.

**Your Alpamayo head does one lift.** 80 queries cross-attend 720 image tokens taken from
decoder layer 24. 3D position is implicit: the query learns to associate token patterns with
box coordinates. There is no depth, no explicit volume, and nothing geometric about how the
four cameras combine — the queries simply see all tokens at once.

**Qwen-Drive does two lifts, in opposite directions, and fuses them.**

| | "push" (LSS / UVTR) | "pull" (BEVFormer) |
|---|---|---|
| direction | pixel → 3D | 3D → pixel |
| mechanism | 118-bin depth distribution per pixel, scattered into voxels | BEV cell projects 4 pillar points into every camera, samples there |
| reads | ViT pre-merge patches | LLM last layer |
| adaptation cost | **0.853 M** | **33.663 M** |

and the join is one line: `bev_queries = bev_embedding.weight + uvtr_bev_feat`. **The pushed
geometry volume is the initial value of the pulled queries.**

Three consequences that bear on your head:

1. **Multi-camera fusion is geometric, not learned.** `point_sampling()` has no parameters.
   It projects each BEV cell into all cameras, keeps the projections that land in-frame, and
   averages. Measured on a 6-camera frame: **86.3 % of BEV cells are seen by one camera,
   13.5 % by two, 0.2 % by none.** That 13.5 % is where a concatenate-everything design
   double-counts and this one does not.
2. **`point_sampling` is portable on its own.** You do not need the 40 000-cell BEV grid to
   borrow it. Giving your 80 queries explicit 3D reference points and projecting them into
   your four cameras is a small, self-contained change that makes fusion geometric.
3. **Explicit 3D is inspectable.** A 200×200×16 volume can be rendered, diffed against
   ground truth, and debugged. An implicit query cannot.

One measured detail worth carrying over regardless: Qwen-Drive's 900 learned query priors
span x ∈ [−49, 49] m and y ∈ [−48, 46] m but **z ∈ [−0.3, 0.9] m**. Training collapsed the
vertical degree of freedom to a 1.2 m band, because road users rest on the ground. If your 80
queries carry an unconstrained 3D reference point, most of that capacity is being spent on
height the data never uses.

---

## 5. The comparison that actually matters for your project

You are adding a perception head to Alpamayo. Here is that specific work, side by side.

| | **your Alpamayo head** | **Qwen-Drive's shipped head** |
|---|---|---|
| params | 28.44 M | 125.06 M |
| trained by | you, on 4133 samples | the authors, on nuScenes + OpenScene/nuPlan |
| tap | **one** — layer 24 of 36, image tokens, 720 × 4096 | **two** — ViT pre-merge patches (geometry) *and* the final LLM hidden states (semantics) |
| tap choice | empirical sweep; 24 beat 12 and 36 | no sweep needed — last layer for semantics, bypass the LLM entirely for geometry |
| queries | 80 | 900 |
| outputs | class, 2D box, 3D box, camera | 3D boxes **+ semantic occupancy 200×200×16 + BEV map segmentation** |
| BEV representation | none — queries cross-attend image tokens directly | explicit 200 × 200 BEV grid, BEVFormer encoder, depth-aware voxel pooling |
| status | 0.48 m train / 1.07 m val 3D centre error — **overfitting**, which is what `run_round1.sh` is currently probing | released, evaluated in the technical report |

**Read that table again.** The v10 sweep running on your GPU right now — `v10_polar`,
`v10_reg`, `v10_half`, `v10_small` — is testing whether the 2.2× train/val gap is loss
shape, regularisation, data volume, or capacity. Qwen-Drive's answer to the same question is
structural: **give the head a real BEV representation and a geometry tap that has not been
through 24 layers of language modelling.** The `v10_half` run in particular — measuring the
data-scaling slope before spending 5–7 h extracting more timestamps — is answering "should I
buy more data", and the Qwen-Drive design suggests the cheaper win is a better tap.

That does **not** mean abandon the Alpamayo work. It means:

- Your head is a **probe**; so is theirs, explicitly ("a probe of the 3D information
  accessible from the shared representations"). You are doing the same experiment they did.
- The two-tap idea is portable. Alpamayo's Qwen3-VL vision tower also has pre-merge patches
  available through a hook on its own merger. Adding a geometry tap alongside your layer-24
  semantic tap is a **contained experiment on the model you already have**, and it is the
  single most transferable idea in the Qwen-Drive release.
- Their 900 queries + BEV grid vs your 80 queries + direct image cross-attention is the
  other portable lever, and the more expensive one.

---

## 6. So: is Qwen-Drive-1.0 better than Alpamayo 1.5?

| question | answer |
|---|---|
| Is it a better **vision-language model**? | **Yes, decisively.** It kept its general ability; Alpamayo lost its. Ego3D 7.78 vs 25.31 is the number that matters for 3D work. |
| Is it a better **driver**? | **Unknown.** No head-to-head planning number exists. Its horizon is shorter and its action space has no feasibility guarantee. |
| Is it a better **platform for 3D perception research**? | **Yes.** Official multi-task BEV head, two taps, half the memory, and it leaves 12 GB free on your 3090 instead of forcing CPU offload. |
| Is it a better **base to fine-tune**? | **No.** Inference-only release, no training code. Alpamayo has published recipes. |
| Should you switch? | **No — run both.** Qwen-Drive is the better *reference* for what a shared representation should be able to do, and the cheapest way to use it is as a target and an idea source for the Alpamayo head you are already training. |

---

## 7. The honest caveats on this document

- The benchmark numbers in §2 are **transcribed from the Qwen-Drive README**, which is the
  authors evaluating themselves and their competitor. I have not reproduced any of them.
- The structural numbers in §1 and §5 **are** measured — from the downloaded checkpoints,
  by [`local/anatomy.py`](../local/anatomy.py). Those you can trust.
- No planning quality comparison in this document is empirical. The experiment that would
  make it empirical is Session 6 of [`00_START_HERE.md`](00_START_HERE.md).
