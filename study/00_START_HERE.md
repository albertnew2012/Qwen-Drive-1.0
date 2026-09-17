# Qwen-Drive-1.0 — start here

One Qwen3.5-4B vision-language model looks at the road. **Three heads read the same
representation**: a BEV perception head (3D boxes + occupancy + map), a planning expert
(a 5-second trajectory), and the model's own untouched LLM decoder (text).

Total **5.70 B parameters, 11.66 GB in bf16** — about half of Alpamayo 1.5, and it leaves
12 GB free on your 3090.

---

## The 60-second mental model

```
3 or 6-8 cameras ──► Qwen3.5-4B VLM (4.54 B, frozen and shared) ──┬─► LLM decoder ──► text
                          │                    │                  │
                   ViT pre-merge          LLM last layer          └─► 8 KV caches
                   (GEOMETRY)             (SEMANTICS)                      │
                          └────────┬───────────┘                           ▼
                                   ▼                          Planning Expert (1.04 B)
                        BEV Perception Head (125 M)           10 flow-matching steps
                                   │                                       │
                 ┌─────────────────┼─────────────────┐                     ▼
                 ▼                 ▼                 ▼            50 waypoints, 5 s
          300 boxes         200x200x16 occ      200x400 map
```

Two ideas carry the whole design:

1. **One frozen representation, three independent readers.** No head feeds another. The VLM
   is byte-identical in all three modes. That makes the perception head a clean *probe* —
   the same experiment you are running on Alpamayo.
2. **Perception lifts to BEV twice, by opposite methods, and fuses them.** A depth-based
   "push" (pixel → voxel) seeds a query-based "pull" (BEV cell → camera). This is the single
   most transferable idea in the repo. [`07`](07_PUSH_AND_PULL.md) explains what actually
   moves and where it goes; [`06`](06_HOW_3D_PERCEPTION_WORKS.md) is the full mechanism.

---

## Runbooks - commands, not theory

* [**TRAINING.md**](../TRAINING.md) - the recipe and every command to train
* [**ONNX_EXPORT.md**](../ONNX_EXPORT.md) - the full export and how to validate it

---

## Read in this order

| # | doc | what you get |
|---|---|---|
| **06** | [**How 3D perception and online mapping work**](06_HOW_3D_PERCEPTION_WORKS.md) | **the main event** — the two lifts, the seeding, all three heads, mechanism by mechanism |
| **07** | [**Push and Pull, made concrete**](07_PUSH_AND_PULL.md) | start here if "push" and "pull" are words rather than pictures. What literally moves, out of what, into what, with every shape. |
| **08** | [**Training this repo**](08_TRAINING.md) | the release cannot back-propagate at all - two kernels have no backward. What that means, and a working 3-stage pipeline. |
| **11** | [**Grounding the plan in perception**](11_GROUNDING_PLAN_IN_PERCEPTION.md) | no shipped loss reads both heads, so the planner may drive through a car it has correctly boxed. A stage that closes the loop - the three ways of writing the cost that quietly do not work, and an honest null on held-out scenes. |
| **09** | [**ONNX export**](09_ONNX_EXPORT.md) | both heads exported and verified, and the silent bug that made a passing export return wrong numbers. |
| **10** | [**Expected results**](10_EXPECTED_RESULTS.md) | every measured number, what counts as a regression, and how to re-verify. |
| 01 | [Model structure](01_MODEL_STRUCTURE.md) | every parameter located, token layout, the hybrid attention stack, the planning expert |
| 02 | [vs Alpamayo 1.5](02_VS_ALPAMAYO_1_5.md) | is it better, at what, and what the benchmark tables do and do not show |
| 05 | [Porting the two-tap idea](05_PORTING_THE_TWO_TAP_IDEA.md) | the concrete change to your Alpamayo head, written against features you already extracted |
| 04 | [Measured results](04_RESULTS.md) | ADE/FDE, timings, and seven corrections where a real measurement supported a wrong conclusion |
| 03 | [Local setup](03_LOCAL_SETUP.md) | the environment, the one source change, the dtype trap |

---

## Run in this order  (VS Code: F5, pick the config)

| config | what it shows |
|---|---|
| **1a** | **Perception internals** — the two lifts, camera-overlap statistics, NMS-free decoding, printed from a live forward. Read alongside doc 06. |
| 1b | The eight perception stages with every intermediate tensor |
| 1d / 1e | The planning expert unrolled; **1e proves it bit-exact** against the library |
| 1f | Reasoning mode — the chain of thought that conditions the trajectory |
| 2a | **The session video**: perception + trajectory + chain of thought over a real nuScenes sequence |
| 3e | How many flow-matching steps the sampler actually needs (answer: fewer than 10) |

---

## Look at these first (no running required)

> **Images do not open from chat links.** Open
> [`study/FIGURES.md`](FIGURES.md) and press **Ctrl+Shift+V** — VS Code's markdown preview
> renders every figure inline, with a still frame for each video.

| file | what it shows |
|---|---|
| [`outputs/two_lifts.png`](../outputs/two_lifts.png) | **the whole design in one image** — the depth input, the pushed volume, the six camera-overlap wedges, and the fused BEV |
| [`outputs/nuscenes_session_0.75x.mp4`](../outputs/nuscenes_session_0.75x.mp4) | **148 frames at the 10 Hz sweep rate**, 0.75x real speed: camera ring + BEV + occupancy + map, predicted trajectory drawn on the road in every camera, chain of thought beside it |
| [`outputs/nuscenes_session_scene1_0.75x.mp4`](../outputs/nuscenes_session_scene1_0.75x.mp4) | the same on a busier scene (pedestrians, cyclist, yellow light) |
| [`outputs/perception_walkthrough.png`](../outputs/perception_walkthrough.png) | one frame's full perception output against ground truth |
| [`outputs/perception_walkthrough.mp4`](../outputs/perception_walkthrough.mp4) | the eight perception stages, narrated |

---

## The numbers worth memorising

| | |
|---|---|
| VLM | Qwen3.5-4B, **4.5393 B**, 32 layers, hybrid |
| — of which **linear attention** | 24 layers, **no KV cache** |
| — of which **full attention** | 8 layers `[3,7,…,31]` — the only ones the planner can read |
| Planning expert | **1.0398 B**, 32 layers, 4 share each VLM cache |
| Perception head | **125.06 M**, fp32 |
| — geometry adaptation (`vit_neck`) | **0.853 M** |
| — semantic adaptation (`adaptor`) | **33.663 M** (39×) |
| BEV grid | 200 × 200 @ 0.512 m, ego frame |
| Detection | 900 queries → 300 boxes, 7 classes, **no NMS** |
| Occupancy | 200 × 200 × 16, 10 classes |
| Map | 200 × 400 @ 0.15 m, 6 classes, **raster not vectors** |
| Trajectory | 50 waypoints @ 10 Hz = **5.0 s**, `(x, y, heading)` |
| Inference, 3090 | planning **~2 s**, perception **~2 s/frame** |

---

## Six things that will bite you

1. **`num_inference_steps` > 10 silently breaks the sampler.** It is coupled to
   `min_one_minus_t = 0.1`; the defaults sit exactly on the boundary. At 20 steps ADE goes
   0.275 → 3.529 while *FDE improves*. [04 §1.6](04_RESULTS.md)
2. **Do not evaluate `planner-rl` with ADE.** RL optimised PDMS/RFS and *degraded* minADE —
   the README's own table shows 0.34 → 0.38. Use `planner-sft` for displacement work.
3. **Predictions come out in the lidar frame; the packed ground truth is in the ego frame.**
   Mixing them rotates every box ~90° on nuScenes.
4. **You cannot differentiate a bf16 trajectory twice.** Waypoints are bf16, so acceleration
   and jerk computed from them measure the number format, not the model. [04 §1.4](04_RESULTS.md)
5. **`VQA_DECODE_DEFAULTS` uses `top_k=1, repetition_penalty=1.0`** — pure greedy. Correct for
   reproducing benchmarks, and it loops forever on open-ended enumeration.
6. **Perception and planning are never trained against each other.** No loss in the recipe
   reads both heads, so a trajectory that contradicts the model's own occupancy grid costs
   nothing. [11](11_GROUNDING_PLAN_IN_PERCEPTION.md)

---

## What is still open

- **No head-to-head planning number against Alpamayo exists** — not in either release, not in
  the literature. Both models can run on NVIDIA PhysicalAI and you have 80 GB of it cached.
- **nuScenes planning is out-of-distribution** for this model (trained on NAVSIM / WOD-E2E /
  PhysicalAI), which is why the session video shows ADE ~2.2 m rather than ~0.35 m. Perception
  on nuScenes is *in*-distribution — possibly on training data.
- **No training code ships with this repo.** Anything you want to train on top has to be your
  own head, exactly like the Alpamayo one.
