# Porting Qwen-Drive's two-tap idea to your Alpamayo detection head

The single most transferable thing in the Qwen-Drive release is **where its perception head
reads from**. This document is the concrete version of Session 7 in
[`00_START_HERE.md`](00_START_HERE.md).

---

## 1. The difference in one picture

```
YOUR ALPAMAYO HEAD                          QWEN-DRIVE'S HEAD
                                            
 vision tower (27 blocks)                    vision tower (24 blocks)
        │                                           │
        │  post-merge tokens                        ├── PRE-MERGE patches ──┐
        ▼                                           │   after merger.norm   │
 language model, 36 layers                          ▼                       │  GEOMETRY
        │                                    language model, 32 layers      │
        ├── TAP layer 24 ──► 720 x 4096              │                      │
        │   (found by sweep: beat 12 and 36)         └── TAP last layer ────┤  SEMANTICS
        ▼                                                after final norm   │
   36 layers of trajectory work                                             ▼
                                                             DepthNet + voxel pooling
   ONE tap. Semantics only.                                   + BEVFormer encoder
   Geometry has been through 24 layers                
   of language modelling before you see it.          TWO taps. Geometry never
                                                     enters the language model.
```

Your head reads image tokens **after 24 decoder layers**. Those layers are optimised for
next-token prediction, and the tap layer had to be found by sweep precisely because there is
a trade-off: too early and the features are not semantic, too late and they are specialised
for language. Qwen-Drive sidesteps the trade-off — it takes semantics from the **last** layer
(no sweep needed) and gets geometry from a **different source entirely**, the ViT patch grid
that never passed through the language model at all.

## 2. Why the pre-merge patches are the right geometry source

Three properties, all of which your layer-24 tap lacks:

| | pre-merge ViT patches | your layer-24 tokens |
|---|---|---|
| spatial resolution | full patch grid (Qwen-Drive: 32 x 56 per camera) | 2x2-merged (Qwen-Drive: 16 x 28) |
| passed through the LLM | **no** | yes, 24 layers |
| what they encode | appearance + position, pre-fusion | fused, language-conditioned semantics |

Qwen-Drive spends only **0.853 M** parameters adapting the ViT stream (`vit_neck`) against
**33.663 M** for the LLM stream (`adaptor`) — the ViT features need far less adaptation
because they are already close to what a view transform wants.

## 3. You have already extracted both taps

**This section was written before checking `alpamayo1.5/perception/features/`. The extraction
step is done.**

```
features/premerge/*.layer0.npy    (10, 2880, 1152) fp16   42 GB, 675 clips, 6659 samples
   json: patch_grid [20, 36], tokens_per_image 720, layers [0]
   2880 = 4 cameras x 720 patches      <- GEOMETRY tap, pre-merge

features/sweep/*.layer12.npy      (10,  720, 4096) fp16
   720 = 4 cameras x 180 merged tokens <- SEMANTICS tap, layer 24 (and 12/36 in the sweep)
```

That is exactly Qwen-Drive's pairing, at comparable ratios:

| | your Alpamayo | Qwen-Drive |
|---|---|---|
| geometry tap | pre-merge, **20 x 36**, dim **1152** | pre-merge, 32 x 56, dim 1024 |
| semantics tap | post-merge, 10 x 18, dim 4096 | post-merge, 16 x 28, dim 2560 |
| spatial ratio | **4x** | 4x |

`extract_premerge.log` also records something worth knowing: the run used the **vision tower
only** — 576.4 M params, **1.64 GiB peak VRAM**, 2.31 samples/s, 48 minutes for all 6659
samples. Extracting geometry features does not require the 11 B model, because they come from
before the language model. **The geometry tap is cheap to re-extract and cheap to iterate on**,
unlike the layer-24 features which need a full VLM prefill.

So the open work is not extraction. It is **fusion**: what the head does with two feature
streams of different resolution and width. Qwen-Drive's answer, and the parameter budget it
assigns to each, is the informative part:

| | module | params | what it does |
|---|---|---|---|
| geometry | `vit_neck` SimpleFPN, scales (1.0,) | **0.853 M** | one scale, minimal adaptation |
| | `DepthNet` | 5.0 M | per-pixel depth distribution, 118 bins |
| | `voxel_pool_depth` | 0 | unproject into a 200x200x16 volume |
| semantics | `adaptor` SimpleFPN, scales (4.0, 2.0, 1.0, 0.5) | **33.663 M** | four scales, heavy adaptation |
| both | BEVFormer encoder x6 | 61.6 M | BEV queries attend the semantic stream; the geometry volume initialises them |

Note the asymmetry: **0.853 M to adapt geometry, 33.663 M to adapt semantics.** The pre-merge
features are already close to what a view transform wants; the LLM features are not.

### The cheapest version that tests the idea

You do not need the depth network or voxel pooling to find out whether the geometry tap
carries signal your head is missing. The minimal experiment reuses your existing
`nn.TransformerDecoderLayer` stack and just gives the queries **more to cross-attend to**:

```python
# current: queries cross-attend 720 x 4096 (layer 24, 4 cams x 180)
# proposed: additionally cross-attend 2880 x 1152 (pre-merge, 4 cams x 720)

self.in_proj_sem  = nn.Linear(4096, 512)     # existing
self.in_proj_geo  = nn.Linear(1152, 512)     # new, 0.59 M params
# separate 2-D position embeddings, since the grids differ:
#   semantics 10 x 18   geometry 20 x 36
memory = torch.cat([
    self.in_proj_sem(sem) + pos_sem,         # [B,  720, 512]
    self.in_proj_geo(geo) + pos_geo,         # [B, 2880, 512]
], dim=1)                                    # [B, 3600, 512]
```

That is **+0.59 M parameters** on a 28.44 M head, and 5x the memory length. If 3600 keys is
too slow, drop the geometry stream to the last frame per camera as you already do for
semantics, or 2x2-pool it back to 10 x 18 and concatenate on the channel axis instead — the
latter keeps the sequence at 720 and costs nothing in attention time.

## 4. What to expect, and how to not fool yourself

Your `v10_*` sweep is testing four hypotheses against a **0.48 m train / 1.07 m val** 3D
centre error gap: loss shape, regularisation, data volume, capacity. **A second tap is none
of those.** It changes the *information available*, not the fit.

So the prediction is specific: a geometry tap should help **val** error more than train
error, because it supplies depth-relevant signal the head currently has to infer from
language-conditioned features. If it improves train and val equally, it is acting as extra
capacity, not extra information — and `v10_small` already tells you whether capacity is the
binding constraint.

Two cautions from this session's own measurements:

- **Watch out for per-scene variance.** On the Qwen-Drive demo set, the reasoning-vs-direct
  effect flipped sign between two scenes with a mean change of −0.002 m. If your val split
  is small, a 2-way comparison can be decided by which clips landed where. Report paired
  per-clip deltas, not just split means.
- **Do not read bias off endpoints.** Measuring the trajectory error at the final waypoint
  showed a systematic −0.85 m under-shoot that vanished under a trajectory mean. The
  analogous trap for detection is reading bias off the furthest range bin.

## 5. The cheaper experiment first

Before building the tap, run the ablation that costs nothing: your head currently reads
**720 x 4096** at layer 24. Qwen-Drive reads **N x 16 x 28 x 2560** semantics *plus*
**N x 32 x 56 x 1024** geometry. The intermediate step is to keep your single tap but stop
throwing away spatial resolution — check whether your `col_embed(18) ⊕ row_embed(10)`
factorised position is actually recovering the grid, by ablating to a flat
`Embedding(180)` and confirming it gets **worse**. Your notes say a flat embedding made the
head learn the row/col split from scratch; that is evidence the spatial structure is
load-bearing, which is the same evidence that says a higher-resolution geometry tap should
help.
