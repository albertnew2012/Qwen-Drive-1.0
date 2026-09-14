# Push and Pull — the two lifts, made concrete

> **The one question this answers:** a camera flattens 3D to 2D. Something has to undo
> that. This repo undoes it **twice, by opposite methods**, and fuses the results.
>
> Read this with [outputs/two_lifts.png](../outputs/two_lifts.png) open (debug config
> **1a1** redraws it).

---

## 0. First: what is actually moving?

This is the part that trips everyone, so before any geometry:

**Nothing that moves is a pixel, a colour, or an image patch.** What moves is a
**256-number vector** — a learned description of what is at one small square of one camera
image. Think of it as the model's private notes: *"pale metal panel, horizontal edge below,
wheel-ish thing at lower left."* 256 numbers, meaningless individually, and never converted
back into a picture.

The front camera is cut into a **32 x 56 grid** of such squares. Six cameras gives
`6 x 32 x 56 = 10,752` squares, each holding a 256-vector:

```
vit_neck output    (6, 256, 32, 56)      <- 6 cameras, 256 channels, 32x56 grid
```

That tensor is the *entire* input to the push lift. Everything below is about **where those
10,752 vectors should go in 3D**.

---

## 0b. A word about the words (three vocabularies, one operation)

**"Push" and "pull" are teaching words, not the repo's words.** Grep the source and you get
**zero** hits for `lift`, `push`, `pull` or `splat`. The repo says **"view transform"** and
otherwise names things by paper lineage: `Uni3DVoxelPoolDepth` (**UVTR**) and
`BEVFormerEncoder` (**BEVFormer**).

**And "lift" is not the umbrella term for both** - that is worth knowing before you read
papers. It comes from **Lift-Splat-Shoot** (Philion & Fidler, ECCV 2020), which names three
steps: *Lift* (give each pixel a depth distribution, making a frustum of 3D points), *Splat*
(pool those into BEV cells), *Shoot* (plan). So **"lift" describes the push side only**.
BEVFormer never calls its operation a lift; it is *spatial cross-attention* over *BEV
queries*.

| this doc | the repo | the papers | surveys also say |
|---|---|---|---|
| **push** | UVTR view transform, `view_transform.py` | LSS "lift" + "splat"; BEVDet, BEVDepth, UVTR | forward projection, depth-based, **scatter** |
| **pull** | BEVFormer encoder, `bev_encoder.py` | BEVFormer spatial cross-attention; DETR3D, PETR | backward projection, query-based, **gather** |
| both | **"view transform"** | - | view transformation, 2D-to-BEV |

`push`/`pull` is borrowed from graphics, where forward texture mapping *pushes* source pixels
to destinations and inverse mapping *pulls* from sources - the same scatter-vs-gather split.
It is widely used in conversation and almost never in print.

The title of `two_lifts.png` says "lifts to BEV twice", which is the loose colloquial usage.
Precisely it is **two view transforms, one of which is a lift**.

---

## 1. PUSH — "I'm an image square. Where do I belong?"

### Pushed **out of**: the image plane. Pushed **into**: a 3D voxel volume.

The destination is a box around the car, `200 x 200` cells on the ground (±50 m at 0.5 m)
and **16 slices of height**. Every one of those `200 x 200 x 16 = 640,000` cells holds its
own 256-vector:

```
voxel volume       (1, 256, 16, 200, 200)
```

So: **10,752 vectors in, 640,000 cells out.**

### The problem, and the trick

Take one square of the front camera — say the one on the white van's side panel. We know the
*direction* it lies in exactly (that is just camera calibration). We do **not** know how far
along that direction it sits. Is the van 8 m away or 30 m?

Nothing in a single camera image can answer that. So the model does not answer it. It
**hedges**, with a probability distribution over 118 candidate depths from 1.0 m to 60.0 m
in 0.5 m steps:

```
depth distribution  (6, 118, 32, 56)     sums to 1.0 along the 118 axis
```

Then it deposits that square's 256-vector into **every** voxel along the ray, each scaled by
how likely that depth is:

```python
out[b, x, y, z, c] += feats[img, c, h, w] * depth[img, d, h, w]
#                     ^ the 256-vector      ^ how much of it goes to THIS depth
```

If the model is 60 % sure the van is at 8.5 m, 60 % of its vector lands in the voxel at
8.5 m, and the rest is scattered along the rest of the ray.

**That smearing is exactly what you see as streaks in panel 2.** It is not noise and not a
rendering artifact — it is uncertainty made visible. Panel 2 of
[backprojection.png](../outputs/backprojection.png) proves it: force the distribution
uniform and the streaks explode (25,692 cells lit); force it to a single point and they
collapse (395). Same operation as unfiltered backprojection in PET or CT.

Only **52.6 %** of frustum points land inside the box at all; the rest are past the walls or
above the ceiling and are dropped.

### Then squash the height away

Detection, mapping and planning all work on the **ground plane**, not in a 3D volume. So the
16 height slices are collapsed by a 1x1 convolution:

```
(1, 256, 16, 200, 200)   ->   (40000, 256)      # 200 x 200 ground cells, 256 numbers each
```

That result is `uvtr_bev_feat`. **This is what push produces.** It knows *what* is around the
car, and is fuzzy about *where*.

> **Cost note.** Before the cameras are merged, this stage holds one volume *per camera*:
> `(1, 6, 256, 16, 200, 200)` = **1875 MiB**, the single largest tensor in the pipeline.

---

## 2. PULL — "I'm a spot on the ground. Who can see me?"

### Pulled **into**: the same 200 x 200 BEV grid. Pulled **from**: the camera images.

Now run the arrow backwards. Stand at one BEV cell — say 12 m ahead, 3 m left. Its position
is **known exactly**; there is nothing to guess. So:

1. Take 4 sample points up a vertical **pillar** at that spot (4 different heights) — because
   a 2-D ground cell still has to decide which height to look at.
2. Project each into all six cameras with `lidar2img @ inv(lidar2ego)` — pure calibration,
   **no network, no depth estimate, no guessing**.
3. Wherever they land in an image, sample the feature there and mix it into the cell.

Panel 3 of the figure is precisely this step's bookkeeping: which cameras can see each BEV
cell. **99.8 %** of cells are seen by at least one camera; **13.5 %** by two (the yellow
overlap wedges); the dark patch at the centre is the blind spot around the vehicle.

Pull's geometry is **exact**. Its weakness is the mirror image of push's: it knows precisely
*where* it is standing, but has no idea whether anything is *there*.

---

## 3. The join — one line

Push knows **what**, and guesses **where**.
Pull knows **where**, and must ask **what**.

So push's output becomes pull's **starting value**:

```python
# heads.py:242-245
bev_queries = self.bev_embedding.weight      # a learned prior, 40000 x 256
bev_queries = bev_queries + uvtr_bev_feat    # + what push deposited, 40000 x 256
```

Both are `40000 x 256`, so this is a plain element-wise sum. The pushed volume is not a
parallel branch to be concatenated later — it is **the initial value of the attention
queries**. Six encoder layers then refine it.

**Geometry seeds; semantics refine.** That is the thesis of the whole figure.

You can watch the refinement work: panel 2 (push alone) is streaky and diffuse; panel 4
(after 6 layers of pull) is tight and blobby, with ground-truth objects landing at the
**98th percentile** of feature magnitude.

---

## 4. Why not just one of them?

| | knows *where* | knows *what* | fails how |
|---|---|---|---|
| **push** (LSS / UVTR) | guessed, from DepthNet | yes, carries real features | smears features along rays when depth is uncertain |
| **pull** (BEVFormer) | exact, from calibration | only what it samples | nothing to sample *toward*; starts from a blank prior |

Push alone gives a blurry but populated scene. Pull alone starts from nothing and must
discover the world through attention. Together: push proposes, pull refines.

**Two taps feed them, and the split is deliberate.** Push is fed from the **ViT** (raw
patches, before the language model) through a tiny `vit_neck` — **0.853 M** parameters. Pull
is fed from the **LLM's last layer** through `adaptor` — **33.663 M**, *39x more*. Geometry
comes from features that never entered the language model and barely need adapting; meaning
comes from features that went all the way through it. See
[01_MODEL_STRUCTURE.md](01_MODEL_STRUCTURE.md).

---

## 5. Every shape, in order

| stage | tensor | what it means |
|---|---|---|
| ViT tap | `(6, 32, 56, 1024)` | raw patch features, 6 cameras |
| `vit_neck` | `(6, 256, 32, 56)` | **what push carries** — 10,752 vectors of 256 |
| `depth_net` | `(6, 118, 32, 56)` | **where push guesses** — a distribution per square |
| frustum mask | `(1, 1, 6, 118, 32, 56)` | which of those land in the box (52.6 %) |
| per-camera volume | `(1, 6, 256, 16, 200, 200)` | 1875 MiB, largest tensor in the pipeline |
| merged volume | `(1, 256, 16, 200, 200)` | 640,000 voxels |
| `uvtr_bev_feat` | `(40000, 256)` | **push's output**, height squashed away |
| `bev_embedding.weight` | `(40000, 256)` | the learned prior pull starts from |
| **sum of the two** | `(40000, 256)` | **the seed** — heads.py:242-245 |
| after 6 encoder layers | `(40000, 256)` | **panel 4**, and the only thing the heads read |

---

## 6. Three traps

1. **The depth distribution is bimodal.** A correct peak at the true distance *plus* a large
   spike at the 59.5 m clip that means "discard this". Average the two and road 5 m ahead
   reads 56 m — *anti*-correlated with lidar. See [06 §2.1](06_HOW_3D_PERCEPTION_WORKS.md).
2. **Frames.** `gt.npz` boxes are in the **ego** frame; model predictions come out in the
   **lidar** frame, and for nuScenes those differ by a **90 degree yaw**.
3. **Axes.** The BEV tensor is `[y, x]` and the repo draws forward **up**
   (`visualize.py`: *"right is -Y, up is +X"*). Get it wrong and the picture is a quarter
   turn off, with no error message.

---

## 7. Run it

| config | what it shows |
|---|---|
| **1a** | live numbers behind every claim here (sections A-H) |
| **1a1** | redraws [two_lifts.png](../outputs/two_lifts.png) |
| **1a3** | the backprojection proof — streaks are depth uncertainty |
| **1b** | all 8 stages with the shapes in §5 |
