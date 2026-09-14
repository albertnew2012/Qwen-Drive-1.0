# Every figure, viewable inline

**Open this file and press `Ctrl+Shift+V`** (or the preview icon, top right) — VS Code's
markdown preview renders images inline. Clicking an image path in the chat may not open it;
this always works.

Videos cannot render inline in markdown preview. Their paths are listed with a still frame
above each one — click the path in VS Code's **Explorer** to open the video, or run the
`xdg-open` command shown.

---

## 1. The whole architecture in one image

The single most useful picture in the repo. Left to right: what the "push" lift takes in,
what it produces, the geometry the "pull" lift uses, and the fusion the three heads read.

![the two lifts](../outputs/two_lifts.png)

**How to read it**

| panel | what it shows |
|---|---|
| 0 | The front camera at 896x512 - exactly what the VLM sees. Panels 0 and 1 are **one** camera; panels 2, 3, 4 are **all six**. |
| 1 | DepthNet's depth on that same camera. It predicts a *distribution* over 118 bins from 1-60 m, and that distribution is **bimodal** - do not take its plain mean (see figure 1b). Grey = the network deposits nothing there, which is the sky. |
| 2 | That depth used to scatter features into voxels, collapsed to BEV. The **radial streaks** are the signature of LSS-style lifting: each pixel's feature is smeared along its ray. |
| 3 | Which cameras see each BEV cell. **Six yellow wedges** = the overlaps between adjacent cameras (13.5% of cells, averaged over both). Teal = one camera. The dark patch at the centre is the **blind spot right around the vehicle**. |
| 4 | The BEV after 6 encoder layers. This single tensor feeds detection, occupancy *and* mapping. Green circles are ground-truth objects; they land at the **98th percentile** of feature magnitude, so the blobs are real. |

**Axes**: all BEV panels put **ego-forward UP** and +y (the vehicle's left) on the left, matching `src/qwen_drive_perception/visualize.py` ("right is -Y, up is +X") and the camera image above them. Note the frame trap: `gt.npz` boxes are in the **ego** frame, model predictions come out in the **lidar** frame, and for nuScenes those differ by a **90 degree yaw** - mixing them silently rotates one set a quarter turn.

Regenerate: debug config **1a1**, or
`python study/scripts/09_perception_internals.py --figure outputs/two_lifts.png`
(**1a** is the same script without the figure - the section A-H walkthrough only.)

---

## 1b. Is that predicted depth actually correct?

Worth its own figure, because the obvious way to read panel 1 above gives the wrong answer,
and the demo frames ship lidar so it can be settled rather than argued.

![depth vs lidar](../outputs/depth_vs_lidar.png)

DepthNet's 118-bin distribution is **bimodal**: a correct peak at the true distance plus a
big spike at the 59.5 m far clip, with the 15-40 m band empty. Collapse it with a plain
expectation and road 5 m ahead reads 56 m - *anti*-correlated with lidar at **-0.547**.
Weight each bin by whether it survives the BEV range mask and the same cell reads 4.9 m,
against a lidar truth of 5.2 m.

| reducer | corr vs lidar | median err |
|---|---|---|
| naive expectation | **-0.547** | 36.1 m |
| argmax over all bins | -0.355 | 47.8 m |
| **mask-weighted** (used everywhere) | **+0.788** | **1.4 m** |

**Score only what the grid can hold.** The BEV volume has a z ceiling, so an upward-looking
ray leaves it at some depth and the network is not permitted to place anything past that.
**85 % of lidar cells beyond 40 m are outside the volume**, so comparing there measures the
grid, not the model:

| | corr | median err | bias |
|---|---|---|---|
| all lidar cells | +0.788 | 1.4 m | +0.0 m |
| **truth inside the volume** | **+0.879** | **1.2 m** | +0.2 m |

Raw bias at 40-60 m is -22.7 m; on representable cells it is **-5.3 m**. The near road is
exact (-0.1 m). The top third of the image looks worst and mostly is not real: in rows 8-12
the ray exits at 24.2 m while the wall is at 33.2 m, so 76 % of those cells are unreachable.

**What panel 1 draws.** Cells are hidden where `trunc > 0.15` (the network wanted a distance
past where the ray leaves the BEV volume, so the truncated mean is a ceiling artifact) or
`kept < 0.005` (nothing survives at all - row 0 is 97.7 % reject). Drawn cells have a median
error of **1.2 m** against lidar; hidden cells **14.2 m**. Note that a low `kept` alone is
*not* a defect: the near road deposits only ~11 % of its mass yet is the most accurate part
of the image, so gating on `kept` throws away the best cells.

Two things that look wrong by eye but are not: the red horizon band is **correct** (lidar
median 36.0 m there), and near objects are never painted far (**0 %** of cells lidar puts
under 15 m are predicted beyond 25 m).

Regenerate: **1a2**, or
`python study/scripts/10_depth_vs_lidar.py --figure outputs/depth_vs_lidar.png`

---

## 1c. Why panel 2 is full of streaks (it is a backprojection)

![backprojection](../outputs/backprojection.png)

The push stream is a **weighted backprojection** - the same operation as unfiltered
backprojection in PET or CT. Each pixel's feature is deposited along its camera ray,
weighted by DepthNet's 118-bin distribution. The streaks are not noise and not a rendering
artifact: they are the depth *uncertainty*, smeared along the ray.

Proved by changing one thing only, a temperature on the DepthNet logits:

| depth distribution | effective cells lit | % energy within 3 m of a real object |
|---|---|---|
| uniform (= pure backprojection) | **25,692** | 11.7 % |
| **as predicted** | 4,197 | **14.5 %** |
| sharpened x8 | 413 | 7.6 % |
| near one-hot | **395** | 7.3 % |

Sharpening collapses the streaks - **65x** concentration from uniform to one-hot - so the
streaks are depth uncertainty and nothing else.

The second column is the surprise: energy landing **on objects** peaks at the model's *own*
sharpness and falls either side. A hard argmax is *worse* than hedging, because for ~60 % of
cells the argmax is the 59.5 m reject bin, so committing throws the features at the far clip
- visible as the bright corner bands in the right-hand panels.

Where the PET analogy stops: PET measures line integrals through a transparent volume, so
activity genuinely is distributed along the ray, and a full detector ring gives hundreds of
angles. Here only one point per ray is real (the first opaque surface), and there are only
6 view directions with 13.5 % overlap - far more ill-posed angularly. The cleanup is also
not iterative reconstruction: it is the 6 learned encoder layers of the pull stream, which
is what turns panel 2 into panel 4.

Regenerate: config **1a3**, or
`python study/scripts/11_backprojection.py`

---

## 2. One frame's full perception output

Camera ring with projected 3D boxes, BEV with lidar, then occupancy and map each shown as
**prediction next to ground truth**.

![perception summary](../outputs/perception_walkthrough.png)

Regenerate: config **2f**.

---

## 3. The session video — scene 0

148 frames at the 10 Hz sweep rate, 0.75x real speed. Every frame is the layout above **plus**
the predicted trajectory drawn on the road in every camera, and the chain of thought that
produced it.

![session scene 0](../outputs/stills/session_scene0.png)

```
outputs/nuscenes_session_0.75x.mp4          19.7s   ADE mean 1.39 m
outputs/nuscenes_session_1.0x.mp4           14.8s   same frames, real time
xdg-open outputs/nuscenes_session_0.75x.mp4
```

Cyan = predicted 5 s path. Amber dashed = the path actually driven. Both are projected onto
the ground plane through the same `lidar2img` the boxes use.

---

## 4. The session video — scene 1

Busier: pedestrians, a cyclist, a yellow light. 152 frames.

![session scene 1](../outputs/stills/session_scene1.png)

```
outputs/nuscenes_session_scene1_0.75x.mp4   20.3s   ADE mean 2.24 m
xdg-open outputs/nuscenes_session_scene1_0.75x.mp4
```

---

## 5. The perception pipeline, stage by stage

Eight shots walking the pipeline: camera ring, both taps as PCA-to-RGB, the FPN scales,
DepthNet, the frustum, the voxel volume, the BEV queries, and the three outputs.

![perception stages](../outputs/stills/perception_stages.png)

```
outputs/perception_walkthrough.mp4          57.9s
xdg-open outputs/perception_walkthrough.mp4
```

---

## 6. The bundled-demo video

All three heads on the four demo scenes plus the VQA probes.

![demo](../outputs/stills/demo.png)

```
outputs/qwen_drive_demo.mp4                 77.4s
xdg-open outputs/qwen_drive_demo.mp4
```
