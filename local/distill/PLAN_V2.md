# PLAN v2: a student that is actually good, not one that is 90% of a weak teacher

Written 2026-09-24 after the user's push-back: "even 61% is a very bad model, not
deliverable." They are right, and the reason is structural, not a tuning gap.

## 1. Why the current approach caps out at mediocre

1. **The target was a ceiling.** "< 10% degradation vs the teacher" made the teacher the
   bar. The teacher (Qwen-Drive-1.0's 4 B perception head) scores ~68% F1@2 m against
   nuScenes ground truth: it was trained on nuPlan/WOD, nuScenes is out of distribution
   for it, and it is a VLM whose strength is language, not state-of-the-art detection.
   Matching it caps the student at mediocre by construction.
2. **The metric is home-made.** F1 at 2 m centre distance over the first 250 cached
   frames. Nobody can judge "deliverable" from it and nothing published reports it.
   nuScenes has an official evaluator (`nuscenes.eval.detection`) producing mAP / NDS
   that every camera-only paper reports; `nuscenes-devkit` is installed and imports.
3. **Single-frame.** The measured recall gap is 10-30 m (student 63-76% vs teacher
   74-91%), i.e. mid-range depth ambiguity -- exactly what temporal fusion resolves via
   motion parallax. Every camera-only line since 2022 (BEVDet4D, BEVFormer, SOLOFusion,
   StreamPETR) gets its largest single gain from temporal fusion (+5 to +10 mAP).
4. **No training recipe.** Zero augmentation, no class-balanced sampling (CBGS), no
   EMA, ~20 epochs. In every published ablation these are worth several mAP each.
5. **Backbone / resolution unspent.** ResNet-50 at 896x512, ImageNet init, single
   scale into the lift-splat. The speed budget is 2x (20.6 Hz measured vs 10 Hz needed)
   and none of it has been spent on the perception trunk.
6. **Occupancy and map inherit the same ceiling** -- they distil from the teacher for
   lack of ground truth. nuScenes ships the HD map (`maps/`, rasterisable to segmentation
   GT) and Occ3D-nuScenes occupancy labels exist for download.

What stays: the export-friendly design that makes 10 Hz possible -- scatter-based
lift-splat, dense CenterPoint head (the only detection head that trained here), top-k
decode in-graph, no deformable attention. Temporal fusion is added the same way.

## 2. The target, restated in the official metric

Camera-only, nuScenes val, published reference points (R50 unless noted):

| model | frames | mAP | NDS |
|---|---|---|---|
| BEVDet-R50 (704x256) | 1 | ~0.30 | ~0.38 |
| BEVDet4D-R50 | 2 | ~0.32 | ~0.46 |
| BEVDepth-R50, 4D | 2 | ~0.35 | ~0.48 |
| SOLOFusion-R50 | 16 | ~0.43 | ~0.53 |
| StreamPETR-R50 | stream | ~0.45 | ~0.55 |

| | mAP | NDS | plus |
|---|---|---|---|
| **deliverable v1** | **>= 0.35** | **>= 0.47** | >= 10 Hz single GPU, all five outputs in one graph |
| stretch | 0.42 | 0.52 | |

The teacher's official mAP/NDS on nuScenes val is unmeasured; Phase 0 measures it as
context. It is no longer the target.

## 3. Phases, each with a gate and a kill criterion

Screening runs stay short (20k steps); only a recipe that clears its gate gets the long
run. Every architecture change re-measures single-GPU speed before it is kept.

### Phase 0 -- fix the yardstick (1 day, no GPU training)
- Official `DetectionEval` on the nuScenes **val split** (6,019 keyframes; the cache
  covers trainval so val is available) for the teacher and for E5c/E6. Output mAP, NDS,
  and the per-class / per-range breakdowns the evaluator already produces.
- Rasterise map segmentation GT from the nuScenes map API (check `maps/expansion`).
- Confirm Occ3D-nuScenes availability and size; request the download.
- **Gate:** the teacher and student land in the published table on a comparable footing.

### Phase 1 -- training recipe (2 days)
Same architecture as E5c/E6. Add: image augmentation (flip, scale 0.8-1.2, crop),
BEV augmentation (rotation, flip, scale -- `bev_index` is recomputed per sample by
`geometry.py`, so this is a target+index transform, not a cache rebuild), CBGS sampler,
EMA weights, 24-epoch cosine schedule, and a sweep of the lidar depth-loss weight
(already implemented, never tuned).
- **Gate:** >= +4 mAP over the Phase 0 baseline. **Kill:** < +1.5.

### Phase 2 -- temporal fusion (2-3 days) -- the expected largest gain
BEVDet4D construction: warp the previous keyframe's BEV feature into the current ego
frame with the ego-motion transform (one 4-D `grid_sample`; its CUDA kernel at opset 20
was verified in this repo), concatenate, fuse with a conv. Start at 2 frames; extend to
4-8 (SOLOFusion shows long history keeps paying). Sweeps are present (164k CAM_FRONT).
The ONNX interface becomes stateful: `prev_bev` and `ego_motion` in, `bev` out --
standard for deployment and cheaper than re-running the backbone on history.
- **Gate:** >= +5 mAP and the 10-30 m recall gap to the teacher under 5 points.
  **Kill:** < +2 with 4 frames.

### Phase 3 -- spend the speed budget on the trunk (1-2 days)
ResNet-101 or ConvNeXt-T; FPN multi-scale into the lift-splat; 118 depth bins (the
teacher's) instead of 64; resolution 1408x768 if the budget allows. Spend down to
~12 Hz, not 10, to leave margin for the deployment target.
- **Gate:** >= +3 mAP at >= 10 Hz. **Kill:** speed < 10 Hz or < +1.5 mAP.

### Phase 4 -- occupancy and map on ground truth (2 days)
Occupancy against Occ3D labels (official mIoU); map against rasterised HD-map GT.
Teacher features stay as an *auxiliary* signal (the other session's feature
distillation), never as the target.
- **Gate:** occupancy mIoU and map mIoU reported in the official protocols.

### Phase 5 -- final (1-2 days)
Long run with the full recipe; export; clean single-GPU speed on an idle card; report in
official metrics with the teacher and the published R50 rows beside the student.

**Total: ~10-12 days of machine time**, largely unattended. The E5c/E6 line keeps
running meanwhile as the baseline v2 must beat.

## 4. Rules carried over from the failures (PLAN.md sections 6-17)

1. Verify the instrument before the reading: same weights must score the same on any
   evaluation path; config comes from the checkpoint; baselines are fit on non-eval data
   and scored on the exact eval frames.
2. A loss printing exactly 0.0000 is a target bug.
3. Bisect before optimising: a 20-minute probe found the head was the bottleneck after
   ~12 GPU-hours of tuning the wrong component.
4. Smoke-test in the launch configuration (DDP), and an edit plus the launch that depends
   on it go in one `&&` chain ending with a check of the number the edit should change.
5. No metric without its baseline beside it.

## 5. Decisions needed before Phase 0 starts

1. Accept the reframed target: official nuScenes mAP/NDS, absolute bar (v1: 0.35 / 0.47),
   teacher as context only.
2. Time: ~10-12 days of mostly unattended machine time.
3. Interface: temporal fusion makes the ONNX stateful (previous BEV in/out). Acceptable?
4. Data: permission to download Occ3D-nuScenes labels for Phase 4.

## 6. Amendments (2026-09-24 23:50)

**Temporal fusion is unblocked on the data side.** `frame.json` has no previous-keyframe
link or global pose, but the devkit does (`sample.prev`, `ego_pose`). A one-off index,
`data/distill/temporal_index.json`, now stores each cached frame's previous cached
keyframe and its ego-to-global 4x4; the warp is `T = inv(E_curr) @ E_prev` (prev-ego ->
current-ego), applied to the previous BEV with one 4-D `grid_sample`. This is needed under
either interface (stateful, or two-frame-in-graph), so it was built ahead of decision 3.

**The official metric needs a taxonomy decision (decision 5).** Teacher/student classes:
`vehicle, czone_sign, bicycle, generic_object, pedestrian, traffic_cone, barrier`.
nuScenes detection classes: `car, truck, bus, trailer, construction_vehicle, pedestrian,
motorcycle, bicycle, traffic_cone, barrier`. `vehicle` merges five; `czone_sign` and
`generic_object` are not nuScenes classes; `motorcycle` is absent. A 7-class score is not
comparable to the published table. Since the student trains on ground truth it is not
bound to the teacher's taxonomy: **option A** keep 7 classes and report a 7-class mAP
(honest, not comparable); **option B** switch the student head to the 10 nuScenes classes
(fine labels are in the devkit; `nusc_gt_boxes.py` collapses them and would emit the
fine class instead), making the official mAP/NDS apply cleanly. B costs nothing extra in
compute and is the recommendation for "deliverable".
