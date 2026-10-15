# Experiment ledger (generated 2026-10-02 06:47 by experiments_ledger.py)

Teacher references: f1 val scenes (eval250) = 0.62; f1 frame split (sorted 250) = 0.682; official mAP / NDS = 0.2155 / 0.222; trajectory ADE (val scenes, 161 frames) = 1.629 m; occupancy visible mIoU on Occ3D's native grid (eval250) = 0.270 (the 0.032 in older records is a grid-mismatch artefact).

F1 = detection F1 at 2 m centre distance against nuScenes GT (student at its best threshold), retained % = student F1 / teacher F1 on the same frames. `tokens=scene` are frames from the 150 held-out val scenes (honest); `sorted`/blank are frame-split frames whose 0.5 s neighbours were in training (optimistic by ~12 points). Official = nuScenes DetectionEval on all 6,019 val keyframes.

## e0

Baseline: 900-query DETR-style head distilled from the teacher's boxes, 896x512, six cameras, frame-level split (leaks neighbours)

Directory: `outputs/distill/student`
Checkpoints: `student.pt` (952 MB, 09-23 08:20)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0-baseline | 78000 | 250 |  | 0.329 |  |  |  | 1.486 |  |  |  | `outputs/distill/student/student.pt` |
| e0-nms2 | 78000 | 250 |  | 0.377 |  |  |  | 1.486 |  |  |  | `outputs/distill/student/student.pt` |

## e1

Query head variants (matching / objectness changes): the DETR head does not converge in this budget

Directory: `outputs/distill/exp/e1`
Checkpoints: `student.pt` (952 MB, 09-23 14:10)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 20000 | 250 |  | 0.000 |  |  |  | 1.511 |  |  |  | `outputs/distill/exp/e1/student.pt` |
| e1-nms2 | 20000 | 250 |  | 0.000 |  |  |  | 1.511 |  |  |  | `outputs/distill/exp/e1/student.pt` |

## e1b

Query head, short run

Directory: `outputs/distill/exp/e1b`
Checkpoints: `student.pt` (952 MB, 09-23 14:32)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1b-mid | 2000 | 80 |  | 0.000 |  |  |  | 1.529 |  |  |  | `outputs/distill/exp/e1b/student.pt` |
| e1b | 2000 | 250 |  | 0.000 |  |  |  | 1.655 |  |  |  | `outputs/distill/exp/e1b/student.pt` |
| e1b-nms2 | 2000 | 250 |  | 0.000 |  |  |  | 1.655 |  |  |  | `outputs/distill/exp/e1b/student.pt` |

## e2

Query head with teacher-asserted query supervision

Directory: `outputs/distill/exp/e2`
Checkpoints: `student.pt` (952 MB, 09-23 17:10)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e2 | 20000 | 250 |  | 0.011 |  |  |  | 1.510 |  |  |  | `outputs/distill/exp/e2/student.pt` |
| e2-nms2 | 20000 | 250 |  | 0.011 |  |  |  | 1.510 |  |  |  | `outputs/distill/exp/e2/student.pt` |

## e3

Query head, 98k steps (first read 2% was diagnose.py using the default cfg; re-eval 28%)

Directory: `outputs/distill/exp/e3`
Checkpoints: `student.pt` (952 MB, 09-23 19:58)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e3 | 98000 | 250 |  | 0.022 |  |  |  | 1.477 |  |  |  | `outputs/distill/exp/e3/student.pt` |
| e3-nms2 | 98000 | 250 |  | 0.022 |  |  |  | 1.477 |  |  |  | `outputs/distill/exp/e3/student.pt` |
| e3-reeval | 98000 | 250 |  | 0.283 |  |  |  | 1.477 |  |  |  | `outputs/distill/exp/e3/student.pt` |

## e4

Query head, 98k steps, variant (re-eval 33%)

Directory: `outputs/distill/exp/e4`
Checkpoints: `student.pt` (952 MB, 09-23 22:45)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4 | 98000 | 250 |  | 0.017 |  |  |  | 1.476 |  |  |  | `outputs/distill/exp/e4/student.pt` |
| e4-nms2 | 98000 | 250 |  | 0.017 |  |  |  | 1.476 |  |  |  | `outputs/distill/exp/e4/student.pt` |
| e4-reeval | 98000 | 250 |  | 0.330 |  |  |  | 1.476 |  |  |  | `outputs/distill/exp/e4/student.pt` |

## e5

First dense CenterPoint head (hidden 64, 2 blocks); e5b continued in place; the 0.0 reads are the heatmap-target bug (Gaussian on fractional coords, no positives)

Directory: `outputs/distill/exp/e5`
Checkpoints: `student.pt` (559 MB, 09-24 18:23)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e5 | 78000 | 250 |  | 0.027 |  |  |  | 1.486 |  |  |  | `outputs/distill/exp/e5/student.pt` |
| e5-nms2 | 78000 | 250 |  | 0.027 |  |  |  | 1.486 |  |  |  | `outputs/distill/exp/e5/student.pt` |

## e5c

Center head with the target fix, 154.5k steps: 55.2% (80.9% of the teacher on the leaked split)

Directory: `outputs/distill/exp/e5c`
Checkpoints: `student.pt` (559 MB, 09-24 23:52)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e5c | 114500 | 250 |  | 0.000 |  |  |  | 1.464 |  |  |  | `outputs/distill/exp/e5c/student.pt` |
| e5c-nms2 | 114500 | 250 |  | 0.000 |  |  |  | 1.464 |  |  |  | `outputs/distill/exp/e5c/student.pt` |
| e5c | 114500 | 250 |  | 0.000 |  |  |  | 1.464 |  |  |  | `outputs/distill/exp/e5c/student.pt` |
| e5c-nms2 | 114500 | 250 |  | 0.000 |  |  |  | 1.464 |  |  |  | `outputs/distill/exp/e5c/student.pt` |
| e5c-mid | 120000 | 100 |  | 0.455 |  |  |  | 1.248 |  |  |  | `outputs/distill/exp/e5c/student.pt` |
| e5c | 154500 | 250 | sorted | 0.552 | 80.9 |  |  | 1.470 |  |  |  | `outputs/distill/exp/e5c/student.pt` |
| e5c-nms2 | 154500 | 250 | sorted | 0.540 | 79.2 |  |  | 1.470 |  |  |  | `outputs/distill/exp/e5c/student.pt` |

## e6

Wider/deeper center head (hidden 128, 3 blocks) + min radius 2

Directory: `outputs/distill/exp/e6`
Checkpoints: `student.pt` (563 MB, 09-25 02:39)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e6 | 20000 | 250 | sorted | 0.583 | 85.5 |  |  | 1.458 |  |  |  | `outputs/distill/exp/e6/student.pt` |
| e6-nms2 | 20000 | 250 | sorted | 0.570 | 83.6 |  |  | 1.458 |  |  |  | `outputs/distill/exp/e6/student.pt` |

## e7

+ camera-flip augmentation (with the mirrored depth target)

Directory: `outputs/distill/exp/e7`
Checkpoints: `student.pt` (750 MB, 09-25 05:31)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e7 | 20000 | 250 | sorted | 0.592 | 86.8 |  |  | 1.461 |  |  |  | `outputs/distill/exp/e7/student.pt` |
| e7-nms2 | 20000 | 250 | sorted | 0.581 | 85.2 |  |  | 1.461 |  |  |  | `outputs/distill/exp/e7/student.pt` |

## e8

+ temporal BEV fusion (history 1), velocity, 10 nuScenes classes, CBGS, EMA: 94.1% of the teacher -- on the leaked split

Directory: `outputs/distill/exp/e8`
Checkpoints: `student.pt` (791 MB, 09-25 10:10)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e8 | 20000 | 250 | sorted | 0.642 | 94.1 |  |  | 1.460 |  |  |  | `outputs/distill/exp/e8/student.pt` |
| e8-nms2 | 20000 | 250 | sorted | 0.630 | 92.4 |  |  | 1.460 |  |  |  | `outputs/distill/exp/e8/student.pt` |

## final

e8 recipe from scratch on the official 700/150 SCENE split (scene-all), 896x512, 80k: the first honest number

Directory: `outputs/distill/exp/final`
Checkpoints: `student.pt` (791 MB, 09-25 23:01)
H200 speed: onnx_h200.json: 24.6 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final-mid | 19000 | 100 | scene | 0.564 | 89.4 |  |  | 1.313 |  |  |  | `outputs/distill/exp/final/student.pt` |
| final | 80000 | 250 | scene | 0.572 | 92.3 |  |  | 1.540 | 0.050 |  |  | `outputs/distill/exp/final/student.pt` |
| final-nms2 | 80000 | 250 | scene | 0.557 | 89.8 |  |  | 1.540 | 0.050 |  |  | `outputs/distill/exp/final/student.pt` |
| final-trainscenes | 80000 | 250 | sorted | 0.792 | 110.0 |  |  | 1.425 | 0.051 |  |  | `outputs/distill/exp/final/student.pt` |
| final-official | 80000 | 6019 |  |  |  | 0.3045 | 0.3941 |  |  |  |  | `outputs/distill/exp/final/student.pt` |
| final-official-cached | 80000 | 4452 |  |  |  | 0.3084 | 0.3947 |  |  |  |  | `outputs/distill/exp/final/student.pt` |

## r1

final recipe at 1152x640, 80k -- poisoned by depth targets binned in 896 pixel space

Directory: `outputs/distill/exp/r1_hires`
Checkpoints: `r1_19k.pt` (791 MB, 09-25 21:54), `student.pt` (791 MB, 09-26 09:22)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1-mid19k | 19000 | 100 | scene | 0.572 | 90.7 |  |  | 1.315 | 0.055 |  |  | `outputs/distill/exp/r1_hires/r1_19k.pt` |
| r1 | 80000 | 250 | scene | 0.566 | 91.4 |  |  | 1.514 | 0.049 |  |  | `outputs/distill/exp/r1_hires/student.pt` |
| r1-nms2 | 80000 | 250 | scene | 0.554 | 89.4 |  |  | 1.514 | 0.049 |  |  | `outputs/distill/exp/r1_hires/student.pt` |
| r1-official | 80000 | 6019 |  |  |  | 0.2887 | 0.3705 |  |  |  |  | `outputs/distill/exp/r1_hires/student.pt` |

## r2_long

1152x640 with corrected depth targets + Occ3D occupancy GT, 120k one-cycle, snapshots every 5k in the last quarter. 80k snapshot = deliverable (61.0%, 98.4%); end-of-run 59.8%, 7-snapshot average 60.1%. Graph v2 (de-contended scatter, same weights): 41 ms loop / 24.1 Hz idle H200 vs 93 ms / 10.7 Hz for v1

Directory: `outputs/distill/exp/r2_long`
Checkpoints: `snap_100000.pt` (791 MB, 09-26 21:20), `snap_19000.pt` (791 MB, 09-26 05:51), `snap_40000.pt` (791 MB, 09-26 09:50), `snap_60000.pt` (791 MB, 09-26 13:49), `snap_80000.pt` (791 MB, 09-26 17:35), `student.pt` (791 MB, 09-27 01:06), `student_100000.pt` (791 MB, 09-26 21:20), `student_105000.pt` (791 MB, 09-26 22:17), `student_110000.pt` (791 MB, 09-26 23:13), `student_115000.pt` (791 MB, 09-27 00:10), `student_120000.pt` (791 MB, 09-27 01:06), `student_90000.pt` (791 MB, 09-26 19:27), `student_95000.pt` (791 MB, 09-26 20:23), `student_avg.pt` (198 MB, 09-27 01:13)
H200 speed: onnx_shape_h200.json: 14.3 Hz; onnx_h200.json: 18.7 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2_long-mid19000 | 19000 | 100 | scene | 0.600 | 95.2 |  |  | 1.331 | 0.295 |  |  | `outputs/distill/exp/r2_long/snap_19000.pt` |
| r2_long-mid19k-250 | 19000 | 250 | scene | 0.587 | 94.7 |  |  | 1.536 | 0.282 |  |  | `outputs/distill/exp/r2_long/snap_19000.pt` |
| r2_long-mid40000 | 40000 | 250 | scene | 0.603 | 97.3 |  |  | 1.510 | 0.302 |  |  | `outputs/distill/exp/r2_long/snap_40000.pt` |
| r2_long-mid40000-official | 40000 | 6019 |  |  |  | 0.3500 | 0.4248 |  |  |  |  | `outputs/distill/exp/r2_long/snap_40000.pt` |
| r2_long-mid60000 | 60000 | 250 | scene | 0.602 | 97.2 |  |  | 1.510 | 0.312 |  |  | `outputs/distill/exp/r2_long/snap_60000.pt` |
| r2_long-mid80000 | 80000 | 250 | scene | 0.610 | 98.4 |  |  | 1.510 | 0.322 |  |  | `outputs/distill/exp/r2_long/snap_80000.pt` |
| r2_long-mid100000 | 100000 | 250 | scene | 0.599 | 96.6 |  |  | 1.509 | 0.329 |  |  | `outputs/distill/exp/r2_long/snap_100000.pt` |
| r2_long | 120000 | 250 | scene | 0.598 | 96.5 |  |  | 1.560 | 0.331 |  |  | `outputs/distill/exp/r2_long/student.pt` |
| r2_long-nms2 | 120000 | 250 | scene | 0.582 | 93.8 |  |  | 1.560 | 0.331 |  |  | `outputs/distill/exp/r2_long/student.pt` |
| r2_long-avg | 120000 | 250 | scene | 0.601 | 96.9 |  |  | 1.527 | 0.330 |  |  | `outputs/distill/exp/r2_long/student_avg.pt` |
| r2_long-official | 120000 | 6019 |  |  |  | 0.3132 | 0.4052 |  |  |  |  | `outputs/distill/exp/r2_long/student_avg.pt` |
| r2_long-80000-valall-nms2 | 80000 | 4452 | valall | 0.582 | 95.3 |  |  | 1.580 | 0.277 |  |  | `outputs/distill/exp/r2_long/snap_80000.pt` |
| r2_long-80000-valall | 80000 | 4452 | valall | 0.595 | 97.5 |  |  | 1.580 | 0.277 |  |  | `outputs/distill/exp/r2_long/snap_80000.pt` |
| r2_long-80000-valfoldB | 80000 | 2219 | valfoldB | 0.598 | 97.5 |  |  | 1.615 | 0.277 |  |  | `outputs/distill/exp/r2_long/snap_80000.pt` |
| r2_long-80000-valfoldA | 80000 | 2233 | valfoldA | 0.593 | 97.6 |  |  | 1.545 | 0.278 |  |  | `outputs/distill/exp/r2_long/snap_80000.pt` |
| r2_long-80000-valfoldB-calA | 80000 | 2219 | valfoldB | 0.598 | 97.5 |  |  | 1.615 | 0.277 |  |  | `outputs/distill/exp/r2_long/snap_80000.pt` |
| r2_long-80000-valfoldA-calB | 80000 | 2233 | valfoldA | 0.593 | 97.6 |  |  | 1.545 | 0.278 |  |  | `outputs/distill/exp/r2_long/snap_80000.pt` |

## r2_hist3

r2_long recipe + three-frame BEV history (K=3), 60k. Final 60.6% (97.8%), 4-snapshot average 60.85% (98.2%): tied with r2_long@80k, so the deliverable did not change. Official mAP 0.342 / NDS 0.437 (avg ckpt). On ALL 4,452 val frames the final weights score 60.3% = 98.8% of the teacher, ahead of r2_long@80k (59.5%, 97.5%): the ranking now uses the full set. Graph (de-contended scatter, K=3): 55 ms real-frame loop / 18.1 Hz idle H200

Directory: `outputs/distill/exp/r2_hist3`
Checkpoints: `snap_19000.pt` (831 MB, 09-26 16:02), `student.pt` (831 MB, 09-27 05:07), `student_45000.pt` (831 MB, 09-27 00:15), `student_50000.pt` (831 MB, 09-27 01:51), `student_55000.pt` (831 MB, 09-27 03:29), `student_60000.pt` (831 MB, 09-27 05:07), `student_avg.pt` (208 MB, 09-27 05:20)
H200 speed: onnx_shape_h200.json: 19.2 Hz; onnx_h200.json: 11.4 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2_hist3-mid19000 | 19000 | 100 | scene | 0.614 | 97.3 |  |  | 1.321 | 0.308 |  |  | `outputs/distill/exp/r2_hist3/snap_19000.pt` |
| r2_hist3 | 60000 | 250 | scene | 0.606 | 97.8 |  |  | 1.562 | 0.323 |  |  | `outputs/distill/exp/r2_hist3/student.pt` |
| r2_hist3-nms2 | 60000 | 250 | scene | 0.593 | 95.7 |  |  | 1.562 | 0.323 |  |  | `outputs/distill/exp/r2_hist3/student.pt` |
| r2_hist3-avg | 60000 | 250 | scene | 0.608 | 98.2 |  |  | 1.565 | 0.322 |  |  | `outputs/distill/exp/r2_hist3/student_avg.pt` |
| r2_hist3-official | 60000 | 6019 |  |  |  | 0.3422 | 0.4373 |  |  |  |  | `outputs/distill/exp/r2_hist3/student_avg.pt` |
| r2_hist3-valall | 60000 | 4452 | valall | 0.603 | 98.8 |  |  | 1.651 | 0.282 |  |  | `outputs/distill/exp/r2_hist3/student.pt` |
| r2_hist3-avg-valfoldA | 60000 | 2233 | valfoldA | 0.607 | 99.9 |  |  | 1.617 | 0.281 |  |  | `outputs/distill/exp/r2_hist3/student_avg.pt` |
| r2_hist3-avg-valfoldB | 60000 | 2219 | valfoldB | 0.603 | 98.3 |  |  | 1.689 | 0.281 |  |  | `outputs/distill/exp/r2_hist3/student_avg.pt` |

## r3_final

r2_hist3 recipe (1152x640, history 3), 75k one-cycle, GPUs 0-3; try 1 stopped at step 150 so the lane restarts with the de-contended scatter (try 2 launched 2026-09-27 01:56); snapshots every 10k are scored individually. 20k: 60.8% gate; 30k: 61.5% gate (99.3%) and 60.85% on ALL 4,452 val frames = 99.7% of the teacher, the leader of the full-val ranking (release v4); 40k 61.3, 50k 60.8, 60k 60.9, final 75k 60.9, avg(60-75k) 61.0 on the gate; student_avg30-50.pt = average of the 30/40/50k plateau snapshots: 61.4% on the gate (99.1%) and 61.2% on ALL val frames = 100.3% of the teacher, the best detector of the project

Directory: `outputs/distill/exp/r3_final`
Checkpoints: `snap_10000.pt` (831 MB, 09-27 04:41), `snap_20000.pt` (831 MB, 09-27 07:20), `snap_30000.pt` (831 MB, 09-27 10:20), `snap_40000.pt` (831 MB, 09-27 13:19), `snap_50000.pt` (831 MB, 09-27 16:05), `snap_60000.pt` (831 MB, 09-27 18:49), `snap_70000.pt` (831 MB, 09-27 21:25), `student.pt` (831 MB, 09-27 22:33), `student_60000.pt` (831 MB, 09-27 18:49), `student_65000.pt` (831 MB, 09-27 20:11), `student_70000.pt` (831 MB, 09-27 21:25), `student_75000.pt` (831 MB, 09-27 22:32), `student_avg.pt` (208 MB, 09-27 22:42), `student_avg30-50.pt` (208 MB, 09-27 23:14)
H200 speed: onnx_shape_h200.json: 17.5 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r3_final-mid10000 | 10000 | 250 | scene | 0.557 | 89.9 |  |  | 1.574 | 0.268 |  |  | `outputs/distill/exp/r3_final/snap_10000.pt` |
| r3_final-mid20000 | 20000 | 250 | scene | 0.608 | 98.1 |  |  | 1.526 | 0.290 |  |  | `outputs/distill/exp/r3_final/snap_20000.pt` |
| r3_final-mid30000 | 30000 | 250 | scene | 0.615 | 99.3 |  |  | 1.515 | 0.299 |  |  | `outputs/distill/exp/r3_final/snap_30000.pt` |
| r3_final-mid30000-valfoldB | 30000 | 2219 | valfoldB | 0.606 | 98.8 |  |  | 1.626 | 0.259 |  |  | `outputs/distill/exp/r3_final/snap_30000.pt` |
| r3_final-mid30000-valfoldA | 30000 | 2233 | valfoldA | 0.611 | 100.5 |  |  | 1.557 | 0.260 |  |  | `outputs/distill/exp/r3_final/snap_30000.pt` |
| r3_final-mid40000 | 40000 | 250 | scene | 0.613 | 99.0 |  |  | 1.510 | 0.305 |  |  | `outputs/distill/exp/r3_final/snap_40000.pt` |
| r3_final-mid50000 | 50000 | 250 | scene | 0.608 | 98.2 |  |  | 1.511 | 0.308 |  |  | `outputs/distill/exp/r3_final/snap_50000.pt` |
| r3_final-mid60000 | 60000 | 250 | scene | 0.609 | 98.3 |  |  | 1.517 | 0.315 |  |  | `outputs/distill/exp/r3_final/snap_60000.pt` |
| r3_final | 75000 | 250 | scene | 0.609 | 98.2 |  |  | 1.522 | 0.318 |  |  | `outputs/distill/exp/r3_final/student.pt` |
| r3_final-nms2 | 75000 | 250 | scene | 0.600 | 96.8 |  |  | 1.522 | 0.318 |  |  | `outputs/distill/exp/r3_final/student.pt` |
| r3_final-avg | 75000 | 250 | scene | 0.610 | 98.3 |  |  | 1.519 | 0.317 |  |  | `outputs/distill/exp/r3_final/student_avg.pt` |
| r3_final-mid10000 | 10000 | 250 | scene | 0.557 | 89.9 |  |  | 1.574 | 0.268 |  |  | `outputs/distill/exp/r3_final/snap_10000.pt` |
| r3_final-mid20000 | 20000 | 250 | scene | 0.608 | 98.1 |  |  | 1.526 | 0.290 |  |  | `outputs/distill/exp/r3_final/snap_20000.pt` |
| r3_final-mid30000 | 30000 | 250 | scene | 0.615 | 99.3 |  |  | 1.515 | 0.299 |  |  | `outputs/distill/exp/r3_final/snap_30000.pt` |
| r3_final-avg30-50 | 50000 | 250 | scene | 0.614 | 99.1 |  |  | 1.512 | 0.303 |  |  | `outputs/distill/exp/r3_final/student_avg30-50.pt` |
| r3_final-avg30-50-valfoldB | 50000 | 2219 | valfoldB | 0.609 | 99.3 |  |  | 1.622 | 0.264 |  |  | `outputs/distill/exp/r3_final/student_avg30-50.pt` |
| r3_final-avg30-50-valfoldA | 50000 | 2233 | valfoldA | 0.615 | 101.2 |  |  | 1.554 | 0.264 |  |  | `outputs/distill/exp/r3_final/student_avg30-50.pt` |
| r3_final-mid30000-official | 30000 | 6019 |  |  |  | 0.3652 | 0.4367 |  |  |  |  | `outputs/distill/exp/r3_final/snap_30000.pt` |

## r4_occfix

r3_final 30k snapshot fine-tuned 30k->35k (annealing tail of a 35k one-cycle, ~1.4e-5 -> 0) with --occ-outside-empty 0.2: voxels outside the Occ3D camera-visible mask pulled toward empty so the exported occupancy stops painting the unobserved 88% of the grid. Result: gate 61.2% (98.7%, vs 61.5% for the seed), trajectory 1.514 m, occupancy mIoU 0.299 on the gate frames; raw occupancy outside the mask down from ~300k to 75-98k voxels per frame (contiguous extensions of the visible surfaces, no blanket)

Directory: `outputs/distill/exp/r4_occfix`
Checkpoints: `student.pt` (831 MB, 09-28 00:43), `student_35000.pt` (831 MB, 09-28 00:43)
H200 speed: onnx_h200.json: 14.2 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r4_occfix | 35000 | 250 | scene | 0.612 | 98.7 |  |  | 1.514 | 0.299 |  |  | `outputs/distill/exp/r4_occfix/student.pt` |
| r4_occfix-nms2 | 35000 | 250 | scene | 0.606 | 97.8 |  |  | 1.514 | 0.299 |  |  | `outputs/distill/exp/r4_occfix/student.pt` |
| r4_occfix-official | 35000 | 6019 |  |  |  | 0.3656 | 0.4401 |  |  |  |  | `outputs/distill/exp/r4_occfix/student.pt` |
| r4_occfix-final-valfoldB | 35000 | 2219 | valfoldB | 0.608 | 99.1 |  |  | 1.625 | 0.260 |  |  | `outputs/distill/exp/r4_occfix/student.pt` |
| r4_occfix-final-valfoldA | 35000 | 2233 | valfoldA | 0.611 | 100.5 |  |  | 1.556 | 0.261 |  |  | `outputs/distill/exp/r4_occfix/student.pt` |

## r7_head2

Round 7 (24 h extension, 2026-09-28): the deliverable (r4_small avg) re-expressed EXACTLY as a 2x-grid detection head (transposed-conv entry = nearest 2x upsample of the trained 1x1, dilation-2 3x3 blocks; heatmap difference 0.00 at the transplant) and fine-tuned 10k annealing steps with the r4_small recipe on 0.256 m targets: sub-cell localisation and separation of adjacent pedestrians/cones without losing the trained large-object behaviour

Directory: `outputs/distill/exp/r7_head2`
Checkpoints: `snap_40000.pt` (834 MB, 09-28 13:24), `student.pt` (834 MB, 09-28 13:24), `student_35000.pt` (834 MB, 09-28 12:02), `student_40000.pt` (834 MB, 09-28 13:24), `student_avg.pt` (209 MB, 09-28 13:34)
H200 speed: onnx_h200.json: 16.1 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r7_head2 | 40000 | 250 | scene | 0.586 | 94.5 |  |  | 1.506 | 0.292 |  |  | `outputs/distill/exp/r7_head2/student.pt` |
| r7_head2-nms2 | 40000 | 250 | scene | 0.600 | 96.8 |  |  | 1.506 | 0.292 |  |  | `outputs/distill/exp/r7_head2/student.pt` |
| r7_head2-avg | 40000 | 250 | scene | 0.591 | 95.3 |  |  | 1.509 | 0.294 |  |  | `outputs/distill/exp/r7_head2/student_avg.pt` |
| r7_head2-mid40000 | 40000 | 250 | scene | 0.586 | 94.6 |  |  | 1.506 | 0.292 |  |  | `outputs/distill/exp/r7_head2/snap_40000.pt` |
| r7_head2-official | 40000 | 6019 |  |  |  | 0.3631 | 0.4387 |  |  |  |  | `outputs/distill/exp/r7_head2/student_avg.pt` |

## r7_occ3d

Round 7: the deliverable with a new 3D occupancy head (1x1 lift to 16 ch x 16 pillars, two 3x3x3 conv blocks, 1x1x1 classifier) trained ALONE (--train-only occ_head, backbone and other heads frozen, LR 1.8e-4 -> 0 over 10k steps): every other output stays bit-identical; target = occupancy mIoU 0.30 -> 0.35+ and sharper renders

Directory: `outputs/distill/exp/r7_occ3d`
Checkpoints: `snap_20000.pt` (417 MB, 09-28 13:26), `student.pt` (417 MB, 09-28 13:26), `student_15000.pt` (417 MB, 09-28 12:05), `student_20000.pt` (417 MB, 09-28 13:26), `student_avg.pt` (208 MB, 09-28 13:36)
H200 speed: onnx_h200.json: 17.9 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r7_occ3d | 20000 | 250 | scene | 0.613 | 98.9 |  |  | 1.511 | 0.308 |  |  | `outputs/distill/exp/r7_occ3d/student.pt` |
| r7_occ3d-nms2 | 20000 | 250 | scene | 0.610 | 98.5 |  |  | 1.511 | 0.308 |  |  | `outputs/distill/exp/r7_occ3d/student.pt` |
| r7_occ3d-avg | 20000 | 250 | scene | 0.613 | 98.9 |  |  | 1.511 | 0.310 |  |  | `outputs/distill/exp/r7_occ3d/student_avg.pt` |
| r7_occ3d-mid20000 | 20000 | 250 | scene | 0.614 | 99.0 |  |  | 1.511 | 0.308 |  |  | `outputs/distill/exp/r7_occ3d/snap_20000.pt` |
| r7_occ3d-official | 20000 | 6019 |  |  |  | 0.3760 | 0.4474 |  |  |  |  | `outputs/distill/exp/r7_occ3d/snap_20000.pt` |

## r6a_avg_small

LAST LANE (GPUs 0-3, from 2026-09-28 03:47): the r3_final 30/40/50k plateau average (100.3% full-val, best pure detector) fine-tuned with the r4_small recipe (heatmap loss x2 on pedestrian/motorcycle/bicycle/cone + occ-outside-empty 0.2), 10k annealing steps; Result: final 61.0% gate / 100.0% full-val, 35k/40k average 61.5% gate / 100.1% full-val, official mAP 0.370 / NDS 0.444, 27.1 Hz -- did not beat r4_small-avg (100.5%), v6 stands

Directory: `outputs/distill/exp/r6a_avg_small`
Checkpoints: `snap_40000.pt` (831 MB, 09-28 05:35), `student.pt` (831 MB, 09-28 05:35), `student_35000.pt` (831 MB, 09-28 04:43), `student_40000.pt` (831 MB, 09-28 05:35), `student_avg.pt` (208 MB, 09-28 05:44)
H200 speed: onnx_h200.json: 16.4 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r6a_avg_small | 40000 | 250 | scene | 0.610 | 98.4 |  |  | 1.508 | 0.305 |  |  | `outputs/distill/exp/r6a_avg_small/student.pt` |
| r6a_avg_small-nms2 | 40000 | 250 | scene | 0.605 | 97.5 |  |  | 1.508 | 0.305 |  |  | `outputs/distill/exp/r6a_avg_small/student.pt` |
| r6a_avg_small-avg | 40000 | 250 | scene | 0.615 | 99.2 |  |  | 1.514 | 0.299 |  |  | `outputs/distill/exp/r6a_avg_small/student_avg.pt` |
| r6a_avg_small-mid40000 | 40000 | 250 | scene | 0.610 | 98.3 |  |  | 1.508 | 0.305 |  |  | `outputs/distill/exp/r6a_avg_small/snap_40000.pt` |
| r6a_avg_small-official | 40000 | 6019 |  |  |  | 0.3697 | 0.4439 |  |  |  |  | `outputs/distill/exp/r6a_avg_small/student_avg.pt` |
| r6a_avg_small-final-valfoldB | 40000 | 2219 | valfoldB | 0.608 | 99.1 |  |  | 1.619 | 0.266 |  |  | `outputs/distill/exp/r6a_avg_small/student.pt` |
| r6a_avg_small-avg-valfoldA | 40000 | 2233 | valfoldA | 0.613 | 100.9 |  |  | 1.551 | 0.262 |  |  | `outputs/distill/exp/r6a_avg_small/student_avg.pt` |
| r6a_avg_small-avg-valfoldB | 40000 | 2219 | valfoldB | 0.609 | 99.3 |  |  | 1.622 | 0.262 |  |  | `outputs/distill/exp/r6a_avg_small/student_avg.pt` |
| r6a_avg_small-final-valfoldA | 40000 | 2233 | valfoldA | 0.612 | 100.7 |  |  | 1.549 | 0.266 |  |  | `outputs/distill/exp/r6a_avg_small/student.pt` |

## r5_avg_occfix

r3_final 30/40/50k plateau average (100.3% full-val, the best detector) fine-tuned 5k annealing steps with --occ-outside-empty 0.2 (same schedule as r4_occfix; seed step relabelled 30000, fresh Adam): the best detector with the occupancy fix, launched 2026-09-28 01:50 on GPUs 4-7

Directory: `outputs/distill/exp/r5_avg_occfix`
Checkpoints: `student.pt` (831 MB, 09-28 02:48), `student_35000.pt` (831 MB, 09-28 02:48)
H200 speed: onnx_h200.json: 16.9 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r5_avg_occfix | 35000 | 250 | scene | 0.613 | 99.0 |  |  | 1.512 | 0.298 |  |  | `outputs/distill/exp/r5_avg_occfix/student.pt` |
| r5_avg_occfix-nms2 | 35000 | 250 | scene | 0.608 | 98.1 |  |  | 1.512 | 0.298 |  |  | `outputs/distill/exp/r5_avg_occfix/student.pt` |
| r5_avg_occfix-official | 35000 | 6019 |  |  |  | 0.3643 | 0.4463 |  |  |  |  | `outputs/distill/exp/r5_avg_occfix/student.pt` |
| r5_avg_occfix-final-valfoldB | 35000 | 2219 | valfoldB | 0.607 | 98.9 |  |  | 1.619 | 0.259 |  |  | `outputs/distill/exp/r5_avg_occfix/student.pt` |
| r5_avg_occfix-final-valfoldA | 35000 | 2233 | valfoldA | 0.609 | 100.2 |  |  | 1.550 | 0.258 |  |  | `outputs/distill/exp/r5_avg_occfix/student.pt` |

## r4_small

r3_final 30k fine-tuned 30k->40k (anneal ~5.5e-5 -> 0, GPUs 0-3, 2026-09-27 23:19 - 09-28 02:00) with --hm-class-weight 5:2,6:2,7:2,8:2 (pedestrian, motorcycle, bicycle, traffic cone heatmap loss x2) + --occ-outside-empty 0.2. Gate 60.7% (97.9%, seed 61.5%): motorcycles 0.49 and bicycles 0.46 now ABOVE the teacher (0.46 / 0.42), barriers 0.65 vs 0.53, pedestrians 0.45 vs 0.51, cones 0.65 vs 0.68, cars equal; the class re-weighting trades ~0.5 pt of count-weighted F1 for small-class recall in the final weights. Its 35k/40k AVERAGE is the project winner: gate 61.6%, full val 61.35% = 100.5% of the teacher, official mAP 0.375 / NDS 0.444, 27.3 Hz real-frame loop, occupancy fix intact (release v6). Weight soups tried at the end: soup_small_occfix = mean(r4_small-avg, r4_occfix-final), soup_small_r5 = mean(r4_small-avg, r5_avg_occfix-final)

Directory: `outputs/distill/exp/r4_small`
Checkpoints: `snap_40000.pt` (831 MB, 09-28 02:00), `soup_small_occfix.pt` (208 MB, 09-28 03:47), `soup_small_r5.pt` (208 MB, 09-28 03:47), `student.pt` (831 MB, 09-28 02:00), `student_35000.pt` (831 MB, 09-28 00:56), `student_40000.pt` (831 MB, 09-28 02:00), `student_avg.pt` (208 MB, 09-28 02:14)
H200 speed: onnx_h200.json: 17.3 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r4_small | 40000 | 250 | scene | 0.607 | 97.9 |  |  | 1.508 | 0.297 |  |  | `outputs/distill/exp/r4_small/student.pt` |
| r4_small-nms2 | 40000 | 250 | scene | 0.602 | 97.2 |  |  | 1.508 | 0.297 |  |  | `outputs/distill/exp/r4_small/student.pt` |
| r4_small-avg | 40000 | 250 | scene | 0.616 | 99.4 |  |  | 1.511 | 0.295 |  |  | `outputs/distill/exp/r4_small/student_avg.pt` |
| r4_small-mid40000 | 40000 | 250 | scene | 0.607 | 97.9 |  |  | 1.508 | 0.297 |  |  | `outputs/distill/exp/r4_small/snap_40000.pt` |
| r4_small-final-valfoldB | 40000 | 2219 | valfoldB | 0.606 | 98.8 |  |  | 1.619 | 0.259 |  |  | `outputs/distill/exp/r4_small/student.pt` |
| r4_small-final-valfoldA | 40000 | 2233 | valfoldA | 0.612 | 100.8 |  |  | 1.551 | 0.259 |  |  | `outputs/distill/exp/r4_small/student.pt` |
| r4_small-official | 40000 | 6019 |  |  |  | 0.3749 | 0.4439 |  |  |  |  | `outputs/distill/exp/r4_small/student_avg.pt` |
| r4_small-avg-valfoldB | 40000 | 2219 | valfoldB | 0.611 | 99.5 |  |  | 1.621 | 0.258 |  |  | `outputs/distill/exp/r4_small/student_avg.pt` |
| r4_small-avg-valfoldA | 40000 | 2233 | valfoldA | 0.616 | 101.4 |  |  | 1.552 | 0.259 |  |  | `outputs/distill/exp/r4_small/student_avg.pt` |
| r4_small-soup_small_r5 | 35000 | 250 | scene | 0.619 | 99.9 |  |  | 1.511 | 0.296 |  |  | `outputs/distill/exp/r4_small/soup_small_r5.pt` |
| r4_small-soup_small_occfix | 35000 | 250 | scene | 0.615 | 99.3 |  |  | 1.511 | 0.297 |  |  | `outputs/distill/exp/r4_small/soup_small_occfix.pt` |
| r4_small-soup_small_r5-valfoldB | 35000 | 2219 | valfoldB | 0.609 | 99.3 |  |  | 1.620 | 0.258 |  |  | `outputs/distill/exp/r4_small/soup_small_r5.pt` |
| r4_small-soup_small_r5-valfoldA | 35000 | 2233 | valfoldA | 0.615 | 101.2 |  |  | 1.552 | 0.258 |  |  | `outputs/distill/exp/r4_small/soup_small_r5.pt` |

## occfix-baseline

Round 8 baseline (2026-10-01): release v6 and the teacher re-scored on Occ3D's NATIVE grid (0.4 m, +-40 m, (x, y, z)), the teacher's own format. v6's occupancy was on the 0.512 m detection grid in (y, x) order: every render was transposed and 28% too large, and the teacher's old 0.03 mIoU was a grid-mismatch artefact (real: 27%). v6 visible mIoU 33.5% vs teacher 27.0%, but 116k occupied voxels per frame outside the camera-visible region vs the teacher's 18k (the blanket)

Directory: `outputs/student_release_v6`
Checkpoints: `student_r4_small_avg.pt` (208 MB, 09-28 03:32)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-baseline-v6 | 40000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 | 0.295 | 0.293 / 0.236 | 116,460 / 18,016 | `outputs/student_release_v6/student_r4_small_avg.pt` |
| occfix-baseline-v6b | 40000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 | 0.295 | 0.335 / 0.270 | 116,461 / 18,016 | `outputs/student_release_v6/student_r4_small_avg.pt` |
| occfix-baseline-v6-valfoldB | 40000 | 2219 | valfoldB | 0.611 | 99.6 |  |  | 1.621 | 0.258 | 0.331 / 0.257 | 117,089 / 17,682 | `outputs/student_release_v6/student_r4_small_avg.pt` |
| occfix-baseline-v6-valfoldA | 40000 | 2233 | valfoldA | 0.616 | 101.4 |  |  | 1.552 | 0.259 | 0.332 / 0.266 | 111,184 / 16,887 | `outputs/student_release_v6/student_r4_small_avg.pt` |

## occfix-r8_occ_a

Round 8: v6 + a new NATIVE-grid occupancy head (grid_sample BEV->Occ3D grid, 3x3 ctx conv, lift to 16 slices x 16 ch, two 3x3x3 Conv3d, 1x1x1 classifier; 1.26 M params) trained alone 8k steps (frozen BN), class-weighted CE on visible voxels + 1.0 x CE on unobserved voxels vs their label. 33.8% visible mIoU, 36.7k unseen fill raw. Round 9: + 'empty' offset 0.75 (b075 records) 32.6% at teacher fill; + PER-CLASS logit offsets fitted on 200 fold-A frames outside eval250 (cb records): 35.3% on eval250, 34.9% / 34.3% on val folds A / B at the teacher's fill (v7 30.7 / 30.2) = RELEASE v8

Directory: `outputs/distill/exp/r8_occ_a`
Checkpoints: `avg_6k_8k.pt` (212 MB, 10-02 04:58), `avg_6k_8k_classbias.pt` (212 MB, 10-02 05:01), `snap_2000.pt` (435 MB, 10-01 18:56), `snap_2000_classbias.pt` (425 MB, 10-02 05:30), `snap_4000.pt` (435 MB, 10-01 19:29), `snap_4000_classbias.pt` (425 MB, 10-02 05:01), `snap_6000.pt` (435 MB, 10-01 20:02), `snap_6000_classbias.pt` (425 MB, 10-02 05:01), `snap_8000.pt` (435 MB, 10-01 20:35), `snap_8000_bias075.pt` (425 MB, 10-02 00:05), `snap_8000_classbias.pt` (425 MB, 10-02 00:28), `student.pt` (435 MB, 10-01 20:35)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r8_occ_a-2k | 2000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.366 / 0.270 | 40,568 / 18,016 | `outputs/distill/exp/r8_occ_a/snap_2000.pt` |
| occfix-r8_occ_a-4k | 4000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.343 / 0.270 | 36,835 / 18,016 | `outputs/distill/exp/r8_occ_a/snap_4000.pt` |
| occfix-r8_occ_a-6k | 6000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.338 / 0.270 | 36,619 / 18,016 | `outputs/distill/exp/r8_occ_a/snap_6000.pt` |
| occfix-r8_occ_a-8k | 8000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.338 / 0.270 | 36,655 / 18,016 | `outputs/distill/exp/r8_occ_a/snap_8000.pt` |
| occfix-r8_occ_a-8k-b075 | 8000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.326 / 0.270 | 17,772 / 18,016 | `outputs/distill/exp/r8_occ_a/snap_8000_bias075.pt` |
| occfix-r8_occ_a-8k-b075-valfoldA | 8000 | 2233 | valfoldA | 0.616 | 101.4 |  |  | 1.552 |  | 0.323 / 0.266 | 17,104 / 16,887 | `outputs/distill/exp/r8_occ_a/snap_8000_bias075.pt` |
| occfix-r8_occ_a-8k-b075-valfoldB | 8000 | 2219 | valfoldB | 0.611 | 99.5 |  |  | 1.621 |  | 0.319 / 0.257 | 17,387 / 17,682 | `outputs/distill/exp/r8_occ_a/snap_8000_bias075.pt` |
| occfix-r8_occ_a-8k-cb | 8000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.353 / 0.270 | 18,002 / 18,016 | `outputs/distill/exp/r8_occ_a/snap_8000_classbias.pt` |
| occfix-r8_occ_a-8k-cb-valfoldB | 8000 | 2219 | valfoldB | 0.611 | 99.5 |  |  | 1.621 |  | 0.343 / 0.257 | 17,628 / 17,682 | `outputs/distill/exp/r8_occ_a/snap_8000_classbias.pt` |
| occfix-r8_occ_a-8k-cb-valfoldA | 8000 | 2233 | valfoldA | 0.616 | 101.4 |  |  | 1.552 |  | 0.349 / 0.266 | 17,267 / 16,887 | `outputs/distill/exp/r8_occ_a/snap_8000_classbias.pt` |

## occfix-r8_occ_b

Round 8: as r8_occ_a with unseen weight 0.3; stopped after its 2k snapshot (37.0%, 93.5k unseen fill: too weak)

Directory: `outputs/distill/exp/r8_occ_b`
Checkpoints: `snap_2000.pt` (435 MB, 10-01 18:52), `student.pt` (435 MB, 10-01 18:52)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r8_occ_b-2k | 2000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.370 / 0.270 | 93,524 / 18,016 | `outputs/distill/exp/r8_occ_b/snap_2000.pt` |

## occfix-r8_occ_d

Round 8: as r8_occ_a with unseen weight 3.0. Final (8k): 31.0% visible mIoU on eval250, 30.7% / 30.2% on val folds A / B (teacher 26.6 / 25.7), unseen fill 14.6k / 14.7k (teacher 16.9k / 17.7k); detection, map, trajectory bit-identical to v6. RELEASE v7

Directory: `outputs/distill/exp/r8_occ_d`
Checkpoints: `snap_2000.pt` (435 MB, 10-01 19:31), `snap_4000.pt` (435 MB, 10-01 20:01), `snap_6000.pt` (435 MB, 10-01 20:32), `snap_8000.pt` (435 MB, 10-01 21:06), `student.pt` (435 MB, 10-01 21:06)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r8_occ_d-2k | 2000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.271 / 0.270 | 17,405 / 18,016 | `outputs/distill/exp/r8_occ_d/snap_2000.pt` |
| occfix-r8_occ_d-8k | 8000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.310 / 0.270 | 15,019 / 18,016 | `outputs/distill/exp/r8_occ_d/snap_8000.pt` |
| occfix-r8_occ_d-8k-valfoldB | 8000 | 2219 | valfoldB | 0.611 | 99.6 |  |  | 1.621 |  | 0.302 / 0.257 | 14,743 / 17,682 | `outputs/distill/exp/r8_occ_d/snap_8000.pt` |
| occfix-r8_occ_d-8k-valfoldA | 8000 | 2233 | valfoldA | 0.616 | 101.4 |  |  | 1.552 |  | 0.307 / 0.266 | 14,564 / 16,887 | `outputs/distill/exp/r8_occ_d/snap_8000.pt` |

## occfix-r8_occ_e

Round 8: --occ-vis -- a learned camera-visibility channel gates unseen voxels to empty in-graph, classes trained on visible voxels only. Visibility too blunt from frozen features: at teacher-like fill ~5 points below r8_occ_d; stopped at ~5.4k

Directory: `outputs/distill/exp/r8_occ_e`
Checkpoints: `snap_2000.pt` (435 MB, 10-01 21:09), `snap_4000.pt` (435 MB, 10-01 21:45), `snap_4000_viscal.pt` (435 MB, 10-01 21:55), `student.pt` (435 MB, 10-01 22:03)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r8_occ_e-2k | 2000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.099 / 0.270 | 1,669 / 18,016 | `outputs/distill/exp/r8_occ_e/snap_2000.pt` |

## occfix-r8_occ_f

Round 8: r8_occ_d's recipe with a 32-channel 3-D head: ahead at 2k (33.1%), converged to r8_occ_d's point by 4k (30.9% / 15.6k)

Directory: `outputs/distill/exp/r8_occ_f`
Checkpoints: `snap_2000.pt` (436 MB, 10-01 21:41), `snap_4000.pt` (436 MB, 10-01 22:10), `snap_6000.pt` (436 MB, 10-01 22:39), `snap_8000.pt` (436 MB, 10-01 23:09), `student.pt` (436 MB, 10-01 23:09)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r8_occ_f-2k | 2000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.331 / 0.270 | 21,460 / 18,016 | `outputs/distill/exp/r8_occ_f/snap_2000.pt` |
| occfix-r8_occ_f-4k | 4000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.309 / 0.270 | 15,597 / 18,016 | `outputs/distill/exp/r8_occ_f/snap_4000.pt` |
| occfix-r8_occ_f-6k | 6000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.306 / 0.270 | 15,137 / 18,016 | `outputs/distill/exp/r8_occ_f/snap_6000.pt` |
| occfix-r8_occ_f-8k | 8000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.307 / 0.270 | 15,128 / 18,016 | `outputs/distill/exp/r8_occ_f/snap_8000.pt` |

## occfix-r8_occ_g

Round 8: r8_occ_d's recipe with a 48-channel 3-D head (does capacity keep paying?)

Directory: `outputs/distill/exp/r8_occ_g`
Checkpoints: `snap_2000.pt` (438 MB, 10-01 22:42), `snap_4000.pt` (438 MB, 10-01 23:11), `snap_6000.pt` (438 MB, 10-01 23:39), `student.pt` (438 MB, 10-01 23:52)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r8_occ_g-2k | 2000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.336 / 0.270 | 19,527 / 18,016 | `outputs/distill/exp/r8_occ_g/snap_2000.pt` |
| occfix-r8_occ_g-4k | 4000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.311 / 0.270 | 15,475 / 18,016 | `outputs/distill/exp/r8_occ_g/snap_4000.pt` |

## occfix-r9_ray_a

Round 9: native head + --occ-raymap (the lift's depth mass passing / stopping / sampled per BEV cell, splatted with the feature indices: the cameras' own visibility evidence), unobserved weight 1.0, 8k steps head-only

Directory: `outputs/distill/exp/r9_ray_a`
Checkpoints: `snap_2000.pt` (435 MB, 10-02 00:37), `snap_2000_classbias.pt` (425 MB, 10-02 01:08), `snap_4000.pt` (435 MB, 10-02 01:06), `snap_6000.pt` (435 MB, 10-02 01:35), `snap_8000.pt` (435 MB, 10-02 02:03), `snap_8000_classbias.pt` (425 MB, 10-02 02:08), `student.pt` (435 MB, 10-02 02:03)

(no evaluation records yet)

## occfix-r10_joint

Round 9: from r8_occ_a 8k, EVERYTHING but the backbone fine-tuned 4k steps (LR 5e-5, all losses, v6 recipe + native occupancy weight 1.0): detection 61.8% gate, map 59.0%, occupancy after per-class offsets = r8_occ_a's -- safe but no gain

Directory: `outputs/distill/exp/r10_joint`
Checkpoints: `snap_1000.pt` (659 MB, 10-02 00:24), `snap_2000.pt` (659 MB, 10-02 00:41), `snap_3000.pt` (659 MB, 10-02 00:58), `snap_4000.pt` (659 MB, 10-02 01:15), `snap_4000_classbias.pt` (425 MB, 10-02 01:29), `student.pt` (659 MB, 10-02 01:15)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r10_joint-2k | 2000 | 250 | scene | 0.617 | 99.6 |  |  | 1.511 |  | 0.342 / 0.270 | 35,106 / 18,016 | `outputs/distill/exp/r10_joint/snap_2000.pt` |
| occfix-r10_joint-4k | 4000 | 250 | scene | 0.618 | 99.7 |  |  | 1.513 |  | 0.342 / 0.270 | 34,961 / 18,016 | `outputs/distill/exp/r10_joint/snap_4000.pt` |

## occfix-r11_ray_joint

Round 9: from r9_ray_a 4k, everything but the backbone with the raymap head, 4k steps (LR 5e-5, all losses)

Directory: `outputs/distill/exp/r11_ray_joint`
Checkpoints: `snap_1000.pt` (659 MB, 10-02 01:33), `snap_2000.pt` (659 MB, 10-02 01:50), `snap_3000.pt` (659 MB, 10-02 02:06), `snap_4000.pt` (659 MB, 10-02 02:23), `snap_4000_classbias.pt` (425 MB, 10-02 02:43), `student.pt` (659 MB, 10-02 02:24)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r11_ray_joint-2k | 2000 | 250 | scene | 0.620 | 100.0 |  |  | 1.511 |  | 0.338 / 0.270 | 35,011 / 18,016 | `outputs/distill/exp/r11_ray_joint/snap_2000.pt` |
| occfix-r11_ray_joint-4k | 4000 | 250 | scene | 0.616 | 99.4 |  |  | 1.513 |  | 0.338 / 0.270 | 34,867 / 18,016 | `outputs/distill/exp/r11_ray_joint/snap_4000.pt` |

## occfix-r12_ray_lovasz

Round 9: r9_ray_a's recipe + Lovasz-softmax on visible voxels (--occ-lovasz 1.0), 8k steps head-only

Directory: `outputs/distill/exp/r12_ray_lovasz`
Checkpoints: `snap_2000.pt` (435 MB, 10-02 02:32), `snap_2000_classbias.pt` (425 MB, 10-02 02:43), `snap_4000.pt` (435 MB, 10-02 03:01), `snap_4000_classbias.pt` (425 MB, 10-02 03:23), `snap_6000.pt` (435 MB, 10-02 03:29), `snap_8000.pt` (435 MB, 10-02 03:57), `snap_8000_classbias.pt` (425 MB, 10-02 04:21), `student.pt` (435 MB, 10-02 03:57)

(no evaluation records yet)

## occfix-r13_ray_lovasz_c32

Round 9: r12's recipe with a 32-channel 3-D head. Lovasz (r12/r13): raw 35.6-35.7% at 8k but only 32.1-32.8% after per-class offsets at the teacher's fill -- it rewards filling

Directory: `outputs/distill/exp/r13_ray_lovasz_c32`
Checkpoints: `snap_2000.pt` (436 MB, 10-02 02:55), `snap_2000_classbias.pt` (426 MB, 10-02 03:23), `snap_4000.pt` (436 MB, 10-02 03:24), `snap_4000_classbias.pt` (426 MB, 10-02 04:01), `snap_6000.pt` (436 MB, 10-02 03:53), `snap_8000.pt` (436 MB, 10-02 04:22), `snap_8000_classbias.pt` (426 MB, 10-02 04:41), `student.pt` (436 MB, 10-02 04:22)

(no evaluation records yet)

## occfix-r14_teacher_out

Round 10: the TEACHER's occupancy as the target for unobserved voxels (--occ-outside-target teacher), weight 1.0: 25.9% calibrated at 2k, raw fill 52k, two classes the teacher never predicts collapse to IoU 0 -- stopped (r15_teacher_out_w3, weight 3.0, stopped with it)

Directory: `outputs/distill/exp/r14_teacher_out`
Checkpoints: `snap_2000.pt` (435 MB, 10-02 04:27), `snap_2000_classbias.pt` (425 MB, 10-02 04:40), `snap_4000.pt` (435 MB, 10-02 04:56), `student.pt` (435 MB, 10-02 04:56)

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfix-r14_teacher_out-2k | 2000 | 250 | scene | 0.616 | 99.3 |  |  | 1.511 |  | 0.300 / 0.270 | 51,929 / 18,016 | `outputs/distill/exp/r14_teacher_out/snap_2000.pt` |

## occfix-r16_w06

Last check: v8's recipe (r8_occ_a) with unobserved weight 0.6, 6000 steps: 34.08% after per-class offsets on the calibration frames (v8's head 34.40%)

Directory: `outputs/distill/exp/r16_w06`
Checkpoints: `snap_2000.pt` (435 MB, 10-02 05:26), `snap_2000_classbias.pt` (425 MB, 10-02 05:30), `snap_4000.pt` (435 MB, 10-02 05:54), `snap_4000_classbias.pt` (425 MB, 10-02 06:10), `snap_6000.pt` (435 MB, 10-02 06:21), `snap_6000_classbias.pt` (425 MB, 10-02 06:30), `student.pt` (435 MB, 10-02 06:21), `student_5000.pt` (435 MB, 10-02 06:07)

(no evaluation records yet)

## occfix-r17_w15

Last check: v8's recipe (r8_occ_a) with unobserved weight 1.5, 6000 steps: 34.36% after per-class offsets on the calibration frames (v8's head 34.40%) -- the weight is flat around 1.0; v8 stands

Directory: `outputs/distill/exp/r17_w15`
Checkpoints: `snap_2000.pt` (435 MB, 10-02 05:27), `snap_2000_classbias.pt` (425 MB, 10-02 05:50), `snap_4000.pt` (435 MB, 10-02 05:55), `snap_4000_classbias.pt` (425 MB, 10-02 06:10), `snap_6000.pt` (435 MB, 10-02 06:22), `snap_6000_classbias.pt` (425 MB, 10-02 06:30), `student.pt` (435 MB, 10-02 06:22), `student_5000.pt` (435 MB, 10-02 06:08)

(no evaluation records yet)

## r3_fine

r2_hist3 recipe + 2x detection-head grid (head_upsample 2, 0.256 m heatmap) aimed at the pedestrian / cone / <20 m gap, 50k steps on GPUs 4-7 (2026-09-27 06:01-20:43). Gate: 54.0 / 59.5 / 60.1 / 59.9 / 59.6% at 10/20/30/40/50k (96-97% retained): the small-class recall it was built for did come (pedestrians 47-49%, cones 63-70%, bicycles 42-54%, barriers 58-65%) but cars and trucks stayed 2-5 points under r3_final, so it did not win overall

Directory: `outputs/distill/exp/r3_fine`
Checkpoints: `snap_10000.pt` (834 MB, 09-27 09:15), `snap_20000.pt` (834 MB, 09-27 12:17), `snap_30000.pt` (834 MB, 09-27 15:08), `snap_40000.pt` (834 MB, 09-27 17:55), `snap_50000.pt` (834 MB, 09-27 20:42), `student.pt` (834 MB, 09-27 20:42), `student_40000.pt` (834 MB, 09-27 17:55), `student_45000.pt` (834 MB, 09-27 19:19), `student_50000.pt` (834 MB, 09-27 20:42), `student_avg.pt` (209 MB, 09-27 21:01)
H200 speed: onnx_shape_h200.json: 13.2 Hz; onnx_h200.json: 13.8 Hz

| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r3_fine-mid10000 | 10000 | 250 | scene | 0.540 | 87.2 |  |  | 1.561 | 0.270 |  |  | `outputs/distill/exp/r3_fine/snap_10000.pt` |
| r3_fine-mid20000 | 20000 | 250 | scene | 0.595 | 96.0 |  |  | 1.525 | 0.293 |  |  | `outputs/distill/exp/r3_fine/snap_20000.pt` |
| r3_fine-mid30000 | 30000 | 250 | scene | 0.601 | 97.0 |  |  | 1.516 | 0.305 |  |  | `outputs/distill/exp/r3_fine/snap_30000.pt` |
| r3_fine-mid40000 | 40000 | 250 | scene | 0.599 | 96.7 |  |  | 1.514 | 0.312 |  |  | `outputs/distill/exp/r3_fine/snap_40000.pt` |
| r3_fine | 50000 | 250 | scene | 0.596 | 96.1 |  |  | 1.514 | 0.315 |  |  | `outputs/distill/exp/r3_fine/student.pt` |
| r3_fine-nms2 | 50000 | 250 | scene | 0.587 | 94.7 |  |  | 1.514 | 0.315 |  |  | `outputs/distill/exp/r3_fine/student.pt` |
| r3_fine-avg | 50000 | 250 | scene | 0.596 | 96.2 |  |  | 1.514 | 0.314 |  |  | `outputs/distill/exp/r3_fine/student_avg.pt` |
| r3_fine-mid10000 | 10000 | 250 | scene | 0.540 | 87.2 |  |  | 1.561 | 0.270 |  |  | `outputs/distill/exp/r3_fine/snap_10000.pt` |
| r3_fine-mid20000 | 20000 | 250 | scene | 0.595 | 96.0 |  |  | 1.525 | 0.293 |  |  | `outputs/distill/exp/r3_fine/snap_20000.pt` |
| r3_fine-mid30000 | 30000 | 250 | scene | 0.601 | 97.0 |  |  | 1.516 | 0.305 |  |  | `outputs/distill/exp/r3_fine/snap_30000.pt` |
| r3_fine-mid40000 | 40000 | 250 | scene | 0.599 | 96.7 |  |  | 1.514 | 0.312 |  |  | `outputs/distill/exp/r3_fine/snap_40000.pt` |
| r3_fine-mid50000 | 50000 | 250 | scene | 0.596 | 96.1 |  |  | 1.514 | 0.315 |  |  | `outputs/distill/exp/r3_fine/snap_50000.pt` |
| r3_fine-final-valfoldB | 50000 | 2219 | valfoldB | 0.599 | 97.6 |  |  | 1.624 | 0.279 |  |  | `outputs/distill/exp/r3_fine/student.pt` |
| r3_fine-final-valfoldA | 50000 | 2233 | valfoldA | 0.599 | 98.7 |  |  | 1.555 | 0.279 |  |  | `outputs/distill/exp/r3_fine/student.pt` |
| r3_fine-official | 30000 | 6019 |  |  |  | 0.3667 | 0.4369 |  |  |  |  | `outputs/distill/exp/r3_fine/snap_30000.pt` |

