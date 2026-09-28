# Qwen-Drive student: one ONNX graph, >10 Hz on a single GPU, ≥95% of the teacher on unseen scenes

*Live document — numbers are updated as lanes finish; the machine-written scoreboard is
`REPORT.md`, the full lab notebook `outputs/distill/lab_notebook.json`, the chronology `PLAN.md`.*
*Last update: 2026-09-26 17:50.*

## 1. Target and result

| requirement | status | evidence |
|---|---|---|
| everything in one ONNX graph: 3-D detection, occupancy, map, trajectory | done | 608-node graph, inputs `image, bev_index, valid, ego, prev_bev, warp_grid`, outputs `cls, box, occ, seg, trajectory, bev_state` |
| faster than 10 Hz on one GPU (H200) | done | graph 16.9–19.2 Hz for the 1152×640 shape on an idle H200; stateful deployment loop (state kept on device, host-side warp included) 20.1 Hz for the 896×512 shape, ≈15 Hz projected for 1152×640 — clean re-measurement of the deliverable pending an idle GPU |
| ≥95% of the teacher's detection | **met (98.4%)** | F1 61.0% vs teacher 62.0% on 250 frames from 150 nuScenes val scenes the student never saw; snapshot at step 80k of 120k (40k: 60.3%, 60k: 60.2%), the low-LR tail still running |
| ≥95% of the teacher's trajectory | met | ADE 1.51 m vs teacher 1.63 m on the same frames |
| occupancy | far above teacher | 32.2% mIoU vs Occ3D ground truth (teacher 3.2%); the teacher's occupancy is not usable on real labels |
| official nuScenes detection metric (all 6,019 val keyframes, 10 classes) | **mAP 0.350 / NDS 0.425** | teacher 0.216 / 0.222 on this metric; published camera-only R50 models: BEVDet4D ≈ 0.32/0.46, BEVDepth-4D ≈ 0.35/0.48 |

Deliverable files: `outputs/onnx/deliverable/student.onnx` (+ `deliverable.json`), checkpoint
`outputs/distill/exp/r2_long/snap_40000.pt`; the lane `r2_long` finishes its 120k steps
~01:30 Sep 27 and will replace the snapshot if its final/averaged checkpoint scores higher.

## 2. What the numbers mean (and why they are lower than earlier ones)

**Protocol.** The student is trained only on the 700 official nuScenes *train* scenes and
scored on 250 frames from the 150 *val* scenes. The teacher (the original 4B Qwen-Drive
perception + planner heads, never trained on nuScenes) is scored on exactly the same frames
with the same ground truth: nuScenes boxes in the 10 official classes, boxes without any
lidar/radar return dropped as the official evaluator does, a detection counts when its
centre lies within 2 m of a same-class object (teacher and student compared in the
teacher's 7 class groups so its coarser taxonomy is not penalised). F1 = harmonic mean of
recall and precision over all objects in the 250 frames.

**Why 60.3 vs 62.0 rather than 64 vs 68.** The first two days of this project evaluated on
a *frame-level* split: 90% of every scene's keyframes in training, the rest held out. The
held-out frames' 0.5 s neighbours were in the training set, and a camera model memorises
scenes. Measured on 2026-09-25: the same model scores F1 79.2% on held-out frames of its
training scenes and 57.2% on frames of unseen scenes (the teacher: 71.9% vs 62.0%) — about
12 points of scene memorisation. Every number in this report is on unseen scenes.

**Official metric.** nuScenes `DetectionEval` on all 6,019 val keyframes, 10 classes:
the 896×512 model reached mAP 0.304 / NDS 0.394; the 1152×640 snapshot at step 40k reaches
**mAP 0.350 / NDS 0.425** (per class: car 0.57, barrier 0.57, cone 0.56, pedestrian 0.38, bus
0.34, motorcycle 0.33, bicycle 0.28, truck 0.27, trailer 0.12, construction 0.09; ATE 0.67 m,
ASE 0.29, AOE 0.71 rad, AVE 0.59 m/s, AAE 0.25). For scale: BEVDet-R50 ≈ 0.30/0.38,
BEVDet4D-R50 ≈ 0.32/0.46, BEVDepth-R50-4D ≈ 0.35/0.48 in the literature.
The teacher, mapped onto the 10-class metric, scores 0.216 / 0.222 because it has one
"vehicle" class (truck, bus, trailer, construction vehicle, motorcycle cannot score).

## 3. The model

ResNet-50 at 1152×640 on six cameras → per-camera depth distribution (64 bins, lidar
supervised) → scatter lift-splat into a 200×200 BEV (0.512 m) → previous keyframe's BEV
warped by ego motion and fused (BEVDet4D-style; the previous frame's state comes back from
the graph's own `bev_state` output, so the backbone runs once per frame) → BEV encoder →
dense CenterPoint head (10 classes + velocity), occupancy head (200×200×16×10), map head
(6×200×400), trajectory head (50×3, constant-velocity anchor + residual). 51.75 M
parameters. Trained from scratch (ImageNet init) with CBGS class balancing, EMA 0.999,
horizontal camera flip, occupancy supervised by Occ3D-nuScenes labels, one-cycle schedule.

## 4. What worked, what failed (chronological, condensed; details in PLAN.md §19–40)

Worked, in order of measured effect on unseen scenes: correctly placed lidar depth targets
at the training resolution (+2.8 F1 at equal steps once a binning bug was fixed); adding
the 6,983 train-scene keyframes the teacher had never processed (+33% data); the wider
dense head (+3.1 on the earlier split); temporal fusion of the previous keyframe (+5.0 on
the earlier split; on unseen scenes the mid-range recall gap to the teacher fell from 19 to
4 points); 1152×640 input (+0.8 at equal steps, more at 30 m+); Occ3D labels for occupancy
(5% → 30% mIoU); training long enough to use the whole low-LR tail.

Failed or found wrong: the frame-level split (inflated every early number by ~10 points);
the DETR-style query head (≤2% F1 in five attempts, replaced by a dense head); a no-grad
"previous frame" pass placed inside the training autocast region (silently cut every
backbone gradient — caught by DDP's reducer); camera flip without flipping the depth
target; depth targets binned in the wrong pixel space for 1152×640 (cost R1 its result);
a CBGS/recipe lane that measured flat on the leaked split; ResNet-101 not yet tested.

## 5. Demo video

`outputs/student_video/scene-0276/student_scene-0276.mp4` — the student on scene-0276 (a held-out
intersection: pedestrians, scooters, parked bicycles), 40 keyframes at real time: camera ring with
3-D boxes and predicted vs recorded ego path, lidar BEV, occupancy, map. Any scene:
`python local/distill/student_video.py --scene scene-XXXX --gpu N`.

## 6. Reproduce

```
# labels and caches (once): scene split, Occ3D labels, depth targets, GT in 10 classes
python local/distill/occ3d_labels.py --split train ; python local/distill/occ3d_labels.py --split val
python local/distill/nusc_depth.py --root data/nuscenes --version v1.0-trainval --image-size 1152 640 --out-name depth_1152x640.npz
# train (4 GPUs), evaluate, export, time
bash local/distill/scripts/lane.sh r2_long "0,1,2,3" 4 120000 "--image-size 1152 640" outputs/logs/lane_r2_long.log
python local/distill/diagnose.py --ckpt outputs/distill/exp/r2_long/student.pt --tokens scene --limit 250
python local/distill/eval_official.py --ckpt outputs/distill/exp/r2_long/student.pt
python local/distill/export_student.py --ckpt outputs/distill/exp/r2_long/student.pt --out outputs/onnx/r2_long/student.onnx
python local/distill/run_stateful.py --ckpt outputs/distill/exp/r2_long/student.pt --onnx outputs/onnx/r2_long/student.onnx --check
```

## 7. Still running (unattended, `scripts/orchestrate_95.py`)

`r2_long` to 120k steps (~01:30 Sep 27, then averaging of its late snapshots); `r2_hist3`
(three previous keyframes) on the other four GPUs; a 1408×768 lane; and a final run
combining whatever helped. The scoreboard, the deliverable copy and this document's §1
table are updated after every finished lane.

## Addendum 2026-09-27: real-frame speed and the scatter fix

Speed measured the strict way (idle H200, real nuScenes frames, full loop: H2D + graph + D2H, BEV state kept
on the device, `run_stateful.py` pinned loop):

| graph | graph run | full loop | Hz |
|---|---|---|---|
| r2_long 80k, original export (v1) | 90 ms | 93 ms | 10.7 |
| r2_long 80k, de-contended scatter (v2, same weights) | 32 ms | 41 ms | 24.1 |

The v1 number is what the shipped ONNX really did; the exporter's 60 ms came from all-valid dummy rays that
never hit the single scratch row every invalid ray was aimed at. v2 changes only how invalid rays are
scattered (spread over scratch rows, then dropped): outputs are the same model to 1e-5 relative. Both meet the
10 Hz gate; v2 is the deliverable graph from here on (`outputs/onnx/deliverable/student_r2_long_80000_v2.onnx`).

## Full-val confirmation (2026-09-27 02:30)

Same protocol as the 250-frame gate, run on every cached keyframe of the 150 val scenes (4,452 frames, 117k GT
objects): teacher F1 61.0%, student (r2_long 80k) 59.5% -> **97.5% retained** (95.3% with 2 m NMS). With this
many objects the F1 noise is about 0.3 points, so the 95% gate is cleared with margin. Per class the student is
at or above the teacher on cars, trucks, trailers, construction vehicles, bicycles and barriers (0.60 vs 0.53);
it trails on pedestrians (0.46 vs 0.50), motorcycles (0.41 vs 0.45) and traffic cones (0.59 vs 0.68). Per range
it trails only inside 20 m (0.85 vs 0.92 at 0-10 m, 0.78 vs 0.82 at 10-20 m) and is at parity or better beyond.
Occupancy 27.7% vs 2.9% mIoU on the same frames; map agreement with the teacher 0.60.

## Addendum 2026-09-27 06:00: r2_hist3 outcome

Three-frame BEV history (r2_hist3, 60k steps): final 60.6% F1 (97.8% retained), 4-snapshot average 60.85%
(98.2%), official mAP 0.342 / NDS 0.437. That is a tie with the r2_long 80k deliverable (61.0%, 98.4%), not the
+1.4 points its 19k read on 100 frames suggested, so the deliverable is unchanged. Its graph with three history
frames runs the real-frame deployment loop at 55 ms (18.1 Hz) on an idle H200. Lanes still running: r3_final
(history 3, 75k) and r3_fine (history 3 + 2x detection-head grid, 50k, aimed at the pedestrian / cone / <20 m gap).

## Addendum 2026-09-27 08:35: full-val ranking favours r2_hist3

On all 4,452 cached val-scene keyframes (117k objects, noise ~0.3 pt) the three-frame-history model beats the
current deliverable: r2_hist3 final weights (60k) 60.3% F1 = 98.8% of the teacher's 61.0%, against r2_long 80k at
59.5% = 97.5%. On the 250-frame gate the order was reversed by 0.4 points (60.6 vs 61.0), which is inside that
protocol's ±1.5-point noise. The full set is the more reliable ranking, so the final deliverable choice will be
made on it: candidates are scored on the 4,452 frames (in two scene folds run in parallel) once their lanes
finish, and the 250-frame protocol stays the quick gate. r2_hist3's graph runs the real-frame loop at 18.1 Hz.

## Addendum 2026-09-27 10:20: deliverable moves to r2_hist3 (averaged), 99.2% on the full val set

Exact full-val numbers (117,417 GT objects, the two scene folds merged from tp/fp/fn):

| model | F1 | retained | gate (250 frames) | official mAP / NDS | loop on idle H200 |
|---|---|---|---|---|---|
| teacher | 61.0% | | 62.0% | 0.215 / 0.222 | ~1 Hz |
| r2_hist3 averaged 45-60k (history 3) | **60.5%** | **99.2%** | 60.85% (98.2%) | 0.342 / 0.437 | 18.1 Hz |
| r2_hist3 final 60k | 60.3% | 98.8% | 60.6% (97.8%) | | 18.1 Hz |
| r2_long 80k (history 1) | 59.5% | 97.5% | 61.0% (98.4%) | 0.350 / 0.425 (40k) | 24.1 Hz (v2 graph) |

The fold merge reproduces the single-run full-val number for r2_long 80k (59.53% vs 59.54%), so the two
protocols agree. Release bundle: `outputs/student_release_v3/`. r3_final (20k snapshot already 60.8% on the gate)
and r3_fine (2x head grid: pedestrian / cone / bicycle recall above the deliverable at 10k) may still move this.

## Addendum 2026-09-27 12:50: r3_final 30k snapshot leads the full-val ranking

| candidate | full-val F1 (4,452 frames, teacher 61.04%) | retained | source |
|---|---|---|---|
| r3_final-mid30000 | 60.85% | 99.70% | folds A+B |
| r2_hist3-avg | 60.52% | 99.16% | folds A+B |
| r2_hist3 | 60.32% | 98.83% | valall |
| r2_long-80000 | 59.54% | 97.54% | valall |

r3_final (three-frame history, de-contended scatter, 75k one-cycle) at 30k of 75k already leads; the lane runs
on and every 10k snapshot is scored on the gate, the leaders on the full set. r3_fine (2x head grid) reads 96.0%
on the gate at 20k: its small-object recall advantage at 10k has not yet turned into an overall win.
Release bundle for the r3_final 30k snapshot: `outputs/student_release_v4/`.

## Addendum 2026-09-27 22:35: idle-GPU speed of the leader, r3_fine on the full set

The r3_final 30k graph (three history frames, de-contended scatter) in the pinned real-frame loop on an idle
H200: graph run 33.8 ms, full loop 41.7 ms = **24.0 Hz**. The extra history frames cost nothing at inference:
the previous BEV states stay on the device. r3_fine (2x head grid) final weights on the full val set:
59.9% = 98.1% retained; its lane picked the 30k snapshot (60.1% on the gate) for export. Remaining candidates
are r3_final's 40k / 70k / final / averaged checkpoints, being fold-scored as the lane finishes.

## Addendum 2026-09-27 22:50: r3_final done; r3_fine wins the official mAP

r3_final finished 75k steps. Gate scores: 30k 61.5%, 40k 61.3%, 50k 60.8%, 60k 60.9%, final 60.9%, 4-snapshot
average (60-75k) 61.0%. The plateau is flat from 30k on; the full-val folds of the 40k, 70k, final and averaged
checkpoints are running to pick the deliverable among them (30k currently leads at 99.7%).

r3_fine (2x detection-head grid), best snapshot 30k: official nuScenes DetectionEval mAP **0.367** / NDS 0.437,
the highest mAP of any student (r2_long 40k 0.350, r2_hist3-avg 0.342; teacher 0.215). The official metric
weights all ten classes equally, so its pedestrian / cone / bicycle gains count fully there, while the
object-count-weighted F1 that decides the deliverable is dominated by cars. Its graph (three history frames,
2x head) runs the real-frame loop at 51.7 ms = 19.4 Hz on an idle H200.

## Addendum 2026-09-28 00:55: occupancy fine-tune (r4_occfix)

r3_final 30k fine-tuned 5k annealing steps with a 0.2-weight cross-entropy pull toward "empty" outside the Occ3D
camera-visible mask. Raw, unmasked output on scene-0276: non-empty voxels outside the mask fall from ~300k to
75-98k per frame and now form contiguous extensions of the visible surfaces instead of a grid-wide blanket;
inside the mask the non-empty count matches the ground truth to within 2% (15.3k vs 15.0k, 14.4k vs 14.2k).
Detection is being re-scored (gate + full-val folds) before this checkpoint can replace the deliverable.

## Addendum 2026-09-28 01:35: r4_occfix scored

Official nuScenes DetectionEval (all 6,019 val keyframes): mAP 0.3656 / NDS 0.4401. Real-frame deployment
loop on an idle H200: 40.5 ms/frame = 24.7 Hz. Gate 61.2% (98.7%). Full-val folds: fold B 60.8% (99.1% of the
teacher on that fold; r3_final 30k had 98.8% there); fold A pending. Raw occupancy no longer blankets the grid.

## Addendum 2026-09-28 02:00: full-val ranking with the fine-tunes; deliverable = r4_occfix

| candidate | full-val F1 (teacher 61.04%) | retained |
|---|---|---|
| r3_final-avg30-50 | 61.22% | 100.30% |
| r4_occfix-final | 60.94% | 99.85% |
| r3_final-mid30000 | 60.85% | 99.70% |
| r2_hist3-avg | 60.52% | 99.16% |
| r2_hist3 | 60.32% | 98.83% |
| r3_fine-final | 59.90% | 98.14% |
| r2_long-80000 | 59.54% | 97.54% |

The 30/40/50k plateau average edges the teacher (100.3%) but still paints the unobserved grid; r4_occfix is
0.3 points behind it (inside the ±0.3 noise), fixes the occupancy, has the best NDS (0.440) and runs at 24.7 Hz,
so it is the recommended deliverable (release v5). Running: r5_avg_occfix = the plateau average fine-tuned with
the same occupancy term (GPUs 4-7, ~03:20) and r4_small (small-class heatmap weights, finishing now); either
replaces v5 only if it ranks higher on the full val set with the occupancy fix intact.

## Addendum 2026-09-28 02:05: official numbers of the r3_final 30k seed

Official nuScenes DetectionEval, all 6,019 val keyframes, r3_final 30k snapshot: mAP 0.3652 / NDS 0.4367
(r4_occfix, its occupancy fine-tune: 0.3656 / 0.4401; r3_fine: 0.3667 / 0.4369; teacher 0.2155 / 0.2220).
r4_small (small-class heatmap weights + occupancy term) finished its 40k steps at 02:00 and is being scored;
r5_avg_occfix passed its launch check and trains on GPUs 4-7 (~03:05 end).

## Addendum 2026-09-28 03:05: r4_small scored; r5 in post

r4_small (small-class heatmap weights + occupancy term, 30k->40k): final weights 60.95% on the full val set
(99.9%), gate 60.7%; its 35k/40k average scores 61.6% on the gate (best of the project) and official mAP
**0.375** / NDS **0.444** (best of the project; teacher 0.215 / 0.222); real-frame loop 36.7 ms = 27.3 Hz. Its
average's full-val folds are running. r5_avg_occfix (plateau average + occupancy term): gate 61.3% (99.0%),
official eval running. The final choice is made on the full val set among r4_occfix, r4_small-avg and r5, all
of which carry the occupancy fix.

## Addendum 2026-09-28 03:35: FINAL deliverable = r4_small averaged (release v6)

| candidate | full-val F1 (teacher 61.04%) | retained |
|---|---|---|
| r4_small-avg | 61.35% | 100.52% |
| r3_final-avg30-50 | 61.22% | 100.30% |
| r4_small-final | 60.95% | 99.87% |
| r4_occfix-final | 60.94% | 99.85% |
| r3_final-mid30000 | 60.85% | 99.70% |
| r2_hist3-avg | 60.52% | 99.16% |

r4_small's 35k/40k average leads the full val set (100.5%), the gate (61.6%) and the official metric
(mAP 0.375 / NDS 0.444), carries the occupancy fix, and runs the real-frame loop at 27.3 Hz on an idle H200.
r5_avg_occfix (plateau average + occupancy term): gate 61.3%, official mAP 0.364 / NDS 0.446; its folds finish next.

## Addendum 2026-09-28 04:05: r5 scored; weight soups; last lane

r5_avg_occfix (plateau average + occupancy term) on the full val set: 60.8% = 99.6%, below r4_small-avg
(100.5%); official mAP 0.364 / NDS 0.446. Weight soups of the fine-tunes (all in the r3_final basin): mean of
r4_small-avg and r5 scores 61.9% on the gate (best gate number of the project), mean of r4_small-avg and r4_occfix
61.5%; the first is being fold-scored. r6a (plateau average + the r4_small recipe, 10k steps) trains on GPUs 0-3
until ~06:30. Cutoff for replacing release v6: 09:15.

## Addendum 2026-09-28 04:50: soup scored

The r4_small-avg + r5 weight soup: 61.9% on the gate but 61.2% = 100.3% on the full val set, below r4_small-avg
(100.5%). The gate and the full set disagree again inside the gate's noise; the full set decides, v6 stands.
r6a is at 35k of 40k.

## Final per-class and per-range recall of the deliverable (all 4,452 val frames, best single threshold)

| class | GT objects | teacher recall | student recall (r4_small-avg) |
|---|---|---|---|
| car | 48,518 | 60.5% | 61.4% |
| truck | 10,136 | 41.3% | 41.6% |
| bus | 2,265 | 46.7% | 42.8% |
| trailer | 2,779 | 20.5% | 25.1% |
| construction_vehicle | 1,745 | 16.8% | 20.6% |
| pedestrian | 22,368 | 50.1% | 45.6% |
| motorcycle | 1,645 | 45.4% | 47.3% |
| bicycle | 1,549 | 38.9% | 47.9% |
| traffic_cone | 8,978 | 68.5% | 66.6% |
| barrier | 17,434 | 52.6% | 66.5% |

| range | GT objects | teacher recall | student recall |
|---|---|---|---|
| 0-10 m | 11,659 | 92.3% | 87.2% |
| 10-20 m | 27,274 | 82.1% | 80.9% |
| 20-30 m | 25,662 | 64.5% | 69.8% |
| 30-40 m | 20,293 | 44.6% | 51.1% |
| 40+ m | 32,529 | 14.1% | 14.9% |

## Closing (2026-09-28 06:35): every candidate scored, release v6 is final

| candidate | full-val F1 (4,452 frames, teacher 61.04%) | retained |
|---|---|---|
| r4_small-avg | 61.35% | 100.52% |
| r3_final-avg30-50 | 61.22% | 100.30% |
| r4_small-soup_small_r5 | 61.20% | 100.27% |
| r6a_avg_small-avg | 61.11% | 100.13% |
| r6a_avg_small-final | 60.99% | 99.93% |
| r4_small-final | 60.95% | 99.87% |
| r4_occfix-final | 60.94% | 99.85% |
| r3_final-mid30000 | 60.85% | 99.70% |
| r5_avg_occfix-final | 60.77% | 99.56% |
| r2_hist3-avg | 60.52% | 99.16% |
| r2_hist3 | 60.32% | 98.83% |
| r3_fine-final | 59.90% | 98.14% |
| r2_long-80000 | 59.54% | 97.54% |

r6a (plateau average + the r4_small recipe): average 61.1% = 100.1%, final 61.0%, gate 61.5%, official mAP
0.370 / NDS 0.444, 27.1 Hz. It did not beat r4_small-avg, so the deliverable stays release v6 (sent to the user
at 03:40 with its README, MD5s and demo video). All lanes, evaluations and the orchestrator are finished; no
process is running; nothing was overwritten.
