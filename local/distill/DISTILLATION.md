# Distilling Qwen-Drive-1.0 into a 10 Hz student — what was tried, what worked, what failed, where it stands

*Written 2026-09-26 21:40 on the 8-GPU machine. Detailed chronology with timestamps: `PLAN.md` (§1–42) and
`PLAN_V2.md`; deliverable summary: `REPORT_FINAL.md`; machine-written scoreboard: `REPORT.md`;
every evaluation record: `outputs/distill/lab_notebook.json` (58 records).*

---

## 1. The task and how the bar moved

| when | requirement as stated | how it was scored |
|---|---|---|
| Sep 22 | export the whole model to one ONNX running ≥10 Hz; the teacher's ONNX cannot, so distil, "degradation < 10%" for 3-D perception and trajectory | initially undefined; became F1 at 2 m centre distance vs nuScenes GT, student/teacher on the same frames |
| Sep 24 | "the student should keep at least 80% of its teacher" | same F1 ratio |
| Sep 25 | 10 Hz on a single GPU (first A100, then H200 after the move to the 8-GPU machine); "the ONNX should have everything: perception and trajectory" | one graph, all five outputs |
| Sep 25 evening | **≥95% of the teacher's performance**, 3 days, 8×H200 | F1 ratio on the *honest* split (see §4), plus trajectory and occupancy reported |

Compute: 4×A100 until Sep 25 17:30, then 8×H200. Data: nuScenes
trainval — 25,599 keyframes with cached teacher outputs, 34,149 keyframes with images + GT.

## 2. Teacher, student, deliverable

**Teacher:** Qwen-Drive-1.0 (4B VLM: Qwen3.5 text stack with 24 Gated-DeltaNet + 8 attention
layers) with the perception head (900-query DETR-style detector on a 200×200 BEV, occupancy
200×200×16×10, map 6×200×400) and the planner-sft head (50×3 trajectory). ~1 Hz on one GPU.
Not an oracle on nuScenes: against nuScenes GT it scores F1 68.2% on the frame-split frames,
62.0% on the val-scene frames; official mAP 0.215 / NDS 0.222 (its 7-class taxonomy cannot
score truck, bus, trailer, construction vehicle, motorcycle); its occupancy is 3.2% mIoU
against Occ3D labels; its trajectory ADE 1.63 m on the val-scene frames.

**Student (final recipe):** ResNet-50 (ImageNet init) on six cameras at 1152×640 → per-camera
64-bin depth distribution supervised by lidar → scatter lift-splat into a 200×200 BEV
(0.512 m) → the previous keyframe's BEV warped by ego motion and fused (`--history 1`; a
3-keyframe variant `--history 3` is in training) → BEV encoder → dense CenterPoint head with
the 10 nuScenes classes + velocity → occupancy head (10 teacher classes, trained on Occ3D
labels) → map head (teacher-distilled) → trajectory head (constant-velocity anchor +
learned residual). 51.75 M parameters. Trained from scratch on the 700 official train
scenes (28,130 keyframes) with CBGS class balancing, EMA 0.999, horizontal camera flip,
one-cycle LR, snapshot averaging over the last quarter.

**Deliverable:** `outputs/onnx/deliverable/student.onnx` — one graph, 608 nodes, opset 20,
inputs `image, bev_index, valid, ego, prev_bev, warp_grid`, outputs `cls, box, occ, seg,
trajectory, bev_state` (the state is fed back next frame; the host builds the 200×200 warp
grid from two ego poses). Validated by `onnx.checker`, a CPU forward, and a proof that the
recurrent loop equals the training-time two-frame path (max heatmap diff 3e-3).

## 3. Results at a glance

### 3.1 Honest protocol (student trained on train scenes only, scored on 250 frames from the 150 val scenes it never saw; teacher on the same frames)

| model | steps | F1 | teacher F1 | retained | traj ADE (teacher 1.63) | occ mIoU vs Occ3D (teacher 3.2%) | official mAP / NDS (6,019 val frames) |
|---|---|---|---|---|---|---|---|
| FINAL 896×512 | 80k | 57.2% | 62.0% | 92.3% | 1.54 m | 5.0% | 0.304 / 0.394 |
| R1 1152×640 (trained on misaligned depth targets) | 80k | 56.6% | 62.0% | 91.4% | 1.51 m | 4.9% | 0.289 / 0.370 |
| **r2_long 1152×640, corrected recipe** @19k | 19k | 58.7% | 62.0% | 94.7% | 1.54 m | 28.2% | — |
| same @40k | 40k | 60.3% | 62.0% | 97.3% | 1.51 m | 30.2% | **0.350 / 0.425** |
| same @60k | 60k | 60.2% | 62.0% | 97.2% | 1.51 m | 31.2% | — |
| **same @80k (current deliverable)** | 80k | **61.0%** | 62.0% | **98.4%** | 1.51 m | 32.2% | — |
| same @100k | 100k | 59.9% | 62.0% | 96.6% | 1.51 m | 32.9% | — |
| r2_hist3 (3 keyframes) @19k, 100-frame subset | 19k | 61.4% | 63.1% | 97.3% | 1.32 m | 30.8% | — |

Equal-step reads on the same 100 frames at step 19k: FINAL 56.4 → R1 57.2 → r2_long 60.0 →
r2_hist3 61.4 — each line differs from the previous by one change (resolution; correct
depth targets + more data; a longer temporal history).

### 3.2 The earlier ladder (frame-level split — leaked, see §4; teacher 68.2% on those frames)

| run | what changed | F1 | note |
|---|---|---|---|
| E0 (first student, 78k steps) | teacher-distilled DETR-style head, 83 M params, 20.6 Hz | 32.9% (37.7% with NMS) | 47% of the teacher; speed target met |
| E1, E1b, E2 | Hungarian matching + focal loss, then prior bias + reference points | 0.0 / 0.0 / 1.1% | query head never converged in 20k steps |
| E3, E4 | warm start + Hungarian / hybrid | 28.3 / 33.0% | first reported as 2.2 / 1.7% — evaluator read the wrong config |
| E5, E5b | dense CenterPoint head | 0.0% after 36k steps | heatmap target bug (Gaussian centred on fractional coords → no positives) |
| E5c | same head, target fixed | 45.5% @120k → **55.2%** @154.5k | +10 F1 from the low-LR tail alone |
| E6 | wider head (128×3), min radius 2 | 58.3% | +3.1 |
| E7 | + CBGS, EMA, camera flip | 59.2% | +0.9 (this split under-rewards regularisers) |
| E8 | + temporal fusion (previous keyframe) | **64.2%** | +5.0; beyond 30 m above the teacher |

### 3.3 Speed (one H200, all five outputs, batch 1)

| graph | ONNX graph | stateful deployment loop |
|---|---|---|
| 896×512, history 1 | 44 ms → 22.6 Hz | 49.8 ms → 20.1 Hz (state kept on device via IO binding; 66.9 ms → 14.9 Hz before that) |
| 1152×640, history 1 (deliverable shape) | 52–59 ms → 16.9–19.2 Hz (idle-GPU measurements of the shape) | ≈15 Hz projected; clean measurement pending an idle GPU |
| 1152×640, history 3 | ~62 ms → 16 Hz | — |
| 1408×768 | ~80 ms → ~12.5 Hz | — |

Numbers measured while another lane was training on the same GPU (e.g. 9.2 Hz for the
40k snapshot) are lower bounds and are excluded by rule.

## 4. The two findings that reshaped the project

**The evaluation split leaked scenes.** The cache held 30 of every scene's 40 keyframes, and
the first two days held out the first 10% of *sorted tokens* — random frames, not scenes.
Of the 250 eval frames' 500 neighbouring keyframes (0.5 s away), 324 were in training.
Measured on the first from-scratch scene-split model: F1 79.2% on held-out frames of its
own train scenes vs 57.2% on unseen scenes (teacher 71.9 vs 62.0) — about 12 points of
scene memorisation. Every number in §3.2 carries that inflation; every number in §3.1 does
not. Consequence: the deliverable had to be trained from scratch on the 700 train scenes
(a warm start from any earlier checkpoint would carry the val scenes with it).

**The teacher is not the ceiling on the public metric.** On the official nuScenes
DetectionEval the student passed the teacher at 896×512 already (0.304 vs 0.215 mAP)
because the teacher cannot express five of the ten classes and its occupancy does not
survive contact with real labels (3.2% mIoU). The "95% of the teacher" bar is therefore
applied to the strict reading — F1 with the teacher's own class groups — where the teacher
is strongest.

## 5. What worked (with the measured effect)

1. **Dense CenterPoint head instead of the DETR-style query decoder.** Five query-head
   attempts produced ≤2% F1 (a linear probe showed the BEV features separated objects at
   AUC 0.84 — the head was the bottleneck); the dense head reached 45% within one run.
2. **Temporal fusion of the previous keyframe (BEVDet4D construction), exported as a
   stateful graph.** +5.0 F1 on the leaked split; on unseen scenes the 10–30 m recall gap
   to the teacher fell from 19 to ~5 points. `bev_state` must be the *unfused* BEV (the
   fused one would be a recurrent state the model never trained on) — found and fixed
   before any deliverable was cut.
3. **Correctly placed lidar depth targets at the training resolution.** After the binning
   bug was fixed, the 1152×640 lane gained +2.8 F1 over the bugged one at equal steps.
4. **A third more training data.** The 6,983 train-scene keyframes the teacher had never
   processed (each scene's last ~10, without a 5 s future) have images, GT boxes, Occ3D
   labels and — after the extractor fix — depth targets; the map loss is masked where the
   teacher target is missing.
5. **Input resolution 1152×640.** +0.8 F1 at equal steps with the gain at 20 m+ and on
   pedestrians; costs 22.6 → ~17 Hz.
6. **Three keyframes of history.** +1.4 F1 over two frames at equal steps (20–40 m recall).
7. **Occ3D-nuScenes labels for occupancy** (via the UniOcc repackaging on Hugging Face;
   its label ids had to be inferred from GT-box overlap and height profiles): 5% → 32% mIoU.
8. **The wider head** (+3.1), **CBGS/EMA/flip** (+0.9 on a split that under-rewards them),
   **the low-LR tail** (+10 on E5c), **snapshot averaging** (in place for every lane ending
   from now on), **IO binding** for the deployment loop (14.9 → 20.1 Hz).
9. **Instruments:** the scene split; the official evaluator (`eval_official.py`) covering
   all 6,019 val keyframes with calibration rebuilt for teacher-less frames; per-range and
   per-class recall; the teacher scored on the same frames for F1, trajectory and
   occupancy; equal-step snapshots to compare recipes hours before runs end; a lane driver
   with speed gate, smoke, self-check, resume and full evaluation; an orchestrator that
   re-attaches to running lanes after a restart.

## 6. What failed, and what it cost

| what | symptom | cause | cost | fix / rule |
|---|---|---|---|---|
| Query (DETR-style) head, 5 variants | 0–2% F1 | needs 50–500 epochs; 878 of 900 queries with unconstrained boxes; unstable matching | ~2 days | dense head |
| Frame-level split | every early number ~10 points too high | eval frames' neighbours in training | credibility of days 1–3 | scene split; final from scratch |
| `diagnose.py` used default cfg | E3/E4 reported at 2% instead of 28/33% | `ref_points` default differed from the checkpoint | wrong decisions for half a day | config read from the checkpoint |
| Heatmap Gaussian centred on fractional coords | loss exactly 0.0000, zero detections | no cell equalled 1.0 → no positives | 36k steps | integer centre; "a loss at exactly 0 is a target bug" |
| Matching cost in raw metres | loss ranked bad above good | scale | one run | normalise by BEV extent; `verify_objective.py` |
| no_grad previous-frame pass inside the training autocast region | every backbone conv weight received no gradient (found by DDP's reducer only) | autocast caches bf16 weights without autograd edges | caught in the smoke | own autocast context for the pre-pass |
| Camera flip without flipping the depth target | depth loss opened at 3.65 instead of 1.28 | image-space target not mirrored | one relaunch | mirror every image-space target |
| 1152×640 depth targets binned in 896×512 pixel space | R1 lost 0.6 F1 to the 896 run it had led | extractor reused the teacher's 896 `lidar2img` without rescaling | R1's 15 h, r2_long's first 2.5 h | rescale; regenerate; "trace a geometry flag through every consumer" |
| `bev_index[0]` shared across the batch | samples 1–3 trained on sample 0's projection | loop bug | E5c/E6 quality | per-sample indices |
| `bev_state` = fused BEV | deployment loop would have differed from training | wrong tensor returned | none (caught) | unfused state; loop equivalence test |
| GT yaw stored in the global frame (7-class file) | orientation targets meaningless (AOE 1.05 rad) | ego rotation not applied to yaw | orientation of the chain models | `gt_boxes10.npz` with ego yaw |
| NFS latency at launch | E8 killed twice by its own 300 s check with nothing printed | 23k GT reads + 74k directory stats per rank, cold cache | ~1 h + 4 orphaned rank processes | one-file label index; one stat per token; 600 s window |
| Orchestrator liveness check on `endswith(script)` | launched a duplicate resume on the same GPUs/checkpoint dir | driver cmdline ended with a pid | 2 minutes (caught) | substring check; state persisted and reloaded on restart |
| `pkill -f` / `pgrep -f` / /proc scans matching the tool shell's own command text | killed my own shell four times mid-chain | heredoc text lives in the shell's cmdline | several 10-minute detours | exclude own ancestry; match torchrun as a substring |
| A100 session closed | final run's launcher died at step 36k | driver attached to the interactive session | 20 min + a resume | nohup drivers; resume-capable lanes |
| `/tmp` (2 GB) filled by scratch ONNX / checkpoints | writes failed | small local /tmp | minutes | scratch on NFS or `/local` |
| Occ3D label download from Google Drive | permission-gated | — | 1 h | Hugging Face UniOcc mirror, rate-limit-aware fetch |
| Google-Drive/README taxonomy for UniOcc | undocumented ids | — | 1 h | inferred from GT-box overlap (car 0.96, ped 0.88, cone 0.83) and height profiles |
| Demo video conventions | boxes drawn wide; scene-end frames without a path | renderer expects lidar-frame, along-heading-first boxes; ego cache skipped scene-end frames | 2 renders | conversions in `student_video.py`; ego cache extended with `has_future` |
| Feature distillation lane | never run | ViT cache covers 10k frames at 256-d vs 1024 expected | — | dropped |
| ResNet-101 lane | not yet run | would fail the 10 Hz gate at 1152 | — | backlog behind the 1408 candidate |

## 7. Where we are now (21:40, Sep 26)

- **Target status:** met on the honest protocol by `r2_long` at step 80k — F1 61.0% vs
  62.0% (**98.4%**), trajectory 1.51 vs 1.63 m, occupancy 32% vs 3% mIoU, official mAP
  0.350 / NDS 0.425 at 40k, one ONNX graph, ~17 Hz graph / ~15 Hz loop for its shape on an
  idle H200. Deliverable files: `outputs/onnx/deliverable/student.onnx` + `deliverable.json`,
  checkpoint `outputs/distill/exp/r2_long/snap_80000.pt`, demo
  `outputs/student_video/scene-0276/student_scene-0276.mp4`.
- **Still owed:** (1) a clean deployment-loop timing of the shipped ONNX on an idle GPU
  (all eight GPUs are training until `r2_long` ends); (2) `r2_long`'s finished checkpoint —
  the last 20k low-LR steps, averaging of the 90k–120k snapshots, its own full evaluation,
  official metric, export, loop check and timing (~01:30–03:00 Sep 27); the deliverable
  swaps automatically if it scores higher.
- **Running / queued for margin (unattended, `scripts/orchestrate_95.py`):** `r2_hist3`
  (3 keyframes, 60k steps, GPUs 4–7, ~06:30 Sep 27); `r3_final` (1152×640 + 3 keyframes,
  100k steps) on GPUs 0–3 as soon as `r2_long` ends (~08:30 Sep 28 + evaluation); a
  1408×768 + history-3 candidate on 4–7 after the history lane, with a history-1 fallback
  if the speed gate rejects it. Deadline Sep 28 18:00.
- **Known remaining deficits:** pedestrians (43% vs 51% recall) and traffic cones (55% vs
  68%) — small objects; the history-3 and 1408 lanes target them. NDS is held back by
  velocity (0.59 m/s) and orientation (0.71 rad) errors.
- **Repository:** all of this is uncommitted on `dev_albertl` (8 modified files, 25 new);
  nothing pushed since `bd84c40`.

## 8. Rules that came out of it (also in the memory notes)

1. Verify the instrument before the reading: config from the checkpoint; baselines on the
   exact eval frames; the teacher scored on the same frames; no metric without its baseline.
2. Split by scene (any temporal dataset), and report only what a from-scratch scene-split
   model scores.
3. Smoke-test every new training path in the launch configuration (DDP with
   `find_unused_parameters=False` is the free gradient-coverage check).
4. A loss term that opens 2–3× higher than the previous run from the same weights is a
   target mismatch, not "the model adapting"; a loss at exactly 0.0000 is a target bug.
5. A geometry knob (image size) must be traced through every artefact a lane consumes,
   including offline extractors; regenerate or prove invariant.
6. Compare dense tensors, never sorted top-k lists, when testing equivalence.
7. Never `pkill -f` from a call whose text contains the pattern; exclude the scanner's
   ancestry; drivers under `nohup`, resume-capable, with markers a supervisor can read.
8. Keep NFS quiet during launch windows; copy small-file caches to local disk; 600 s
   self-checks.
9. Time speed on an idle GPU and name the GPU; numbers under load are lower bounds.
10. Launch with `nohup` and return; no sleep-polling in the foreground.

## 9. Reproduce the deliverable

```
# one-time data: scene split, 10-class GT, Occ3D labels, depth targets at 1152x640, ego states
python local/distill/nusc_gt_boxes.py --root data/nuscenes --taxonomy nuscenes10
python local/distill/occ3d_labels.py --split train; python local/distill/occ3d_labels.py --split val
python local/distill/nusc_depth.py --root data/nuscenes --version v1.0-trainval --image-size 1152 640 --out-name depth_1152x640.npz
python local/distill/nusc_ego.py --root data/nuscenes --version v1.0-trainval --include-short-future
# train + evaluate + export + time (one lane, 4 GPUs)
bash local/distill/scripts/lane.sh r2_long "0,1,2,3" 4 120000 "--image-size 1152 640" outputs/logs/lane_r2_long.log
# demo video on any scene
python local/distill/student_video.py --scene scene-0276 --gpu 0
```

## Release bundle and ONNX pipeline (2026-09-26)

`outputs/student_release/` holds the deliverable for use outside this machine: `student_r2_long_80k.pt`
(slim checkpoint: EMA weights + `StudentConfig`, 207 MB), `student_r2_long_80k.onnx` (216 MB; a fresh
export from the slim checkpoint is weight-for-weight identical), `student_release_code.tar.gz`
(`local/distill/`), `README.md` (setup, ONNX contract, KPIs) and `parts/` (29 MiB splits, because the
chat upload cap is 30 MiB; MD5s in `MD5SUMS`).

`local/distill/scripts/onnx_pipeline.sh CKPT [SCENE] [GPU] [OUT]` runs the whole export-and-demo path:
`export_student.py` (checkpoint -> one graph with every head + recurrent state, timed) followed by
`demo_onnx.py`, an onnxruntime-only demo that needs no training caches: calibration, images and the
ego-pose track come from the nuScenes devkit, the 116-d ego state is rebuilt the way `nusc_ego.py`
built it (checked against the cache: max difference 1e-3 m on a val keyframe), the BEV state stays on
the device between keyframes through IO binding, and the session-demo layout is rendered to MP4.

## Experiment naming and checkpoint immutability (rule from 2026-09-26 23:30)

Every experiment has its own symbol and directory: `e0..e8` (frame-split series), `final`, `r1`, `r2_long`,
`r2_hist3`, `r3_final`, `r3_fine` under `outputs/distill/exp/<name>/`. Snapshots (`snap_<step>.pt`,
`student_<step>.pt`) are written once and never replaced; `student.pt` is only that run's rolling resume
point. A new idea always gets a new name, never a re-run into an existing directory. Deliverable ONNX copies
are versioned (`outputs/onnx/deliverable/student_<lane>_<step>.onnx`, history in `deliverable_history.jsonl`),
release folders refuse to overwrite. `local/distill/EXPERIMENTS.md` (regenerated by
`experiments_ledger.py` after every report) lists, per experiment, what changed, where its checkpoints are,
and every evaluation record.

## Speed: the scatter hot-spot (2026-09-27 02:00)

The exporter timed the 1152x640 graph at 60 ms with dummy inputs, but the deployment loop on real frames
(idle H200) took 90-93 ms. Phase timers put all of it in the graph run, and the cause was in
`LiftSplat.forward`: every ray outside the BEV grid (13-45% of the 1.1M rays per frame) was scattered into
ONE scratch row, so ScatterElements(add) serialised hundreds of thousands of atomics on 384 addresses. The
all-valid dummy never exercised that path. Fix: spread invalid rays over `cells` scratch rows with a static
arange (a constant in the graph) and drop those rows after the scatter. Same outputs (186/186 boxes within
0.24 mm on three val keyframes, occupancy/map/state within 1e-5 relative, trajectory bit-identical) and the
loop went from 93 ms (10.7 Hz) to 41 ms (24.1 Hz); graph run 90 -> 32 ms. Re-exports live next to the
originals as `outputs/onnx/<lane>_snapNNk_v2/`; the deliverable copy is `student_r2_long_80000_v2.onnx`;
`outputs/student_release_v2/` bundles the same .pt with the v2 graph. The exporter now uses a realistic
invalid fraction (45%) in its dummies, and `run_stateful.py` times a pinned-buffer loop with the GPU kept
busy, so the reported number is the per-frame device cost, not a GPU clocking down between Python frames.
Lanes launched from now on (r3_final, r3_fine) also train with the de-contended scatter.

## Full-val confirmation (2026-09-27 02:30)

Same protocol as the 250-frame gate, run on every cached keyframe of the 150 val scenes (4,452 frames, 117k GT
objects): teacher F1 61.0%, student (r2_long 80k) 59.5% -> **97.5% retained** (95.3% with 2 m NMS). With this
many objects the F1 noise is about 0.3 points, so the 95% gate is cleared with margin. Per class the student is
at or above the teacher on cars, trucks, trailers, construction vehicles, bicycles and barriers (0.60 vs 0.53);
it trails on pedestrians (0.46 vs 0.50), motorcycles (0.41 vs 0.45) and traffic cones (0.59 vs 0.68). Per range
it trails only inside 20 m (0.85 vs 0.92 at 0-10 m, 0.78 vs 0.82 at 10-20 m) and is at parity or better beyond.
Occupancy 27.7% vs 2.9% mIoU on the same frames; map agreement with the teacher 0.60.

## Per-class operating points (2026-09-27 07:40): examined, no gain

Per-class score thresholds were fitted on one half of the 150 val scenes and scored on the other half
(out-of-sample by scene), for the r2_long 80k deliverable. Fold A: 59.3% F1 at the single best threshold vs
59.4% with fold-B thresholds; fold B: 59.8% vs 60.0% with fold-A thresholds. +0.1 to +0.2 points, inside the
noise: the classes whose thresholds move (trailer, construction vehicle, motorcycle, bicycle) are too rare to
shift the aggregate. The deliverable keeps one threshold (0.3-0.4). `demo_onnx.py --calib` can still apply a
per-class file (`outputs/distill/calib/r2_long80k_fold{A,B}.json`) for deployment.

## Ranking protocol tightened (2026-09-27 08:35)

The 250-frame gate ranked r2_long 80k (61.0%) above r2_hist3 (60.6%); on all 4,452 val frames the order flips
(59.5% vs 60.3%, i.e. 97.5% vs 98.8% retained). 250 frames carry ±1.5 points of noise, 4,452 carry ±0.3, so the
deliverable is now chosen on the full val set (`diagnose.py --tokens valall`, or `valfoldA`/`valfoldB` in
parallel and merged), with eval250 kept as the fast gate inside the lanes.

## Occupancy looked bad in the demo: why, and the fix (2026-09-27 23:05)

Occ3D supervises only camera-visible voxels (12% of the 200x200x16 grid on a typical frame) and the loss ignored
the rest, so the head was free to fill the other 88% with "driveable"/"background": a blanket over the whole
+/-51 m grid in every render, while inside the visible region the prediction is close to the ground truth
(per-class IoU on scene-0276 frames: driveable 0.74-0.87, background 0.53-0.59, cars 0.38-0.56, pedestrians
0.12-0.14; val mIoU 32% vs the teacher's 3%). Two fixes: (1) `demo_onnx.py --occ-mask-dir` draws only the
camera-visible voxels when the frame's occ3d.npz exists (benchmark convention; visualisation only), and
(2) lane `r4_occfix` fine-tunes the leading checkpoint (r3_final 30k) for 5k annealing steps with
`--occ-outside-empty 0.2`, a light cross-entropy pull toward "empty" outside the mask, so the exported model
itself stops painting the invisible region. Detection is expected to stay put (low LR); it is re-scored on the
gate and the full val set before it can replace the deliverable.

## Where we are now (final, 2026-09-28 03:45)

Deliverable: `outputs/student_release_v6/` = the r4_small averaged checkpoint (r3_final 30k fine-tuned 10k
annealing steps with the heatmap loss doubled on pedestrians / motorcycles / bicycles / cones and a light pull to
"empty" for occupancy outside the camera mask, averaged over its 35k and 40k snapshots).

| metric (150 held-out val scenes) | student | teacher |
|---|---|---|
| detection F1, all 4,452 val frames (117k objects) | 61.35% | 61.04% (100.5% retained) |
| detection F1, 250-frame gate | 61.6% | 62.0% (99.6%) |
| official nuScenes mAP / NDS (6,019 keyframes) | 0.375 / 0.444 | 0.215 / 0.222 |
| trajectory ADE (5 s) | 1.51 m | 1.63 m |
| occupancy mIoU vs Occ3D (camera-visible) | 0.30 | 0.03 |
| real-frame deployment loop, idle H200, all outputs | 36.7 ms = 27.3 Hz | ~1 Hz |

Runner-ups: r3_final 30/40/50k average 100.3% (no occupancy fix), r4_occfix 99.9% (occupancy fix, NDS 0.440),
r3_final 30k 99.7%, r2_hist3-avg 99.2%, r3_fine 98.1% (best small-object recall before r4_small), r2_long 80k
97.5% (the first download). Every lane, snapshot and evaluation is in `EXPERIMENTS.md`; nothing was overwritten.
The orchestrator and all lanes are stopped; GPUs are free.

Closing note (2026-09-28 06:35): the last probes (r5, r6a, two weight soups) all landed between 99.6% and 100.3%
on the full val set, below the deliverable's 100.5%; release v6 is final. See REPORT_FINAL.md for the table.

## How to reproduce the deliverable (commands, in order)

```bash
# 0. data caches (once): frames/teacher/ego/depth/occ3d under /local/$USER/distill (see PLAN.md), scene split in data/distill/scene_split.json
# 1. base lane: three-frame history at 1152x640, 75k one-cycle, snapshots every 10k (lane.sh carries the RECIPE flags)
local/distill/scripts/lane.sh r3_final "0,1,2,3" 4 75000 "--image-size 1152 640 --history 3" outputs/logs/lane_r3_final.log
# 2. pick the peak snapshot on the gate (here 30k), seed a fine-tune lane from it
mkdir -p outputs/distill/exp/r4_small && cp outputs/distill/exp/r3_final/snap_30000.pt outputs/distill/exp/r4_small/student.pt
# 3. fine-tune 30k -> 40k (resume path of lane.sh: no gate/smoke), heatmap loss x2 on ped/moto/bicycle/cone + occupancy outside-mask term
OMP_NUM_THREADS=1 local/distill/scripts/lane.sh r4_small "0,1,2,3" 4 40000 \
  "--image-size 1152 640 --history 3 --hm-class-weight 5:2,6:2,7:2,8:2 --occ-outside-empty 0.2" outputs/logs/lane_r4_small.log
# 4. the lane post averages the last-quarter snapshots (35k, 40k) -> student_avg.pt, scores the gate, runs the official eval, exports, times the loop
# 5. full-val ranking (two scene folds per candidate, merged exactly)
local/distill/scripts/fold_score.sh outputs/distill/exp/r4_small/student_avg.pt r4_small-avg 5 6 && python local/distill/fullval_rank.py
# 6. release bundle (new folder, never overwrites) and the ONNX demo pipeline
local/distill/scripts/make_release.sh outputs/distill/exp/r4_small/student_avg.pt r4_small_avg outputs/student_release_v6 0
local/distill/scripts/onnx_pipeline.sh outputs/student_release_v6/student_r4_small_avg.pt scene-0276 0 outputs/student_release_v6/run
```

## Round 7 (24 h extension granted 2026-09-28 10:30): two structural fine-tunes from the deliverable

Reflection. Six fine-tunes and soups from the r3_final peak all landed between 99.6% and 100.5% of the teacher's
F1 on the full val set: the recipe is saturated for count-weighted F1. What is left is structural: (1) the
0.512 m BEV cell caps localisation and separation of pedestrians / cones inside 20 m (the only remaining deficit
vs the teacher); (2) the occupancy head is a single 1x1 conv (0.30 mIoU inside the Occ3D mask; camera-only
methods with a real 3D decoder reach ~0.40). The 27 Hz margin can pay for both.

Two lanes, both warm-started from the deliverable so nothing trained is thrown away:
- `r7_head2`: the trained 1x head re-expressed EXACTLY as a 2x-grid head (transposed conv = nearest upsample of
  the 1x1 entry conv; dilation-2 3x3 blocks; heatmap difference 0.00 at the transplant), then 10k annealing steps
  with the r4_small recipe on 0.256 m targets. Unlike r3_fine (2x head from scratch), the large-object behaviour
  is inherited, and the fine-tune only has to learn sub-cell structure.
- `r7_occ3d`: a new 3D occupancy head (1x1 lift to 16 channels x 16 pillars, two 3x3x3 conv blocks, 1x1x1
  classifier, 0.15 M params) trained alone with the backbone and every other head frozen (`--train-only occ_head`,
  LR 1.8e-4 -> 0 over 10k steps): detection, map and trajectory stay bit-identical, only occupancy can change.
If both help, the 3D head is retrained on top of the r7_head2 weights (another head-only run). Speed is
re-measured in the pinned loop; the 10 Hz gate has ~2.7x margin.
