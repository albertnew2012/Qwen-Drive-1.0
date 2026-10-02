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

### 2.1 The teacher

Qwen-Drive-1.0 is about 5.7 B parameters in three parts, counted from its weight files:

| part | parameters | what it does |
|---|---|---|
| VLM | 4.54 B | vision encoder + Qwen3.5 text stack (24 Gated-DeltaNet + 8 full-attention layers); reads the six cameras and the prompt, optionally writes reasoning text |
| perception head | 0.13 B | 900-query DETR-style detector on a 200×200 BEV, occupancy 200×200×16×10, map 6×200×400 |
| planner (`planner-sft`; `planner-rl` has the same shape) | 1.04 B | 32-layer flow-matching transformer: 50 waypoint tokens attend to the VLM's key/value cache and are denoised from Gaussian noise in 10 sequential steps |

It runs at about 1 Hz on one GPU, dominated by the VLM prefill; in reasoning mode, token-by-token text
generation adds more. It is not an oracle on nuScenes: against ground truth it scores detection F1 61.0%
on all 4,452 cached val-scene frames, official mAP 0.215 / NDS 0.222 (its 7-class taxonomy cannot score
truck, bus, trailer, construction vehicle or motorcycle), occupancy 3.2% mIoU against Occ3D, trajectory
ADE 1.63 m.

### 2.2 What the student is

A single 54.4 M-parameter convolutional network (`distill/student.py`) that produces everything the
teacher's perception and planning produce, in one forward pass per frame, at 27.3 Hz on one H200. It has
no language model, no tokens and no denoising loop.

**Data flow on one page.** Measured from the release v6 checkpoint: parameter counts from its weights,
tensor shapes traced through one forward pass at batch 1.

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                THE STUDENT — one convolutional graph, no language model               ║
║         54.408 M params · fp32 · one 620-node ONNX graph · 27.3 Hz on one H200        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝

  GRAPH INPUT, every frame              │  BUILT ON THE HOST, from calibration and ego poses
  image  [1, 6, 3, 640, 1152]           │  bev_index  [1 105 920] int64  ray → BEV cell
  6 cameras, FRONT first, clockwise     │  valid      [1 105 920] bool   ray inside the grid
  RGB, (x/255 − 0.5) / 0.5              │      = 6 cams × 64 depth bins × 40×72 pixels
                                        │  ego        [1, 116]  16×3 history · 16×2 velocity
                                        │                       · 16×2 accel · 3 nav · speed
                                        │  warp_grid  [1, 3, 200, 200, 2]   ego motion
                                        │  prev_bev   [1, 3, 384, 200, 200] last 3 states
                                        ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │  BACKBONE  ResNet-50, ImageNet init         24.885 M  (45.74 %)     │
     │  the 6 cameras run as one batch of 6, shared weights                │
     │  stem + layer1     256 ch  160×288   stride 4                       │
     │  layer2            512 ch   80×144   stride 8                       │
     │  layer3           1024 ch   40×72    stride 16     7.098 M          │
     │  layer4           2048 ch   20×36    stride 32    14.965 M          │
     │  lat3 + upsampled lat4 → 3×3 fuse   → [6, 256, 40, 72]              │
     └──────────────────────────────────┬──────────────────────────────────┘
                                        ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │  LIFT-SPLAT  0.115 M  (0.21 %)                                      │
     │  depth  1×1 → 64 bins, 1–60 m, softmax       [6, 64, 40, 72]        │
     │  feat   1×1 → 384 ch                         [6, 384, 40, 72]       │
     │  outer product → 1 105 920 rays × 384 ch                            │
     │  ONE scatter-add by bev_index → BEV [1, 384, 200, 200]              │
     │  200×200 cells of 0.512 m = ±51.2 m, all 6 cameras in one grid      │
     │  invalid rays land on spread-out scratch rows, dropped afterwards   │
     └──────────────────────────────────┬──────────────────────────────────┘
                                        │ ── bev_state: this un-fused BEV is also an OUTPUT,
                                        │    fed back next frame as part of prev_bev
                                        ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │  TEMPORAL FUSE  5.309 M  (9.76 %)                                   │
     │  3 previous states, 0.5 / 1.0 / 1.5 s back, each grid_sample'd      │
     │  into the current ego frame by its own warp_grid                    │
     │  concat 4 × 384 = 1536 ch → 3×3 conv + BN + ReLU → 384 ch           │
     └──────────────────────────────────┬──────────────────────────────────┘
                                        ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │  BEV ENCODER  8.410 M  (15.46 %)                                    │
     │  5 × (3×3 conv 384→384 + BN + ReLU)     1.328 M each                │
     │  1 × (3×3 conv 384→512 + BN + ReLU)     1.770 M                     │
     │  ► ONE shared scene map  [1, 512, 200, 200], read by every head     │
     └─┬────────────────┬────────────────┬────────────────────┬────────────┘
       ▼                ▼                ▼                    ▼
   DETECTION        OCCUPANCY        MAP                  TRAJECTORY  14.062 M  (25.85 %)
   CenterPoint      0.082 M          1.182 M              avg-pool 5×5 → 40×40 = 1600 tokens
   0.364 M          1×1 conv         crop 60 m × 30 m     + position [1600, 512]      0.819 M
   1×1 512→128      512 → 16 × 10      = 118×58 cells     ego MLP 116→512→512         0.323 M
   2 × 3×3 128      per BEV cell:    bilinear → 200×400   50 waypoint queries [50, 512]
   heatmap 10 cls   16 height slices   cells of 0.15 m    3 × decoder layer      4.204 M each
   regression 10:   × 10 classes     3×3 512→256            self-attn, cross-attn to the
   dx dy z,                          1×1 256→6 classes      1600 tokens, FFN 2048, 8 heads
   log w l h,                                             head 512→512→3              0.264 M
   sin cos, vx vy                                         + ego_direct 116→150        0.018 M
   3×3 max-pool,                                          + constant-velocity anchor
   top-900 decode
   inside the graph
       │                │                │                    │
       ▼                ▼                ▼                    ▼
   cls [1, 900, 10] occ [1, 200,     seg [1, 6,           trajectory [1, 50, 3]
   box [1, 900, 10]   200, 16, 10]     200, 400]          x, y, heading · 5 s @ 10 Hz,
                                                          ego frame

   + bev_state [1, 384, 200, 200]: this frame's memory, fed back as prev_bev next frame
```

Two choices make it fast. Geometry leaves the graph: the host turns calibration into 1.1 million ray-to-cell
indices, so the whole image-to-BEV lift costs two 1×1 convolutions (0.115 M parameters, 0.2 %) and one
scatter-add, with no camera math inside the network. And every output reads the same 512-channel BEV map:
occupancy and map add 1.26 M parameters between them, and the trajectory decoder attends to that map pooled
to 1,600 tokens. The parameters sit where the work is: 46 % in the image backbone, 26 % in the trajectory
decoder, 15 % in the BEV encoder and 10 % in the temporal fusion.

**One frame, step by step**

1. **Images to features.** The six camera images, resized to 1152×640, go through a ResNet-50 backbone
   (ImageNet initialisation).
2. **Features to bird's-eye view.** For every image feature pixel the network predicts a distribution over
   64 depth bins from 1 to 60 m. Each feature is spread along its camera ray by that distribution and
   scattered into a 200×200 grid around the car, 0.512 m per cell, covering ±51.2 m. All six cameras land
   in the same grid, so it sees 360°. The ray-to-cell indices are computed on the host from calibration and
   passed in, so the graph is one gather and one scatter-add.
3. **Memory of the past.** The BEV grids of the previous three keyframes stay on the GPU, are warped into
   the current ego frame by `grid_sample` using the ego motion, and are fused with the current grid. This
   is what lets it keep briefly hidden objects and estimate velocity.
4. **BEV encoder.** Six convolutional blocks refine the fused grid into one shared 512-channel scene map.
5. **Five heads read that shared map:**

| head | how it works | output |
|---|---|---|
| 3-D detection | CenterPoint style: a heatmap per class on the 200×200 grid, plus sub-cell offset, height, size, heading and velocity at each cell; a 3×3 max-pool keeps local peaks and the top 900 are decoded inside the graph | up to 900 boxes, 10 nuScenes classes |
| occupancy | a 1×1 conv turns each BEV cell into 16 height slices × 10 classes | 200×200×16 voxel grid |
| map | crops a 60 m × 30 m window centred on the car, resizes it to 200×400 cells of 0.15 m and classifies them | 6 classes: driveable, lane line, road edge, crosswalk, walkway, background |
| trajectory | starts from a constant-velocity extrapolation, adds a linear correction from the 116-number ego state (16-point history, velocities, accelerations, nav command, speed), then 50 waypoint queries attend to a 40×40 pooled version of the scene map through 3 layers and add a learned residual | 50 points, 5 s at 10 Hz: x, y, heading |
| state | this frame's un-fused BEV grid | fed back as next frame's memory |

**What it learns from.** Despite the word distillation, most heads learn from ground truth, not from the
teacher:

| head | target |
|---|---|
| detection | nuScenes annotated boxes (10 classes, ego-frame yaw, velocity) with Gaussian-focal heatmap + L1 box loss; the teacher-box term exists in the code but is unused under the CenterPoint objective |
| depth (inside the lift-splat) | lidar points projected into each camera |
| occupancy | Occ3D labels, camera-visible voxels only, plus a light pull toward "empty" outside them |
| map | the teacher's map predictions: the only head distilled from the teacher |
| trajectory | the path the car actually drove over the next 5 s |

The teacher's main role was the bar to beat: the student is scored against ground truth next to the
teacher on the same 150 held-out val scenes, which it never trained on.

**Teacher and student side by side**

| | teacher | student |
|---|---|---|
| size | about 5.7 B | 54.4 M |
| scene representation | image tokens inside a language model | a metric 200×200 BEV grid |
| trajectory | 10 denoising steps of a 32-layer transformer reading the VLM cache | one pass: anchor + linear term + 3 attention layers |
| reasoning text | yes, optional | none |
| detection F1, all 4,452 val frames | 61.04% | 61.35% (100.5%) |
| official mAP / NDS | 0.215 / 0.222 | 0.375 / 0.444 |
| trajectory ADE, 5 s | 1.63 m | 1.51 m |
| occupancy mIoU vs Occ3D | 0.03 | 0.30 |
| speed, one H200, real-frame loop | about 1 Hz | 27.3 Hz |

**Where it still trails.** Pedestrians (45.6 vs 50.1% recall), traffic cones (66.6 vs 68.5%) and buses
(42.8 vs 46.7%), all inside 20 m, where the 0.5 m cell is coarse for small objects; and occupancy detail,
where the single 1×1 head has no vertical context. Round 7 (end of this file) targets both.

### 2.3 The deliverable (release v6)

`outputs/student_release_v6/`: the r4_small averaged checkpoint (fp32) and its ONNX graph, 620 nodes,
opset 20. Inputs: `image (1,6,3,640,1152)`, `bev_index`, `valid`, `ego (1,116)`,
`prev_bev (1,3,384,200,200)`, `warp_grid (1,3,200,200,2)`. Outputs: `cls (1,900,10)`, `box (1,900,10)`,
`occ (1,200,200,16,10)`, `seg (1,6,200,400)`, `trajectory (1,50,3)`, `bev_state (1,384,200,200)`; the
state is fed back next frame and the host builds the warp grids from ego poses. The recurrent ONNX loop
matches the training-time path to a heatmap difference of 3e-3. How it was reached is in sections 5 to 7
and the dated sections below; every run is in `EXPERIMENTS.md`.

### 2.4 How the student is trained: data, targets and process

**The gist.** The student is trained like an ordinary supervised perception-and-planning network.
Each training step shows it one nuScenes keyframe from the 700 training scenes, it predicts all five
outputs, each output is compared with a target for that keyframe, and the five errors are added into one
loss that updates every weight. Four of the five targets are nuScenes ground truth. The teacher supplies
the fifth, the map: the nuScenes map expansion that could have provided ground truth was not available here.

```
  ONE TRAINING STEP

  a keyframe from the 700 training scenes                 targets for the same keyframe,
  ┌──────────────────────────────────────┐                built once, offline
  │ 6 camera images                      │
  │ the 3 previous keyframes' BEV maps   │
  │ ego history: 16 points over 1.5 s    │
  └──────────────────┬───────────────────┘
                     ▼
              ┌─────────────┐
              │   student   │── boxes ────── vs ── nuScenes annotated boxes ........ ground truth
              │  54 M, CNN  │── depth ────── vs ── lidar points seen by each camera  ground truth
              │             │── occupancy ── vs ── Occ3D voxel labels .............. ground truth
              │             │── map ──────── vs ── the TEACHER's map prediction .... teacher
              └─────────────┘── trajectory ─ vs ── the path actually driven, 5 s ... ground truth
                                     │
                                     ▼
    5 losses, weighted and summed → one gradient step on all 54 M weights → next batch of 16 keyframes
```

**The run, in four phases** (the recipe that produced release v6):

1. **Base training, 75,000 steps, about 21 h on 4 H200s.** Starts from a backbone pretrained on ImageNet;
   everything else starts from scratch. A snapshot is kept every 10,000 steps.
2. **Pick the best snapshot.** Each snapshot is scored on 250 frames from the 150 held-out scenes. The best
   came at 30,000 steps; later ones plateaued or slipped slightly.
3. **Fine-tune that snapshot, 10,000 steps, about 3 h,** at a low and decaying learning rate, with two loss
   changes: double weight on pedestrians, motorcycles, bicycles and traffic cones, the only classes where
   the student trailed the teacher, and a pull toward "empty" for occupancy the cameras cannot see.
4. **Average** the fine-tune's last two checkpoints, weight by weight. That average is release v6.

**Why "distillation" when most targets are ground truth.** The project started as pure copying: the
student learned to reproduce the teacher's 900 detection outputs, with no ground truth at all
(experiments e0 to e4). That capped it at the teacher's own mistakes, and the teacher is not an oracle: it
scores 61% F1 against nuScenes ground truth, and its 7-class taxonomy cannot express trucks, buses or
motorcycles. Switching to ground truth let the student match and then slightly pass the teacher. The
teacher's role became three things: the source of the map targets, the output format the student keeps
(the teacher's 900-box list, occupancy grid and map grid, so the same decoding and rendering code serves
both, with a 10-to-7 class mapping for detections), and the bar it is scored against, on the same held-out
scenes. So "distillation" here means replacing a
5.7 B model with a 54 M one that meets its bar; only the map is distilled in the classic sense.

**The three training tricks that mattered most:** class-balanced sampling, so rare classes are seen as
often as cars; mirroring every camera image left-right at random, with its projection adjusted so the
targets stay valid; and keeping a running average of the weights (decay 0.999), which is what is evaluated
and shipped.

The details follow: the data, the cached files, each loss, the exact settings, and the commands.

**Training data.**

| source | what is used | size |
|---|---|---|
| nuScenes v1.0-trainval, official scene split | keyframes at 2 Hz, six cameras each, lidar, annotations, ego poses | 700 train scenes = 28,130 keyframes; 150 val scenes = 6,019 keyframes, never trained on |
| Occ3D-nuScenes, via UniOcc on Hugging Face (`tasl-lab/uniocc`, in `data/occ3d/uniocc`) | 3-D occupancy labels with a camera-visibility mask | every keyframe |
| the teacher, Qwen-Drive-1.0 perception head, run once offline at 896×512 | map predictions (and boxes, occupancy, logits, used only by early experiments) | 25,599 keyframes: 21,147 train + 4,452 val |
| nuScenes ego-pose table, about 50 Hz | past and future ego path | every keyframe |

6,983 train keyframes have no teacher record. They still train every head except the map, using calibration
from `calib.npz`. Evaluation uses the 4,452 val keyframes that have a teacher record, so student and teacher
are scored on identical frames.

**What is cached per keyframe** (built once, then read by every run):

| file | contents | used for |
|---|---|---|
| `data/distill/frames/<token>/images/` | the six camera JPEGs, symlinked from nuScenes | input |
| `.../frame.json`, `calib.npz` | camera order; intrinsics, camera-to-lidar and lidar-to-ego transforms | input geometry |
| `.../gt_boxes10.npz` | ground-truth boxes in the ego frame, 10 classes, lidar point count, visibility, velocity | detection target |
| `.../depth_1152x640.npz` | lidar points projected into each camera, as a 64-bin depth per 40×72 feature cell (−1 = no return) | depth target |
| `.../occ3d.npz` | 200×200×16 occupancy classes on the student's grid, plus the camera-visible mask | occupancy target |
| `data/distill/teacher/<token>.npz` | teacher 900×7 logits and boxes, 200×200×16 occupancy, 200×400 map, its projection matrices | map target |
| `data/distill/ego/<token>.npz` | 16×3 history, velocities, accelerations, nav command, speed, 50×3 future, `has_future` | ego input, trajectory target |
| `data/distill/temporal_index.json` | previous keyframe and ego pose of every keyframe | history frames and their warps |
| `data/distill/scene_split.json` | token lists: train and val keyframes with and without a teacher record, and `eval250`, the fixed 250-frame check | which frames train, which evaluate |
| `data/distill/occ3d_freq.npy` | occupancy class counts over the camera-visible voxels | occupancy loss weights |

The frame cache is 8.0 GB, because the images are symlinks; the teacher cache is 1.2 GB.

**Targets and losses, per head.**

| head | target | loss |
|---|---|---|
| detection | ground-truth boxes with at least one lidar point, drawn as a Gaussian heatmap per class (minimum radius 2 cells) plus a 10-number regression at each centre cell | Gaussian focal loss + L1. The final fine-tune doubles the heatmap loss for pedestrians, motorcycles, bicycles and cones. |
| depth, inside the lift | the lidar depth bin of each camera feature cell | cross-entropy; cells without a lidar return are ignored |
| occupancy | Occ3D labels, camera-visible voxels only | cross-entropy weighted by class frequency. The final fine-tune adds 0.2 × a pull toward "empty" outside the visible region. |
| map | the teacher's map, 6 classes on 200×400 cells | cross-entropy with gently boosted rare classes (frequency exponent 0.3); skipped on frames without a teacher record |
| trajectory | the path actually driven over the next 5 s, 50 points at 10 Hz | L1, each dimension scaled by its spread; masked where the scene ends before 5 s |

Total loss = detection + 5 × trajectory + occupancy + map + depth.

**The training process.**

- **Class-balanced sampling (CBGS).** Frames are re-drawn so every class gets about an equal share: one epoch
  becomes 128,110 samples instead of 28,130. Without it the model sees a barrier one frame in three.
- **Augmentation.** With probability 0.5, every camera image is mirrored left-right and its projection matrix
  adjusted to match, so the bird's-eye-view targets stay valid; the depth target is mirrored with it.
- **History.** For each sample, the three previous keyframes (0.5, 1.0 and 1.5 s back) go through the backbone
  and lift without gradients, in their own mixed-precision context, and are warped into the current ego frame
  before fusion. The first frames of a scene repeat the oldest available frame.
- **Optimisation.** AdamW, learning rate 3e-4 with a one-cycle schedule (10% warm-up), weight decay 0.01,
  gradient clipping at 1.0, bfloat16 autocast, batch 4 per GPU × 4 GPUs = 16. An exponential moving average of
  the weights (decay 0.999) is kept, and that average is what is evaluated and shipped.
- **Checkpoints.** A resumable `student.pt` every 500 steps, a permanent `snap_<step>.pt` every 10k steps,
  and `student_<step>.pt` every 5k steps over the last quarter, for averaging.
- **Selection.** Every snapshot is scored on the 250-frame check; the leaders are scored on all 4,452 val
  frames (two scene folds in parallel) and with the official nuScenes metric. Nothing replaces the
  deliverable without winning on the full set.

**How v6 was produced**, on 4 H200 GPUs: `r3_final` trained 75k steps from an ImageNet backbone, about 21 h,
with its best snapshot at 30k. From that snapshot, `r4_small` ran the 10k-step fine-tune with the class-weighted
heatmap and the occupancy term, about 3 h. Its 35k and 40k checkpoints, averaged, are release v6. The commands
are under "How to reproduce the deliverable" below.

**One-time data preparation**, in order (`--limit 0` = all frames; the teacher step needs a GPU and the 4B model):

```bash
export PYTHONPATH=src:.
python distill/nusc_frames.py --root data/nuscenes --version v1.0-trainval --limit 0     # frames/<token>/
python distill/cache_teacher.py --frames data/distill/frames --limit 0                  # teacher/<token>.npz
python distill/nusc_gt_boxes.py --root data/nuscenes --version v1.0-trainval            # gt_boxes10.npz
python distill/nusc_depth.py --root data/nuscenes --version v1.0-trainval \
    --image-size 1152 640 --out-name depth_1152x640.npz                                 # depth targets
python distill/nusc_ego.py --root data/nuscenes --version v1.0-trainval --include-short-future   # ego/<token>.npz
python distill/occ3d_labels.py --split train && python distill/occ3d_labels.py --split val       # occ3d.npz
python distill/build_temporal_index.py --root data/nuscenes --version v1.0-trainval     # temporal_index.json
```

Three small files complete the set:

- `occ3d_freq.npy` is written by `distill/scripts/occ3d_train_labels.sh` after the train labels. It counts classes over
  train and val frames together, so the loss weights saw validation label statistics, though no validation samples.
- `scene_split.json` was written once by a snippet that is not in the repository: the official nuScenes split
  (`nuscenes.utils.splits.create_splits_scenes()`) restricted to the cached frames, plus the `eval250` list. Keep
  the file: `eval250` is the fixed check every 250-frame number in this document uses.
- The trainer builds `traj_scale.npy` on first use. `gt_labels_index.json` is an optional cache of each frame's
  classes for the balanced sampler; without it the trainer reads the label files directly.

`box_stats.npz` (`distill/box_stats.py`) only mattered for the early query-head experiments.

**The whole process as one script: `distill/scripts/run_distillation.sh`.** It runs every stage above in
order and is resumable: a stage whose outputs already exist is skipped, an interrupted training run resumes
from its last checkpoint, and earlier experiments are never overwritten.

| stage | what it does |
|---|---|
| `prep` | builds any missing cache: frames, teacher outputs, ground-truth boxes, lidar depth, ego tracks, Occ3D labels, temporal index, scene split, occupancy class frequencies |
| `base` | trains the base model from an ImageNet backbone (the `r3_final` recipe) |
| `select` | scores every base snapshot on the 250-frame check and keeps the best |
| `finetune` | the small-object and occupancy fine-tune from that snapshot (the `r4_small` recipe) |
| `average` | averages the fine-tune's last-quarter checkpoints |
| `evaluate` | 250-frame check and all 4,452 val frames for the final and averaged weights, picks the better, official nuScenes mAP/NDS |
| `release` | checkpoint, ONNX and download chunks via `make_release.sh`, then the real-frame loop timing |
| `demo` | the `nuscenes_session.gif` render with the released ONNX |

```bash
distill/scripts/run_distillation.sh                    # everything, resuming wherever the last run stopped
distill/scripts/run_distillation.sh --from select      # start at a stage
distill/scripts/run_distillation.sh --only evaluate    # one stage
distill/scripts/run_distillation.sh --dry-run          # print the commands only
NAME=v8 GPUS=0,1,2,3 distill/scripts/run_distillation.sh   # new experiment name and GPUs
```

Settings are environment variables: `NAME` (default `v7`, giving `outputs/distill/exp/v7_base`, `v7_ft` and
`outputs/student_release_v7`), `GPUS`, `BATCH` (auto: 4 per GPU with at least 100 GB of memory, else 2, which
is untested at full length), `BASE_STEPS=75000`, `FT_STEPS=10000`, `FRAMES`, `TEACHER`, `NUSC_ROOT`, and
`OFFICIAL=0` or `DEMO=0` to skip those stages. If `scene_split.json` is missing the script writes a new one,
but its 250-frame list will differ from the original, so 250-frame numbers would not be comparable.
`--help` prints the full usage.

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
python distill/nusc_gt_boxes.py --root data/nuscenes --taxonomy nuscenes10
python distill/occ3d_labels.py --split train; python distill/occ3d_labels.py --split val
python distill/nusc_depth.py --root data/nuscenes --version v1.0-trainval --image-size 1152 640 --out-name depth_1152x640.npz
python distill/nusc_ego.py --root data/nuscenes --version v1.0-trainval --include-short-future
# train + evaluate + export + time (one lane, 4 GPUs)
bash distill/scripts/lane.sh r2_long "0,1,2,3" 4 120000 "--image-size 1152 640" outputs/logs/lane_r2_long.log
# demo video on any scene
python distill/student_video.py --scene scene-0276 --gpu 0
```

## Release bundle and ONNX pipeline (2026-09-26)

`outputs/student_release/` holds the deliverable for use outside this machine: `student_r2_long_80k.pt`
(slim checkpoint: EMA weights + `StudentConfig`, 207 MB), `student_r2_long_80k.onnx` (216 MB; a fresh
export from the slim checkpoint is weight-for-weight identical), `student_release_code.tar.gz`
(`distill/`), `README.md` (setup, ONNX contract, KPIs) and `parts/` (29 MiB splits, because the
chat upload cap is 30 MiB; MD5s in `MD5SUMS`).

`distill/scripts/onnx_pipeline.sh CKPT [SCENE] [GPU] [OUT]` runs the whole export-and-demo path:
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
release folders refuse to overwrite. `distill/EXPERIMENTS.md` (regenerated by
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
distill/scripts/lane.sh r3_final "0,1,2,3" 4 75000 "--image-size 1152 640 --history 3" outputs/logs/lane_r3_final.log
# 2. pick the peak snapshot on the gate (here 30k), seed a fine-tune lane from it
mkdir -p outputs/distill/exp/r4_small && cp outputs/distill/exp/r3_final/snap_30000.pt outputs/distill/exp/r4_small/student.pt
# 3. fine-tune 30k -> 40k (resume path of lane.sh: no gate/smoke), heatmap loss x2 on ped/moto/bicycle/cone + occupancy outside-mask term
OMP_NUM_THREADS=1 distill/scripts/lane.sh r4_small "0,1,2,3" 4 40000 \
  "--image-size 1152 640 --history 3 --hm-class-weight 5:2,6:2,7:2,8:2 --occ-outside-empty 0.2" outputs/logs/lane_r4_small.log
# 4. the lane post averages the last-quarter snapshots (35k, 40k) -> student_avg.pt, scores the gate, runs the official eval, exports, times the loop
# 5. full-val ranking (two scene folds per candidate, merged exactly)
distill/scripts/fold_score.sh outputs/distill/exp/r4_small/student_avg.pt r4_small-avg 5 6 && python distill/fullval_rank.py
# 6. release bundle (new folder, never overwrites) and the ONNX demo pipeline
distill/scripts/make_release.sh outputs/distill/exp/r4_small/student_avg.pt r4_small_avg outputs/student_release_v6 0
distill/scripts/onnx_pipeline.sh outputs/student_release_v6/student_r4_small_avg.pt scene-0276 0 outputs/student_release_v6/run
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

**Round 7 results (lanes finished 2026-09-28 14:10; never fold-scored, deliverable unchanged).**
`r7_head2` (exact 2x-head transplant + fine-tune): gate 58.6% (94.5%), official mAP 0.363 / NDS 0.439, worse
than v6 on every metric. `r7_occ3d` (3D occupancy head, everything else frozen): occupancy mIoU 0.308-0.310 on
the gate frames vs 0.30, official mAP 0.376 / NDS 0.447 vs 0.375 / 0.444, but gate F1 61.3% vs 61.6%: detection
moved although its weights were frozen, most likely because BatchNorm running statistics keep updating in train
mode (`--train-only` freezes parameters, not BN buffers). Net: no clear winner; release v6 stands.

## The teacher's session GIF, re-rendered with the student (2026-10-01)

`distill/student_session.py` reproduces `assets/nuscenes_session.gif` with the student ONNX: same scene
(nuScenes mini scene 0 = scene-0061), the same 148 instants at the ~10 Hz camera-sweep rate, the same layout and
stats rows, 0.75x speed, and the same GIF recipe (first 12 s at 4 fps: source frames 1, 3, 5, ... 90; 48 frames,
250 ms, 1238x1506). The student's three BEV history states are taken 0.5 / 1.0 / 1.5 s back so the spacing
matches training, after a 1.5 s warm-up. Outputs: `assets/nuscenes_session_student.gif`,
`outputs/student_session/student_scene-0061_0.75x.mp4`, frames in `outputs/student_session/frames/`.
Over the 148 frames: mean ADE 1.38 m (teacher 1.39 m), 81 boxes >0.3 per frame (teacher 87, GT 63).
Caveat: scene-0061 is in the official TRAIN split, so its keyframes were student training data; the
held-out mini scenes are scene-0103, 0553, 0796 and 0916 (`--scene 1, 2, 5, 6`).

Run it from the repo root, in three steps (each render takes about 6.5 min on one A100). `env_gpu.sh` puts
torch's CUDA/cuDNN libraries on the loader path; without it onnxruntime silently falls back to the CPU.

```bash
source export_onnx/env_gpu.sh
export PYTHONPATH=src:.:tools

# 1. the PyTorch checkpoint: render the session, then export the same checkpoint to ONNX
python distill/student_session.py --ckpt outputs/student_release_v6/student_r4_small_avg.pt --scene 0 --gpu 0 \
  --out outputs/student_session_pytorch \
  --gif outputs/student_session_pytorch/nuscenes_session_student_pytorch.gif \
  --export-onnx outputs/student_export/student_r4_small_avg.onnx

# 2. the exported ONNX (onnxruntime only): same inputs, same render, compared frame by frame with step 1
python distill/student_session.py --onnx outputs/student_export/student_r4_small_avg.onnx --scene 0 --gpu 0 \
  --out outputs/student_session --gif assets/nuscenes_session_student.gif \
  --compare outputs/student_session_pytorch

# 3. rebuild only a GIF from frames already rendered
python distill/student_session.py --gif-only --out outputs/student_session --gif assets/nuscenes_session_student.gif
```

The three steps are VS Code launch configurations `9a`, `9b` and `9c` (group `9-student` in
`.vscode/launch.json`; their `env` block reproduces `env_gpu.sh`). Run 9a before 9b: 9b reads the ONNX file 9a
exports and compares against 9a's render. Verified 2026-10-01 on scene-0061: over the 148 frames the ONNX
trajectories match the PyTorch ones to 2.7 mm at worst (median 0), mean ADE 1.38 m in both; the number of boxes
above 0.3 differs on 32 frames by at most 2, from scores sitting at the threshold, as expected from float-level
differences between PyTorch and onnxruntime kernels. The re-exported graph has the same 620 nodes and 236 weight
tensors as the released `student_release_v6/student_r4_small_avg.onnx`, equal up to float rounding (max
difference 2.4e-7); the files differ byte-wise only because constant folding rounds slightly differently per run.
