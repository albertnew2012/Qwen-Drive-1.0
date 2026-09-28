# Handoff: continuing on the 8-GPU machine (8x H200) -- written 2026-09-25 17:30 on the 4-GPU machine

> **2026-09-25 17:50 update:** the A100 run in §1 was killed at step 36k when that session closed; it is being resumed on the 8-GPU machine from its checkpoint (`local/distill/scripts/final_resume_8gpu.sh`, log `outputs/logs/final_8gpu.log`). §1 no longer applies; PLAN.md §28 has the details.

The new machine sees the repo, `.venv`, caches, weights, logs, this note and the Claude memory
directory as they are.
Then the repo, `.venv`, caches, weights, logs and the Claude memory directory
(`~/.claude/projects/<project>/memory/`) are already
there; nothing needs copying except, for speed, the caches to local NVMe (see 3).

## 1. The deliverable run: stopped here at step 36,000, resume it on the 8-GPU machine FIRST

`final.sh` (the from-scratch scene-split run: 700 train scenes, 10 nuScenes classes,
velocity, CBGS + EMA + camera flip, temporal fusion, 80k steps) was stopped on the A100
node at the user's request on 2026-09-25 17:32, right after its step-36,000 checkpoint
(`outputs/distill/exp/final/student.pt`: model, EMA, optimiser, step; the one-cycle
schedule fast-forwards on resume). Nothing of it was lost. Resume on the 8-GPU machine:

```
cd Qwen-Drive-1.0
nohup bash local/distill/scripts/final_resume_8gpu.sh > outputs/logs/final_8gpu.log 2>&1 &
```
(4 ranks on GPUs 0-3 keep the run's effective batch; `GPUS=`, `FRAMES=`, `TEACHER=` env
vars override. 44k steps at H200 speed ~5-6 h.) The script then runs the whole evaluation:
grouped F1 vs the teacher on 250 val-scene frames (`final`, `final-nms2`), the same model
on frames of its own train scenes (`final-trainscenes`, sizes the leak), official mAP/NDS
on all 6,019 val keyframes (`final-official`) and on the cached 4,452
(`final-official-cached`), ONNX export (`outputs/onnx/student_final/student.onnx`), the
deployment-loop equivalence check and the timing (on an H200 -- say so; the A100 figure
for the E5c shape was 48.6 ms = 20.6 Hz). Records land in `outputs/distill/lab_notebook.json`.

Mid-run snapshot at step 19k on 100 unseen val-scene frames: F1 56.4% vs teacher 63.1%
(89% retained; bar is 80%). Best leaked-split model: E8, 64.2% vs 68.2%
(`outputs/distill/exp/e8/student.pt`, `outputs/onnx/student_e8/student.onnx`).
The other four H200s are free for section 4 from the start.

## 2. Read first

`local/distill/PLAN.md` sections 19-27 (the last 24 h: temporal fusion, the scene leak,
the official yardstick, every bug and what it cost) and `PLAN_V2.md`. The memory index
(`MEMORY.md` in the directory above) has the rules: split by scene not frame; gt_boxes.npz
has the global yaw (use gt_boxes10.npz); autocast + no_grad pre-pass freezes weights;
augment every image-space target; never `pkill -f` from a call whose text contains the
pattern; give launches a 600 s window and keep NFS quiet during it.

## 3. Bring-up on the 8-GPU machine (do in this order)

```
cd Qwen-Drive-1.0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv   # 8x H200; driver must support CUDA 12.8 (>= 570)
ls "$CUDA_HOME"                                                         # CUDA 12.8 toolkit; the scripts require CUDA_HOME
export PYTHONPATH=src:. PATH="$PWD/.venv/bin:$PATH"
.venv/bin/python -c "import torch, onnxruntime as ort; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0), ort.get_device())"
```
Local NVMe copy (7 GB) -- NFS latency cost two launches last night:
```
L=/local/scratch/$USER/distill      # or whatever the new machine's local disk is
mkdir -p $L && rsync -a data/distill/frames data/distill/teacher data/distill/ego $L/
```
`train_student.py`, `diagnose.py`, `eval_official.py`, `run_stateful.py` take `--frames $L/frames
--teacher $L/teacher`; the ego cache is found as `<frames>/../ego`, so keep that layout. The
small index files (`temporal_index.json`, `scene_split.json`, `gt_labels_index.json`,
`box_stats.npz`, `occ_freq.npy`, `traj_scale.npy`) are read from `data/distill/` on NFS and
are fine there.

Smoke in the launch configuration (2 GPUs, 30 steps, ~2 min), before anything else:
```
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 --master_port=29640 \
  local/distill/train_student.py --det-objective center --det-head center --ref-points off \
  --center-hidden 128 --center-blocks 3 --center-min-radius 2 --classes nuscenes10 --split scene \
  --velocity --cbgs --ema 0.999 --cam-flip --temporal --steps 30 --batch 2 --workers 4 \
  --log-every 10 --frames $L/frames --teacher $L/teacher --out outputs/distill/exp/smoke_n17
```
Expect `student 51.75 M params`, 30 step lines, no `did not receive grad`, a checkpoint.
Then `diagnose.py --ckpt outputs/distill/exp/smoke_n17/student.pt --tokens scene --limit 8`
and `export_student.py --ckpt ... --out outputs/onnx/smoke_n17/student.onnx` (times the
graph on an H200; report the GPU with any Hz figure). Delete the smoke outputs and their
`lab_notebook.json` records (tags starting `smoke`/`test`).

## 4. What to run there (after the smoke; can start before the A100 run ends)

The final run's own evaluation decides the next step, but the deficit is already known
from every run so far: pedestrians (33% vs 53% recall), small classes, and 30 m+. The
lever that has not been pulled is input resolution -- the trunk runs at 896x512 (stride
16 -> 56x32 features) on 1600x900 images.

Resolution run (H200 has the memory; 8 GPUs make it ~3x faster than here):
1. Re-extract the lidar depth targets at the new size: `local/distill/nusc_depth.py`
   (`--stride 16`, bins/min/max as now) writes `depth.npz` per frame at `image_size/16`;
   write to a NEW cache root so the running A100 job is untouched.
2. The loader scales nothing: the teacher's `lidar2img` is stored at 896x512. Apply the
   same `S = diag(w/896, h/512, 1, 1)` the calib fallback uses (`DistillSet._teacher_or_calib`)
   when `cfg.image_size != (896, 512)`; the camera-flip `M` already uses `w`.
3. Re-measure speed FIRST with an untrained export at the new size on the target GPU
   (`export_student.py` with no checkpoint exports the shape). 1152x640 is ~1.6x the trunk
   cost; 1408x768 ~2.4x. On A100 the current graph is ~19-20 Hz; keep >= 12 Hz margin.
4. Same recipe as the final run, 8 ranks: `--nproc_per_node=8`, per-GPU batch 4
   (effective 32), 40k-50k steps, LR 3e-4 (keep; compare like for like).

Second lever, cheaper: temporal history of 4 keyframes instead of 2 (SOLOFusion-style);
the stateful interface stays the same if the state is the fused BEV -- that is a
different model from the current one (see `temporal.py` docstring) and must be trained as
such. Third: Occ3D-nuScenes labels for a real occupancy target (download approved).

## 5. Output hygiene

Nothing runs on the 4-GPU machine any more; rules while the resumed run is alive:
- never write under `outputs/distill/exp/final`, `outputs/onnx/student_final`,
  `outputs/distill/official/final-*`; use `--out outputs/distill/exp/b8_*` and tags
  prefixed `b8-` (the tag is what the gate scripts key on);
- `outputs/distill/lab_notebook.json` is appended by read-modify-write with no lock; do not
  run `diagnose.py` or `eval_official.py` while the resumed run is in its evaluation phase;
- do not edit `local/distill/*.py` while the resumed run is between training and its
  evaluation: the script imports them fresh for each evaluation step;
- `/tmp` is machine-local: the chain scripts were copied into
  `local/distill/scripts/`; new scripts belong there too.

## 6. Ports and processes

Give each concurrent launch its own `--master_port`. One launch at a time, nothing
else touching the data mount during its first 10 minutes. To stop a run, kill the driver
script first (`bash .../x.sh`, ppid 1), then the torchrun launcher and rank processes by
PID from a `/proc` scan -- see the memory note `pkill-self-match`.
