# R1: the final recipe at 1152x640 input (72x40 feature grid), from scratch on the 700 train
# scenes, on GPUs 4-7 of the 8-GPU machine. Self-contained: waits for its prerequisites, smoke-tests,
# launches with a 600 s check, then evaluates, exports and times. Log: outputs/logs/r1.log
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
L=${DISTILL_CACHE:-/local/$USER/distill}
RECIPE="--det-objective center --det-head center --ref-points off --center-hidden 128 --center-blocks 3 --center-min-radius 2 --classes nuscenes10 --split scene --velocity --cbgs --ema 0.999 --cam-flip --temporal --image-size 1152 640"
echo "[$(date +%H:%M:%S)] R1 queued: waiting for depth_1152x640.npz on all 25599 cached frames and the local image copy"
until [ "$(find data/distill/frames -maxdepth 2 -name depth_1152x640.npz | wc -l)" -ge 25599 ]; do sleep 30; done
until ! pgrep -f "rsync -aL" >/dev/null; do sleep 30; done
rsync -a --include='*/' --include='depth_1152x640.npz' --exclude='*' data/distill/frames/ $L/frames_real/
echo "[$(date +%H:%M:%S)] prerequisites done: $(find $L/frames_real -maxdepth 2 -name depth_1152x640.npz | wc -l) local depth files, images real: $(test -f $(readlink -f $L/frames_real/$(ls $L/frames_real | head -1)/images/CAM_FRONT.jpg) && echo yes)"
rm -rf outputs/distill/exp/r1_smoke && mkdir -p outputs/distill/exp/r1_smoke
CUDA_VISIBLE_DEVICES=4,5 .venv/bin/torchrun --nproc_per_node=2 --master_port=29643 local/distill/train_student.py $RECIPE \
  --steps 30 --batch 2 --workers 4 --log-every 10 --frames $L/frames_real --teacher $L/teacher --out outputs/distill/exp/r1_smoke > outputs/logs/r1_smoke.log 2>&1
if ! grep -q "trained 30 steps" outputs/logs/r1_smoke.log || grep -qE "did not receive grad|Traceback" outputs/logs/r1_smoke.log \
   || grep -oE "depth [0-9.]+" outputs/logs/r1_smoke.log | grep -q "depth 0.0000"; then
  echo "R1_SMOKE_FAILED"; grep -E "Traceback|Error|depth" outputs/logs/r1_smoke.log | head -5; exit 1; fi
echo "[$(date +%H:%M:%S)] R1 smoke passed: $(grep -oE 'depth [0-9.]+' outputs/logs/r1_smoke.log | head -3 | tr '\n' ' ')"; rm -rf outputs/distill/exp/r1_smoke
rm -rf outputs/distill/exp/r1_hires && mkdir -p outputs/distill/exp/r1_hires
echo "=== R1 ($(date +%H:%M)): final recipe at 1152x640, from scratch, 700 train scenes, 4 ranks x batch 4 on GPUs 4-7, 80k steps ==="
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/torchrun --nproc_per_node=4 --master_port=29642 local/distill/train_student.py $RECIPE \
  --steps 80000 --workers 12 --frames $L/frames_real --teacher $L/teacher --out outputs/distill/exp/r1_hires &
TPID=$!
sleep 600
if grep -qE "Traceback|Error" outputs/logs/r1.log || ! grep -qE "^  step " outputs/logs/r1.log || ! grep -q "scene split:" outputs/logs/r1.log; then
  echo "R1_LAUNCH_FAILED"; kill -9 $TPID 2>/dev/null; exit 1; fi
echo "[$(date +%H:%M:%S)] R1 10-minute check passed: $(grep -oE 'cls [0-9.]+' outputs/logs/r1.log | head -4 | tr '\n' ' ')"
wait $TPID; echo "TRAIN_RC=$?"
source export_onnx/env_gpu.sh
NOTE="R1: final recipe at 1152x640 input, from scratch on the 700 train scenes, 80k steps on 4xH200 (EMA weights); scored on the 150 held-out val scenes"
for x in "r1::" "r1-nms2::--nms 2.0"; do t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/r1_hires/student.pt --tag "$t" --note "$NOTE" --tokens scene --limit 250 --frames $L/frames_real --teacher $L/teacher $e; done
CUDA_VISIBLE_DEVICES=4 .venv/bin/python local/distill/eval_official.py --ckpt outputs/distill/exp/r1_hires/student.pt --tag r1-official --note "$NOTE" --frames $L/frames_real --teacher $L/teacher
CUDA_VISIBLE_DEVICES=4 .venv/bin/python local/distill/export_student.py --ckpt outputs/distill/exp/r1_hires/student.pt --out outputs/onnx/student_r1/student.onnx
CUDA_VISIBLE_DEVICES=4 .venv/bin/python local/distill/run_stateful.py --ckpt outputs/distill/exp/r1_hires/student.pt --check --frames 20
CUDA_VISIBLE_DEVICES=4 .venv/bin/python local/distill/run_stateful.py --ckpt outputs/distill/exp/r1_hires/student.pt --onnx outputs/onnx/student_r1/student.onnx --gpu 4 --frames 40
echo "R1_DONE"
