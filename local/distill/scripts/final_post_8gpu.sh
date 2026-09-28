# Post-training steps for the resumed final run on the 8-GPU machine. Blocks on the training
# launcher's PID (no polling in the foreground), then evaluates, exports and times.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
TRAIN_PID=${1:?torchrun pid}
echo "[$(date +%H:%M:%S)] waiting for training launcher pid $TRAIN_PID to exit"
tail --pid=$TRAIN_PID -f /dev/null
STEP=$(.venv/bin/python -c "import torch;print(torch.load('outputs/distill/exp/final/student.pt',map_location='cpu').get('step'))" 2>/dev/null)
echo "[$(date +%H:%M:%S)] training launcher exited; checkpoint step $STEP"
if [ "$STEP" != "80000" ]; then echo "TRAINING_ENDED_EARLY at step $STEP -- rerun final_resume_8gpu.sh"; exit 2; fi
source export_onnx/env_gpu.sh
NOTE="FINAL: from scratch on the 700 train scenes, 10 classes + velocity, E7 recipe + temporal, 80k steps (36k on 4xA100, rest on 4xH200; EMA weights); scored on the 150 held-out val scenes"
for x in "final::" "final-nms2::--nms 2.0"; do t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/final/student.pt \
    --tag "$t" --note "$NOTE" --tokens scene --limit 250 --frames ${DISTILL_CACHE:-/local/$USER/distill}/frames --teacher ${DISTILL_CACHE:-/local/$USER/distill}/teacher $e; done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/final/student.pt \
    --tag "final-trainscenes" --note "$NOTE -- eval on frame-split val frames, i.e. held-out frames of TRAIN scenes (measures the scene leak)" --limit 250 --frames ${DISTILL_CACHE:-/local/$USER/distill}/frames --teacher ${DISTILL_CACHE:-/local/$USER/distill}/teacher
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_official.py --ckpt outputs/distill/exp/final/student.pt --tag final-official --note "$NOTE" --frames ${DISTILL_CACHE:-/local/$USER/distill}/frames --teacher ${DISTILL_CACHE:-/local/$USER/distill}/teacher
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_official.py --ckpt outputs/distill/exp/final/student.pt --subset cached --tag final-official-cached --note "$NOTE -- cached val subset, like-for-like with teacher-official" --frames ${DISTILL_CACHE:-/local/$USER/distill}/frames --teacher ${DISTILL_CACHE:-/local/$USER/distill}/teacher
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/export_student.py --ckpt outputs/distill/exp/final/student.pt --out outputs/onnx/student_final/student.onnx
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/run_stateful.py --ckpt outputs/distill/exp/final/student.pt --check --frames 20
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/run_stateful.py --ckpt outputs/distill/exp/final/student.pt --onnx outputs/onnx/student_final/student.onnx --gpu 0 --frames 40
echo "FINAL_DONE"
