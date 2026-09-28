# Resume the deliverable run on the 8-GPU machine (8x H200) from outputs/distill/exp/final/student.pt.
# The A100 launcher was killed at step ~36,050 on 2026-09-25 17:31; the checkpoint at step
# 36,000 holds model, EMA, optimiser and step, and train_student.py fast-forwards the
# one-cycle schedule. Same recipe: 4 ranks x batch 4 = effective batch 16 (BatchNorm sees
# the same per-GPU batch), same flags, same 80k total. Post-training steps as in final.sh.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
test -f outputs/distill/exp/final/student.pt || { echo "no checkpoint to resume"; exit 1; }
test -d ${DISTILL_CACHE:-/local/$USER/distill}/ego || { echo "local cache copy missing (${DISTILL_CACHE:-/local/$USER/distill})"; exit 1; }
echo "=== FINAL resume on $(hostname) ($(date +%H:%M)): from step $(.venv/bin/python -c "import torch;print(torch.load('outputs/distill/exp/final/student.pt',map_location='cpu').get('step'))" 2>/dev/null) of 80000, temporal, 8 ranks x batch 4 ==="
.venv/bin/torchrun --nproc_per_node=8 --master_port=29641 \
  local/distill/train_student.py --det-objective center --det-head center --ref-points off \
  --center-hidden 128 --center-blocks 3 --center-min-radius 2 \
  --classes nuscenes10 --split scene --velocity --cbgs --ema 0.999 --cam-flip --temporal \
  --steps 80000 --workers 12 --frames ${DISTILL_CACHE:-/local/$USER/distill}/frames --teacher ${DISTILL_CACHE:-/local/$USER/distill}/teacher \
  --out outputs/distill/exp/final &
TPID=$!
sleep 600
if grep -qE "Traceback|Error" outputs/logs/final_8gpu.log || ! grep -qE "^  step " outputs/logs/final_8gpu.log \
   || ! grep -q "scene split:" outputs/logs/final_8gpu.log || ! grep -qE "resumed at step 36" outputs/logs/final_8gpu.log; then
  echo "FINAL_LAUNCH_FAILED"; kill -9 $TPID 2>/dev/null; exit 1; fi
echo "[$(date +%H:%M:%S)] FINAL resume 10-minute check passed: $(grep -oE 'step +[0-9]+/80000' outputs/logs/final_8gpu.log | head -1) ... $(grep -oE 'cls [0-9.]+' outputs/logs/final_8gpu.log | tail -1)"
wait $TPID; echo "TRAIN_RC=$?"
source export_onnx/env_gpu.sh
NOTE="FINAL: from scratch on the 700 train scenes, 10 classes + velocity, E7 recipe + temporal, 80k steps (36k on 4xA100, rest on 8xH200 at effective batch 32; EMA weights); scored on the 150 held-out val scenes"
for x in "final::" "final-nms2::--nms 2.0"; do t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/final/student.pt \
    --tag "$t" --note "$NOTE" --tokens scene --limit 250 $e; done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/final/student.pt \
    --tag "final-trainscenes" --note "$NOTE -- eval on frame-split val frames, i.e. held-out frames of TRAIN scenes (measures the scene leak)" --limit 250
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_official.py --ckpt outputs/distill/exp/final/student.pt --tag final-official --note "$NOTE"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_official.py --ckpt outputs/distill/exp/final/student.pt --subset cached --tag final-official-cached --note "$NOTE -- cached val subset, like-for-like with teacher-official"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/export_student.py --ckpt outputs/distill/exp/final/student.pt --out outputs/onnx/student_final/student.onnx
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/run_stateful.py --ckpt outputs/distill/exp/final/student.pt --check --frames 20
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/run_stateful.py --ckpt outputs/distill/exp/final/student.pt --onnx outputs/onnx/student_final/student.onnx --gpu 0 --frames 40
echo "FINAL_DONE"
