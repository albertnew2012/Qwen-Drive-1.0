# Equal-step comparison: snapshot R1 at its first checkpoint >= 19,000 steps and score it on
# the same 100 unseen val-scene frames as the final run's 19k snapshot (record 'final-mid').
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
CK=outputs/distill/exp/r1_hires/student.pt
step(){ .venv/bin/python -c "import torch;print(torch.load('$CK',map_location='cpu').get('step',0))" 2>/dev/null || echo 0; }
until [ "$(step)" -ge 19000 ]; do sleep 60; done
cp $CK outputs/distill/exp/r1_hires/r1_19k.pt
echo "[$(date +%H:%M:%S)] snapshot taken at step $(step)"
source export_onnx/env_gpu.sh
CUDA_VISIBLE_DEVICES=4 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/r1_hires/r1_19k.pt --tag r1-mid19k \
  --note "R1 (1152x640) at step ~19k on the same 100 val-scene frames as final-mid (step 19k, 56.4%)" --tokens scene --limit 100 \
  --frames ${DISTILL_CACHE:-/local/$USER/distill}/frames_real --teacher ${DISTILL_CACHE:-/local/$USER/distill}/teacher 2>&1 | grep -E "checkpoint step|TEACHER|STUDENT best|retained|by range|teacher  |student @|occupancy vs"
echo "R1_MID_DONE"
