# mid_snapshot.sh NAME STEP [GPU]: when the lane's checkpoint reaches STEP, score a copy on the
# 100 comparison frames (same frames as final-mid 56.4% / r1-mid19k 57.2% at step 19k).
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
NAME=$1; STEP=$2; GPU=${3:-0}; LIMIT=${4:-100}; CK=outputs/distill/exp/$NAME/student.pt
step(){ .venv/bin/python -c "import torch;print(torch.load('$CK',map_location='cpu').get('step',0))" 2>/dev/null || echo 0; }
until [ "$(step)" -ge $STEP ]; do sleep 120; done
cp $CK outputs/distill/exp/$NAME/snap_${STEP}.pt
echo "[$(date +%H:%M:%S)] $NAME snapshot at step $(step)"
source export_onnx/env_gpu.sh
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/$NAME/snap_${STEP}.pt --tag ${NAME}-mid${STEP} \
  --note "$NAME at step ~$STEP on the 100 comparison frames (final-mid 56.4%, r1-mid19k 57.2% at 19k)" --tokens scene --limit $LIMIT \
  --frames ${DISTILL_CACHE:-/local/$USER/distill}/frames_real --teacher ${DISTILL_CACHE:-/local/$USER/distill}/teacher 2>&1 | grep -E "checkpoint step|STUDENT best|retained|by range|teacher  |student @|occupancy vs"
echo "MID_SNAPSHOT_DONE"
