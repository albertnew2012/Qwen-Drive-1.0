#!/usr/bin/env bash
# Full-val scoring of one checkpoint as two parallel scene folds (merge with fullval_rank.py).
#   local/distill/scripts/fold_score.sh CKPT TAG GPU_A GPU_B [NOTE]
# Records land in the lab notebook as TAG-valfoldA / TAG-valfoldB; logs in outputs/logs/idle/.
set -uo pipefail
CKPT=${1:?ckpt}; TAG=${2:?tag}; GA=${3:?gpu a}; GB=${4:?gpu b}; NOTE=${5:-"$TAG on val fold (full-val in two halves)"}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export PYTHONPATH=src:. PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false; source export_onnx/env_gpu.sh
# each diagnose otherwise spawns a CPU thread per core (240): four of them pushed the load to 330 and starved the
# training lanes. 16 threads each keeps them at full GPU pace with the box left for the lanes.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
L=${DISTILL_CACHE:-/local/$USER/distill}; mkdir -p outputs/logs/idle
for F in A B; do G=$([ $F = A ] && echo $GA || echo $GB)
  CUDA_VISIBLE_DEVICES=$G nohup .venv/bin/python local/distill/diagnose.py --ckpt "$CKPT" --tag "$TAG-valfold$F" --tokens valfold$F --limit 5000 \
      --note "$NOTE $F" --frames $L/frames_real --teacher $L/teacher > outputs/logs/idle/${TAG}_valfold${F}_gpu$G.log 2>&1 &
done
echo "fold scoring of $CKPT launched as $TAG-valfoldA (GPU $GA) and $TAG-valfoldB (GPU $GB)"
