#!/bin/bash
# Full Qwen-Drive-1.0 demo pipeline. GPU by default; DEVICE=cpu falls back and is
# polite to whatever owns the card.
#
#   bash local/run_all.sh              # GPU (default)
#   DEVICE=cpu bash local/run_all.sh   # CPU, safe to run beside a GPU job
#
# Runs the stages SEQUENTIALLY on purpose: two 4.5 B inference jobs at once starved
# the Alpamayo trainers' dataloaders (300 s -> 356 s/epoch) and pushed free RAM to 11 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-12}
NICE=${NICE:-19}
OUT=${OUT:-outputs}

if [ "$DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""
  ATTN=sdpa; PERC_DTYPE=float32
else
  ATTN=${ATTN:-flash_attention_2}; PERC_DTYPE=bfloat16
fi
export PYTHONPATH=src
export OMP_NUM_THREADS=$THREADS
# torch JIT-compiles the two perception CUDA kernels by shelling out to `ninja`,
# which lives in the venv bin - not on PATH when python is called by absolute path.
export PATH="$PWD/.venv/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr}

# renice the process AND torch's intra-op threads, which the process nice does not cover
# RESERVE_CORES physical cores are left untouched for whatever owns the GPU.
# On this box cpu i and cpu i+12 are the two threads of physical core i.
RESERVE_CORES=${RESERVE_CORES:-4}

lower () {
  local pid=$1
  renice -n "$NICE" -p "$pid" >/dev/null 2>&1 || true
  for t in $(ls /proc/"$pid"/task 2>/dev/null); do
    renice -n "$NICE" -p "$t" >/dev/null 2>&1 || true
  done
  if [ "$DEVICE" = "cpu" ] && [ "$RESERVE_CORES" -gt 0 ]; then
    local phys=$(lscpu -p=CORE | grep -vc '^#')
    phys=$(( phys / 2 ))                       # physical cores (SMT 2)
    if [ "$phys" -gt "$RESERVE_CORES" ]; then
      taskset -apc "$RESERVE_CORES-$((phys-1)),$((phys+RESERVE_CORES))-$((2*phys-1))" \
              "$pid" >/dev/null 2>&1 || true
    fi
  fi
}

run () {                      # run <logfile> <script> [args...]
  local log=$1; shift
  echo ">>> $* "
  "$@" > "$log" 2>&1 &
  local pid=$!
  sleep 30 && lower "$pid" &   # threads only exist once torch has spun them up
  wait "$pid"
  tail -3 "$log"
}

mkdir -p "$OUT"

run "$OUT/planning.log" .venv/bin/python -u local/run_planning_demo.py \
    --device "$DEVICE" --attn "$ATTN" --threads "$THREADS" \
    --num-samples 6 --max-new-tokens 220 --output "$OUT/planning_demo"

# repetition_penalty: the released VQA default is 1.0 with top_k=1, i.e. pure greedy,
# which loops on open-ended enumeration ("read every sign") - measured, see 04_RESULTS.md.
# 1.05 is a demo-quality choice; use 1.0 to reproduce the paper's benchmark protocol.
run "$OUT/vqa.log" .venv/bin/python -u local/run_vqa_probe.py \
    --device "$DEVICE" --attn "$ATTN" --threads "$THREADS" \
    --max-new-tokens 200 --repetition-penalty "${VQA_REP_PENALTY:-1.05}" \
    --output "$OUT/vqa_probe.json"

run "$OUT/perception.log" .venv/bin/python -u local/run_perception_demo.py \
    --device "$DEVICE" --dtype "$PERC_DTYPE" --attn "$ATTN" --threads "$THREADS" \
    --output "$OUT/perception_demo"

.venv/bin/python local/summarize.py | tee "$OUT/summary.txt"
.venv/bin/python local/make_video.py --output "$OUT/qwen_drive_demo.mp4"
echo "done -> $OUT/qwen_drive_demo.mp4"
