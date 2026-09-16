#!/bin/bash
# Parallel variant of run_export.sh. Same artifacts, same validation, ~2x faster.
#
#   bash export_onnx/run_export_parallel.sh            # export + optimize + validate
#   JOBS=2 bash export_onnx/run_export_parallel.sh     # fewer workers, less RAM
#   LAYER_JOBS=2 bash export_onnx/run_export_parallel.sh   # fewer workers per layer export
#   STAGE=export bash export_onnx/run_export_parallel.sh
#
# WHAT IS PARALLELISED
#   Two levels.
#   * Across steps: the six export steps are independent - no step reads
#     another's output - so they run concurrently in RAM-sized waves. The opset
#     retarget is parallel too: independent files, no model load.
#   * Inside a layer export: `--jobs N` forks N workers that each trace a
#     disjoint subset of the 32 decoder layers. Measured 3.4x on the layer phase
#     (92s -> 27s for four layers, four workers), and the output was verified
#     byte-identical to the sequential export, graph by graph.
#
#   `--jobs` is NOT the same thing as the `--layers` flag, which cannot shard:
#   it skips the *export* of a layer but not its *forward*, so a shard covering
#   layers 24-31 would still run 0-23, and every shard would rewrite
#   manifest.json (last writer wins) and final_norm.onnx (concurrent writes to
#   one path). Forking instead keeps one forward pass and one writer.
#
# WHAT IS NOT, AND WHY
#   * Validation stays sequential. `--phase compare` holds the ONNX graphs and a
#     PyTorch forward at once; two of those together get OOM-killed silently.
#
# RAM, not cores, is the limit: each exporter loads the 4.5 B VLM at ~31 GB.
# Forked workers share those weights copy-on-write and add only ~5 GB each, so
# two layer exports at LAYER_JOBS=4 cost ~2*31 + 8*5 = ~100 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
STAGE=${STAGE:-all}
FRAME=${FRAME:-90162f90eceb4ada9e595bc1adb71b5f}
SCENE=${SCENE:-a53176b07ad432affda912ef26737f02}
GB_PER_JOB=${GB_PER_JOB:-32}

export PYTHONPATH=src:.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs/export_parallel
mkdir -p "$LOG"

avail=$(free -g | awk '/Mem:/{print $7}')
cap=$(( avail / GB_PER_JOB )); [ "$cap" -lt 1 ] && cap=1
JOBS=${JOBS:-$(( cap < 4 ? cap : 4 ))}
# Forked layer workers are cheap in RAM (copy-on-write) but each wants a core or
# two, and two layer exports run side by side, so keep 2*LAYER_JOBS under nproc.
half_cores=$(( $(nproc) / 2 ))
LAYER_JOBS=${LAYER_JOBS:-$(( half_cores < 4 ? half_cores : 4 ))}
[ "$LAYER_JOBS" -lt 1 ] && LAYER_JOBS=1

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

PIDS=(); NAMES=()
spawn() {                     # spawn <name> <cmd...>
  local name=$1; shift
  "$@" >"$LOG/$name.log" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
  printf '  started  %-26s pid %-7s -> %s.log\n' "$name" "$!" "$name"
}
reap() {                      # wait for the wave, report every failure
  local rc=0 i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      printf '  \033[32mok\033[0m       %s\n' "${NAMES[$i]}"
    else
      printf '  \033[31mFAILED\033[0m   %-26s see %s\n' "${NAMES[$i]}" "$LOG/${NAMES[$i]}.log"
      rc=1
    fi
  done
  PIDS=(); NAMES=()
  return $rc
}

printf 'workers: %s   (%s GB available / %s GB per job)\n' "$JOBS" "$avail" "$GB_PER_JOB"
printf 'layer workers per export: %s   (forked, weights shared copy-on-write)\n' "$LAYER_JOBS"

# ---------------------------------------------------------------- prerequisites
if [ ! -f "data/train_cache/${FRAME}.pt" ]; then
  step "caching VLM feature taps (one-off)"; $PY training/cache_features.py
fi
if [ ! -d data/train_cache_plan ] || [ -z "$(ls -A data/train_cache_plan 2>/dev/null)" ]; then
  step "caching planner scene caches (one-off)"; $PY training/cache_planner_features.py
fi

# ------------------------------------------------------------------- 1. export
if [ "$STAGE" = all ] || [ "$STAGE" = export ]; then
  export CUDA_VISIBLE_DEVICES=""
  t0=$SECONDS

  # Wave 1: the two 12-13 min layer exports, started first so the short jobs
  # fill the remaining slots beside them rather than after them.
  step "wave 1 - the long poles + whatever fits beside them"
  spawn layers_perception $PY export_onnx/export_vlm_layers.py --task perception \
        --frame "$FRAME" --jobs "$LAYER_JOBS" --out outputs/onnx/vlm_layers_v2
  spawn layers_planning   $PY export_onnx/export_vlm_layers.py --task planning \
        --jobs "$LAYER_JOBS" --out outputs/onnx/vlm_layers_plan_v2
  [ "$JOBS" -ge 3 ] && spawn perception_head $PY export_onnx/export_perception.py \
        --no-fold --opset 20 --record "${FRAME}.pt" \
        --out outputs/onnx/perception/perception.onnx
  [ "$JOBS" -ge 4 ] && spawn planner_step $PY export_onnx/export_planner.py \
        --opset 20 --record "$SCENE" --out outputs/onnx/planner/planner_step.onnx
  reap

  step "wave 2 - the rest"
  [ "$JOBS" -lt 3 ] && spawn perception_head $PY export_onnx/export_perception.py \
        --no-fold --opset 20 --record "${FRAME}.pt" \
        --out outputs/onnx/perception/perception.onnx
  [ "$JOBS" -lt 4 ] && spawn planner_step $PY export_onnx/export_planner.py \
        --opset 20 --record "$SCENE" --out outputs/onnx/planner/planner_step.onnx
  spawn vision_perception $PY export_onnx/export_vlm.py --part vision --frame "$FRAME"
  spawn vision_planning   $PY export_onnx/export_vlm.py --part vision --task planning
  reap

  printf '\nexport stage: %d min\n' "$(( (SECONDS - t0) / 60 ))"
fi

# ----------------------------------------------------------------- 2. optimize
# Both rewrites change WHERE a node runs, not what it computes.
if [ "$STAGE" = all ] || [ "$STAGE" = optimize ]; then
  export CUDA_VISIBLE_DEVICES=""
  t0=$SECONDS

  step "5-D GridSample -> gather (perception head)"
  $PY export_onnx/gridsample5d_to_gather.py outputs/onnx/perception/perception.onnx

  step "retarget opset 20 -> 18, three targets in parallel"
  spawn retarget_perception $PY export_onnx/retarget_opset.py \
        outputs/onnx/perception/perception.onnx --opset 18 --verify
  spawn retarget_layers     $PY export_onnx/retarget_opset.py \
        outputs/onnx/vlm_layers_v2 --opset 18 --verify
  spawn retarget_layers_plan $PY export_onnx/retarget_opset.py \
        outputs/onnx/vlm_layers_plan_v2 --opset 18 --verify
  reap

  printf '\noptimize stage: %d min\n' "$(( (SECONDS - t0) / 60 ))"
fi

# ----------------------------------------------------------------- 3. validate
# Sequential on purpose - see the header.
if [ "$STAGE" = all ] || [ "$STAGE" = validate ]; then
  export CUDA_VISIBLE_DEVICES=""
  t0=$SECONDS

  step "perception: run"
  $PY export_onnx/run_onnx_pipeline.py --phase run     --frame "$FRAME" 2>&1 | tee "$LOG/validate_perception_run.log"
  step "perception: compare vs PyTorch"
  $PY export_onnx/run_onnx_pipeline.py --phase compare --frame "$FRAME" 2>&1 | tee "$LOG/validate_perception_cmp.log"
  step "planning: run"
  $PY export_onnx/run_onnx_planner.py --phase run     2>&1 | tee "$LOG/validate_planning_run.log"
  step "planning: compare vs PyTorch"
  $PY export_onnx/run_onnx_planner.py --phase compare 2>&1 | tee "$LOG/validate_planning_cmp.log"

  step "verdict"
  grep -hE "PASS|FAIL" "$LOG"/validate_*_cmp.log || echo "  no verdict line found"
  printf '\nvalidate stage: %d min\n' "$(( (SECONDS - t0) / 60 ))"
fi

step "done"
