#!/bin/bash
# Export Qwen-Drive-1.0 to ONNX, apply the GPU-placement rewrites, and validate
# every graph against PyTorch.
#
#   bash export_onnx/run_export.sh                 # export + optimize + validate
#   STAGE=export   bash export_onnx/run_export.sh  # graphs only
#   STAGE=optimize bash export_onnx/run_export.sh  # rewrites only (needs the graphs)
#   STAGE=validate bash export_onnx/run_export.sh  # equivalence only
#   STAGE=bench    bash export_onnx/run_export.sh  # GPU timing, needs .venv-ortgpu
#
# Export and validation run on the CPU in fp32 on purpose: this is a numerical
# equivalence check, and the GPU is not involved in deciding whether it passes.
#
# NOT exported here:
#   planner-rl    - planner-sft is the released default and the one both runners
#                   use; exporting both doubles 2 GB for no extra coverage.
#   vlm_decode    - the autoregressive text path (chain of thought). ~15 GB of
#                   artifacts that perception and trajectory never touch.
#                   Add it with:  $PY export_onnx/export_vlm_decode.py --prefill 64
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
STAGE=${STAGE:-all}
FRAME=${FRAME:-90162f90eceb4ada9e595bc1adb71b5f}
SCENE=${SCENE:-a53176b07ad432affda912ef26737f02}

export PYTHONPATH=src:.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

LOG=outputs/logs
mkdir -p "$LOG"

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
timed() { local t0=$SECONDS; "$@"; printf '    (%d s)\n' "$((SECONDS - t0))"; }

# ---------------------------------------------------------------- prerequisites
# The exporters trace against a cached feature record, not against raw images.
if [ ! -f "data/train_cache/${FRAME}.pt" ]; then
  step "caching VLM feature taps (one-off)"
  timed $PY training/cache_features.py
fi
if [ ! -d data/train_cache_plan ] || [ -z "$(ls -A data/train_cache_plan 2>/dev/null)" ]; then
  step "caching planner scene caches (one-off)"
  timed $PY training/cache_planner_features.py
fi

# ------------------------------------------------------------------- 1. export
if [ "$STAGE" = all ] || [ "$STAGE" = export ]; then
  export CUDA_VISIBLE_DEVICES=""

  step "1/6  BEV perception head          (~4 min)"
  timed $PY export_onnx/export_perception.py --no-fold --opset 20 \
      --record "${FRAME}.pt" \
      --out outputs/onnx/perception/perception.onnx

  step "2/6  VLM vision tower, perception (~1 min)"
  timed $PY export_onnx/export_vlm.py --part vision --frame "$FRAME"

  step "3/6  VLM text, 33 per-layer graphs, perception shape (~12 min)"
  timed $PY export_onnx/export_vlm_layers.py --task perception --frame "$FRAME" \
      --out outputs/onnx/vlm_layers_v2

  step "4/6  planner denoise step         (~1 min)"
  timed $PY export_onnx/export_planner.py --opset 20 \
      --record "$SCENE" \
      --out outputs/onnx/planner/planner_step.onnx

  step "5/6  VLM vision tower, planning   (~1 min)"
  timed $PY export_onnx/export_vlm.py --part vision --task planning

  step "6/6  VLM text, 33 per-layer graphs, planning shape (~13 min)"
  timed $PY export_onnx/export_vlm_layers.py --task planning \
      --out outputs/onnx/vlm_layers_plan_v2
fi

# ----------------------------------------------------------------- 2. optimize
# Both rewrites are about WHERE a node runs, not what it computes. ORT silently
# places a node on the CPU when it has no CUDA kernel for that opset or rank,
# and the device copies around it cost more than the node.
if [ "$STAGE" = all ] || [ "$STAGE" = optimize ]; then
  export CUDA_VISIBLE_DEVICES=""

  step "5-D GridSample -> gather-based trilinear (perception head)"
  timed $PY export_onnx/gridsample5d_to_gather.py \
      outputs/onnx/perception/perception.onnx

  step "retarget opset 20 -> 18 so Pad/Resize keep their CUDA kernels"
  timed $PY export_onnx/retarget_opset.py \
      outputs/onnx/perception/perception.onnx \
      outputs/onnx/vlm_layers_v2 \
      outputs/onnx/vlm_layers_plan_v2 \
      --opset 18 --verify
fi

# ----------------------------------------------------------------- 3. validate
# run and compare MUST stay separate processes: holding the 17 GiB fp32 decoder
# and a 20 GiB PyTorch perception forward at once gets the process OOM-killed,
# silently, with no traceback.
if [ "$STAGE" = all ] || [ "$STAGE" = validate ]; then
  export CUDA_VISIBLE_DEVICES=""

  step "perception: run the graphs   (~5 min)"
  timed $PY export_onnx/run_onnx_pipeline.py --phase run     --frame "$FRAME" \
      2>&1 | tee "$LOG/onnx_validate_perception_run.log"
  step "perception: compare vs PyTorch (~3 min)"
  timed $PY export_onnx/run_onnx_pipeline.py --phase compare --frame "$FRAME" \
      2>&1 | tee "$LOG/onnx_validate_perception_cmp.log"

  step "planning: run the graphs     (~6 min)"
  timed $PY export_onnx/run_onnx_planner.py --phase run     \
      2>&1 | tee "$LOG/onnx_validate_planning_run.log"
  step "planning: compare vs PyTorch  (~4 min)"
  timed $PY export_onnx/run_onnx_planner.py --phase compare \
      2>&1 | tee "$LOG/onnx_validate_planning_cmp.log"

  step "verdict"
  grep -hE "PASS|FAIL" "$LOG"/onnx_validate_*_cmp.log || echo "  no verdict line found"
fi

# -------------------------------------------------------------------- 4. bench
# Separate venv: onnxruntime-gpu and onnxruntime cannot coexist, and .venv holds
# the CPU build that every number above was validated with.
if [ "$STAGE" = bench ]; then
  [ -x .venv-ortgpu/bin/python ] || {
    echo "need .venv-ortgpu:  uv venv --python 3.12 .venv-ortgpu && \\"
    echo "  VIRTUAL_ENV=.venv-ortgpu uv pip install onnxruntime-gpu==1.22.0 onnx numpy==2.2.6"
    exit 1
  }
  unset CUDA_VISIBLE_DEVICES
  NV=$PWD/.venv/lib/python3.12/site-packages/nvidia
  export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusparse/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

  step "GPU inputs (needs torch, so it runs in .venv)"
  CUDA_VISIBLE_DEVICES="" timed $PY export_onnx/run_onnx_drive.py --phase prep

  step "perception, fp32, streamed"
  timed .venv-ortgpu/bin/python -u export_onnx/run_onnx_drive.py --phase run \
      --precision fp32 --layers outputs/onnx/vlm_layers_v2 \
      --head-dir outputs/onnx/perception --chunk 8 --skip-planning --reps 3

  step "trajectory, fp16, 20 layers resident"
  timed .venv-ortgpu/bin/python -u export_onnx/run_onnx_drive.py --phase run \
      --precision fp16 --layers-plan outputs/onnx/vlm_layers_plan_v2 \
      --planner-dir outputs/onnx/planner --chunk 0 --resident-plan 20 \
      --skip-perception --reps 3
fi

step "done"
