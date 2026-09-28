#!/usr/bin/env bash
# Student release pipeline: checkpoint -> ONNX -> ONNX-only demo video on a nuScenes scene.
#
#   local/distill/scripts/onnx_pipeline.sh CKPT [SCENE] [GPU] [OUT_DIR] [DATAROOT]
#
#   CKPT      student checkpoint (full training checkpoint or the slim release .pt)
#   SCENE     nuScenes scene name for the demo            (default scene-0276, a val scene)
#   GPU       CUDA device index for export + onnxruntime  (default 0; -1 = CPU demo)
#   OUT_DIR   where student.onnx, the report and the video go (default outputs/student_release/run)
#   DATAROOT  nuScenes root with v1.0-trainval (or v1.0-mini: set NUSC_VERSION=v1.0-mini)
#
# Steps: 1) export_student.py traces the checkpoint into one graph with every head
#        (detection, occupancy, map, trajectory, recurrent BEV state) and times it;
#        2) demo_onnx.py runs that graph with onnxruntime only, straight from the devkit
#        (calibration, images, ego poses), and renders the session-demo video.
set -euo pipefail
CKPT=${1:?checkpoint path}; SCENE=${2:-scene-0276}; GPU=${3:-0}; OUT=${4:-outputs/student_release/run}; DATAROOT=${5:-data/nuscenes}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
export PYTHONPATH=src:.:local${PYTHONPATH:+:$PYTHONPATH}
[ -f export_onnx/env_gpu.sh ] && source export_onnx/env_gpu.sh || true     # cuDNN/TensorRT library paths, if the repo has them
mkdir -p "$OUT"
echo "== 1/2 export  $CKPT -> $OUT/student.onnx"
if [ "$GPU" -ge 0 ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
"$PY" local/distill/export_student.py --ckpt "$CKPT" --out "$OUT/student.onnx" --report "$OUT/onnx_timing.json" 2>&1 | grep -vE "Warning|warn" || true
[ -s "$OUT/student.onnx" ] || { echo "export failed"; exit 1; }
echo "== 2/2 demo    $SCENE -> $OUT/demo/$SCENE/student_onnx_$SCENE.mp4"
DEMO_GPU=0; [ "$GPU" -ge 0 ] || DEMO_GPU=-1
"$PY" local/distill/demo_onnx.py --onnx "$OUT/student.onnx" --scene "$SCENE" --gpu $DEMO_GPU --out "$OUT/demo" \
      --dataroot "$DATAROOT" --version "${NUSC_VERSION:-v1.0-trainval}" ${DEMO_ARGS:-} 2>&1 | grep -vE "Warning|warn" || true
[ -s "$OUT/demo/$SCENE/student_onnx_$SCENE.mp4" ] || { echo "demo failed"; exit 1; }
echo "== done:  $OUT/student.onnx   $OUT/onnx_timing.json   $OUT/demo/$SCENE/student_onnx_$SCENE.mp4"
