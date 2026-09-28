# After the final run's own post steps: write its H200 timing json where the orchestrator's
# report looks for it (final_post_8gpu.sh predates the --report flag).
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
until grep -q "FINAL_DONE" outputs/logs/final_post_8gpu.log 2>/dev/null; do sleep 120; done
source export_onnx/env_gpu.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/export_student.py --ckpt outputs/distill/exp/final/student.pt \
  --out outputs/onnx/student_final/student.onnx --report outputs/distill/exp/final/onnx_h200.json 2>&1 | grep -E "WHOLE|exported"
echo "FINAL_TIMING_DONE"
