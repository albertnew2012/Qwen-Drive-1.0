cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3
echo "[$(date +%H:%M:%S)] FINAL queued: waiting for E8 to finish (or fail)"
until grep -qE "E8_DONE|E8_LAUNCH_FAILED" outputs/logs/e8.log 2>/dev/null; do sleep 60; done
# Gate decision from the lab notebook: temporal fusion goes in if E8 beat its seed by >= 2 F1
# (it costs ~1 ms; even a small gain is worth keeping), the E7 recipe goes in regardless.
TEMPORAL=$(.venv/bin/python - <<'PY'
import json
def f1(r):
    v = r.get("student_vs_gt_best_thr"); return float(v["f1"]) if isinstance(v, dict) else -1.0
best = {}
for r in json.load(open("outputs/distill/lab_notebook.json")):
    t = str(r.get("tag", "")).split("-")[0]
    if t in ("e6", "e7", "e8"): best[t] = max(best.get(t, -1.0), f1(r))
seed = max(best.get("e6", -1.0), best.get("e7", -1.0))
gain = best.get("e8", -1.0) - seed
print(1 if (best.get("e8", -1.0) > 0 and gain >= 0.02) else 0)
print(f"  gate: E6 {best.get('e6')} E7 {best.get('e7')} E8 {best.get('e8')} -> temporal gain {gain:+.3f}", file=__import__('sys').stderr)
PY
)
echo "  temporal in final run: $TEMPORAL"
TFLAG=""; STEPS=100000
if [ "$TEMPORAL" = "1" ]; then TFLAG="--temporal"; STEPS=80000; fi
rm -rf outputs/distill/exp/final && mkdir -p outputs/distill/exp/final
echo "=== FINAL ($(date +%H:%M)): from scratch, 700 train scenes, 10 classes, velocity, E7 recipe $TFLAG, $STEPS steps ==="
.venv/bin/torchrun --nproc_per_node=4 --master_port=29625 \
  local/distill/train_student.py --det-objective center --det-head center --ref-points off \
  --center-hidden 128 --center-blocks 3 --center-min-radius 2 \
  --classes nuscenes10 --split scene --velocity --cbgs --ema 0.999 --cam-flip $TFLAG \
  --steps $STEPS --workers 8 --out outputs/distill/exp/final &
TPID=$!
sleep 600
if grep -qE "Traceback|Error" outputs/logs/final.log || ! grep -qE "^  step " outputs/logs/final.log \
   || ! grep -q "scene split:" outputs/logs/final.log || ! grep -q "EMA weights on" outputs/logs/final.log; then
  echo "FINAL_LAUNCH_FAILED"; kill -9 $TPID 2>/dev/null; pkill -9 -f "master_port=2962[5]"; exit 1; fi
echo "[$(date +%H:%M:%S)] FINAL 10-minute check passed: $(grep -oE 'cls [0-9.]+' outputs/logs/final.log | head -4 | tr '\n' ' ')"
wait $TPID; echo "TRAIN_RC=$?"
source export_onnx/env_gpu.sh
NOTE="FINAL: from scratch on the 700 train scenes, 10 classes + velocity, E7 recipe${TFLAG:+ + temporal}, $STEPS steps (EMA weights); scored on the 150 held-out val scenes"
for x in "final::" "final-nms2::--nms 2.0"; do t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/final/student.pt \
    --tag "$t" --note "$NOTE" --tokens scene --limit 250 $e; done
# the inflation check: the same model on held-out FRAMES of the train scenes it trained on
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/final/student.pt \
    --tag "final-trainscenes" --note "$NOTE -- eval on frame-split val frames, i.e. held-out frames of TRAIN scenes (measures the scene leak)" --limit 250
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_official.py --ckpt outputs/distill/exp/final/student.pt --tag final-official --note "$NOTE"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_official.py --ckpt outputs/distill/exp/final/student.pt --subset cached --tag final-official-cached --note "$NOTE -- cached val subset, like-for-like with teacher-official"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/export_student.py --ckpt outputs/distill/exp/final/student.pt --out outputs/onnx/student_final/student.onnx
if [ "$TEMPORAL" = "1" ]; then
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/run_stateful.py --ckpt outputs/distill/exp/final/student.pt --check --frames 20
fi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/run_stateful.py --ckpt outputs/distill/exp/final/student.pt --onnx outputs/onnx/student_final/student.onnx --gpu 0 --frames 40
echo "FINAL_DONE"
