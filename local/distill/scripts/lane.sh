#!/bin/bash
# lane.sh NAME "GPUS" NPROC STEPS "EXTRA FLAGS" LOGFILE
# One experiment, unattended: A100 speed gate -> smoke (skipped on resume) -> launch with a
# 600 s self-check -> wait -> verify the checkpoint reached STEPS -> evaluate on the held-out
# val scenes -> official mAP/NDS -> ONNX export + H200 timing -> A100 timing -> loop check.
# Markers on stdout: LANE_SKIPPED (too slow), LANE_FAILED, LANE_INCOMPLETE (rerun = resume), LANE_DONE.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
NAME=$1; GPUS=$2; NPROC=$3; STEPS=$4; EXTRA=($5); LOG=$6
L=${DISTILL_CACHE:-/local/$USER/distill}; FR=$L/frames_real; TE=$L/teacher
OUT=outputs/distill/exp/$NAME; mkdir -p $OUT
G1=${GPUS%%,*}; G2=$(echo $GPUS | cut -d, -f2); PORT=$((29660 + G1))
RECIPE=(--det-objective center --det-head center --ref-points off --center-hidden 128 --center-blocks 3 --center-min-radius 2
        --classes nuscenes10 --split scene-all --velocity --cbgs --ema 0.999 --cam-flip --temporal --occ-gt occ3d)
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $NAME: $*"; }
ckstep(){ .venv/bin/python -c "import torch;print(torch.load('$OUT/student.pt',map_location='cpu').get('step'))" 2>/dev/null; }
SZ=$(echo "${EXTRA[*]}" | grep -oE "image-size [0-9]+ [0-9]+" | awk '{print $2, $3}'); [ -z "$SZ" ] && SZ="896 512"
ARCH=$(echo "${EXTRA[*]}" | grep -oE "arch [A-Za-z0-9_]+" | awk '{print $2}'); [ -z "$ARCH" ] && ARCH=resnet50
HUP=$(echo "${EXTRA[*]}" | grep -oE "head-upsample [0-9]+" | awk '{print $2}'); [ -z "$HUP" ] && HUP=1   # the gate must time the head it will train
say "gpus $GPUS x$NPROC, steps $STEPS, size $SZ, arch $ARCH, extra: ${EXTRA[*]}"
if [ ! -f $OUT/student.pt ]; then
  source export_onnx/env_gpu.sh
  HZ=$(CUDA_VISIBLE_DEVICES=$G1 timeout 1500 .venv/bin/python local/distill/export_student.py --ckpt /nonexistent.pt --image-size $SZ --arch $ARCH --head-upsample $HUP \
       --out outputs/onnx/shape_probe/${NAME}.onnx --report $OUT/onnx_shape_h200.json 2>/dev/null | grep -oE "[0-9.]+ Hz" | head -1 | awk '{print $1}')
  if [ -n "$HZ" ]; then
    say "H200 speed of the untrained shape (GPU $G1): $HZ Hz"
    if [ "$(echo "$HZ < 10" | bc -l)" = "1" ]; then say "LANE_SKIPPED: below 10 Hz on H200"; exit 3; fi
  else say "speed gate produced no number; continuing"; fi
  rm -rf ${OUT}_smoke; mkdir -p ${OUT}_smoke
  CUDA_VISIBLE_DEVICES=$G1,$G2 .venv/bin/torchrun --nproc_per_node=2 --master_port=$((PORT+1)) local/distill/train_student.py \
    "${RECIPE[@]}" "${EXTRA[@]}" --steps 30 --batch 2 --workers 4 --log-every 10 --frames $FR --teacher $TE --out ${OUT}_smoke > outputs/logs/lane_${NAME}_smoke.log 2>&1
  if ! grep -q "trained 30 steps" outputs/logs/lane_${NAME}_smoke.log || grep -qE "did not receive grad|Traceback" outputs/logs/lane_${NAME}_smoke.log \
     || grep -oE "depth [0-9.]+" outputs/logs/lane_${NAME}_smoke.log | grep -q "depth 0.0000"; then
    say "LANE_FAILED: smoke"; grep -E "Traceback|Error|depth" outputs/logs/lane_${NAME}_smoke.log | head -4; exit 1; fi
  say "smoke passed ($(grep -oE 'depth [0-9.]+' outputs/logs/lane_${NAME}_smoke.log | head -2 | tr '\n' ' '))"; rm -rf ${OUT}_smoke
else
  say "resuming from checkpoint step $(ckstep)"
fi
CUDA_VISIBLE_DEVICES=$GPUS .venv/bin/torchrun --nproc_per_node=$NPROC --master_port=$PORT local/distill/train_student.py \
  "${RECIPE[@]}" "${EXTRA[@]}" --steps $STEPS --workers 12 --frames $FR --teacher $TE --out $OUT &
TPID=$!
sleep 600
if grep -qE "Traceback|Error" $LOG || ! grep -qE "^  step " $LOG || ! grep -q "scene split:" $LOG; then
  say "LANE_FAILED: launch check"; kill -9 $TPID 2>/dev/null; sleep 5
  .venv/bin/python - "$OUT" <<'PY'
import os, sys, signal
out = sys.argv[1]
for pid in os.listdir("/proc"):
    if not pid.isdigit(): continue
    try: c = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except Exception: continue
    if "train_student.py" in c and f"--out {out}" in c and c.split(" ")[0].endswith("python"):
        os.kill(int(pid), signal.SIGKILL)
PY
  exit 1; fi
say "launch check passed: $(grep -oE 'step +[0-9]+/[0-9]+' $LOG | tail -1) $(grep -oE 'cls [0-9.]+' $LOG | tail -1)"
wait $TPID; RC=$?
STEP=$(ckstep); say "training exited rc=$RC at checkpoint step $STEP"
if [ "$STEP" != "$STEPS" ]; then say "LANE_INCOMPLETE at step $STEP"; exit 2; fi
source export_onnx/env_gpu.sh
NOTE="$NAME: final recipe ${EXTRA[*]}, from scratch on the 700 train scenes, $STEPS steps, $NPROC x H200 (EMA weights); scored on the 150 held-out val scenes"
for x in "$NAME::" "$NAME-nms2::--nms 2.0"; do t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=$G1 .venv/bin/python local/distill/diagnose.py --ckpt $OUT/student.pt --tag "$t" --note "$NOTE" --tokens scene --limit 250 --frames $FR --teacher $TE $e; done
# checkpoint averaging over the late snapshots; keep whichever scores better
BEST=$OUT/student.pt
if .venv/bin/python local/distill/average_checkpoints.py $OUT; then
  CUDA_VISIBLE_DEVICES=$G1 .venv/bin/python local/distill/diagnose.py --ckpt $OUT/student_avg.pt --tag "$NAME-avg" --note "$NOTE -- average of the late snapshots" --tokens scene --limit 250 --frames $FR --teacher $TE
  BEST=$(.venv/bin/python - "$NAME" "$OUT" <<'PY'
import json, sys
name, out = sys.argv[1], sys.argv[2]
f1 = {}
for r in json.load(open("outputs/distill/lab_notebook.json")):
    if r.get("tag") in (name, name + "-avg") and r.get("eval_tokens") == "scene":
        f1[r["tag"]] = r["student_vs_gt_best_thr"]["f1"]
print(out + "/student_avg.pt" if f1.get(name + "-avg", -1) > f1.get(name, -1) else out + "/student.pt")
PY
); fi
# every periodic snapshot on the same 250 frames (tags NAME-mid<step>, files snap_<step>.pt), then the best of final / avg / snapshots
for sp in $(ls $OUT/snap_*.pt 2>/dev/null | sort -V); do st=$(basename $sp .pt | cut -d_ -f2)
  CUDA_VISIBLE_DEVICES=$G1 .venv/bin/python local/distill/diagnose.py --ckpt $sp --tag "$NAME-mid$st" --note "$NOTE -- snapshot at step $st" --tokens scene --limit 250 --frames $FR --teacher $TE; done
BEST=$(.venv/bin/python - "$NAME" "$OUT" "$BEST" <<'PY'
import json, re, sys
name, out, best = sys.argv[1:4]
scores = {}
for r in json.load(open("outputs/distill/lab_notebook.json")):
    t = str(r.get("tag", ""))
    if r.get("eval_tokens") != "scene" or r.get("frames") != 250 or r.get("official"): continue
    if t == name or t == name + "-avg" or re.fullmatch(re.escape(name) + r"-mid\d+", t):
        scores[r.get("ckpt")] = max(scores.get(r.get("ckpt"), -1), r["student_vs_gt_best_thr"]["f1"])
print(max(scores, key=scores.get) if scores else best)
PY
)
say "exporting and scoring $BEST"
CUDA_VISIBLE_DEVICES=$G1 .venv/bin/python local/distill/eval_official.py --ckpt $BEST --tag "$NAME-official" --note "$NOTE ($BEST)" --frames $FR --teacher $TE
# ONNX location follows the orchestrator's onnx_for(): snapshots go to outputs/onnx/<name>_snap<k>k/, the run's final/avg to outputs/onnx/<name>/
case "$(basename $BEST)" in snap_*) SK=$(( $(basename $BEST .pt | cut -d_ -f2) / 1000 )); ONNX_DIR=outputs/onnx/${NAME}_snap${SK}k;; *) ONNX_DIR=outputs/onnx/$NAME;; esac; mkdir -p $ONNX_DIR
CUDA_VISIBLE_DEVICES=$G1 .venv/bin/python local/distill/export_student.py --ckpt $BEST --out $ONNX_DIR/student.onnx --report $ONNX_DIR/onnx_h200.json; cp $ONNX_DIR/onnx_h200.json $OUT/onnx_h200.json 2>/dev/null
CUDA_VISIBLE_DEVICES=$G1 .venv/bin/python local/distill/run_stateful.py --ckpt $BEST --check --frames 20
CUDA_VISIBLE_DEVICES=$G1 .venv/bin/python local/distill/run_stateful.py --ckpt $BEST --onnx $ONNX_DIR/student.onnx --gpu 0 --frames 40   # index 0 = the only visible GPU
say "LANE_DONE"
