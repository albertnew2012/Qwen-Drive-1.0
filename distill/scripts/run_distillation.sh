#!/usr/bin/env bash
# The whole Qwen-Drive-1.0 -> student distillation, end to end, as one resumable script.
#
#   distill/scripts/run_distillation.sh                 # every stage, from wherever the last run stopped
#   distill/scripts/run_distillation.sh --from select   # start at a stage (earlier ones are assumed done)
#   distill/scripts/run_distillation.sh --only evaluate # one stage
#   distill/scripts/run_distillation.sh --dry-run       # print the commands, run nothing
#
# Stages, in order (each one skips work whose outputs already exist, so rerunning resumes):
#   prep      one-time caches: frames, teacher outputs, GT boxes, lidar depth, ego tracks, Occ3D labels,
#             temporal index, scene split, occupancy class frequencies
#   base      train the base model from an ImageNet backbone: three-frame history, 1152x640  (r3_final recipe)
#   select    score every base snapshot on the 250-frame check, pick the best one
#   finetune  from that snapshot: heatmap loss x2 on ped/moto/bicycle/cone + occupancy outside-mask term
#             (r4_small recipe)
#   average   average the fine-tune's last-quarter checkpoints
#   evaluate  250-frame check + all 4,452 val frames (two scene folds) for the fine-tune's final and averaged
#             weights, pick the better one; official nuScenes mAP/NDS on it
#   release   checkpoint + ONNX + chunks (make_release.sh), then the real-frame loop timing
#   demo      the nuscenes_session.gif render with the released ONNX
#
# Settings (environment variables):
#   NAME=v7            experiment name: outputs/distill/exp/<NAME>_base, <NAME>_ft, outputs/student_release_<NAME>
#   GPUS=0,1,2,3       GPUs for training and parallel evaluation
#   BATCH=auto         per-GPU batch; auto = 4 on GPUs with >=100 GB, else 2 (batch 2 is not validated at full length)
#   BASE_STEPS=75000   FT_STEPS=10000   SNAP_EVERY=10000
#   FRAMES=data/distill/frames   TEACHER=data/distill/teacher   NUSC_ROOT=data/nuscenes
#   EVAL_LIMIT=0       0 = full evaluation sets; N = first N frames (smoke tests only)
#   OFFICIAL=1  DEMO=1 set to 0 to skip the official nuScenes eval / the demo render
# Never overwrites an earlier experiment: a finished stage is skipped, an interrupted one resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

NAME=${NAME:-v7}; GPUS=${GPUS:-0,1,2,3}; BATCH=${BATCH:-auto}
BASE_STEPS=${BASE_STEPS:-75000}; FT_STEPS=${FT_STEPS:-10000}; SNAP_EVERY=${SNAP_EVERY:-10000}
FRAMES=${FRAMES:-data/distill/frames}; TEACHER=${TEACHER:-data/distill/teacher}; NUSC_ROOT=${NUSC_ROOT:-data/nuscenes}
EVAL_LIMIT=${EVAL_LIMIT:-0}; OFFICIAL=${OFFICIAL:-1}; DEMO=${DEMO:-1}
BASE=${NAME}_base; FT=${NAME}_ft; EXP=outputs/distill/exp; LOGS=outputs/logs/distillation_${NAME}
STAGES=(prep base select finetune average evaluate release demo)

FROM=prep; ONLY=""; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --from) FROM=$2; shift 2;; --only) ONLY=$2; shift 2;; --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,34p' "$0"; exit 0;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done

export PYTHONPATH=src:.:tools PATH="$ROOT/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
[ -f export_onnx/env_gpu.sh ] && source export_onnx/env_gpu.sh
PY=.venv/bin/python
IFS=, read -r -a GPU_LIST <<< "$GPUS"; NPROC=${#GPU_LIST[@]}; G0=${GPU_LIST[0]}
mkdir -p "$LOGS"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
logged() { local log=$1; shift; echo "+ $* > $log"; [ "$DRY" = 1 ] || "$@" > "$log" 2>&1; }
want() {   # is stage $1 selected by --from / --only?
  if [ -n "$ONLY" ]; then [ "$1" = "$ONLY" ]; return; fi
  local hit=0; for s in "${STAGES[@]}"; do [ "$s" = "$FROM" ] && hit=1; [ "$s" = "$1" ] && return $((1 - hit)); done; return 1
}
ckstep() { [ -f "$1" ] && $PY -c "import torch;print(torch.load('$1',map_location='cpu').get('step'))" 2>/dev/null || echo 0; }
notebook_f1() {   # best gate F1 per checkpoint for a tag prefix -> "f1 ckpt" lines, best first
  $PY - "$1" <<'PY'
import json, re, sys
pre = sys.argv[1]; best = {}
for r in json.load(open("outputs/distill/lab_notebook.json")):
    t = str(r.get("tag", ""))
    if t.startswith(pre) and r.get("eval_tokens") == "scene" and "student_vs_gt_best_thr" in r:
        best[r["ckpt"]] = max(best.get(r["ckpt"], -1), r["student_vs_gt_best_thr"]["f1"])
for c, f in sorted(best.items(), key=lambda kv: -kv[1]): print(f"{f:.4f} {c}")
PY
}
diag() {   # diag GPU CKPT TAG TOKENS [LIMIT]  -> one lab-notebook record (skipped if the tag exists)
  local gpu=$1 ck=$2 tag=$3 tokens=$4 limit=${5:-0}
  if grep -q "\"tag\": \"$tag\"" outputs/distill/lab_notebook.json 2>/dev/null; then echo "  $tag: already scored"; return; fi
  [ "$limit" = 0 ] && limit=$([ "$tokens" = scene ] && echo 250 || echo 5000)
  [ "$EVAL_LIMIT" != 0 ] && limit=$EVAL_LIMIT
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 logged "$LOGS/$tag.log" \
    $PY distill/diagnose.py --ckpt "$ck" --tag "$tag" --tokens "$tokens" --limit "$limit" \
    --frames "$FRAMES" --teacher "$TEACHER" --note "run_distillation.sh $NAME: $tag"
}
train() {  # train OUT STEPS EXTRA...   (resumes automatically from OUT/student.pt)
  local out=$1 steps=$2; shift 2
  local at; at=$(ckstep "$out/student.pt")
  if [ "$at" = "$steps" ]; then say "$out already at step $steps"; return; fi
  say "training $out to step $steps (currently at ${at:-0})"
  CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 run .venv/bin/torchrun --nproc_per_node="$NPROC" --master_port=$((29700 + G0)) distill/train_student.py \
    --det-objective center --det-head center --ref-points off --center-hidden 128 --center-blocks 3 \
    --center-min-radius 2 --classes nuscenes10 --split scene-all --velocity --cbgs --ema 0.999 --cam-flip \
    --temporal --occ-gt occ3d --image-size 1152 640 --history 3 --batch "$BATCH" --workers 12 \
    --snap-every "$SNAP_EVERY" --frames "$FRAMES" --teacher "$TEACHER" --steps "$steps" --out "$out" "$@" \
    2>&1 | tee -a "$LOGS/$(basename "$out")_train.log" || true
  [ "$DRY" = 1 ] || [ "$(ckstep "$out/student.pt")" = "$steps" ] || { say "training stopped before step $steps: rerun to resume"; exit 3; }
}

if [ "$BATCH" = auto ]; then
  MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$G0" 2>/dev/null || echo 0)
  if [ "${MEM:-0}" -ge 100000 ]; then BATCH=4; else BATCH=2; fi
fi
say "distillation $NAME on GPUs $GPUS (${NPROC} x batch $BATCH); logs in $LOGS"
[ "$BATCH" -lt 4 ] && say "note: batch $BATCH per GPU differs from the validated recipe (4 per GPU on 4 GPUs = 16)"

# ------------------------------------------------------------------ prep
if want prep; then
  say "== prep: one-time caches"
  [ -d "$NUSC_ROOT/v1.0-trainval" ] || { echo "nuScenes v1.0-trainval not found under $NUSC_ROOT"; exit 1; }
  mkdir -p "$FRAMES" "$TEACHER"
  read -r N_FRAMES MISS_GT MISS_DEPTH MISS_OCC <<< "$($PY - "$FRAMES" <<'PY'
import os, sys
fr = sys.argv[1]; toks = [t for t in os.listdir(fr) if os.path.isdir(os.path.join(fr, t))] if os.path.isdir(fr) else []
miss = lambda f: sum(not os.path.exists(os.path.join(fr, t, f)) for t in toks)
print(len(toks), miss("gt_boxes10.npz"), miss("depth_1152x640.npz"), miss("occ3d.npz"))
PY
)"
  say "  $N_FRAMES frames; missing: gt_boxes10 $MISS_GT, depth_1152x640 $MISS_DEPTH, occ3d $MISS_OCC"
  [ "$N_FRAMES" -gt 0 ] || run $PY distill/nusc_frames.py --root "$NUSC_ROOT" --version v1.0-trainval --limit 0 --out "$FRAMES"
  if [ "$(ls "$TEACHER" | wc -l)" -eq 0 ]; then   # the 4B teacher over every frame: needs a GPU and the weights
    CUDA_VISIBLE_DEVICES=$G0 run $PY distill/cache_teacher.py --frames "$FRAMES" --out "$TEACHER" --limit 0
  fi
  [ "$MISS_GT" -eq 0 ] || run $PY distill/nusc_gt_boxes.py --root "$NUSC_ROOT" --version v1.0-trainval --frames "$FRAMES"
  [ "$MISS_DEPTH" -eq 0 ] || run $PY distill/nusc_depth.py --root "$NUSC_ROOT" --version v1.0-trainval \
      --frames "$FRAMES" --teacher "$TEACHER" --image-size 1152 640 --out-name depth_1152x640.npz
  [ "$(ls data/distill/ego 2>/dev/null | wc -l)" -ge "$N_FRAMES" ] || \
      run $PY distill/nusc_ego.py --root "$NUSC_ROOT" --version v1.0-trainval --include-short-future
  if [ "$MISS_OCC" -gt 0 ]; then
    [ -d data/occ3d/uniocc ] || { echo "Occ3D labels missing: download UniOcc (tasl-lab/uniocc) to data/occ3d/uniocc"; exit 1; }
    run $PY distill/occ3d_labels.py --split train --frames "$FRAMES" --teacher "$TEACHER"
    run $PY distill/occ3d_labels.py --split val --frames "$FRAMES" --teacher "$TEACHER"
  fi
  [ -f data/distill/temporal_index.json ] || run $PY distill/build_temporal_index.py --root "$NUSC_ROOT" --version v1.0-trainval
  if [ ! -f data/distill/scene_split.json ]; then
    say "  writing data/distill/scene_split.json (eval250 will NOT match the original fixed list)"
    [ "$DRY" = 1 ] || $PY - "$NUSC_ROOT" "$FRAMES" "$TEACHER" <<'PY'
import json, os, random, sys
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
root, fr, te = sys.argv[1:4]
nusc = NuScenes(version="v1.0-trainval", dataroot=root, verbose=False)
split = create_splits_scenes(); scene_of = {s["token"]: s["name"] for s in nusc.scene}
have = set(os.listdir(fr)); cached = {f[:-4] for f in os.listdir(te) if f.endswith(".npz")}
out = {k: [] for k in ("train_tokens", "val_tokens", "uncached_train_tokens", "uncached_val_tokens")}
for smp in nusc.sample:
    if smp["token"] not in have: continue
    part = "train" if scene_of[smp["scene_token"]] in split["train"] else "val"
    out[(part + "_tokens") if smp["token"] in cached else f"uncached_{part}_tokens"].append(smp["token"])
random.seed(0); out["eval250"] = sorted(random.sample(sorted(out["val_tokens"]), 250))
out["note"] = "official nuScenes scene split (700 train / 150 val scenes); cached = has teacher npz; eval250 regenerated, seed 0"
json.dump(out, open("data/distill/scene_split.json", "w")); print({k: len(v) for k, v in out.items() if isinstance(v, list)})
PY
  fi
  if [ ! -f data/distill/occ3d_freq.npy ]; then   # as the original: class counts over train AND val frames
    say "  writing data/distill/occ3d_freq.npy"
    [ "$DRY" = 1 ] || $PY - "$FRAMES" <<'PY'
import json, sys, numpy as np
fr = sys.argv[1]; ss = json.load(open("data/distill/scene_split.json")); counts = np.zeros(10, np.int64)
for tok in ss["train_tokens"] + ss["val_tokens"]:
    try: o = np.load(f"{fr}/{tok}/occ3d.npz")
    except FileNotFoundError: continue
    counts += np.bincount(o["occ"][o["mask"] > 0].astype(np.int64), minlength=10)[:10]
np.save("data/distill/occ3d_freq.npy", counts)
PY
  fi
fi

# ------------------------------------------------------------------ base
if want base; then
  say "== base: ${BASE_STEPS} steps from an ImageNet backbone -> $EXP/$BASE"
  train "$EXP/$BASE" "$BASE_STEPS"
fi

# ------------------------------------------------------------------ select
SEL_JSON=$EXP/$BASE/selected.json
if want select; then
  say "== select: every base snapshot on the 250-frame check"
  SNAPS=($(ls "$EXP/$BASE"/snap_*.pt 2>/dev/null | sort -V || true))
  [ "$DRY" = 1 ] || [ ${#SNAPS[@]} -gt 0 ] || { echo "no snapshots in $EXP/$BASE"; exit 1; }
  i=0
  for sp in "${SNAPS[@]}"; do
    st=$(basename "$sp" .pt | cut -d_ -f2)
    diag "${GPU_LIST[$((i % NPROC))]}" "$sp" "$BASE-mid$st" scene &
    i=$((i + 1)); [ $((i % NPROC)) -eq 0 ] && wait
  done; wait
  if [ "$DRY" = 0 ]; then
    BEST=$(notebook_f1 "$BASE-mid" | head -1 | awk '{print $2}')
    [ -n "$BEST" ] || { echo "no snapshot was scored: see $LOGS/$BASE-mid*.log"; exit 1; }
    STEP=$(basename "$BEST" .pt | cut -d_ -f2)
    echo "{\"snapshot\": \"$BEST\", \"step\": $STEP}" > "$SEL_JSON"
    say "  best snapshot: $BEST"; notebook_f1 "$BASE-mid" | head -5 | sed 's/^/    gate F1 /'
  fi
fi

# ------------------------------------------------------------------ finetune
if want finetune; then
  [ "$DRY" = 1 ] && SEL_STEP=30000 || SEL_STEP=$($PY -c "import json;print(json.load(open('$SEL_JSON'))['step'])")
  [ "$DRY" = 1 ] && SEL_CK=$EXP/$BASE/snap_30000.pt || SEL_CK=$($PY -c "import json;print(json.load(open('$SEL_JSON'))['snapshot'])")
  FT_TOTAL=$((SEL_STEP + FT_STEPS))
  say "== finetune: $SEL_CK -> $EXP/$FT, steps $SEL_STEP -> $FT_TOTAL"
  if [ ! -f "$EXP/$FT/student.pt" ]; then run mkdir -p "$EXP/$FT"; run cp "$SEL_CK" "$EXP/$FT/student.pt"; fi
  # the one-cycle schedule runs over FT_TOTAL steps and resumes at SEL_STEP: a short, low-LR anneal
  train "$EXP/$FT" "$FT_TOTAL" --hm-class-weight 5:2,6:2,7:2,8:2 --occ-outside-empty 0.2
fi

# ------------------------------------------------------------------ average
if want average; then
  say "== average: the fine-tune's last-quarter checkpoints"
  if [ -f "$EXP/$FT/student_avg.pt" ]; then say "  student_avg.pt exists"
  else run $PY distill/average_checkpoints.py "$EXP/$FT" || say "  fewer than two snapshots to average: skipped"; fi
fi

# ------------------------------------------------------------------ evaluate
CHOSEN_JSON=$EXP/$FT/chosen.json
if want evaluate; then
  say "== evaluate: 250-frame check and all val frames"
  CANDS=("$EXP/$FT/student.pt:$FT-final"); [ -f "$EXP/$FT/student_avg.pt" ] && CANDS+=("$EXP/$FT/student_avg.pt:$FT-avg")
  i=0
  for c in "${CANDS[@]}"; do ck=${c%%:*}; tag=${c##*:}
    for t in scene valfoldA valfoldB; do
      tg=$tag; [ $t != scene ] && tg=$tag-$t
      diag "${GPU_LIST[$((i % NPROC))]}" "$ck" "$tg" "$t" & i=$((i + 1)); [ $((i % NPROC)) -eq 0 ] && wait
    done
  done; wait
  if [ "$DRY" = 0 ]; then
    $PY distill/fullval_rank.py | tee "$LOGS/fullval_rank.txt" | head -8
    CHOSEN=$($PY - "$FT" <<'PY'
import json, sys
rows = [r for r in json.load(open("outputs/distill/fullval_ranking.json")) if r["candidate"].startswith(sys.argv[1] + "-")]
print(max(rows, key=lambda r: r["student_f1"])["ckpt"] if rows else "")
PY
)
    [ -n "$CHOSEN" ] || { echo "no full-val result for $FT: see $LOGS/$FT-*valfold*.log"; exit 1; }
    echo "{\"checkpoint\": \"$CHOSEN\"}" > "$CHOSEN_JSON"; say "  chosen: $CHOSEN"
    if [ "$OFFICIAL" = 1 ]; then
      CUDA_VISIBLE_DEVICES=$G0 OMP_NUM_THREADS=16 logged "$LOGS/official.log" $PY distill/eval_official.py --ckpt "$CHOSEN" \
        --tag "$FT-official" --frames "$FRAMES" --teacher "$TEACHER" --note "run_distillation.sh $NAME" \
        $([ "$EVAL_LIMIT" != 0 ] && echo --limit "$EVAL_LIMIT")
      grep -E "mAP|NDS" "$LOGS/official.log" | tail -2
    fi
  fi
fi

# ------------------------------------------------------------------ release
REL=outputs/student_release_$NAME
if want release; then
  say "== release -> $REL"
  if [ -d "$REL" ] && [ -n "$(ls -A "$REL" 2>/dev/null)" ]; then say "  $REL exists: kept"
  else
    [ "$DRY" = 1 ] && CHOSEN=$EXP/$FT/student_avg.pt || CHOSEN=$($PY -c "import json;print(json.load(open('$CHOSEN_JSON'))['checkpoint'])")
    run distill/scripts/make_release.sh "$CHOSEN" "$NAME" "$REL" "$G0"
    CUDA_VISIBLE_DEVICES=$G0 logged "$LOGS/loop_timing.log" $PY distill/run_stateful.py --ckpt "$CHOSEN" \
      --onnx "$REL/student_$NAME.onnx" --gpu 0 --frames 40 || true
    [ "$DRY" = 1 ] || grep -E "DEPLOYMENT" "$LOGS/loop_timing.log" | tail -1
  fi
fi

# ------------------------------------------------------------------ demo
if want demo && [ "$DEMO" = 1 ]; then
  say "== demo: the nuscenes_session.gif render with the released ONNX"
  run $PY distill/student_session.py --onnx "$REL/student_$NAME.onnx" --scene 0 --gpu "$G0" \
    --out "outputs/student_session_$NAME" --gif "outputs/student_session_$NAME/nuscenes_session_student.gif"
fi
say "done: $NAME"
