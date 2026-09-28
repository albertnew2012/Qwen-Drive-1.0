# Waits (inside this script) for the UniOcc train-split download, converts the labels into
# the student grid, syncs them into both local frame copies, recomputes the class stats.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. PATH="$PWD/.venv/bin:$PATH"
until grep -q "^DONE" outputs/logs/occ3d_uniocc_train.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] train-split download finished: $(grep '^DONE' outputs/logs/occ3d_uniocc_train.log)"
.venv/bin/python local/distill/occ3d_labels.py --split train 2>&1 | grep -vE "Warning" | tail -2
for D in ${DISTILL_CACHE:-/local/$USER/distill}/frames ${DISTILL_CACHE:-/local/$USER/distill}/frames_real; do
  rsync -a --include='*/' --include='occ3d.npz' --exclude='*' data/distill/frames/ $D/ && echo "$D: $(find $D -maxdepth 2 -name occ3d.npz | wc -l) occ3d files"; done
.venv/bin/python - <<'EOF'
import numpy as np, json
ss = json.load(open("data/distill/scene_split.json")); counts = np.zeros(10, np.int64); n = 0
for tok in ss["train_tokens"] + ss["val_tokens"]:
    try: o = np.load(f"data/distill/frames/{tok}/occ3d.npz")
    except FileNotFoundError: continue
    counts += np.bincount(o["occ"][o["mask"] > 0].astype(np.int64), minlength=10)[:10]; n += 1
np.save("data/distill/occ3d_freq.npy", counts); print(f"occ3d_freq.npy from {n} frames:", [f"{x:.4f}" for x in counts / counts.sum()])
EOF
echo "OCC3D_TRAIN_LABELS_DONE"
