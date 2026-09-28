#!/usr/bin/env bash
# Build the downloadable release set from a training checkpoint:
#   slim fp32 .pt (EMA weights + cfg), ONNX, zip, 29 MiB chunks (chat upload cap is 30 MiB), MD5s. fp32 only: never downgrade the weights.
#   local/distill/scripts/make_release.sh CKPT TAG [OUT_DIR] [GPU]
#   e.g. make_release.sh outputs/distill/r2_long/student.pt r2_long_final outputs/student_release_final 3
set -euo pipefail
CKPT=${1:?checkpoint}; TAG=${2:?tag}; OUT=${3:-outputs/student_release_$TAG}; GPU=${4:-0}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "refusing to overwrite existing release folder $OUT -- pick a new TAG/OUT_DIR"; exit 2; fi
mkdir -p "$OUT"
export PYTHONPATH=src:.:local; [ -f export_onnx/env_gpu.sh ] && source export_onnx/env_gpu.sh
.venv/bin/python - "$CKPT" "$OUT" "$TAG" <<'PY'
import sys, torch
ckpt, out, tag = sys.argv[1:4]
ck = torch.load(ckpt, map_location="cpu")
sd = ck["ema"] if ck.get("ema") else ck["model"]
note = f"{tag}: EMA weights of step {ck.get('step')} from {ckpt}"
torch.save({"model": sd, "cfg": ck["cfg"], "step": ck.get("step"), "note": note}, f"{out}/student_{tag}.pt")
print(f"  slim fp32 checkpoint written for step {ck.get('step')}")
PY
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python local/distill/export_student.py --ckpt "$OUT/student_$TAG.pt" --out "$OUT/student_$TAG.onnx" --report "$OUT/onnx_timing.json" 2>&1 | grep -E "loaded|exported|Hz"
cd "$OUT"
rm -f student_weights_$TAG.zip
zip -q -1 student_weights_$TAG.zip student_$TAG.pt student_$TAG.onnx
md5sum student_$TAG.pt student_$TAG.onnx student_weights_$TAG.zip > MD5SUMS
for Z in student_weights_$TAG.zip; do
  D=chunks_${Z%.zip}; mkdir -p "$D"; split -b 29m -d -a 2 "$Z" "$D/$Z.part"; cp MD5SUMS "$D/"
  echo "  $Z -> $(ls "$D" | grep -c part) chunks in $OUT/$D   (cat $Z.part* > $Z)"
done
cat MD5SUMS
