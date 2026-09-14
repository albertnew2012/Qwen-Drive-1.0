#!/usr/bin/env bash
# Re-run every training check end to end. Each must PASS.
set -u
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src:. CUDA_HOME=/usr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
pass=0; fail=0
run () {  # name, expected-marker, command...
  local name="$1"; local marker="$2"; shift 2
  echo; echo "######## $name"
  if "$@" 2>&1 | tee /tmp/_vt.log | grep -E "$marker" ; then
    if grep -qE "PASS|differentiable end to end" /tmp/_vt.log; then
      echo "   -> PASS"; pass=$((pass+1)); else echo "   -> FAIL"; fail=$((fail+1)); fi
  else echo "   -> FAIL (marker not found)"; fail=$((fail+1)); fi
}
echo "=== 0. the shipped kernels genuinely cannot backward ==="
$PY training/test_gradients.py --skip-patch --dtype bfloat16 2>&1 | \
  grep -q "NotImplementedError" && echo "   -> confirmed: no backward without the patch" \
  || echo "   -> UNEXPECTED: it did not fail"

run "1. gradients flow once patched" "PASS|non-zero gradient" \
    $PY training/test_gradients.py --dtype bfloat16
run "2. stage 1 - perception head overfit" "OVERFIT TEST|loss .*->" \
    $PY training/train_perception.py --steps 60 --overfit --out outputs/verify_s1
run "3. stage 3 - planning expert overfit" "OVERFIT TEST|mean loss" \
    $PY training/train_planner.py --steps 120 --overfit --scratch --out outputs/verify_s3
run "4. stage 2 - joint perception + VLM (LoRA)" "JOINT TEST|L_perc .*->" \
    $PY training/train_joint.py --steps 8 --overfit --out outputs/verify_s2
echo; echo "################ TRAINING VERIFICATION: $pass passed, $fail failed"
exit $fail
