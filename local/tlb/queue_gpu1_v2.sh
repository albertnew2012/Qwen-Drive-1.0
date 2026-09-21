#!/usr/bin/env bash
# GPU1: finish the full-val abstention threshold properly.
# The earlier run hardcoded gov-thr 0.50 while the holdout sweep preferred 0.70 and was
# still rising, so extend the sweep and then score val once at the chosen value.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[e]val_vqa.py" >/dev/null; do sleep 30; done

echo "[1/3] extend the holdout sweep (val NOT read)"
for T in 0.70 0.80 0.90 0.95; do
  $P local/tlb/pipeline_e2e.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --gov-thr $T \
     --gt data/tlb/full_holdout.jsonl --out outputs/tlb/full_cfg2_t$T.json \
     > outputs/tlb/log_full_cfg2_t$T.log 2>&1
  printf "  gov-thr=%-5s %s\n" $T "$(grep -E '^  all ' outputs/tlb/log_full_cfg2_t$T.log|sed 's/^  all *//')"
done

echo "[2/3] pick the best and score the FULL val once"
BEST=$($P - <<'PY'
import glob,re
best=(None,-1)
for f in glob.glob('outputs/tlb/log_full_cfg2_t*.log')+glob.glob('outputs/tlb/log_full_cfg_t*.log'):
    t=re.search(r'_t([0-9.]+)\.log',f).group(1)
    for l in open(f):
        if l.strip().startswith('all '):
            m=re.search(r'acc\s+([0-9.]+)%',l)
            if m and float(m.group(1))>best[1]: best=(t,float(m.group(1)))
print(best[0])
PY
)
echo "  chosen gov-thr = $BEST"
$P local/tlb/pipeline_e2e.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --gov-thr $BEST \
   --gt data/tlb/full_val.jsonl --out outputs/tlb/full_pipeline_v2.json \
   > outputs/tlb/log_full_pipeline_v2.log 2>&1
grep -E "^  all |^  light |confusion" outputs/tlb/log_full_pipeline_v2.log

echo "[3/3] full-val VQA summary"
grep -A 20 "===== full / full" outputs/tlb/log_full_vqa.log | head -22
echo "GPU1_V2 QUEUE DONE"
