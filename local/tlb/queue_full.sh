#!/usr/bin/env bash
# Both methods over the WHOLE validation split (6019 frames), including the 78% where the
# right answer is "none". The abstention threshold is chosen on TRAIN segments first.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[e]val_lanetype.py" >/dev/null; do sleep 30; done

echo "[full 1/3] choose the abstention threshold on TRAIN segments (val untouched)"
$P - <<'PY'
import json, random
rows=[json.loads(l) for l in open('data/tlb/full_train.jsonl')]
segs=sorted({r['segment'] for r in rows}); random.Random(31).shuffle(segs)
hold=set(segs[:int(len(segs)*0.18)])
sub=[r for r in rows if r['segment'] in hold][::6]      # thin: 2 Hz neighbours are alike
with open('data/tlb/full_holdout.jsonl','w') as f:
    for r in sub: f.write(json.dumps(r)+'\n')
from collections import Counter
print(f"    holdout {len(sub)} frames, {len(hold)} segments, labels {Counter(r['label'] for r in sub).most_common()}")
PY
for T in 0.30 0.50 0.70; do
  $P local/tlb/pipeline_e2e.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --gov-thr $T \
     --gt data/tlb/full_holdout.jsonl --out outputs/tlb/full_cfg_t$T.json \
     > outputs/tlb/log_full_cfg_t$T.log 2>&1
  printf "    gov-thr=%-5s %s\n" $T "$(grep -E '^  all ' outputs/tlb/log_full_cfg_t$T.log|sed 's/^  all *//')"
done

echo "[full 2/3] PIPELINE over all 6019 val frames"
$P local/tlb/pipeline_e2e.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --gov-thr 0.50 \
   --gt data/tlb/full_val.jsonl --out outputs/tlb/full_pipeline.json \
   > outputs/tlb/log_full_pipeline.log 2>&1
grep -E "^  all |^  discriminative|confusion" outputs/tlb/log_full_pipeline.log

echo "[full 3/3] VQA over all 6019 val frames (fine-tuned, 'none' allowed)"
$P local/tlb/eval_vqa.py --src data/tlb/full_val.jsonl --cond full --question full \
   --lora outputs/tlb/lora_vqa_nb.pt --out outputs/tlb/full_vqa.json \
   > outputs/tlb/log_full_vqa.log 2>&1
grep -A 18 "===== full / full" outputs/tlb/log_full_vqa.log | head -20
echo "FULL QUEUE DONE"
