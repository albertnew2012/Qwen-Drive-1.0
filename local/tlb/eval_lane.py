#!/usr/bin/env python
"""Does Qwen-Drive know which lane it is in?

The secondary capability: not just the light, but "I am in lane 2 of 3 from the left".
Ground truth comes from the same lane graph as the traffic-light work -- every centerline
running the ego's way at the ego's longitudinal position, ranked left to right (y is
positive left), with the ego's own lane located among them.

Scored three ways, because they fail differently:
  count    is the number of lanes right
  index    is the ego's position right
  both     both right, which is what a planner would actually need

Single-lane roads are reported separately: "1 of 1" is answerable without seeing anything,
so mixing them in flatters the score.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
ROOT = BASE / 'data/OpenLane-V2'

QUESTIONS = {
    'of': ("You are driving this vehicle. Counting lanes that travel in your direction "
           "from the left, which lane are you in and how many are there? "
           "Answer in exactly this form: <lane> of <total>"),
    'count': ("How many lanes travel in your direction on this road? "
              "Answer with a single number."),
    # The base model over-counts by about one, which is what counting the oncoming side
    # would look like. Say so explicitly and see if the bias goes away.
    'same_dir': ("You are driving this vehicle. Count ONLY the lanes going the same "
                 "direction as you; do not count oncoming lanes on the other side of the "
                 "road. Which of those lanes are you in, counting from the left, and how "
                 "many are there? Answer with just two numbers in the form: 2 of 3"),
    # Anchoring probe: identical question, different worked example. If the answer tracks
    # the example rather than the road, the format example is doing the work.
    'anchor14': ("You are driving this vehicle. Count ONLY the lanes going the same "
                 "direction as you; do not count oncoming lanes on the other side of the "
                 "road. Which of those lanes are you in, counting from the left, and how "
                 "many are there? Answer with just two numbers in the form: 1 of 4"),
    'anchor_none': ("You are driving this vehicle. Count ONLY the lanes going the same "
                    "direction as you; do not count oncoming lanes. Reply with your lane "
                    "number, then the word 'of', then the total number of such lanes."),
}
# The model answers by filling the template literally -- "<1> of <2>" -- so the brackets
# have to be tolerated, or 97% of correct answers are scored as unparseable.
PAT = re.compile(r'<?\s*(\d+)\s*>?\s*(?:of|/)\s*<?\s*(\d+)\s*>?')
NUM = re.compile(r'\b(\d+)\b')


def parse_of(t):
    m = PAT.search(t or '')
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_count(t):
    # the 'count' question asks for the TOTAL, so it belongs in the total slot -- putting
    # it in the index slot scored lane-count as 0.0% and lane-index against the wrong field
    m = NUM.search(t or '')
    return (None, int(m.group(1))) if m else (None, None)


def boot(items, stat, iters=2000, seed=0):
    by = defaultdict(list)
    for r in items:
        by[r['ep']].append(r)
    eps = list(by)
    if not eps:
        return 0., 0.
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(iters):
        p = rng.choice(len(eps), len(eps), replace=True)
        v.append(stat([x for i in p for x in by[eps[i]]]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/tlb/lane_val.jsonl')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--question', default='of', choices=list(QUESTIONS))
    ap.add_argument('--lora', default='')
    ap.add_argument('--stride', type=int, default=8,
                    help='keep 1 frame in N: neighbours are the same scene at 2 Hz')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='outputs/tlb/lane_eval.json')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    rows = rows[::args.stride]
    if args.limit:
        rows = rows[:args.limit]
    ep = {}
    n = 0
    for seg, g in groupby(rows, key=lambda r: r['segment']):
        for k, gg in groupby(list(g), key=lambda r: (r['lane_index'], r['n_lanes'])):
            for r in gg:
                ep[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1

    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if args.lora:
        sys.path.insert(0, str(BASE))
        from training.lora import apply_lora
        ck = torch.load(BASE / args.lora, map_location='cpu')
        apply_lora(model.vlm, ck['args'].get('rank', 16), ck['args'].get('alpha', 32))
        model.vlm.load_state_dict({k: v.to(model.device) for k, v in ck['lora'].items()},
                                  strict=False)
        model.eval()

    q = QUESTIONS[args.question]
    q = q.replace("Answer in exactly this form: <lane> of <total>",
                  "Answer with just two numbers in the form: 2 of 3")
    res = []
    t0 = time.time()
    for i, r in enumerate(rows):
        txt = model.generate_text([str(ROOT / r['image'])], q, max_new_tokens=24).text
        # pick the parser by ANSWER SHAPE, not by question name: 'same_dir' also asks
        # for "<lane> of <total>" and was being routed to the single-number parser.
        idx, tot = (parse_count if args.question == 'count' else parse_of)(txt)
        res.append({'key': f"{r['segment']}/{r['timestamp']}", 'gt_index': r['lane_index'],
                    'gt_total': r['n_lanes'], 'pred_index': idx, 'pred_total': tot,
                    'raw': txt.strip()[:50], 'ep': ep[f"{r['segment']}/{r['timestamp']}"],
                    'multi': r['n_lanes'] > 1})
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    def f_count(it):
        v = [x for x in it if x['pred_total'] is not None]
        return sum(x['pred_total'] == x['gt_total'] for x in v) / len(it) if it else 0.
    def f_index(it):
        v = [x for x in it if x['pred_index'] is not None]
        return sum(x['pred_index'] == x['gt_index'] for x in v) / len(it) if it else 0.
    def f_both(it):
        v = [x for x in it if x['pred_index'] is not None and x['pred_total'] is not None]
        return sum(x['pred_index'] == x['gt_index'] and x['pred_total'] == x['gt_total']
                   for x in v) / len(it) if it else 0.

    print(f"\n===== ego lane position ({args.question}) =====")
    print(f"  {len(res)} frames, {len({r['ep'] for r in res})} episodes")
    for name, sub in (('all', res), ('multi-lane only', [r for r in res if r['multi']])):
        if not sub:
            continue
        for lab, fn in (('lane count', f_count), ('lane index', f_index), ('both', f_both)):
            lo, hi = boot(sub, fn)
            print(f"  {name:16s} {lab:11s} {fn(sub):6.1%}  [{lo:.1%},{hi:.1%}]  n={len(sub)}")
    print("  gt totals:", Counter(r['gt_total'] for r in res).most_common())
    print("  predicted totals:", Counter(r['pred_total'] for r in res).most_common(6))
    print("  unparseable:", sum(r['pred_index'] is None for r in res))
    print("  sample:", [r['raw'] for r in res[:4]])
    (BASE / args.out).write_text(json.dumps(res, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
