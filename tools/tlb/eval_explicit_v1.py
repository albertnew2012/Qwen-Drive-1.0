#!/usr/bin/env python
"""The explicit answer: what kind of lane am I in, and what is ITS traffic light?

Composes the two capabilities that now work separately:
  lane type   the fine-tuned lane-type LoRA reading the arrow in the ego's lane
  colour      the detect -> associate -> colour pipeline (97.85% where a light exists)

Both halves are scored on their own and jointly, because a joint number alone hides which
half fails. Evaluated on the frames where the lane graph knows both, which is the only
place the joint claim can be checked.
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE))
ROOT = BASE / 'data/OpenLane-V2'

LANE_Q = ("Look at the arrow painted on the road in your own lane. Which movements does "
          "your lane allow? Reply with all that apply from: left, straight, right.")
MOVES = ['left', 'straight', 'right']
PAT = {'left': re.compile(r'\bleft\b', re.I),
       'straight': re.compile(r'\b(straight|through|ahead)\b', re.I),
       'right': re.compile(r'\bright\b', re.I)}


def parse_moves(t):
    return sorted(m for m in MOVES if PAT[m].search(t or ''))


def boot(items, fn, iters=2000, seed=0):
    by = defaultdict(list)
    for r in items:
        by[r['ep']].append(r)
    eps = list(by)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(iters):
        p = rng.choice(len(eps), len(eps), replace=True)
        v.append(fn([x for i in p for x in by[eps[i]]]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/tlb/joint_val.jsonl')
    ap.add_argument('--lane-lora', default='outputs/tlb/lora_lanetype_v1.pt')
    ap.add_argument('--colour-src', default='outputs/tlb/ens_val.json',
                    help='pipeline predictions, used for the colour half')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--out', default='outputs/tlb/explicit_val.json')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    eid = {}; n = 0
    for s, g in groupby(rows, key=lambda r: r['segment']):
        for k, gg in groupby(list(g), key=lambda r: r['joint']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1
    colour = {x['key']: x['pred'] for x in json.loads((BASE / args.colour_src).read_text())}

    from qwen_drive import QwenDriveForPlanning
    from training.lora import apply_lora
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    ck = torch.load(BASE / args.lane_lora, map_location='cpu')
    apply_lora(model.vlm, ck['args'].get('rank', 32), ck['args'].get('alpha', 64))
    model.vlm.load_state_dict({k: v.to(model.device) for k, v in ck['lora'].items()},
                              strict=False)
    model.eval()
    print(f"  lane-type LoRA r={ck['args'].get('rank')} loaded", flush=True)

    out = []
    miss = 0
    for i, r in enumerate(rows):
        key = f"{r['segment']}/{r['timestamp']}"
        txt = model.generate_text([str(ROOT / r['image'])], LANE_Q, max_new_tokens=24).text
        lt = parse_moves(txt)
        cp = colour.get(key)
        if cp is None:
            miss += 1
        out.append({'key': key, 'ep': eid[key],
                    'gt_lane': sorted(r['allow']), 'pred_lane': lt,
                    'gt_colour': r['label'], 'pred_colour': cp,
                    'gt_joint': r['joint'], 'raw': txt.strip()[:40]})
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(rows)}", flush=True)

    f_lane = lambda it: sum(x['pred_lane'] == x['gt_lane'] for x in it) / len(it)
    f_col = lambda it: sum(x['pred_colour'] == x['gt_colour'] for x in it) / len(it)
    f_joint = lambda it: sum(x['pred_lane'] == x['gt_lane']
                             and x['pred_colour'] == x['gt_colour'] for x in it) / len(it)
    print(f"\n===== EXPLICIT: lane type + its traffic light =====")
    print(f"  {len(out)} frames, {len({x['ep'] for x in out})} episodes"
          + (f"   ({miss} without a pipeline colour)" if miss else ""))
    for name, fn in (('lane type exact', f_lane), ('light colour', f_col),
                     ('BOTH correct', f_joint)):
        lo, hi = boot(out, fn)
        print(f"  {name:18s} {fn(out):6.1%}  [{lo:.1%},{hi:.1%}]")
    ok = [x for x in out if x['pred_lane'] == x['gt_lane'] and x['pred_colour'] == x['gt_colour']]
    print(f"\n  example correct answers:")
    for x in ok[:5]:
        print(f"    \"I am in a {'+'.join(x['pred_lane'])} lane and its light is "
              f"{x['pred_colour']}\"   (truth: {x['gt_joint']})")
    bad = [x for x in out if x['pred_lane'] != x['gt_lane']]
    print(f"\n  lane-type errors dominate: {len(bad)} of {len(out) - len(ok)} total errors")
    print("  most common:", Counter((','.join(x['gt_lane']), ','.join(x['pred_lane']))
                                    for x in bad).most_common(4))
    (BASE / args.out).write_text(json.dumps(out, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
