#!/usr/bin/env python
"""Task 1 explicit: ask for the ego's lane type AND its light colour in one answer.

Scored three ways, because a joint answer can fail in two different places:
    colour only   did it get the light right (comparable to every other Task 1 number)
    type only     did it get the lane right
    joint         both, which is the claim being made

159 val frames over 22 segments: enough to show the capability, far too few to quote a
rate with a tight interval. Reported with episode-bootstrapped bounds so that is visible.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
ROOT = BASE / 'data/OpenLane-V2'

QUESTIONS = {
    'joint': ("You are driving this vehicle. Answer two things about YOUR OWN lane: "
              "first which movements it allows (any of: left, straight, right), then the "
              "colour of the traffic light controlling it (red, yellow, or green). "
              "Reply as: movements = ...; light = ..."),
    'joint_terse': ("For the lane you are driving in, list the movements it allows "
                    "(left/straight/right) and the colour of its traffic light. "
                    "Reply as: movements = ...; light = ..."),
}
MOVES = ['left', 'straight', 'right']
MPAT = {'left': re.compile(r'\bleft\b', re.I),
        'straight': re.compile(r'\b(straight|through|ahead|forward)\b', re.I),
        'right': re.compile(r'\bright\b', re.I)}
CPAT = re.compile(r'\b(red|green|yellow|amber)\b', re.I)


def parse(t):
    t = t or ''
    # split on the light clause so "right" in a movement list is not read as a colour cue
    mpart, lpart = t, t
    m = re.search(r'light\s*=', t, re.I)
    if m:
        mpart, lpart = t[:m.start()], t[m.start():]
    mv = sorted(k for k in MOVES if MPAT[k].search(mpart))
    c = CPAT.search(lpart) or CPAT.search(t)
    col = (c.group(1).lower() if c else None)
    if col == 'amber':
        col = 'yellow'
    return mv, col


def boot(items, stat, iters=3000, seed=0):
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
    ap.add_argument('--gt', default='data/tlb/joint_val.jsonl')
    ap.add_argument('--question', default='joint', choices=list(QUESTIONS))
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--lora', default='')
    ap.add_argument('--out', default='outputs/tlb/joint_eval.json')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    eid = {}; n = 0
    for seg, g in groupby(rows, key=lambda r: r['segment']):
        for k, gg in groupby(list(g), key=lambda r: r['joint']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
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
        print(f"  loaded LoRA from {args.lora}", flush=True)

    q = QUESTIONS[args.question]
    res = []
    t0 = time.time()
    for i, r in enumerate(rows):
        txt = model.generate_text([str(ROOT / r['image'])], q, max_new_tokens=48).text
        mv, col = parse(txt)
        res.append({'key': f"{r['segment']}/{r['timestamp']}", 'gt_type': r['allow'],
                    'gt_colour': r['label'], 'pred_type': mv, 'pred_colour': col,
                    'raw': txt.strip()[:70], 'ep': eid[f"{r['segment']}/{r['timestamp']}"]})
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    f_col = lambda it: sum(x['pred_colour'] == x['gt_colour'] for x in it)/len(it) if it else 0.
    f_typ = lambda it: sum(x['pred_type'] == x['gt_type'] for x in it)/len(it) if it else 0.
    f_j = lambda it: sum(x['pred_colour'] == x['gt_colour'] and x['pred_type'] == x['gt_type']
                         for x in it)/len(it) if it else 0.
    print(f"\n===== Task 1 explicit: lane type + light colour ({args.question}) =====")
    print(f"  {len(res)} frames, {len({r['ep'] for r in res})} episodes")
    for name, fn in (('colour only', f_col), ('lane type only', f_typ), ('JOINT', f_j)):
        lo, hi = boot(res, fn)
        print(f"  {name:15s} {fn(res):6.1%}  [{lo:.1%},{hi:.1%}]")
    print("  gt joint  :", Counter(f"{'+'.join(r['gt_type'])}|{r['gt_colour']}" for r in res).most_common(5))
    print("  pred joint:", Counter(f"{'+'.join(r['pred_type'])}|{r['pred_colour']}" for r in res).most_common(5))
    print("  sample    :", [r['raw'][:52] for r in res[:2]])
    (BASE / args.out).write_text(json.dumps(res, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
