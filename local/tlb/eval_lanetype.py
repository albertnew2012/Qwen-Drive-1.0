#!/usr/bin/env python
"""Can Qwen-Drive tell what kind of lane it is in: left-turn, through, or right-turn?

Multi-label, because real lanes are often "through or left". Scored as:
  exact       the whole permitted set is right
  per-move    precision/recall for each of left / straight / right
  macro F1    mean over the three, so the 83%-prevalent "straight" cannot carry the score

Prompt note: the options are listed (left / straight / right) but no example ANSWER is
given. An earlier lane experiment showed the model copies a worked example -- "answer in
the form: 2 of 3" made it reply "of 3" on 87% of frames -- so a demonstration answer would
measure the prompt, not the road.
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
    'arrow': ("Look at the arrow painted on the road in your own lane. Which movements "
              "does your lane allow? Reply with all that apply from: left, straight, right."),
    'plain': ("You are driving this vehicle. Is your lane a left-turn lane, a through "
              "lane, a right-turn lane, or a combination? Reply with all that apply from: "
              "left, straight, right."),
}
MOVES = ['left', 'straight', 'right']
PAT = {'left': re.compile(r'\bleft\b', re.I),
       'straight': re.compile(r'\b(straight|through|ahead|forward)\b', re.I),
       'right': re.compile(r'\bright\b', re.I)}
NEG = re.compile(r'\b(no|not|cannot|can\'t|only)\s+(\w+\s+){0,2}(left|right|straight)\b', re.I)


def parse(t):
    t = (t or '').strip()
    return sorted(m for m in MOVES if PAT[m].search(t))


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


def f1(items, m):
    tp = sum(1 for r in items if m in r['gt'] and m in r['pred'])
    fp = sum(1 for r in items if m not in r['gt'] and m in r['pred'])
    fn = sum(1 for r in items if m in r['gt'] and m not in r['pred'])
    p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 2 * p * rc / max(p + rc, 1e-9), p, rc


def macro_f1(items):
    return sum(f1(items, m)[0] for m in MOVES) / 3


def exact(items):
    return sum(1 for r in items if r['gt'] == r['pred']) / len(items) if items else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/tlb/lanetype_val.jsonl')
    ap.add_argument('--question', default='arrow', choices=list(QUESTIONS))
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--lora', default='')
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='outputs/tlb/lanetype_eval.json')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    rows = rows[::args.stride]
    if args.limit:
        rows = rows[:args.limit]
    eid = {}; n = 0
    for seg, g in groupby(rows, key=lambda r: r['segment']):
        for k, gg in groupby(list(g), key=lambda r: r['type']):
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

    q = QUESTIONS[args.question]
    res = []
    t0 = time.time()
    for i, r in enumerate(rows):
        txt = model.generate_text([str(ROOT / r['image'])], q, max_new_tokens=32).text
        res.append({'key': f"{r['segment']}/{r['timestamp']}", 'gt': sorted(r['allow']),
                    'pred': parse(txt), 'type': r['type'], 'raw': txt.strip()[:60],
                    'ep': eid[f"{r['segment']}/{r['timestamp']}"]})
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(rows)}  {time.time()-t0:.0f}s  exact {exact(res):.1%}",
                  flush=True)

    print(f"\n===== ego lane type ({args.question}) =====")
    print(f"  {len(res)} frames, {len({r['ep'] for r in res})} episodes")
    lo, hi = boot(res, exact)
    mlo, mhi = boot(res, macro_f1)
    print(f"  exact set match  {exact(res):6.1%}  [{lo:.1%},{hi:.1%}]")
    print(f"  macro F1         {macro_f1(res):6.1%}  [{mlo:.1%},{mhi:.1%}]")
    for m in MOVES:
        f, p, rc = f1(res, m)
        base = sum(1 for r in res if m in r['gt']) / len(res)
        print(f"    {m:9s} F1 {f:6.1%}  P {p:6.1%}  R {rc:6.1%}   (present in {base:.1%} of frames)")
    print("  gt types   :", Counter(r['type'] for r in res).most_common(5))
    print("  pred sets  :", Counter(tuple(r['pred']) for r in res).most_common(5))
    print("  sample     :", [r['raw'][:34] for r in res[:3]])
    (BASE / args.out).write_text(json.dumps(res, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
