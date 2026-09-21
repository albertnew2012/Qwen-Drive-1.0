#!/usr/bin/env python
"""Score the gate, then the decomposed VQA it enables.

  gate    yes/no, does a light govern the ego lane
  colour  the v1 LoRA's 3-way answer, which is 94.7% where a light exists but has no way
          to say "none" -- the gate supplies that

Reported against the single-model alternatives so the decomposition earns its place or
does not.
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
GATE_Q = ("Is there a traffic light that controls the lane the ego vehicle is in? "
          "Traffic lights for cross traffic or for other lanes do not count. "
          "Answer with one word: yes or no.")
YES = re.compile(r'\byes\b', re.I)
NO = re.compile(r'\bno\b', re.I)


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
    ap.add_argument('--gt', default='data/tlb/full_val.jsonl')
    ap.add_argument('--lora', default='outputs/tlb/lora_gate_v1.pt')
    ap.add_argument('--colour-src', default='outputs/tlb/base_assoc_full.json')
    ap.add_argument('--colour-ft', default='outputs/tlb/ft_nb_assoc_full.json')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out', default='outputs/tlb/gate_val.json')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    rows = rows[::args.stride]
    eid = {}; n = 0
    for s, g in groupby(rows, key=lambda r: r['segment']):
        for k, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1
    col = {x['key']: x['pred'] for x in json.loads((BASE / args.colour_ft).read_text())}

    from qwen_drive import QwenDriveForPlanning
    from training.lora import apply_lora
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    ck = torch.load(BASE / args.lora, map_location='cpu')
    apply_lora(model.vlm, ck['args'].get('rank', 16), ck['args'].get('alpha', 32))
    model.vlm.load_state_dict({k: v.to(model.device) for k, v in ck['lora'].items()},
                              strict=False)
    model.eval()

    out = []
    for i, r in enumerate(rows):
        key = f"{r['segment']}/{r['timestamp']}"
        t = model.generate_text([str(ROOT / r['image'])], GATE_Q, max_new_tokens=8).text
        g = 'yes' if (YES.search(t) and not NO.search(t)) else ('no' if NO.search(t) else 'yes')
        c = col.get(key)
        out.append({'key': key, 'ep': eid[key], 'gt': r['label'], 'why': r['why'],
                    'gate': g, 'gate_gt': 'no' if r['label'] == 'none' else 'yes',
                    'colour': c,
                    'pred': 'none' if g == 'no' else (c if c not in (None, 'none') else 'none')})
        if (i + 1) % 500 == 0:
            acc = sum(x['gate'] == x['gate_gt'] for x in out) / len(out)
            print(f"    {i+1}/{len(rows)}  gate {acc:.1%}", flush=True)

    f_gate = lambda it: sum(x['gate'] == x['gate_gt'] for x in it) / len(it)
    f_acc = lambda it: sum(x['pred'] == x['gt'] for x in it) / len(it)
    def macro(it):
        rs = []
        for c in ('red', 'green', 'yellow', 'none'):
            cl = [x for x in it if x['gt'] == c]
            if cl:
                rs.append(sum(x['pred'] == c for x in cl) / len(cl))
        return sum(rs) / len(rs) if rs else 0.
    print(f"\n===== GATE + colour, decomposed VQA =====")
    print(f"  {len(out)} frames, {len({x['ep'] for x in out})} episodes")
    lo, hi = boot(out, f_gate)
    print(f"  gate accuracy (yes/no)     {f_gate(out):6.1%}  [{lo:.1%},{hi:.1%}]")
    for w in ('not_ego', 'no_light', 'governed'):
        s = [x for x in out if x['why'] == w]
        if s:
            print(f"    on {w:9s} n={len(s):5d}  {f_gate(s):6.1%}")
    lo, hi = boot(out, f_acc)
    print(f"  decomposed VQA accuracy    {f_acc(out):6.1%}  [{lo:.1%},{hi:.1%}]   "
          f"macroR {macro(out):.1%}")
    sub = [x for x in out if x['gt'] != 'none']
    print(f"  ...on frames WITH a light  {f_acc(sub):6.1%}   (n={len(sub)})")
    ph = sum(1 for x in out if x['gt'] == 'none' and x['pred'] != 'none')
    ms = sum(1 for x in out if x['gt'] != 'none' and x['pred'] == 'none')
    print(f"  phantom {ph}   missed {ms}")
    print(f"\n  for comparison, single-model VQA on the same frames:")
    print(f"    v3 (4-way)               88.5%   macroR 78.8%   phantom 517  missed 161")
    print(f"    pipeline ensemble        90.1%   macroR 73.3%   phantom 296  missed 293")
    (BASE / args.out).write_text(json.dumps(out, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
