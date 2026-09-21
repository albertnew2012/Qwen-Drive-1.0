#!/usr/bin/env python
"""Colour accuracy given a perfect box: the ceiling for any detect-then-classify pipeline.

Takes the ground-truth governing box for each usable val frame, crops it, and asks the
colour head for the frame's answer. Association is handed over for free, so whatever this
scores is the best a detector-driven pipeline could do on colour alone -- and the gap to
the VQA baseline is what better resolution buys.

Where a frame has several governing lights, the largest box decides, with ties broken by
confidence. That mirrors what the full pipeline will do at inference.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
import sys
sys.path.insert(0, str(BASE / 'local/tlb'))
from train_colour import build, COLOURS                       # noqa: E402
from extract_crops import crop_one                            # noqa: E402

ROOT = BASE / 'data/OpenLane-V2'


def acc(it):
    return sum(r['pred'] == r['gt'] for r in it) / len(it) if it else 0.0


def macro(it):
    rs = []
    for c in ('red', 'green', 'yellow'):
        cl = [r for r in it if r['gt'] == c]
        if cl:
            rs.append(sum(r['pred'] == c for r in cl) / len(cl))
    return sum(rs) / len(rs) if rs else 0.0


def boot(items, stat, iters=3000, seed=0):
    by = defaultdict(list)
    for r in items:
        by[r['ep']].append(r)
    eps = list(by)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(iters):
        p = rng.choice(len(eps), len(eps), replace=True)
        v.append(stat([x for i in p for x in by[eps[i]]]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/tlb/colour_head.pt')
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--out', default='outputs/tlb/colour_oracle.json')
    args = ap.parse_args()

    device = 'cuda'
    ck = torch.load(BASE / args.ckpt, map_location='cpu')
    model = build().to(device)
    model.load_state_dict(ck['model']); model.eval()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    eid = {}; n = 0
    for seg, g in groupby(rows, key=lambda r: r['segment']):
        for lab, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1

    out = []
    buf, meta = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            p = torch.softmax(model(x).float(), 1).cpu().numpy()
        for pr, m in zip(p, meta):
            # the frame's answer comes from the largest governing box.
            # `pred` is the plain 4-way argmax; `pred_rgy` restricts to the three real
            # colours, which is the right call here because a *governing* light is
            # annotated 'unknown' in under 5% of frames.
            rgy = np.array([pr[1], pr[2], pr[3]])
            out.append({**m, 'pred': COLOURS[int(pr.argmax())],
                        'pred_rgy': ['red', 'green', 'yellow'][int(rgy.argmax())],
                        'conf': float(pr.max()), 'conf_rgy': float(rgy.max() / rgy.sum()),
                        'p_rgy': {c: float(v) for c, v in
                                  zip(('red', 'green', 'yellow'), rgy)}})
        buf.clear(); meta.clear()

    for r in rows:
        im = Image.open(ROOT / r['image']).convert('RGB')
        b = max(r['gov_boxes'], key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        c = crop_one(im, b)
        t = torch.from_numpy(np.asarray(c, np.uint8).copy()).permute(2, 0, 1).float() / 255.
        buf.append((t - 0.5) / 0.5)
        meta.append({'key': f"{r['segment']}/{r['timestamp']}", 'gt': r['label'],
                     'disc': r['discriminative'], 'w': r['max_gov_wh'][0],
                     'ep': eid[f"{r['segment']}/{r['timestamp']}"]})
        if len(buf) >= args.bs:
            flush()
    flush()

    print(f"oracle-box colour head on {len(out)} val frames, "
          f"{len({o['ep'] for o in out})} episodes")
    for key in ('pred', 'pred_rgy'):
        for o in out:
            o['pred'] = o[key] if key == 'pred_rgy' else o.get('_p4', o['pred'])
            if key == 'pred':
                o['_p4'] = o['pred']
        lab = '4-way argmax' if key == 'pred' else 'argmax over red/green/yellow'
        lo, hi = boot(out, acc)
        print(f"  {lab:32s} acc {acc(out):6.2%} [{lo:.1%},{hi:.1%}]  macroR {macro(out):6.2%}")
    for o in out:
        o['pred'] = o['pred_rgy']
    print("  (subset table below uses argmax over red/green/yellow)")
    for name, sel in (('all', lambda r: True),
                      ('discriminative', lambda r: r['disc']),
                      ('light <12px', lambda r: r['w'] < 12),
                      ('light 12-20px', lambda r: 12 <= r['w'] < 20),
                      ('light >=20px', lambda r: r['w'] >= 20)):
        s = [r for r in out if sel(r)]
        if not s:
            continue
        lo, hi = boot(s, acc)
        print(f"  {name:16s} n={len(s):5d} eps={len({r['ep'] for r in s}):3d}  "
              f"acc {acc(s):6.2%}  [{lo:.1%},{hi:.1%}]  macroR {macro(s):6.2%}")
    print("  confusion:", {g: dict(Counter(r['pred'] for r in out if r['gt'] == g))
                           for g in ('red', 'green', 'yellow')})
    # abstain when unsure: precision/coverage trade
    for t in (0.0, 0.5, 0.7, 0.9, 0.95):
        s = [r for r in out if r['conf'] >= t and r['pred'] != 'unknown']
        if s:
            print(f"  conf>={t:.2f}: coverage {len(s)/len(out):6.1%}  precision {acc(s):6.2%}")
    (BASE / args.out).write_text(json.dumps(out, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
