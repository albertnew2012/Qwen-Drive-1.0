#!/usr/bin/env python
"""Causal temporal smoothing of per-frame colour predictions.

A traffic light's colour is stable for seconds at a time, but the model's answer is not:
in segment 11047 it alternates red/yellow across frames that are 0.5 s apart and visually
identical. Voting over a short *causal* window (current frame plus the previous k-1)
removes that flicker.

Causal on purpose -- it uses only past frames, so it is deployable, and it cannot leak
the future across a light change. Window is capped in seconds so a long red does not
smear across the transition to green.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')


def smooth(res, gt_rows, k=5, max_gap_s=1.5):
    """Majority vote over the current frame and the previous k-1 within one segment."""
    meta = {f"{r['segment']}/{r['timestamp']}": r for r in gt_rows}
    by_seg = defaultdict(list)
    for x in res:
        m = meta.get(x['key'])
        if m is None:
            continue
        by_seg[m['segment']].append((int(m['timestamp']), x))
    out = []
    for seg, items in by_seg.items():
        items.sort(key=lambda t: t[0])
        for i, (ts, x) in enumerate(items):
            win = []
            for j in range(i, -1, -1):
                tj, xj = items[j]
                if (ts - tj) / 1e6 > max_gap_s * k:      # timestamps are microseconds
                    break
                win.append(xj['pred'])
                if len(win) >= k:
                    break
            votes = Counter(p for p in win if p is not None)
            pred = votes.most_common(1)[0][0] if votes else x['pred']
            out.append({**x, 'pred': pred, 'pred_raw': x['pred'], 'window': len(win)})
    return out


def episodes(rows):
    rows = sorted(rows, key=lambda r: (r['segment'], r['timestamp']))
    eid = {}; n = 0
    for seg, g in groupby(rows, key=lambda r: r['segment']):
        for lab, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1
    return eid


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
        pick = rng.choice(len(eps), len(eps), replace=True)
        v.append(stat([x for i in pick for x in by[eps[i]]]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5, 7, 9])
    ap.add_argument('--save', default='')
    args = ap.parse_args()

    gt_rows = [json.loads(l) for l in open(BASE / args.gt)]
    eid = episodes(gt_rows)
    res = json.loads((BASE / args.res).read_text())
    meta = {f"{r['segment']}/{r['timestamp']}": r for r in gt_rows}

    print(f"{args.res}")
    print(f"  {'k':>3s} {'acc':>7s} {'95% CI':>16s} {'macroR':>8s} {'disc acc':>9s} {'disc macroR':>12s}")
    best = None
    for k in args.ks:
        sm = smooth(res, gt_rows, k=k)
        for x in sm:
            x['ep'] = eid.get(x['key'], -1)
            x['disc'] = meta[x['key']]['discriminative']
        d = [x for x in sm if x['disc']]
        lo, hi = boot(sm, acc)
        print(f"  {k:3d} {acc(sm):7.1%}  [{lo:5.1%},{hi:5.1%}] {macro(sm):8.1%} "
              f"{acc(d):9.1%} {macro(d):12.1%}")
        if best is None or acc(sm) > best[1]:
            best = (k, acc(sm), sm)
    if args.save and best:
        (BASE / args.save).write_text(json.dumps(best[2], indent=1))
        print(f"  saved k={best[0]} to {args.save}")


if __name__ == '__main__':
    main()
