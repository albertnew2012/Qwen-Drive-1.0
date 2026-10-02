#!/usr/bin/env python
"""Score a run with error bars that respect how the data is actually correlated.

Frames are not independent: val's 1300 usable frames come from 64 segments and only 86
label-episodes (a maximal run of consecutive frames in one segment sharing one label),
median 12 frames each. Treating frames as independent would understate the interval by
roughly sqrt(12). Confidence intervals here come from a bootstrap that resamples whole
*episodes*, which is the unit that actually varies.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')


def load(gt_path, res_path):
    rows = [json.loads(l) for l in open(BASE / gt_path)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    # episode id
    eid = {}
    n = 0
    for seg, g in groupby(rows, key=lambda r: r['segment']):
        for lab, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1
    gt = {f"{r['segment']}/{r['timestamp']}": r for r in rows}
    res = json.loads((BASE / res_path).read_text())
    out = []
    for x in res:
        r = gt.get(x['key'])
        if r is None:
            continue
        gov_a = max((b[2]-b[0])*(b[3]-b[1]) for b in r['gov_boxes'])
        big_other = max(((b[2]-b[0])*(b[3]-b[1]) for b in r['other_boxes']), default=0)
        out.append({**x, 'ep': eid[x['key']],
                    'sal_defeat': bool(r['discriminative'] and big_other > gov_a),
                    'w': r['max_gov_wh'][0]})
    return out, n


def boot_ci(items, stat, iters=4000, seed=0):
    """Bootstrap over episodes, not frames."""
    by = defaultdict(list)
    for r in items:
        by[r['ep']].append(r)
    eps = list(by)
    if not eps:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        pick = rng.choice(len(eps), len(eps), replace=True)
        s = [x for i in pick for x in by[eps[i]]]
        vals.append(stat(s))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def acc(items):
    return sum(r['pred'] == r['gt'] for r in items) / len(items) if items else 0.0


def macro(items):
    rs = []
    for c in ('red', 'green', 'yellow'):
        cl = [r for r in items if r['gt'] == c]
        if cl:
            rs.append(sum(r['pred'] == c for r in cl) / len(cl))
    return sum(rs) / len(rs) if rs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--res', nargs='+', required=True)
    args = ap.parse_args()

    for rp in args.res:
        items, nep = load(args.gt, rp)
        print(f"\n===== {rp} =====")
        print(f"  {len(items)} frames over {len({r['ep'] for r in items})} episodes "
              f"(of {nep} in the GT file)")
        subsets = [
            ('all', lambda r: True),
            ('discriminative', lambda r: r['disc']),
            ('salience-defeating', lambda r: r['sal_defeat']),
            ('light <12px', lambda r: r['w'] < 12),
            ('light >=20px', lambda r: r['w'] >= 20),
        ]
        print(f"  {'subset':22s} {'frames':>7s} {'eps':>5s} {'acc':>7s} {'95% CI (episode boot)':>24s} {'macroR':>8s}")
        for name, sel in subsets:
            s = [r for r in items if sel(r)]
            if not s:
                print(f"  {name:22s}       0")
                continue
            lo, hi = boot_ci(s, acc)
            print(f"  {name:22s} {len(s):7d} {len({r['ep'] for r in s}):5d} "
                  f"{acc(s):7.1%}   [{lo:6.1%}, {hi:6.1%}]      {macro(s):7.1%}")
        print("  confusion (gt -> pred):")
        for g in ('red', 'green', 'yellow'):
            c = Counter(r['pred'] for r in items if r['gt'] == g)
            if c:
                print(f"      {g:7s} n={sum(c.values()):4d} -> {dict(c)}")
        bad = [r for r in items if r['pred'] != r['gt']]
        if bad:
            print(f"  {len(bad)} misses over {len({r['ep'] for r in bad})} episodes; first few:")
            for r in bad[:8]:
                print(f"      {r['key']}  gt={r['gt']:6s} pred={str(r['pred']):6s} "
                      f"w={r['w']:.0f}px disc={int(r['disc'])} raw={r['raw']!r}")


if __name__ == '__main__':
    main()
