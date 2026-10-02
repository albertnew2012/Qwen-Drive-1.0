#!/usr/bin/env python
"""Rank candidate questions, and test whether any of them actually associates.

Headline is macro-recall, not accuracy: the label mix is ~55% green, and the model has a
green bias, so accuracy rewards guessing the prior. Intervals resample whole episodes.

The decisive column is the paired comparison against the association-free controls on the
*discriminative* frames -- the ones where another visible light disagrees with the ego's.
A prompt that does not beat `salient` there is not associating, however high it scores.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')


def episodes(gt_path):
    rows = [json.loads(l) for l in open(BASE / gt_path)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    eid = {}
    n = 0
    for seg, g in groupby(rows, key=lambda r: r['segment']):
        for lab, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1
    return eid


def acc(items):
    return sum(r['pred'] == r['gt'] for r in items) / len(items) if items else 0.0


def macro(items):
    rs = []
    for c in ('red', 'green', 'yellow'):
        cl = [r for r in items if r['gt'] == c]
        if cl:
            rs.append(sum(r['pred'] == c for r in cl) / len(cl))
    return sum(rs) / len(rs) if rs else 0.0


def boot(items, stat, iters=3000, seed=0):
    by = defaultdict(list)
    for r in items:
        by[r['ep']].append(r)
    eps = list(by)
    if not eps:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(iters):
        pick = rng.choice(len(eps), len(eps), replace=True)
        v.append(stat([x for i in pick for x in by[eps[i]]]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_delta(a, b, sel, stat, iters=3000, seed=1):
    """Bootstrap the difference stat(a)-stat(b) on the same episodes."""
    ka = {r['key']: r for r in a if sel(r)}
    kb = {r['key']: r for r in b if sel(r)}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        return 0.0, (0.0, 0.0)
    by = defaultdict(list)
    for k in keys:
        by[ka[k]['ep']].append(k)
    eps = list(by)
    rng = np.random.default_rng(seed)
    d = stat([ka[k] for k in keys]) - stat([kb[k] for k in keys])
    vals = []
    for _ in range(iters):
        pick = rng.choice(len(eps), len(eps), replace=True)
        ks = [k for i in pick for k in by[eps[i]]]
        vals.append(stat([ka[k] for k in ks]) - stat([kb[k] for k in ks]))
    return d, (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='outputs/tlb/prompts')
    ap.add_argument('--gt', default='data/tlb/dev.jsonl')
    ap.add_argument('--cond', default='hires')
    args = ap.parse_args()

    eid = episodes(args.gt)
    runs = {}
    for p in sorted((BASE / args.dir).glob(f'*_{args.cond}.json')):
        d = json.loads(p.read_text())
        for r in d['res']:
            r['ep'] = eid.get(r['key'], -1)
        runs[d['name']] = d
    if not runs:
        print('no results yet'); return

    print(f"{len(runs)} prompts, {len(next(iter(runs.values()))['res'])} frames, "
          f"{len({r['ep'] for r in next(iter(runs.values()))['res']})} episodes\n")
    hdr = (f"  {'prompt':20s} {'kind':8s} {'acc':>6s} {'macroR':>7s} "
           f"{'macroR 95% CI':>16s} | {'disc acc':>8s} {'disc macroR':>11s} {'CI':>16s}")
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    rank = []
    for name, d in runs.items():
        res = d['res']; disc = [r for r in res if r['disc']]
        lo, hi = boot(res, macro)
        dlo, dhi = boot(disc, macro)
        rank.append((macro(disc), macro(res), name))
        print(f"  {name:20s} {d['kind']:8s} {acc(res):6.1%} {macro(res):7.1%} "
              f"  [{lo:5.1%},{hi:5.1%}] | {acc(disc):8.1%} {macro(disc):11.1%} "
              f"  [{dlo:5.1%},{dhi:5.1%}]")

    print("\n  prediction distribution (green bias shows up here):")
    for name, d in runs.items():
        c = Counter(r['pred'] for r in d['res'])
        print(f"    {name:20s} {dict(c)}")
    gt = Counter(r['gt'] for r in next(iter(runs.values()))['res'])
    print(f"    {'GROUND TRUTH':20s} {dict(gt)}")

    ctrls = [n for n, d in runs.items() if d['kind'] == 'control']
    if ctrls:
        print(f"\n  paired vs controls on DISCRIMINATIVE frames (macro-recall difference,")
        print(f"  episode-bootstrapped; a CI excluding 0 means the prompt really associates):")
        for name, d in runs.items():
            if d['kind'] == 'control':
                continue
            for c in ctrls:
                delta, (lo, hi) = paired_delta(d['res'], runs[c]['res'],
                                               lambda r: r['disc'], macro)
                sig = '  *' if (lo > 0 or hi < 0) else ''
                print(f"    {name:20s} - {c:12s} {delta:+7.1%}  [{lo:+6.1%},{hi:+6.1%}]{sig}")

    print("\n  ranked by discriminative macro-recall:")
    for m, ma, name in sorted(rank, reverse=True):
        print(f"    {name:20s} disc macroR {m:6.1%}   overall macroR {ma:6.1%}")


if __name__ == '__main__':
    main()
