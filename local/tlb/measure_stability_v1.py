#!/usr/bin/env python
"""How stable is the ego-light choice from frame to frame, and what does stability cost?

Two things are reported because they trade against each other:
  stability  how often the ego-light SET changes between consecutive frames of a segment,
             measured by Jaccard overlap of the boxes (matched by IoU, not index -- the
             detector's ordering is not stable)
  accuracy   the colour answer, which is what the stability is for

A set-valued answer is the honest target: 80% of val frames have more than one governing
light, so a single pick is arbitrary among equals and flips for no reason.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from itertools import groupby

import numpy as np

BASE = '/home/albert/Desktop/Qwen-Drive-1.0/'


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0.0, x2-x1) * max(0.0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/ua if ua > 0 else 0.0


def set_jaccard(A, B, thr=0.3):
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    used = set(); inter = 0
    for a in A:
        for k, b in enumerate(B):
            if k in used:
                continue
            if iou(a, b) >= thr:
                used.add(k); inter += 1; break
    return inter / (len(A) + len(B) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', nargs='+', required=True)
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    args = ap.parse_args()
    gt = {f"{r['segment']}/{r['timestamp']}": r
          for r in (json.loads(l) for l in open(BASE + args.gt))}
    print(f"  {'run':34s} {'acc':>7s} {'set-Jaccard':>12s} {'set changed':>12s} "
          f"{'colour flips':>13s} {'mean |ego|':>11s}")
    for p in args.res:
        d = json.load(open(BASE + p))
        for x in d:
            m = gt.get(x['key'])
            x['_seg'] = x['key'].split('/')[0]
            x['_ts'] = int(x['key'].split('/')[1])
        d.sort(key=lambda x: (x['_seg'], x['_ts']))
        acc = sum(x['pred'] == x['gt'] for x in d) / len(d)
        jac, chg, flip, n, sizes = [], 0, 0, 0, []
        for seg, g in groupby(d, key=lambda x: x['_seg']):
            g = list(g); prev = None
            for x in g:
                cur = x.get('ego_boxes', [])
                sizes.append(len(cur))
                if prev is not None:
                    j = set_jaccard(prev[0], cur)
                    jac.append(j); chg += (j < 1.0); flip += (x['pred'] != prev[1]); n += 1
                prev = (cur, x['pred'])
        print(f"  {p.split('/')[-1]:34s} {acc:7.2%} {np.mean(jac):12.2f} "
              f"{chg/max(n,1):12.0%} {flip/max(n,1):13.0%} {np.mean(sizes):11.2f}")


if __name__ == '__main__':
    main()
