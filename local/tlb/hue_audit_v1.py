#!/usr/bin/env python
"""Find segments where the traffic-light colour is not recoverable from the pixels.

Shooting into the sun blows the lamp to white: hue is destroyed while the label still says
red. Those frames punish any model that reads colour, and they are a sensor limitation
rather than a reasoning failure.

Two measurements per governing lamp, both from the image alone so the audit is independent
of any model:
  sat   peak saturation inside the lamp region -- a lit lamp is saturated, a blown one is not
  agree whether the dominant hue matches the annotated colour

A segment is flagged when most of its lamps are desaturated AND the hue disagrees.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict

import numpy as np
from PIL import Image

BASE = '/home/albert/Desktop/Qwen-Drive-1.0/'
ROOT = BASE + 'data/OpenLane-V2/'


def lamp_stats(img, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.width), min(y2, img.height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = img.crop((x1, y1, x2, y2)).convert('HSV')
    a = np.asarray(crop, np.float32)
    h, s, v = a[..., 0] * 2, a[..., 1] / 255., a[..., 2] / 255.
    lit = v > max(0.55, float(np.percentile(v, 80)))       # the illuminated lamp only
    if lit.sum() < 3:
        return None
    hs, ss = h[lit], s[lit]
    hue = float(np.median(hs))
    # red wraps at 0/360; green ~90-160; yellow ~40-70
    if hue < 25 or hue > 335:
        dom = 'red'
    elif 25 <= hue <= 75:
        dom = 'yellow'
    elif 75 < hue <= 190:
        dom = 'green'
    else:
        dom = 'other'
    return float(np.percentile(ss, 80)), dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--sat-thr', type=float, default=0.35)
    ap.add_argument('--bad-frac', type=float, default=0.5)
    ap.add_argument('--out', default='data/tlb/hue_bad_segments.json')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE + args.gt)]
    per = defaultdict(lambda: {'n': 0, 'desat': 0, 'disagree': 0, 'both': 0, 'sat': []})
    for r in rows:
        img = Image.open(ROOT + r['image'])
        for b in r['gov_boxes']:
            st = lamp_stats(img, b)
            if st is None:
                continue
            sat, dom = st
            p = per[r['segment']]
            p['n'] += 1
            p['sat'].append(sat)
            d = sat < args.sat_thr
            g = (dom != r['label'])
            p['desat'] += d
            p['disagree'] += g
            p['both'] += (d and g)
    bad = []
    print(f"{'segment':>9} {'lamps':>6} {'medsat':>7} {'desat':>7} {'hue!=lbl':>9} {'both':>6}")
    for seg, p in sorted(per.items()):
        if p['n'] < 5:
            continue
        f_both = p['both'] / p['n']
        flag = f_both >= args.bad_frac
        if flag:
            bad.append(seg)
        print(f"{seg:>9} {p['n']:6d} {np.median(p['sat']):7.2f} "
              f"{p['desat']/p['n']:7.0%} {p['disagree']/p['n']:9.0%} {f_both:6.0%}"
              + ("   <== hue unusable" if flag else ""))
    json.dump(bad, open(BASE + args.out, 'w'))
    nb = sum(1 for r in rows if r['segment'] in bad)
    print(f"\nflagged {len(bad)} segments: {bad}")
    print(f"  they hold {nb}/{len(rows)} = {nb/len(rows):.1%} of frames")
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
