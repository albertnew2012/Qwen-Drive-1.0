#!/usr/bin/env python
"""Side-by-side BEV + front view, to check *which* light was tied to the ego lane.

Left: lane centerlines in the ego frame. The ego lane is drawn thick, lanes reachable
from it within the hop budget are drawn medium, everything else thin grey. Right: the
front image, governing lights boxed in their label colour, other lights in white.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

R = Path('/home/albert/Desktop/Qwen-Drive-1.0/data/OpenLane-V2')
import sys; sys.path.insert(0, '/home/albert/Desktop/Qwen-Drive-1.0/local/tlb')
from build_gt import ego_lane_index, reachable, COLOUR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/tlb/val.jsonl')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open('/home/albert/Desktop/Qwen-Drive-1.0/' + args.src)]
    rows = [r for r in rows if r['discriminative'] and r['n_other'] >= 1]
    import random; random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 4.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for k, r in enumerate(rows):
        info = json.load(open(R / r['split'] / r['segment'] / 'info' / f"{r['timestamp']}.json"))
        a = info['annotation']
        L = a['lane_centerline']; tes = a['traffic_element']
        lclc = np.asarray(a['topology_lclc']); lcte = np.asarray(a['topology_lcte'])
        e = ego_lane_index(L); reach = reachable(lclc, e, args.hops)

        ax = axes[k, 0]
        for i, ln in enumerate(L):
            P = np.asarray(ln['points'])
            if i == e:      ax.plot(P[:,1], P[:,0], '-', lw=3.5, color='#00b0ff', zorder=5, label='ego lane')
            elif i in reach: ax.plot(P[:,1], P[:,0], '-', lw=2.0, color='#7fd4ff', zorder=4, label='reachable')
            else:            ax.plot(P[:,1], P[:,0], '-', lw=0.8, color='#999999', zorder=2)
        ax.plot(0, 0, 'k^', ms=13, zorder=6, label='ego')
        ax.set_aspect('equal'); ax.invert_xaxis()
        ax.set_xlabel('y left (m)'); ax.set_ylabel('x forward (m)')
        ax.set_title(f"{r['segment']}/{r['timestamp']}  label={r['label']}  reach={len(reach)} lanes")
        h, l = ax.get_legend_handles_labels()
        d = dict(zip(l, h)); ax.legend(d.values(), d.keys(), fontsize=7, loc='upper right')
        ax.grid(alpha=.2)

        ax = axes[k, 1]
        im = Image.open(R / r['image']).convert('RGB')
        ax.imshow(im); ax.axis('off')
        for b in r['other_boxes']:
            ax.add_patch(plt.Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec='white', lw=1.4))
        for b in r['gov_boxes']:
            ax.add_patch(plt.Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], fill=False,
                                       ec={'red':'#ff3030','green':'#30ff30','yellow':'#ffd030'}[r['label']], lw=2.2))
        ax.set_title(f"gov={r['n_gov']} ({r['label']})  other={r['n_other']} {r['other_colours']}", fontsize=9)
    plt.tight_layout(); plt.savefig(args.out, dpi=85)
    print('wrote', args.out)
    for r in rows: print(f"  {r['segment']}/{r['timestamp']} {r['label']} other={r['other_colours']}")


if __name__ == '__main__':
    main()
