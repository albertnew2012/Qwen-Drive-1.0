#!/usr/bin/env python
"""Build ego-lane traffic-light ground truth from OpenLane-V2 subset_B.

Only a minority of frames pose the problem at all: a frame is usable only when a
*coloured* traffic light is linked, through the lane graph, to the lane the ego is in.
This script finds exactly those frames and writes one JSONL row per usable frame.

The association chain, all of it from published annotation, none of it guessed:

    ego pose (origin)  ->  ego lane centerline      geometric, x fwd / y left
    ego lane           ->  reachable lanes          topology_lclc, i -> j, <=2 hops
    reachable lanes    ->  traffic elements         topology_lcte
    traffic element    ->  colour                   attribute 1=red 2=green 3=yellow

Traffic-element boxes are in CAM_FRONT pixel coordinates (verified visually), so only
CAM_FRONT is used.

A frame is `discriminative` when another visible coloured light disagrees with the ego's
own. Those are the only frames where true association and "report the most salient lamp"
give different answers, so they carry the real signal.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = '/home/albert/Desktop/Qwen-Drive-1.0/data/OpenLane-V2'
COLOUR = {1: 'red', 2: 'green', 3: 'yellow'}
CAM = 'CAM_FRONT'

# ego lane acceptance
MAX_LAT = 1.75        # half a lane width
MAX_HEADING_DEG = 45  # reject cross-traffic lanes that merely pass near the ego
MIN_AHEAD = 2.0       # lane must extend ahead of the ego


def ego_lane_index(lanes):
    """Index of the lane the ego occupies, or None.

    Nearest centerline to the origin that also runs roughly parallel to the ego heading.
    The heading gate is what keeps a perpendicular cross-traffic lane at an intersection
    from being mistaken for the ego's own lane.
    """
    best, best_d = None, 1e9
    for i, ln in enumerate(lanes):
        P = np.asarray(ln['points'], dtype=float)
        if P[:, 0].max() < MIN_AHEAD:
            continue
        d = np.linalg.norm(P[:, :2], axis=1)
        k = int(np.argmin(d))
        if d[k] > MAX_LAT:
            continue
        a, b = max(k - 8, 0), min(k + 8, len(P) - 1)
        if b <= a:
            continue
        t = P[b, :2] - P[a, :2]
        if abs(np.degrees(np.arctan2(t[1], t[0]))) > MAX_HEADING_DEG:
            continue
        if d[k] < best_d:
            best, best_d = i, d[k]
    return best


def reachable(lclc, e, hops):
    """Ego lane plus everything reachable downstream within `hops` edges."""
    seen = {e}
    frontier = {e}
    for _ in range(hops):
        nxt = set()
        for r in frontier:
            nxt |= set(np.nonzero(lclc[r])[0].tolist())
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def do_segment(job):
    split, seg, frames, hops = job
    rows, c = [], Counter()
    for fr in frames:
        path = os.path.join(ROOT, split, seg, 'info', fr)
        d = json.load(open(path))
        a = d['annotation']
        tes = a['traffic_element']
        lanes = a['lane_centerline']
        c['frames'] += 1

        tl = [j for j, t in enumerate(tes) if t['category'] == 1]
        col = [j for j in tl if tes[j]['attribute'] in COLOUR]
        if tl:
            c['has_tl'] += 1
        if not col:
            continue
        c['has_coloured_tl'] += 1

        if not lanes:
            c['no_lanes'] += 1
            continue
        e = ego_lane_index(lanes)
        if e is None:
            c['no_ego_lane'] += 1
            continue

        lclc = np.asarray(a['topology_lclc'])
        lcte = np.asarray(a['topology_lcte'])
        if lcte.size == 0:
            c['no_lcte'] += 1
            continue
        reach = reachable(lclc, e, hops)
        gov = [j for j in col if any(lcte[r][j] for r in reach)]
        if not gov:
            c['no_gov_tl'] += 1
            continue

        cols = {COLOUR[tes[j]['attribute']] for j in gov}
        if len(cols) > 1:
            c['ambiguous'] += 1
            continue
        label = cols.pop()

        other = [j for j in col if j not in gov]
        other_cols = sorted({COLOUR[tes[j]['attribute']] for j in other})
        disc = bool(set(other_cols) - {label})

        def box(j):
            (x1, y1), (x2, y2) = tes[j]['points']
            return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

        gb = [box(j) for j in gov]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in gb]
        rows.append({
            'split': split, 'segment': seg, 'timestamp': fr[:-5],
            'image': d['sensor'][CAM]['image_path'],
            'label': label,
            'discriminative': disc,
            'gov_boxes': gb,
            'other_boxes': [box(j) for j in other],
            'other_colours': other_cols,
            'n_gov': len(gov), 'n_other': len(other), 'n_tl': len(tl),
            'max_gov_area_px': round(max(areas), 1),
            'max_gov_wh': [round(max(b[2] - b[0] for b in gb), 1),
                           round(max(b[3] - b[1] for b in gb), 1)],
            'ego_lane': int(e), 'n_reach': len(reach),
        })
        c['USABLE'] += 1
        c['disc'] += disc
        c[f'label_{label}'] += 1
    return split, rows, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--out-dir', default='data/tlb')
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()

    split_map = json.load(open(os.path.join(ROOT, 'data_dict_subset_B.json')))
    jobs = [(sp, seg, fr, args.hops) for sp in args.splits for seg, fr in split_map[sp].items()]
    out_dir = os.path.join('/home/albert/Desktop/Qwen-Drive-1.0', args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    per_split, tally = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for split, rows, c in ex.map(do_segment, jobs, chunksize=4):
            per_split.setdefault(split, []).extend(rows)
            tally.setdefault(split, Counter()).update(c)

    for split in args.splits:
        rows = sorted(per_split.get(split, []), key=lambda r: (r['segment'], r['timestamp']))
        p = os.path.join(out_dir, f'{split}.jsonl')
        with open(p, 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        c = tally[split]
        segs = len({r['segment'] for r in rows})
        print(f"\n===== {split} =====")
        print(f"  frames scanned            {c['frames']:7d}")
        print(f"  ...with any traffic light {c['has_tl']:7d}  {c['has_tl']/c['frames']:6.1%}")
        print(f"  ...with a coloured one    {c['has_coloured_tl']:7d}  {c['has_coloured_tl']/c['frames']:6.1%}")
        print(f"  dropped: no ego lane {c['no_ego_lane']}, no lcte {c['no_lcte']}, "
              f"not linked to ego {c['no_gov_tl']}, ambiguous {c['ambiguous']}")
        print(f"  USABLE                    {c['USABLE']:7d}  {c['USABLE']/c['frames']:6.1%} of frames, "
              f"{c['USABLE']/max(c['has_coloured_tl'],1):.1%} of coloured-TL frames")
        print(f"  discriminative            {c['disc']:7d}")
        print(f"  labels: red {c['label_red']}  green {c['label_green']}  yellow {c['label_yellow']}")
        print(f"  segments represented      {segs}")
        if rows:
            wh = np.array([r['max_gov_wh'] for r in rows])
            print(f"  governing box width px: median {np.median(wh[:,0]):.0f}  "
                  f"p10 {np.percentile(wh[:,0],10):.0f}  p90 {np.percentile(wh[:,0],90):.0f}")
        print(f"  wrote {p}")


if __name__ == '__main__':
    main()
