#!/usr/bin/env python
"""Ego lane position: which lane of the carriageway is the ego in?

At the ego's own longitudinal position, take every centerline running the same way the
ego is (heading within 45 deg of +x -- this is what excludes oncoming traffic and cross
streets), read each one's lateral offset, and rank them left to right. y is positive to
the left, so sorting by descending y gives lane 1 = leftmost.

Output per frame: lane_index (1-based from the left), n_lanes, and the ego lane's offset.
Frames where the ego lane itself is not identifiable are skipped.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_gt import ego_lane_index, ROOT, CAM, MAX_HEADING_DEG

NEAR_X = 4.0          # a lane counts if it has a vertex within this of the ego's x


def lateral_at_ego(P):
    """Lateral offset where the polyline passes the ego's longitudinal position."""
    near = P[np.abs(P[:, 0]) < NEAR_X]
    if len(near) == 0:
        return None
    return float(near[np.argmin(np.abs(near[:, 0])), 1])


def heading_ok(P):
    k = int(np.argmin(np.abs(P[:, 0])))
    a, b = max(k - 8, 0), min(k + 8, len(P) - 1)
    if b <= a:
        return False
    t = P[b, :2] - P[a, :2]
    return abs(np.degrees(np.arctan2(t[1], t[0]))) <= MAX_HEADING_DEG


def do_segment(job):
    split, seg, frames = job
    rows, c = [], Counter()
    for fr in frames:
        d = json.load(open(os.path.join(ROOT, split, seg, 'info', fr)))
        a = d['annotation']
        lanes = a['lane_centerline']
        c['frames'] += 1
        if not lanes:
            continue
        e = ego_lane_index(lanes)
        if e is None:
            c['no_ego_lane'] += 1
            continue
        cands = []
        for i, ln in enumerate(lanes):
            P = np.asarray(ln['points'], dtype=float)
            if not heading_ok(P):
                continue
            y = lateral_at_ego(P)
            if y is None:
                continue
            cands.append((y, i))
        if not cands:
            c['no_candidates'] += 1
            continue
        # merge near-duplicate centerlines (successive segments of one lane) by offset
        cands.sort(key=lambda t: -t[0])
        merged = []
        for y, i in cands:
            if merged and abs(y - merged[-1][0]) < 1.2:
                if i == e:
                    merged[-1] = (y, i)
                continue
            merged.append((y, i))
        idx = [i for _, i in merged]
        if e not in idx:
            c['ego_merged_away'] += 1
            continue
        pos = idx.index(e) + 1
        rows.append({'split': split, 'segment': seg, 'timestamp': fr[:-5],
                     'image': d['sensor'][CAM]['image_path'],
                     'lane_index': pos, 'n_lanes': len(idx),
                     'ego_offset_m': round(merged[pos - 1][0], 2),
                     'from_right': len(idx) - pos + 1})
        c['USABLE'] += 1
        c[f'lanes_{len(idx)}'] += 1
        c[f'pos_{pos}of{len(idx)}'] += 1
    return split, rows, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()
    split_map = json.load(open(os.path.join(ROOT, 'data_dict_subset_B.json')))
    jobs = [(sp, seg, fr) for sp in args.splits for seg, fr in split_map[sp].items()]
    per, tally = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for split, rows, c in ex.map(do_segment, jobs, chunksize=4):
            per.setdefault(split, []).extend(rows)
            tally.setdefault(split, Counter()).update(c)
    for split in args.splits:
        rows = sorted(per[split], key=lambda r: (r['segment'], r['timestamp']))
        p = f"/home/albert/Desktop/Qwen-Drive-1.0/data/tlb/lane_{split}.jsonl"
        with open(p, 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        c = tally[split]
        print(f"\n===== lane {split} =====")
        print(f"  frames {c['frames']}  usable {c['USABLE']} ({c['USABLE']/c['frames']:.1%})  "
              f"no ego lane {c['no_ego_lane']}")
        print("  lane count distribution:",
              {k[6:]: v for k, v in sorted(c.items()) if k.startswith('lanes_')})
        top = sorted(((k, v) for k, v in c.items() if k.startswith('pos_')), key=lambda x: -x[1])[:8]
        print("  most common (position of total):", {k[4:]: v for k, v in top})
        print(f"  wrote {p}")


if __name__ == '__main__':
    main()
