#!/usr/bin/env python
"""Ego lane TYPE: which movements does the ego's own lane permit?

Left-turn lane, through lane, right-turn lane, or a combination. Derived from the road
markings OpenLane-V2 annotates as traffic elements (category 2), linked to the ego lane by
the same topology_lcte used for the lights.

Signs, not geometry. The lane graph's successors are a subset of the signed movements 94%
of the time -- the graph only connects what is annotated within range, so geometry says
"straight" for a lane whose arrow clearly also permits a left. Signs state the permission,
which is what "lane type" means and what a planner needs.

Attributes: 4 go_straight, 5 turn_left, 6 turn_right, 9 u_turn, 11 slight_left,
12 slight_right, and the prohibitions 7 no_left_turn, 8 no_right_turn, 10 no_u_turn, which
are recorded as explicit negatives rather than dropped.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_gt import ego_lane_index, reachable, ROOT, CAM

ALLOW = {4: 'straight', 5: 'left', 6: 'right', 11: 'left', 12: 'right', 9: 'uturn'}
DENY = {7: 'left', 8: 'right', 10: 'uturn'}
ORDER = ['left', 'straight', 'right']


def type_name(allow):
    """A readable single label, for the cases that have one."""
    s = tuple(m for m in ORDER if m in allow)
    return {('left',): 'left-turn only',
            ('straight',): 'through only',
            ('right',): 'right-turn only',
            ('left', 'straight'): 'through or left',
            ('straight', 'right'): 'through or right',
            ('left', 'right'): 'left or right',
            ('left', 'straight', 'right'): 'left, through or right'}.get(s, '+'.join(s))


def do_segment(job):
    split, seg, frames, hops = job
    rows, c = [], Counter()
    for fr in frames:
        d = json.load(open(os.path.join(ROOT, split, seg, 'info', fr)))
        a = d['annotation']
        lanes = a['lane_centerline']
        tes = a['traffic_element']
        c['frames'] += 1
        if not lanes:
            continue
        e = ego_lane_index(lanes)
        if e is None:
            c['no_ego_lane'] += 1
            continue
        lcte = np.asarray(a['topology_lcte'])
        if not lcte.size:
            continue
        reach = reachable(np.asarray(a['topology_lclc']), e, hops)
        allow, deny = set(), set()
        for j, t in enumerate(tes):
            if t['category'] != 2:
                continue
            if not any(lcte[r][j] for r in reach):
                continue
            if t['attribute'] in ALLOW:
                allow.add(ALLOW[t['attribute']])
            elif t['attribute'] in DENY:
                deny.add(DENY[t['attribute']])
        allow -= deny
        mv = {m for m in allow if m in ORDER}
        if not mv:
            c['no_signed_movement'] += 1
            continue
        rows.append({'split': split, 'segment': seg, 'timestamp': fr[:-5],
                     'image': d['sensor'][CAM]['image_path'],
                     'allow': sorted(mv), 'deny': sorted(deny),
                     'type': type_name(mv),
                     'left': int('left' in mv), 'straight': int('straight' in mv),
                     'right': int('right' in mv)})
        c['USABLE'] += 1
        c['type_' + type_name(mv)] += 1
    return split, rows, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()
    split_map = json.load(open(os.path.join(ROOT, 'data_dict_subset_B.json')))
    jobs = [(sp, seg, fr, args.hops) for sp in args.splits for seg, fr in split_map[sp].items()]
    per, tally = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for split, rows, c in ex.map(do_segment, jobs, chunksize=4):
            per.setdefault(split, []).extend(rows)
            tally.setdefault(split, Counter()).update(c)
    for split in args.splits:
        rows = sorted(per[split], key=lambda r: (r['segment'], r['timestamp']))
        p = f"/home/albert/Desktop/Qwen-Drive-1.0/data/tlb/lanetype_{split}.jsonl"
        with open(p, 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        c = tally[split]
        segs = len({r['segment'] for r in rows})
        print(f"\n===== lane type {split} =====")
        print(f"  frames {c['frames']}   usable {c['USABLE']} ({c['USABLE']/c['frames']:.1%})"
              f"   over {segs} segments")
        print(f"  dropped: no ego lane {c['no_ego_lane']}, no signed movement {c['no_signed_movement']}")
        for k, v in sorted(((k[5:], v) for k, v in c.items() if k.startswith('type_')),
                           key=lambda x: -x[1]):
            print(f"    {k:26s} {v:5d}  {v/max(c['USABLE'],1):5.1%}")
        for m in ORDER:
            n = sum(r[m] for r in rows)
            print(f"  permits {m:9s} {n:5d}  {n/max(len(rows),1):5.1%}")
        print(f"  wrote {p}")


if __name__ == '__main__':
    main()
