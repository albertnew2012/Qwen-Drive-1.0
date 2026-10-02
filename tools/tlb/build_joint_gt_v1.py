#!/usr/bin/env python
"""Task 1, explicit form: the ego's lane TYPE and its light COLOUR in one label.

"I am in a left-turn lane and its signal is red" rather than just "red". Both halves come
from the same lane graph:

    ego lane --topology_lcte--> road arrows   -> permitted movements (lane type)
    ego lane --topology_lcte--> traffic light -> colour

This is the strongest statement the dataset supports. It is NOT per-movement signal state:
traffic lights here carry colour only, and in 3508 val frames there is not one where two
movements show different colours -- so "the left-turn signal is red while the through
signal is green" cannot be expressed, let alone tested.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_gt import ego_lane_index, reachable, COLOUR, ROOT, CAM
from build_lanetype_gt import ALLOW, DENY, ORDER, type_name


def do_segment(job):
    split, seg, frames, hops = job
    rows, c = [], Counter()
    for fr in frames:
        d = json.load(open(os.path.join(ROOT, split, seg, 'info', fr)))
        a = d['annotation']
        lanes, tes = a['lane_centerline'], a['traffic_element']
        c['frames'] += 1
        if not lanes:
            continue
        e = ego_lane_index(lanes)
        if e is None:
            continue
        lcte = np.asarray(a['topology_lcte'])
        if not lcte.size:
            continue
        reach = reachable(np.asarray(a['topology_lclc']), e, hops)
        allow, deny, cols, gov_boxes = set(), set(), set(), []
        for j, t in enumerate(tes):
            if not any(lcte[r][j] for r in reach):
                continue
            if t['category'] == 2:
                if t['attribute'] in ALLOW:
                    allow.add(ALLOW[t['attribute']])
                elif t['attribute'] in DENY:
                    deny.add(DENY[t['attribute']])
            elif t['category'] == 1 and t['attribute'] in COLOUR:
                cols.add(COLOUR[t['attribute']])
                (x1, y1), (x2, y2) = t['points']
                gov_boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
        mv = {m for m in (allow - deny) if m in ORDER}
        if not mv or len(cols) != 1:
            c['incomplete'] += 1
            continue
        colour = cols.pop()
        rows.append({'split': split, 'segment': seg, 'timestamp': fr[:-5],
                     'image': d['sensor'][CAM]['image_path'],
                     'allow': sorted(mv), 'lane_type': type_name(mv),
                     'label': colour, 'gov_boxes': gov_boxes,
                     'joint': f"{type_name(mv)} | {colour}"})
        c['USABLE'] += 1
        c[f"joint_{type_name(mv)}|{colour}"] += 1
    return split, rows, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()
    sm = json.load(open(os.path.join(ROOT, 'data_dict_subset_B.json')))
    jobs = [(sp, seg, fr, args.hops) for sp in args.splits for seg, fr in sm[sp].items()]
    per, tally = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for split, rows, c in ex.map(do_segment, jobs, chunksize=4):
            per.setdefault(split, []).extend(rows)
            tally.setdefault(split, Counter()).update(c)
    for split in args.splits:
        rows = sorted(per[split], key=lambda r: (r['segment'], r['timestamp']))
        p = f"/home/albert/Desktop/Qwen-Drive-1.0/data/tlb/joint_{split}.jsonl"
        with open(p, 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        c = tally[split]
        print(f"\n===== joint (lane type + colour) {split} =====")
        print(f"  frames {c['frames']}   usable {c['USABLE']} ({c['USABLE']/c['frames']:.1%})"
              f"   segments {len({r['segment'] for r in rows})}")
        for k, v in sorted(((k[6:], v) for k, v in c.items() if k.startswith('joint_')),
                           key=lambda x: -x[1])[:10]:
            print(f"    {k:34s} {v}")
        print(f"  wrote {p}")


if __name__ == '__main__':
    main()
