#!/usr/bin/env python
"""Ground truth over the WHOLE val split, including frames with no ego-lane light.

Everything measured so far used the 1300 frames where a coloured light governs the ego
lane. A deployed system also has to answer "none" on the other 4719, and never invent a
colour. This labels all 6019:

    red / green / yellow   a coloured light governs the ego lane   (as before)
    none                   no coloured light governs the ego lane

`none` covers three genuinely different cases, kept as separate flags so the failure modes
stay distinguishable:
    no_light      no traffic light in the frame at all
    not_ego       lights are visible, but none is linked to the ego lane
    unknown_only  a light governs the ego lane but its colour is annotated 'unknown'
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_gt import ego_lane_index, reachable, COLOUR, ROOT, CAM


def do_segment(job):
    split, seg, frames, hops = job
    rows, c = [], Counter()
    for fr in frames:
        d = json.load(open(os.path.join(ROOT, split, seg, 'info', fr)))
        a = d['annotation']
        tes = a['traffic_element']
        lanes = a['lane_centerline']
        tl = [j for j, t in enumerate(tes) if t['category'] == 1]
        col = [j for j in tl if tes[j]['attribute'] in COLOUR]

        label, why = 'none', 'no_light'
        e = ego_lane_index(lanes) if lanes else None
        lcte = np.asarray(a['topology_lcte'])
        gov_any = []
        if tl:
            why = 'not_ego'
        if e is not None and lcte.size and tl:
            reach = reachable(np.asarray(a['topology_lclc']), e, hops)
            gov_any = [j for j in tl if any(lcte[r][j] for r in reach)]
            gov_col = [j for j in gov_any if j in col]
            if gov_col:
                cols = {COLOUR[tes[j]['attribute']] for j in gov_col}
                if len(cols) == 1:
                    label, why = cols.pop(), 'governed'
                else:
                    why = 'ambiguous'
            elif gov_any:
                why = 'unknown_only'
        rows.append({'split': split, 'segment': seg, 'timestamp': fr[:-5],
                     'image': d['sensor'][CAM]['image_path'],
                     'label': label, 'why': why,
                     'n_lights': len(tl), 'n_coloured': len(col),
                     'ego_lane_known': e is not None})
        c[f'{label}:{why}'] += 1
        c['frames'] += 1
    return split, rows, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['val'])
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
        p = f"/home/albert/Desktop/Qwen-Drive-1.0/data/tlb/full_{split}.jsonl"
        with open(p, 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        c = tally[split]
        print(f"\n===== FULL {split}: {c['frames']} frames =====")
        for k, v in sorted(c.items(), key=lambda x: -x[1]):
            if k == 'frames':
                continue
            print(f"  {k:24s} {v:5d}  {v/c['frames']:6.1%}")
        lab = Counter(r['label'] for r in rows)
        print(f"  labels: {lab.most_common()}")
        print(f"  a 'none'-always baseline would score {lab['none']/len(rows):.1%}")
        print(f"  wrote {p}")


if __name__ == '__main__':
    main()
