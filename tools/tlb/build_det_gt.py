#!/usr/bin/env python
"""Per-light detection ground truth: box, colour, and whether it governs the ego lane.

Task 1 asks one question per frame. Detection needs every light in the frame labelled,
so this emits a row per *frame* containing every traffic-light box with:

    colour    red / green / yellow / unknown   (attribute 1,2,3 else unknown)
    is_gov    1 governs the ego lane, 0 does not, -1 undetermined

`is_gov` is -1 when the ego lane itself could not be established, so those frames still
train detection and colour while being ignored by the association loss.

This covers every frame with a traffic light (~12k train), not just the ~5k where a
*coloured* light governs the ego, so the detector sees far more data than the VQA task.
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
        tl = [j for j, t in enumerate(tes) if t['category'] == 1]
        c['frames'] += 1
        if not tl:
            continue
        c['frames_with_tl'] += 1
        lanes = a['lane_centerline']
        lcte = np.asarray(a['topology_lcte'])
        e = ego_lane_index(lanes) if lanes else None
        gov = set()
        if e is not None and lcte.size:
            reach = reachable(np.asarray(a['topology_lclc']), e, hops)
            gov = {j for j in tl if any(lcte[r][j] for r in reach)}
            c['frames_ego_known'] += 1
        lights = []
        for j in tl:
            (x1, y1), (x2, y2) = tes[j]['points']
            col = COLOUR.get(tes[j]['attribute'], 'unknown')
            isg = (1 if j in gov else 0) if e is not None else -1
            lights.append({'box': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                           'colour': col, 'is_gov': isg})
            c[f'light_{col}'] += 1
            if isg == 1:
                c[f'gov_{col}'] += 1
        rows.append({'split': split, 'segment': seg, 'timestamp': fr[:-5],
                     'image': d['sensor'][CAM]['image_path'],
                     'lights': lights, 'ego_known': e is not None})
        c['lights'] += len(lights)
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
    per, tally = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for split, rows, c in ex.map(do_segment, jobs, chunksize=4):
            per.setdefault(split, []).extend(rows)
            tally.setdefault(split, Counter()).update(c)
    for split in args.splits:
        rows = sorted(per[split], key=lambda r: (r['segment'], r['timestamp']))
        p = os.path.join(out_dir, f'det_{split}.jsonl')
        with open(p, 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        c = tally[split]
        print(f"\n===== det {split} =====")
        print(f"  frames with >=1 traffic light  {c['frames_with_tl']:6d} of {c['frames']}")
        print(f"  ...with ego lane established   {c['frames_ego_known']:6d}")
        print(f"  light instances                {c['lights']:6d}")
        print(f"    by colour  " + str({k[6:]: v for k, v in c.items() if k.startswith('light_')}))
        print(f"    governing  " + str({k[4:]: v for k, v in c.items() if k.startswith('gov_')}))
        print(f"  wrote {p}")


if __name__ == '__main__':
    main()
