#!/usr/bin/env python
"""Survey OpenLane-V2 subset_B for ego-lane traffic-light signal.

Answers the feasibility question before any pipeline is built:
  how many frames carry a *coloured traffic light* that `topology_lcte` links to the
  lane the ego is actually in?

Categories: 1 = traffic_light, 2 = road_sign.
Attributes: 0 unknown, 1 red, 2 green, 3 yellow, 4..12 arrow/movement types.
Lane centerline points are in the ego frame (x forward, y left, metres).
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

ROOT = '/home/albert/Desktop/Qwen-Drive-1.0/data/OpenLane-V2'
ATTR = {0: 'unknown', 1: 'red', 2: 'green', 3: 'yellow', 4: 'go_straight',
        5: 'turn_left', 6: 'turn_right', 7: 'no_left_turn', 8: 'no_right_turn',
        9: 'u_turn', 10: 'no_u_turn', 11: 'slight_left', 12: 'slight_right'}
COLOUR = {1: 'red', 2: 'green', 3: 'yellow'}


def ego_lane_index(lanes, max_lat=1.75, min_ahead=2.0):
    """Index of the centerline the ego sits in: passes nearest the origin, heading ahead.

    Uses the polyline's closest vertex to (0,0). A lane the ego occupies must pass within
    half a lane width laterally and must extend ahead of the ego, otherwise it is a lane
    already behind us or a neighbour.
    """
    best, best_d = None, 1e9
    for i, ln in enumerate(lanes):
        pts = ln['points']
        if max(p[0] for p in pts) < min_ahead:
            continue                      # entirely behind the ego
        d = min((p[0] ** 2 + p[1] ** 2) ** 0.5 for p in pts)
        lat = min(abs(p[1]) for p in pts if abs(p[0]) < 3.0) if any(abs(p[0]) < 3.0 for p in pts) else 1e9
        score = min(d, lat)
        if score < best_d:
            best, best_d = i, score
    return (best, best_d) if best_d <= max_lat else (None, best_d)


def do_segment(job):
    split, seg, frames = job
    c = Counter()
    for fr in frames:
        p = os.path.join(ROOT, split, seg, 'info', fr)
        a = json.load(open(p))['annotation']
        tes = a['traffic_element']
        lanes = a['lane_centerline']
        lcte = a['topology_lcte']
        c['frames'] += 1
        for te in tes:
            c[f"cat{te['category']}_attr_{ATTR.get(te['attribute'], te['attribute'])}"] += 1
        tl_idx = [j for j, te in enumerate(tes) if te['category'] == 1]
        tl_col = [j for j in tl_idx if tes[j]['attribute'] in COLOUR]
        if tes:            c['frame_has_te'] += 1
        if tl_idx:         c['frame_has_trafficlight'] += 1
        if tl_col:         c['frame_has_coloured_tl'] += 1
        if any(any(row) for row in lcte): c['frame_has_lcte_link'] += 1

        ego, d = ego_lane_index(lanes)
        if ego is None:
            c['no_ego_lane'] += 1
            continue
        c['ego_lane_found'] += 1
        if not tl_col:
            continue
        gov = [j for j in tl_col if lcte[ego][j]]
        if not gov:
            c['ego_lane_no_gov_tl'] += 1
            continue
        cols = {COLOUR[tes[j]['attribute']] for j in gov}
        c['USABLE'] += 1
        c[f"ego_colour_{'AMBIG' if len(cols) > 1 else cols.pop()}"] += 1
        # frames where another visible light disagrees are the discriminative ones
        gov_cols = {COLOUR[tes[j]['attribute']] for j in gov}
        other = {COLOUR[tes[j]['attribute']] for j in tl_col if j not in gov}
        if other - gov_cols:
            c['DISCRIMINATIVE'] += 1
    return split, c


def main():
    split_map = json.load(open(os.path.join(ROOT, 'data_dict_subset_B.json')))
    splits = sys.argv[1:] or ['train', 'val', 'test']
    jobs = [(sp, seg, fr) for sp in splits for seg, fr in split_map[sp].items()]
    tally = {}
    with ProcessPoolExecutor(max_workers=20) as ex:
        for split, c in ex.map(do_segment, jobs, chunksize=4):
            tally.setdefault(split, Counter()).update(c)
    for split in splits:
        c = tally[split]
        print(f"\n===== {split}  ({c['frames']} frames) =====")
        for k in ('frame_has_te', 'frame_has_trafficlight', 'frame_has_coloured_tl',
                  'frame_has_lcte_link', 'ego_lane_found', 'no_ego_lane',
                  'ego_lane_no_gov_tl', 'USABLE', 'DISCRIMINATIVE'):
            print(f"  {k:24s} {c[k]:7d}   {c[k]/max(c['frames'],1):6.1%}")
        print("  ego colour:", {k.replace('ego_colour_',''): v for k, v in sorted(c.items()) if k.startswith('ego_colour_')})
        print("  element categories/attributes:")
        for k, v in sorted(((k, v) for k, v in c.items() if k.startswith('cat')), key=lambda x: -x[1]):
            print(f"      {k:34s} {v}")


if __name__ == '__main__':
    main()
