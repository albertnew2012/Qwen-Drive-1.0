#!/usr/bin/env python
"""Per-frame 3D supervision for traffic lights: range and height beside each 2D box.

The confirmed 3D positions are one point per light per segment (world frame). A detector
needs them per frame, in the ego frame of that frame, attached to the 2D box it should be
regressed from. The join is by (segment, element id) -- ids are stable within a segment,
which is what made the triangulation possible in the first place.

Only lights confirmed by a camera-visible Occ3D voxel are emitted, and only for frames
where the light is inside the range the labels are trustworthy at (<=40 m, measured
against lidar: 0.84 m error under 25 m, 1.38 m at 25-40 m, no signal beyond 55 m).
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter, defaultdict

import numpy as np

BASE = '/home/albert/Desktop/Qwen-Drive-1.0/'
ROOT = BASE + 'data/OpenLane-V2'
MAX_RANGE = 40.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refined', default='data/tlb/tl3d_train_refined.jsonl')
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', default='data/tlb/det3d_train.jsonl')
    args = ap.parse_args()

    ref = [json.loads(l) for l in open(BASE + args.refined)]
    by_seg = defaultdict(dict)
    for r in ref:
        by_seg[r['segment']][r['id']] = np.asarray(r['xyz_world_refined'], float)
    print(f"confirmed 3D lights: {len(ref)} over {len(by_seg)} segments")

    split_map = json.load(open(os.path.join(ROOT, 'data_dict_subset_B.json')))
    rows, c = [], Counter()
    for seg, frames in split_map[args.split].items():
        if seg not in by_seg:
            continue
        for fr in frames:
            p = os.path.join(ROOT, args.split, seg, 'info', fr)
            d = json.load(open(p))
            a = d['annotation']
            Re = np.asarray(d['pose']['rotation'], float)
            te = np.asarray(d['pose']['translation'], float)
            lights = []
            for t in a['traffic_element']:
                if t['category'] != 1:
                    continue
                X = by_seg[seg].get(t['id'])
                if X is None:
                    c['no_3d'] += 1
                    continue
                Xe = Re.T @ (X - te)
                rng = float(np.linalg.norm(Xe[:2]))
                if rng > MAX_RANGE or Xe[0] < 1.0:
                    c['out_of_range'] += 1
                    continue
                (x1, y1), (x2, y2) = t['points']
                if x2 - x1 < 2 or y2 - y1 < 2:
                    c['tiny_box'] += 1
                    continue
                lights.append({'id': t['id'], 'box': [round(v, 1) for v in (x1, y1, x2, y2)],
                               'attribute': t['attribute'],
                               'range_m': round(rng, 2),
                               'height_m': round(float(Xe[2]), 2),
                               'fwd_m': round(float(Xe[0]), 2),
                               'lat_m': round(float(Xe[1]), 2)})
                c['labelled'] += 1
            if lights:
                rows.append({'split': args.split, 'segment': seg, 'timestamp': fr[:-5],
                             'image': d['sensor']['CAM_FRONT']['image_path'],
                             'lights3d': lights})
                c['frames'] += 1
    p = BASE + args.out
    with open(p, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print(f"  frames with >=1 3D-labelled light: {c['frames']}")
    print(f"  labelled light instances:          {c['labelled']}")
    print(f"  dropped: no 3D {c['no_3d']}, out of range {c['out_of_range']}, tiny {c['tiny_box']}")
    if rows:
        rr = np.array([l['range_m'] for r in rows for l in r['lights3d']])
        hh = np.array([l['height_m'] for r in rows for l in r['lights3d']])
        print(f"  range  m: median {np.median(rr):5.1f}  p10 {np.percentile(rr,10):5.1f}  p90 {np.percentile(rr,90):5.1f}")
        print(f"  height m: median {np.median(hh):5.2f}  p10 {np.percentile(hh,10):5.2f}  p90 {np.percentile(hh,90):5.2f}")
    print(f"  wrote {p}")


if __name__ == '__main__':
    main()
