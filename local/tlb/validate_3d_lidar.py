#!/usr/bin/env python
"""Independent check: is there real 3D structure where triangulation says a light is?

The triangulated positions are derived from the same 2D boxes they would supervise, so
reprojection error is partly circular -- least squares minimises exactly that. LIDAR is
independent evidence: it never entered the triangulation, and a traffic light is a
physical object on a pole or gantry that returns points.

All 10 nuScenes v1.0-mini scenes appear in OpenLane-V2 subset B, and the calibration was
verified identical, so the comparison is done in the EGO frame at one timestamp:

    triangulated point (world) --ego pose--> ego frame
    lidar points (sensor)      --calib-->    ego frame

and the question is simply how far the nearest lidar return is. A correct position should
land on structure; a fabricated one should sit in empty air.
"""
from __future__ import annotations
import argparse, json, glob, os
from collections import defaultdict

import numpy as np

BASE = '/home/albert/Desktop/Qwen-Drive-1.0/'
N = BASE + 'data/nuscenes/'
OLV2 = BASE + 'data/OpenLane-V2/'


def quat2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def load_nuscenes():
    j = lambda f: json.load(open(N + 'v1.0-mini/' + f))
    sensors = {s['token']: s['channel'] for s in j('sensor.json')}
    cs = {c['token']: c for c in j('calibrated_sensor.json')}
    ego = {e['token']: e for e in j('ego_pose.json')}
    sd = j('sample_data.json')
    by_sample = defaultdict(dict)
    for r in sd:
        if not r['is_key_frame']:
            continue
        ch = sensors[cs[r['calibrated_sensor_token']]['sensor_token']]
        by_sample[r['sample_token']][ch] = r
    cam_ts = {}
    for st, chans in by_sample.items():
        if 'CAM_FRONT' in chans:
            cam_ts[chans['CAM_FRONT']['timestamp']] = st
    return by_sample, cam_ts, cs, ego


def lidar_in_ego(rec, cs):
    p = N + rec['filename']
    if not os.path.exists(p):
        return None
    pts = np.fromfile(p, dtype=np.float32).reshape(-1, 5)[:, :3]
    c = cs[rec['calibrated_sensor_token']]
    R = quat2R(c['rotation'])
    t = np.asarray(c['translation'], float)
    return pts @ R.T + t                      # sensor -> ego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tl3d', default='data/tlb/tl3d_mini.jsonl')
    ap.add_argument('--radius', type=float, default=2.0)
    args = ap.parse_args()

    by_sample, cam_ts, cs, ego = load_nuscenes()
    rows = [json.loads(l) for l in open(BASE + args.tl3d)]
    tl = [r for r in rows if r['category'] == 1]
    print(f"triangulated traffic lights in the overlapping scenes: {len(tl)}")

    res = []
    for r in tl:
        seg, split = r['segment'], r['split']
        # the frame where the light is nearest gives the densest lidar on it
        frames = sorted(glob.glob(f"{OLV2}{split}/{seg}/info/*.json"))
        best = None
        X = np.asarray(r['xyz_world'], float)
        for f in frames:
            d = json.load(open(f))
            ts = int(os.path.basename(f)[:-5])
            if ts not in cam_ts:
                continue
            Re = np.asarray(d['pose']['rotation'], float)
            te = np.asarray(d['pose']['translation'], float)
            Xe = Re.T @ (X - te)
            rng = np.linalg.norm(Xe[:2])
            if best is None or rng < best[0]:
                best = (rng, ts, Xe)
        if best is None:
            continue
        rng, ts, Xe = best
        st = cam_ts[ts]
        lid = by_sample[st].get('LIDAR_TOP')
        if lid is None:
            continue
        P = lidar_in_ego(lid, cs)
        if P is None:
            continue
        d = np.linalg.norm(P - Xe, axis=1)
        near = int((d < args.radius).sum())
        res.append({'seg': seg, 'id': r['id'], 'range_m': round(float(rng), 1),
                    'height_m': r['height_m'], 'reproj': r['reproj_px_median'],
                    'nearest_lidar_m': round(float(d.min()), 2),
                    'pts_within_2m': near})

    if not res:
        print("no overlapping frames found"); return
    nl = np.array([x['nearest_lidar_m'] for x in res])
    rr = np.array([x['range_m'] for x in res])
    print(f"\nmatched {len(res)} lights to a lidar keyframe")
    print(f"  distance from triangulated point to NEAREST lidar return:")
    print(f"    median {np.median(nl):5.2f} m   p25 {np.percentile(nl,25):5.2f}   "
          f"p75 {np.percentile(nl,75):5.2f}   p90 {np.percentile(nl,90):5.2f}")
    for t in (0.5, 1.0, 2.0, 3.0):
        print(f"    within {t:.1f} m of structure: {np.mean(nl < t):5.0%}")
    close = rr < 30
    if close.sum():
        print(f"\n  lights closer than 30 m (where lidar is dense): "
              f"median {np.median(nl[close]):.2f} m, within 1 m {np.mean(nl[close] < 1):.0%} "
              f"(n={close.sum()})")
    far = rr >= 30
    if far.sum():
        print(f"  lights beyond 30 m (lidar sparse/absent):        "
              f"median {np.median(nl[far]):.2f} m, within 1 m {np.mean(nl[far] < 1):.0%} "
              f"(n={far.sum()})")
    print("\n  a random point in free space would sit metres from any return;")
    print("  landing on structure is evidence the geometry is right.")
    out = BASE + 'outputs/tlb/tl3d_lidar_check.json'
    json.dump(res, open(out, 'w'), indent=1)
    print(f"  wrote {out}")


if __name__ == '__main__':
    main()
