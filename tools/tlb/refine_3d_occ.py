#!/usr/bin/env python
"""Refine triangulated traffic-light positions against Occ3D-nuScenes occupancy.

Triangulation alone lands ~1 m from real structure (measured against mini lidar). Occ3D
gives dense semantic voxels for all 850 scenes at 0.4 m, in the ego frame, with a camera
visibility mask -- so each point can be snapped to the nearest `manmade` voxel that the
cameras actually observed.

Only lights CONFIRMED this way are kept. An unconfirmed light is not evidence of a wrong
position: Occ3D labels only observed voxels, so absence is often just absence of
observation. Keeping the confirmed subset trades quantity for a label whose accuracy is
bounded by the voxel size rather than by triangulation error.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import Counter

import numpy as np

BASE = '/home/albert/Desktop/Qwen-Drive-1.0/'
GTS = '/home/albert/Desktop/OccWorld/data/nuscenes/gts/'
OLV2 = BASE + 'data/OpenLane-V2/'
MANMADE, FREE = 15, 17
VOX, X0, Y0, Z0 = 0.4, -40.0, -40.0, -1.0
NX = NY = 200
NZ = 16


def load_map():
    p = '/tmp/claude-1000/-home-albert-Desktop-Qwen-Drive-1-0/b272f55a-5e58-4d50-8938-73722dc82138/scratchpad/ts2occ.json'
    return {int(k): tuple(v) for k, v in json.load(open(p)).items()}


def neighbourhood(sem, mask, Xe, radius_m):
    """Camera-visible `manmade` voxel centres within `radius_m` of Xe (ego frame)."""
    r = int(np.ceil(radius_m / VOX))
    ix = int((Xe[0] - X0) / VOX); iy = int((Xe[1] - Y0) / VOX); iz = int((Xe[2] - Z0) / VOX)
    xs = slice(max(ix - r, 0), min(ix + r + 1, NX))
    ys = slice(max(iy - r, 0), min(iy + r + 1, NY))
    zs = slice(max(iz - r, 0), min(iz + r + 1, NZ))
    sub = sem[xs, ys, zs]; msk = mask[xs, ys, zs]
    sel = (sub == MANMADE) & (msk > 0)
    if not sel.any():
        return None
    gx, gy, gz = np.nonzero(sel)
    cx = X0 + (xs.start + gx + 0.5) * VOX
    cy = Y0 + (ys.start + gy + 0.5) * VOX
    cz = Z0 + (zs.start + gz + 0.5) * VOX
    P = np.stack([cx, cy, cz], 1)
    d = np.linalg.norm(P - Xe, axis=1)
    keep = d <= radius_m
    return (P[keep], d[keep]) if keep.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tl3d', default='data/tlb/tl3d_val_all.jsonl')
    ap.add_argument('--radius', type=float, default=1.5)
    ap.add_argument('--out', default='data/tlb/tl3d_val_refined.jsonl')
    args = ap.parse_args()

    ts2occ = load_map()
    rows = [json.loads(l) for l in open(BASE + args.tl3d)]
    tl = [r for r in rows if r['category'] == 1]
    # the plausibility gate stays, but it is a prior, not evidence -- Occ3D is the evidence
    cand = [r for r in tl if r['reproj_px_median'] < 5 and 1.5 < r['height_m'] < 9
            and r['range_m'] < 120]
    print(f"traffic lights: {len(tl)}   passing the geometric gate: {len(cand)}")

    # cache ego poses per segment so each info file is read once
    poses = {}
    for r in cand:
        key = (r['split'], r['segment'])
        if key in poses:
            continue
        pp = {}
        for f in sorted(glob.glob(f"{OLV2}{r['split']}/{r['segment']}/info/*.json")):
            ts = int(os.path.basename(f)[:-5])
            d = json.load(open(f))
            pp[ts] = (np.asarray(d['pose']['rotation'], float),
                      np.asarray(d['pose']['translation'], float))
        poses[key] = pp

    out = []
    stats = Counter()
    occ_cache = {}
    for r in cand:
        pp = poses[(r['split'], r['segment'])]
        X = np.asarray(r['xyz_world'], float)
        # evaluate at the frame where the light is nearest: best observed, densest labels
        best = None
        for ts, (Re, te) in pp.items():
            if ts not in ts2occ:
                continue
            Xe = Re.T @ (X - te)
            rng = np.linalg.norm(Xe[:2])
            if abs(Xe[0]) > 39 or abs(Xe[1]) > 39 or not (-1 < Xe[2] < 5.4):
                continue                       # outside the Occ3D volume at this frame
            if best is None or rng < best[0]:
                best = (rng, ts, Xe, Re, te)
        if best is None:
            stats['outside_occ_volume'] += 1
            continue
        rng, ts, Xe, Re, te = best
        scene, tok = ts2occ[ts]
        f = f"{GTS}{scene}/{tok}/labels.npz"
        if not os.path.exists(f):
            stats['no_occ_file'] += 1
            continue
        if f not in occ_cache:
            if len(occ_cache) > 64:
                occ_cache.clear()
            z = np.load(f)
            occ_cache[f] = (z['semantics'], z['mask_camera'])
        sem, mask = occ_cache[f]
        hood = neighbourhood(sem, mask, Xe, args.radius)
        if hood is None:
            stats['no_visible_manmade'] += 1
            continue
        P, d = hood
        j = int(np.argmin(d))
        Xe_ref = P[j]
        shift = float(d[j])
        X_ref = Re @ Xe_ref + te               # back to world
        out.append({**r, 'confirmed': True, 'snap_shift_m': round(shift, 2),
                    'occ_scene': scene, 'occ_token': tok, 'occ_ts': ts,
                    'xyz_world_refined': [round(float(v), 2) for v in X_ref],
                    'ego_refined': [round(float(v), 2) for v in Xe_ref],
                    'range_refined_m': round(float(np.linalg.norm(Xe_ref[:2])), 1),
                    'height_refined_m': round(float(Xe_ref[2]), 2),
                    'n_manmade_near': int(len(P))})
        stats['confirmed'] += 1

    p = BASE + args.out
    with open(p, 'w') as fh:
        for r in out:
            fh.write(json.dumps(r) + '\n')
    print(f"\n  confirmed by a camera-visible 'manmade' voxel within {args.radius} m: "
          f"{stats['confirmed']} ({stats['confirmed']/max(len(cand),1):.0%} of candidates)")
    for k, v in stats.most_common():
        if k != 'confirmed':
            print(f"  {k:24s} {v}")
    if out:
        s = np.array([r['snap_shift_m'] for r in out])
        h = np.array([r['height_refined_m'] for r in out])
        hb = np.array([r['height_m'] for r in out])
        print(f"\n  snap distance (m):    median {np.median(s):.2f}  p90 {np.percentile(s,90):.2f}")
        print(f"  height before snap:   median {np.median(hb):.2f} m")
        print(f"  height after  snap:   median {np.median(h):.2f} m")
        print(f"  label accuracy is now bounded by the 0.4 m voxel, not by triangulation")
    print(f"  wrote {p}")


if __name__ == '__main__':
    main()
