#!/usr/bin/env python
"""Recover 3D traffic-light positions from the 2D boxes, by triangulation.

OpenLane-V2 annotates traffic elements only as 2D boxes in CAM_FRONT, so a 3D head has
nothing to train against. But the element ids are stable within a segment and every frame
carries the ego pose plus the camera's intrinsics and extrinsics, so the same light seen
from a moving vehicle gives many rays through one point in the world.

For each frame the box centre back-projects to a ray:
    d_cam   = K^-1 [u v 1]
    d_world = R_ego @ R_cam @ d_cam          (unit)
    C_world = R_ego @ t_cam + t_ego          (camera centre)
The point closest to all rays is the least-squares solution of
    (sum_i I - d_i d_i^T) X = (sum_i (I - d_i d_i^T) C_i)

Validated, not assumed: reprojection error in pixels, height above the road, and range.
A light that lands 40 m up or behind the car means the triangulation failed.
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = '/home/albert/Desktop/Qwen-Drive-1.0/data/OpenLane-V2'
CAM = 'CAM_FRONT'


def rays_for_segment(split, seg, frames):
    """All (camera centre, direction, frame) observations, grouped by element id."""
    obs = defaultdict(list)
    meta = {}
    for fr in frames:
        p = os.path.join(ROOT, split, seg, 'info', fr)
        d = json.load(open(p))
        s = d['sensor'][CAM]
        K = np.asarray(s['intrinsic']['K'], float)
        Rc = np.asarray(s['extrinsic']['rotation'], float)
        tc = np.asarray(s['extrinsic']['translation'], float).reshape(3)
        Re = np.asarray(d['pose']['rotation'], float)
        te = np.asarray(d['pose']['translation'], float).reshape(3)
        Kinv = np.linalg.inv(K)
        C = Re @ tc + te
        for t in d['annotation']['traffic_element']:
            (x1, y1), (x2, y2) = t['points']
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            dc = Kinv @ np.array([u, v, 1.0])
            dw = Re @ (Rc @ dc)
            n = np.linalg.norm(dw)
            if n < 1e-9:
                continue
            obs[t['id']].append((C, dw / n, fr, (x1, y1, x2, y2), K, Rc, Re, te, tc))
            meta[t['id']] = (t['category'], t['attribute'])
    return obs, meta


def _solve(rays, w=None):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i, (C, d, *_) in enumerate(rays):
        wi = 1.0 if w is None else w[i]
        if wi <= 0:
            continue
        P = (np.eye(3) - np.outer(d, d)) * wi
        A += P
        b += P @ C
    if np.linalg.cond(A) > 1e10:
        return None
    return np.linalg.solve(A, b)


def triangulate(rays, iters=4):
    """Robust least squares: refit while down-weighting rays that disagree.

    A traffic light's apparent centre drifts as the viewing angle changes (the housing is
    a 3D object seen as a 2D box), and an id occasionally jumps between physical lights.
    Plain least squares lets a few such rays drag the point far off; Tukey weights on the
    perpendicular distance keep the consensus.
    """
    X = _solve(rays)
    if X is None:
        return None
    for _ in range(iters):
        r = np.array([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - C))
                      for C, d, *_ in rays])
        s = max(1.4826 * np.median(r), 0.05)        # robust scale
        u = np.clip(r / (4.685 * s), 0, 1)
        w = (1 - u ** 2) ** 2                        # Tukey biweight
        Xn = _solve(rays, w)
        if Xn is None:
            break
        if np.linalg.norm(Xn - X) < 1e-3:
            X = Xn
            break
        X = Xn
    return X


def reproj_error(X, rays):
    errs = []
    for C, d, fr, box, K, Rc, Re, te, tc in rays:
        Xe = Re.T @ (X - te)              # world -> ego
        Xc = Rc.T @ (Xe - tc)             # ego   -> camera
        if Xc[2] <= 0.1:
            continue
        uv = K @ (Xc / Xc[2])
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        errs.append(float(np.hypot(uv[0] - cx, uv[1] - cy)))
    return errs


def do_segment(job):
    split, seg, frames = job
    obs, meta = rays_for_segment(split, seg, frames)
    out = []
    for tid, rays in obs.items():
        if len(rays) < 3:
            continue
        # a light seen only from nearly one place gives a degenerate baseline
        Cs = np.array([r[0] for r in rays])
        if np.linalg.norm(Cs.max(0) - Cs.min(0)) < 2.0:
            continue
        # angular spread: rays that are nearly parallel cannot fix a depth
        D = np.array([r[1] for r in rays])
        spread = float(np.degrees(np.arccos(np.clip(
            (D @ D.T).min(), -1, 1))))
        if spread < 1.0:
            continue
        X = triangulate(rays)
        if X is None:
            continue
        errs = reproj_error(X, rays)
        if not errs:
            continue
        # range and bearing at the frame where it is nearest
        best = min(rays, key=lambda r: np.linalg.norm(X - r[7]))
        Re, te = best[6], best[7]
        Xe = Re.T @ (X - te)
        out.append({'split': split, 'segment': seg, 'id': tid,
                    'category': meta[tid][0], 'attribute': meta[tid][1],
                    'n_obs': len(rays), 'baseline_m': round(float(
                        np.linalg.norm(Cs.max(0) - Cs.min(0))), 2),
                    'xyz_world': [round(float(v), 2) for v in X],
                    'height_m': round(float(X[2]), 2),
                    'range_m': round(float(np.linalg.norm(Xe[:2])), 1),
                    'fwd_m': round(float(Xe[0]), 1), 'lat_m': round(float(Xe[1]), 1),
                    'ray_spread_deg': round(spread, 2),
                    'reproj_px_median': round(float(np.median(errs)), 1),
                    'reproj_px_p90': round(float(np.percentile(errs, 90)), 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit-segments', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', default='data/tlb/tl3d_val.jsonl')
    args = ap.parse_args()
    split_map = json.load(open(os.path.join(ROOT, 'data_dict_subset_B.json')))
    segs = sorted(split_map[args.split].items())
    if args.limit_segments:
        segs = segs[:args.limit_segments]
    jobs = [(args.split, s, f) for s, f in segs]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(do_segment, jobs, chunksize=2):
            rows += r
    p = '/home/albert/Desktop/Qwen-Drive-1.0/' + args.out
    with open(p, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    tl = [r for r in rows if r['category'] == 1]
    err = np.array([r['reproj_px_median'] for r in tl])
    h = np.array([r['height_m'] for r in tl])
    rg = np.array([r['range_m'] for r in tl])
    print(f"triangulated {len(rows)} elements ({len(tl)} traffic lights) over {len(segs)} segments")
    print(f"\n  reprojection error (px):  median {np.median(err):5.1f}   "
          f"p90 {np.percentile(err, 90):5.1f}   <2px {np.mean(err < 2):.0%}  <5px {np.mean(err < 5):.0%}")
    print(f"  height above road (m):    median {np.median(h):5.2f}   "
          f"p10 {np.percentile(h, 10):5.2f}   p90 {np.percentile(h, 90):5.2f}")
    print(f"  range at closest (m):     median {np.median(rg):5.1f}   "
          f"p10 {np.percentile(rg, 10):5.1f}   p90 {np.percentile(rg, 90):5.1f}")
    good = (err < 5) & (h > 1.5) & (h < 9) & (rg < 120)
    print(f"\n  physically plausible AND well-fit: {good.sum()}/{len(tl)} ({good.mean():.0%})")
    print(f"  (reproj <5 px, height 1.5-9 m, range <120 m)")
    print(f"  wrote {p}")


if __name__ == '__main__':
    main()
