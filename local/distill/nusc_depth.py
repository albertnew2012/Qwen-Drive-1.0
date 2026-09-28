"""Sparse lidar depth targets for the student's lift-splat, at feature resolution.

WHY
The student's LiftSplat predicts a 64-bin depth distribution per feature cell and that
distribution had NO supervision at all -- it was learned only through the detection
gradient. If the depth is wrong the image feature is scattered into the wrong BEV cell,
and no amount of head training recovers it. Supervising depth from lidar is the single
largest known lever in camera-only BEV detection (this is what BEVDepth is), and 18 GB
of nuScenes lidar was sitting unused beside the cache the whole time.

WHAT IS STORED
Per frame, one int8 array of shape (n_cams, H, W) = (6, 32, 56) at the stride-16 feature
grid, holding the depth-bin index of the closest lidar return in that cell, or -1 where
no point landed. 10.7 kB a frame, ~275 MB for all 25,599 -- against ~26 MB a frame if the
dense soft distribution were stored.

Closest return, not mean: a cell straddling an object edge gets foreground and background
points, and the mean of those is a depth where nothing exists.

``lidar2img`` comes from the teacher cache and already has the 896x512 resize folded in,
so pixel coordinates land directly on the image the student consumes.

    python local/distill/nusc_depth.py --root data/nuscenes_trainval --version v1.0-trainval
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes_trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--depth-bins", type=int, default=64)
    ap.add_argument("--depth-min", type=float, default=1.0)
    ap.add_argument("--depth-max", type=float, default=60.0)
    ap.add_argument("--image-size", type=int, nargs=2, default=[896, 512])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-name", default="depth.npz",
                    help="per-frame file name; use depth_WxH.npz for a non-default --image-size")
    args = ap.parse_args()
    os.chdir(_ROOT)

    root = Path(args.root)
    meta = root / args.version

    def load(name):
        return json.loads((meta / f"{name}.json").read_text())

    print("  reading sample_data", flush=True)
    lidar_of = {}
    for d in load("sample_data"):
        if not d.get("is_key_frame"):
            continue
        parts = Path(d["filename"]).parts
        if len(parts) > 1 and parts[1] == "LIDAR_TOP":
            lidar_of[d["sample_token"]] = d["filename"]

    W_img, H_img = args.image_size
    Hf, Wf = H_img // args.stride, W_img // args.stride
    edges = np.linspace(args.depth_min, args.depth_max, args.depth_bins + 1)

    frames = sorted(p for p in Path(args.frames).iterdir() if p.is_dir())
    todo = [f for f in frames if not (f / args.out_name).exists()]
    if args.limit:
        todo = todo[:args.limit]
    if args.of > 1:
        todo = todo[args.shard::args.of]
    print(f"  {len(todo)} frames to do (shard {args.shard}/{args.of})", flush=True)

    done = skipped = 0
    for d in todo:
        tok = d.name
        tc = Path(args.teacher) / f"{tok}.npz"
        if tok not in lidar_of:
            skipped += 1
            continue
        pc_file = root / lidar_of[tok]
        if not pc_file.exists():
            skipped += 1
            continue
        pts = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 5)[:, :3]
        # lidar2img at the STUDENT's image size. The teacher cache stores it at 896x512;
        # rows 0-1 scale with the image (a bug until 2026-09-26 02:00: the 1152x640 and
        # 1408x768 targets were binned in 896x512 pixel space, i.e. compressed by 0.78).
        # Frames the teacher was never run on get the same matrix from calib.npz
        # (1600x900 intrinsics), exactly as DistillSet._teacher_or_calib does.
        if tc.exists():
            l2i = np.load(tc)["lidar2img"].astype(np.float64)     # (6, 4, 4) at 896x512
            S = np.diag([W_img / 896.0, H_img / 512.0, 1.0, 1.0])
        else:
            cal = np.load(d / "calib.npz")
            l2i = []
            for ci in range(cal["cam_intrinsic"].shape[0]):
                s2l = np.eye(4); s2l[:3, :3] = cal["sensor2lidar_rotation"][ci]; s2l[:3, 3] = cal["sensor2lidar_translation"][ci]
                Kp = np.eye(4); Kp[:3, :3] = cal["cam_intrinsic"][ci]
                l2i.append(Kp @ np.linalg.inv(s2l))
            l2i = np.asarray(l2i, np.float64)
            S = np.diag([W_img / 1600.0, H_img / 900.0, 1.0, 1.0])
        l2i = S[None] @ l2i

        out = np.full((l2i.shape[0], Hf, Wf), -1, np.int8)
        hom = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], 1)
        for c in range(l2i.shape[0]):
            p = hom @ l2i[c].T
            z = p[:, 2]
            ok = z > args.depth_min
            if not ok.any():
                continue
            u = p[ok, 0] / z[ok]
            v = p[ok, 1] / z[ok]
            dep = z[ok]
            m = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img) & (dep < args.depth_max)
            if not m.any():
                continue
            fu = (u[m] / args.stride).astype(np.int32)
            fv = (v[m] / args.stride).astype(np.int32)
            db = np.clip(np.digitize(dep[m], edges) - 1, 0, args.depth_bins - 1)
            # closest return wins: sort far-to-near so nearer overwrites
            order = np.argsort(-dep[m])
            out[c, fv[order], fu[order]] = db[order].astype(np.int8)
        tmp = d / (args.out_name + f".tmp{os.getpid()}")
        np.savez_compressed(tmp, depth_bin=out)               # savez appends .npz
        os.replace(str(tmp) + ".npz", d / args.out_name)     # atomic: readers never see a partial file
        done += 1
        if done % 2000 == 0:
            cov = float((out >= 0).mean())
            print(f"  {done}/{len(todo)}  (last frame cell coverage {cov:.1%})", flush=True)

    print(f"  wrote {done} frames, skipped {skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
