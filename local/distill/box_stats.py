"""Per-dimension statistics of the teacher's box targets.

Measured over the queries the teacher actually asserts:

    dim 0 x      std 24.773        dim 5 h      std  0.341
    dim 1 y      std 17.173        dim 6 rot_s  std  0.663
    dim 2 z      std  1.104        dim 7 rot_c  std  0.648
    dim 3 w      std  0.650        dim 8 vx     std  1.889
    dim 4 l      std  0.676        dim 9 vy     std  1.468

An unweighted L1 over those is 94% position: the first two dimensions are two orders of
magnitude wider than size and rotation, so those are effectively unsupervised. The
student therefore regresses a standardised target and a fixed affine restores teacher
units at the output, which keeps the ONNX signature identical while giving every
dimension comparable gradient.
"""
from __future__ import annotations

import argparse, glob, os
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--out", default="data/distill/box_stats.npz")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()
    os.chdir(_ROOT)

    files = sorted(glob.glob(f"{args.teacher}/*.npz"))[:args.limit]
    rows = []
    for f in files:
        d = np.load(f)
        k = d["keep"]
        if len(k):
            rows.append(d["box"][k])
    if not rows:
        print("  nothing cached yet")
        return 1
    x = np.concatenate(rows, 0).astype(np.float64)
    mean = x.mean(0).astype(np.float32)
    std = np.maximum(x.std(0), 1e-2).astype(np.float32)
    np.savez(args.out, mean=mean, std=std, n=np.int64(len(x)))
    print(f"  {len(x)} boxes from {len(files)} frames")
    print("  mean " + " ".join(f"{v:7.3f}" for v in mean))
    print("  std  " + " ".join(f"{v:7.3f}" for v in std))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
