"""Occ3D-nuScenes occupancy labels -> the student's voxel grid and the teacher's 10 classes.

WHY
Occupancy has been distilled from the teacher for lack of ground truth, and scored as
agreement with the teacher (~11% mIoU on occupied classes): a bad target and a bad metric.
Occ3D-nuScenes (via the UniOcc repackaging on Hugging Face) gives voxel labels for every
keyframe: 200x200x16 at 0.4 m over [-40, 40] x [-40, 40] x [-1, 5.4] m, 17 semantic classes
+ free, with a camera-visibility mask. Both the teacher and the student can be scored
against it, and the student can be trained on it.

WHAT IS WRITTEN
Per frame `data/distill/frames/<token>/occ3d.npz`:
  occ   int8  (200, 200, 16)  class in the teacher's taxonomy (vehicle, czone_sign, bicycle,
        generic_object, pedestrian, traffic_cone, barrier, driveable, background, empty=9)
        on the student grid: 0.512 m cells over +-51.2 m, 16 pillars of 0.4 m from -1 m
  mask  uint8 (200, 200, 16)  1 where the cell lies inside the Occ3D range AND is camera
        visible -- only those voxels carry supervision or count in the metric.
The UniOcc arrays are (x, y, z) and the student grid is (y, x, z) (established by GT-box
overlap: 3.1% of occupied voxels fall inside GT boxes under the transpose vs <1.5% for
every other orientation; --transform auto re-runs a teacher-agreement check instead).

    python local/distill/occ3d_labels.py --split val
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))
import numpy as np

# UniOcc's occ_label ids (its own unification of Occ3D's 17 classes; not documented, so
# established empirically on 2026-09-25 from GT-box overlap and height profiles, see
# PLAN.md section 32): 1 vehicle (0.96 of its voxels inside car boxes), 2 bicycle (0.89),
# 4 pedestrian (0.88), 5 traffic cone (0.83), 7 road (ground slices), 8 other flat ground,
# 6 and 9 tall structure / vegetation, 0 others (rare), 3 never seen, 10 free.
UNIOCC_TO_TEACHER = {0: 8, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 8, 9: 8, 10: 9}
TEACHER = ["vehicle", "czone_sign", "bicycle", "generic_object", "pedestrian", "traffic_cone",
           "barrier", "driveable", "background", "empty"]
LUT = np.array([UNIOCC_TO_TEACHER.get(i, 8) for i in range(256)], np.int8)

N, CELL, X0 = 200, 0.512, -51.2            # student grid
ON, OCELL, OX0 = 200, 0.4, -40.0           # Occ3D grid

# student cell centre -> Occ3D index along one axis (or -1 outside)
_c = X0 + (np.arange(N) + 0.5) * CELL
_src = np.floor((_c - OX0) / OCELL).astype(int)
_src[(_src < 0) | (_src >= ON)] = -1
SRC_IY, SRC_IX = np.meshgrid(_src, _src, indexing="ij")      # (N, N) each
INSIDE = (SRC_IY >= 0) & (SRC_IX >= 0)


def resample(label, cam_mask, transform):
    """Occ3D (200,200,16) -> student (200,200,16) under an axis transform of the source."""
    lab = transform(label); msk = transform(cam_mask)
    out = np.full((N, N, 16), 9, np.int8); m = np.zeros((N, N, 16), np.uint8)
    iy, ix = np.clip(SRC_IY, 0, ON - 1), np.clip(SRC_IX, 0, ON - 1)
    out[INSIDE] = LUT[lab[iy[INSIDE], ix[INSIDE], :]]
    m[INSIDE] = msk[iy[INSIDE], ix[INSIDE], :]
    return out, m


TRANSFORMS = {
    "yx": lambda a: a,                                  # source already (y, x, z)
    "xy->yx": lambda a: np.transpose(a, (1, 0, 2)),
    "yx flipx": lambda a: a[:, ::-1],
    "yx flipy": lambda a: a[::-1],
    "yx flipxy": lambda a: a[::-1, ::-1],
    "xy->yx flipx": lambda a: np.transpose(a, (1, 0, 2))[:, ::-1],
    "xy->yx flipy": lambda a: np.transpose(a, (1, 0, 2))[::-1],
    "xy->yx flipxy": lambda a: np.transpose(a, (1, 0, 2))[::-1, ::-1],
}


def iou(a, b, c, m):
    A = (a == c) & (m > 0); B = (b == c) & (m > 0)
    return (A & B).sum() / max((A | B).sum(), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--root", default="data/occ3d/uniocc")
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--transform", default="xy->yx",
                    help="UniOcc arrays are (x, y, z); the student grid is (y, x, z). 'auto' re-runs the agreement check")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.chdir(_ROOT)
    src = Path(args.root) / f"NuScenes-via-Occ3D-2Hz-{args.split}"
    files = sorted(src.glob("scene-*/*.npz"))
    if args.limit: files = files[:args.limit]
    print(f"  {len(files)} Occ3D frames in {src}", flush=True)

    transform = args.transform
    if transform == "auto":
        # pick the orientation by agreement with the teacher's cached occupancy
        scores = {k: [] for k in TRANSFORMS}
        n = 0
        for f in files:
            a = np.load(f, allow_pickle=True); tok = str(a["sample_token"])
            tf = Path(args.teacher) / f"{tok}.npz"
            if not tf.exists(): continue
            t_occ = np.load(tf)["occ"]
            for k, T in TRANSFORMS.items():
                lab, m = resample(a["occ_label"], a["occ_mask_camera"], T)
                scores[k].append((iou(t_occ, lab, 0, m) + iou(t_occ, lab, 7, m) + iou(t_occ, lab, 9, m)) / 3)
            n += 1
            if n >= 40: break
        best = max(scores, key=lambda k: np.mean(scores[k]))
        print("  orientation check (mean IoU of vehicle/driveable/empty vs teacher over %d frames):" % n)
        for k in TRANSFORMS: print(f"    {k:16s} {np.mean(scores[k]):.3f}")
        transform = best; print(f"  -> using '{best}'", flush=True)
    T = TRANSFORMS[transform]
    t0 = time.time(); made = 0; missing = 0
    for i, f in enumerate(files):
        a = np.load(f, allow_pickle=True); tok = str(a["sample_token"])
        d = Path(args.frames) / tok
        if not d.is_dir(): missing += 1; continue
        lab, m = resample(a["occ_label"], a["occ_mask_camera"], T)
        np.savez_compressed(d / "occ3d.npz", occ=lab, mask=m)
        made += 1
        if (i + 1) % 1000 == 0: print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  wrote {made} occ3d.npz ({missing} tokens without a frame dir) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
