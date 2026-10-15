"""Occ3D labels on their NATIVE grid, in the teacher's exact occupancy format.

The teacher predicts occupancy on Occ3D's own grid: 200x200x16 cells of 0.4 m over [-40, 40] x [-40, 40] x
[-1, 5.4] m, indexed (x, y, z) -- established 2026-10-01 against ground-truth vehicle boxes (95.5% of the
teacher's vehicle voxels fall inside a box when read as (x, y), 6.8% as (y, x)) and against these labels
(teacher mIoU 0.353 with the UniOcc arrays used as they are, 0.05 transposed). UniOcc stores the labels in
that same order, so no resampling and no transpose: only the class mapping.

`occ3d_labels.py` (the older converter) resampled the labels onto the student's 0.512 m detection grid in
(y, x) order instead; scoring the teacher against those is what produced its spurious 0.03 mIoU.

Writes per keyframe `data/distill/frames/<token>/occ3d_native.npz`:
  occ   int8  (200, 200, 16)  teacher taxonomy (vehicle ... background, empty = 9), index (x, y, z)
  mask  uint8 (200, 200, 16)  1 = camera-visible voxel (the Occ3D benchmark's evaluation region)
and `data/distill/occ3d_native_freq.npy`: class counts over the camera-visible voxels of the TRAIN split only.

    python distill/occ3d_native_labels.py --workers 40
"""
from __future__ import annotations

import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))
from distill.occ3d_labels import LUT

FRAMES = "data/distill/frames"


def convert(path: str):
    a = np.load(path, allow_pickle=True)
    tok = str(a["sample_token"]); d = Path(FRAMES) / tok
    if not d.is_dir():
        return tok, None
    occ = LUT[a["occ_label"]].astype(np.int8)
    mask = (a["occ_mask_camera"] > 0).astype(np.uint8)
    tmp = d / "occ3d_native.tmp.npz"
    np.savez_compressed(tmp, occ=occ, mask=mask); os.replace(tmp, d / "occ3d_native.npz")
    return tok, np.bincount(occ[mask > 0].astype(np.int64), minlength=10)[:10]


def main() -> int:
    global FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/occ3d/uniocc")
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.chdir(_ROOT); FRAMES = args.frames
    files = sorted(str(p) for s in ("train", "val") for p in (Path(args.root) / f"NuScenes-via-Occ3D-2Hz-{s}").glob("scene-*/*.npz"))
    if args.limit: files = files[:args.limit]
    print(f"  {len(files)} UniOcc frames -> occ3d_native.npz with {args.workers} workers", flush=True)
    train = set(json.load(open("data/distill/scene_split.json"))["train_tokens"]) | \
        set(json.load(open("data/distill/scene_split.json")).get("uncached_train_tokens", []))
    counts = np.zeros(10, np.int64); made = missing = 0; t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (tok, c) in enumerate(pool.imap_unordered(convert, files, chunksize=16)):
            if c is None: missing += 1; continue
            made += 1
            if tok in train: counts += c
            if (i + 1) % 2000 == 0: print(f"  {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    if not args.limit:
        np.save("data/distill/occ3d_native_freq.npy", counts)
    print(f"  wrote {made} occ3d_native.npz ({missing} without a frame dir) in {time.time() - t0:.0f}s; "
          f"train class fractions: {np.round(counts / max(counts.sum(), 1), 4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
