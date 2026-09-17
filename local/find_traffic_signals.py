#!/usr/bin/env python
"""Find traffic signal heads in nuScenes frames and rank them by how legible they are.

nuScenes has no traffic light annotations, so the signals have to be found from the
pixels. Lit lamps are small, highly saturated blobs of one of three hues sitting above
the road, which is distinctive enough to locate them without a detector. Ranking by
blob area then tells you which keyframe shows a signal closest to the camera - that is
the only place you can tell an arrow from a ball.

    .venv/bin/python local/find_traffic_signals.py --scenes scene-0061 --dump 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]


def load_index(root: Path, version: str):
    meta = root / version
    load = lambda n: json.loads((meta / n).read_text())
    sample = {s["token"]: s for s in load("sample.json")}
    cs = {c["token"]: c for c in load("calibrated_sensor.json")}
    sensor = {s["token"]: s for s in load("sensor.json")}
    scenes = {s["name"]: s for s in load("scene.json")}
    bych: dict[str, dict[str, str]] = {}
    for d in load("sample_data.json"):
        if not d["is_key_frame"]:
            continue
        ch = sensor[cs[d["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        bych.setdefault(d["sample_token"], {})[ch] = d["filename"]
    return sample, scenes, bych


def lamps(img: Image.Image):
    """Lit lamp blobs as (colour, x0, y0, x1, y1, area)."""
    a = np.asarray(img.convert("HSV")).astype(int)
    H, S, V = a[..., 0], a[..., 1], a[..., 2]
    masks = {
        "red": ((H < 8) | (H > 245)) & (S > 90) & (V > 110),
        "amber": (H > 12) & (H < 35) & (S > 120) & (V > 150),
        "green": (H > 95) & (H < 135) & (S > 80) & (V > 110),
    }
    found = []
    for colour, m in masks.items():
        m = m.copy()
        m[int(img.height * 0.72):, :] = False          # below this is road, not signals
        lab, n = ndimage.label(m)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            area = int((lab[sl] == i).sum())
            if not 3 <= area <= 400:
                continue
            ys, xs = sl
            # lamps are roughly round; long thin blobs are reflections and railings
            w, h = xs.stop - xs.start, ys.stop - ys.start
            if max(w, h) > 3 * max(1, min(w, h)):
                continue
            found.append((colour, xs.start, ys.start, xs.stop, ys.stop, area))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", default="scene-0061,scene-0103,scene-0553,scene-0757,scene-0796")
    ap.add_argument("--cams", default="CAM_FRONT")
    ap.add_argument("--dump", type=int, default=4, help="zoom crops of the N largest heads")
    ap.add_argument("--out", default="outputs/traffic_light/signals")
    args = ap.parse_args()

    root = Path(args.dataroot)
    sample, scenes, bych = load_index(root, args.version)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = args.cams.split(",")

    for name in args.scenes.split(","):
        toks, tok = [], scenes[name]["first_sample_token"]
        while tok:
            toks.append(tok)
            tok = sample[tok]["next"]
        print(f"\n{name}: {scenes[name]['description'][:64]}")
        rows = []
        for kf, t in enumerate(toks):
            for cam in cams:
                fn = bych[t].get(cam)
                if not fn:
                    continue
                img = Image.open(root / fn)
                for colour, x0, y0, x1, y1, area in lamps(img):
                    rows.append({"kf": kf, "cam": cam, "colour": colour, "area": area,
                                 "box": (x0, y0, x1, y1), "file": fn})
        if not rows:
            print("  no lamps detected")
            continue
        by_colour: dict[str, int] = {}
        for r in rows:
            by_colour[r["colour"]] = by_colour.get(r["colour"], 0) + 1
        print(f"  {len(rows)} lamp detections  {by_colour}")
        rows.sort(key=lambda r: -r["area"])
        print("  largest (most legible) heads:")
        for r in rows[:args.dump]:
            x0, y0, x1, y1 = r["box"]
            print(f"    kf {r['kf']:2d}  {r['cam']:16s} {r['colour']:5s} area {r['area']:3d}  "
                  f"at x{x0}-{x1} y{y0}-{y1}")
        for n, r in enumerate(rows[:args.dump]):
            x0, y0, x1, y1 = r["box"]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            img = Image.open(root / r["file"])
            # a signal head is tall: take a box far bigger than the lit lamp itself
            pad_x, pad_y = 70, 110
            crop = img.crop((max(0, cx - pad_x), max(0, cy - pad_y),
                             min(img.width, cx + pad_x), min(img.height, cy + pad_y)))
            crop = crop.resize((crop.width * 6, crop.height * 6), Image.LANCZOS)
            p = out / f"{name}_kf{r['kf']:02d}_{r['colour']}_{n}.jpg"
            crop.save(p, quality=95)
        print(f"  wrote {min(args.dump, len(rows))} crops to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
