#!/usr/bin/env python
"""Cut every labelled traffic light out of CAM_FRONT into a fixed-size crop.

Colour is easy from a clear, upscaled crop and hard from a 14 px lamp inside a 1600x900
frame -- that asymmetry is most of the baseline's error. This builds the training set for
a dedicated colour head that only ever sees crops.

Crops keep aspect by padding rather than stretching: a traffic light's tall-thin shape is
informative (which third of the housing is lit tells you the colour even when the hue is
washed out), and stretching to a square destroys it.
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

BASE = '/home/albert/Desktop/Qwen-Drive-1.0'
ROOT = os.path.join(BASE, 'data/OpenLane-V2')
COLOURS = ['unknown', 'red', 'green', 'yellow']
CIDX = {c: i for i, c in enumerate(COLOURS)}
SIZE = 64
PAD_FRAC = 0.35


def crop_one(im, box, size=SIZE):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = w * PAD_FRAC, h * PAD_FRAC
    x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    # square the box around its centre so the resize does not distort the housing
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    b = (int(round(cx - side / 2)), int(round(cy - side / 2)),
         int(round(cx + side / 2)), int(round(cy + side / 2)))
    c = im.crop(b)                       # PIL pads out-of-bounds with black
    return c.resize((size, size), Image.BICUBIC)


def do_file(job):
    split, lines = job
    X, y, meta = [], [], []
    c = Counter()
    for r in lines:
        try:
            im = Image.open(os.path.join(ROOT, r['image'])).convert('RGB')
        except Exception:
            c['img_fail'] += 1
            continue
        for L in r['lights']:
            x1, y1, x2, y2 = L['box']
            if x2 - x1 < 2 or y2 - y1 < 2:
                c['too_small'] += 1
                continue
            X.append(np.asarray(crop_one(im, L['box']), dtype=np.uint8))
            y.append(CIDX[L['colour']])
            meta.append((r['segment'], r['timestamp'], round(x2 - x1, 1), round(y2 - y1, 1),
                         L['is_gov']))
            c[L['colour']] += 1
    return split, X, y, meta, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out-dir', default='data/tlb/crops')
    args = ap.parse_args()
    os.makedirs(os.path.join(BASE, args.out_dir), exist_ok=True)
    for split in args.splits:
        rows = [json.loads(l) for l in open(os.path.join(BASE, f'data/tlb/det_{split}.jsonl'))]
        chunks = [(split, rows[i::args.workers]) for i in range(args.workers)]
        X, y, meta = [], [], []
        tot = Counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for _, Xi, yi, mi, ci in ex.map(do_file, chunks):
                X += Xi; y += yi; meta += mi; tot.update(ci)
        X = np.stack(X); y = np.asarray(y, dtype=np.int64)
        p = os.path.join(BASE, args.out_dir, f'{split}.npz')
        np.savez_compressed(p, X=X, y=y,
                            seg=np.array([m[0] for m in meta]),
                            ts=np.array([m[1] for m in meta]),
                            w=np.array([m[2] for m in meta], dtype=np.float32),
                            h=np.array([m[3] for m in meta], dtype=np.float32),
                            is_gov=np.array([m[4] for m in meta], dtype=np.int64))
        print(f"{split}: {len(X)} crops {X.shape}  {dict(tot)}")
        print(f"  colour counts: " + str({COLOURS[i]: int((y == i).sum()) for i in range(4)}))
        print(f"  wrote {p} ({os.path.getsize(p)/1e6:.0f} MB)")


if __name__ == '__main__':
    main()
