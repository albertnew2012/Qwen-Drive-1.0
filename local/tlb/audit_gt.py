#!/usr/bin/env python
"""Render a contact sheet of GT rows so the labels can be checked by eye.

Each tile is a crop around the governing traffic light, with the annotated box drawn in
the colour the label claims. If the label is right, the lamp inside the box shows that
colour. Other visible lights are boxed in white for contrast.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from PIL import Image, ImageDraw

R = Path('/home/albert/Desktop/Qwen-Drive-1.0/data/OpenLane-V2')
RGB = {'red': (255, 60, 60), 'green': (60, 255, 60), 'yellow': (255, 220, 60)}
TILE = 320


def tile(row, pad_frac=1.4, min_pad=60):
    im = Image.open(R / row['image']).convert('RGB')
    bs = row['gov_boxes']
    x1 = min(b[0] for b in bs); y1 = min(b[1] for b in bs)
    x2 = max(b[2] for b in bs); y2 = max(b[3] for b in bs)
    w, h = x2 - x1, y2 - y1
    px, py = max(w * pad_frac, min_pad), max(h * pad_frac, min_pad)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(im.width, x2 + px), min(im.height, y2 + py)
    crop = im.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
    sx = TILE / crop.width; sy = TILE / crop.height
    crop = crop.resize((TILE, TILE), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    for b in row['other_boxes']:
        d.rectangle([(b[0]-cx1)*sx, (b[1]-cy1)*sy, (b[2]-cx1)*sx, (b[3]-cy1)*sy],
                    outline=(255, 255, 255), width=2)
    for b in bs:
        d.rectangle([(b[0]-cx1)*sx, (b[1]-cy1)*sy, (b[2]-cx1)*sx, (b[3]-cy1)*sy],
                    outline=RGB[row['label']], width=3)
    tag = f"{row['label'].upper()}"
    if row['discriminative']:
        tag += f"  DISC(other={','.join(row['other_colours'])})"
    d.rectangle([0, 0, TILE, 18], fill=(0, 0, 0))
    d.text((4, 4), f"{tag}  {row['max_gov_wh'][0]:.0f}x{row['max_gov_wh'][1]:.0f}px", fill=RGB[row['label']])
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/tlb/val.jsonl')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--only', default='all', choices=['all', 'disc', 'red', 'green', 'yellow', 'small'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open('/home/albert/Desktop/Qwen-Drive-1.0/' + args.src)]
    if args.only == 'disc':
        rows = [r for r in rows if r['discriminative']]
    elif args.only == 'small':
        rows = [r for r in rows if r['max_gov_wh'][0] < 14]
    elif args.only in ('red', 'green', 'yellow'):
        rows = [r for r in rows if r['label'] == args.only]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]

    cols = args.cols; rn = (len(rows) + cols - 1) // cols
    sheet = Image.new('RGB', (TILE * cols, TILE * rn), (20, 20, 20))
    for i, r in enumerate(rows):
        sheet.paste(tile(r), (TILE * (i % cols), TILE * (i // cols)))
    sheet.save(args.out)
    print(f"wrote {args.out}  ({len(rows)} tiles)")
    for r in rows:
        print(f"  {r['segment']}/{r['timestamp']} {r['label']:6s} disc={int(r['discriminative'])} "
              f"gov={r['n_gov']} other={r['n_other']}{r['other_colours']} {r['max_gov_wh']}")


if __name__ == '__main__':
    main()
