#!/usr/bin/env python
"""Render one demo video per session: ground truth against both methods, side by side.

The two methods answer the same question from the same frame and are drawn separately so
the difference is visible rather than asserted:

  VQA       Qwen-Drive answering in words, fine-tuned, no boxes at all
  PIPELINE  detector -> selector -> colour head, with every box it used drawn on the image

Colour coding: the annotated governing light is outlined in its true colour, other
annotated lights in thin white, and the detector's boxes in cyan with the one it voted
hardest for thickened. The inset is the governing lamp at 8x, which is the only way to see
what either method is actually looking at.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
ROOT = BASE / 'data/OpenLane-V2'
F = '/usr/share/fonts/truetype/dejavu/'
RGB = {'red': (255, 70, 70), 'green': (60, 235, 110), 'yellow': (255, 210, 60),
       None: (150, 150, 150)}
W, IH = 1280, 720
HDR, FTR = 44, 132
H = HDR + IH + FTR
CYAN = (0, 210, 255)


def fonts():
    return (ImageFont.truetype(F + 'DejaVuSans-Bold.ttf', 26),
            ImageFont.truetype(F + 'DejaVuSans.ttf', 19),
            ImageFont.truetype(F + 'DejaVuSans-Bold.ttf', 34),
            ImageFont.truetype(F + 'DejaVuSans.ttf', 15))


def chip(d, xy, w, h, colour, label, ok, fb, fs):
    x, y = xy
    d.rectangle([x, y, x + w, y + h], fill=(26, 26, 30), outline=(70, 70, 78), width=1)
    d.rectangle([x + 8, y + 8, x + 8 + 34, y + h - 8], fill=RGB[colour])
    d.text((x + 52, y + 9), (label or 'no answer').upper(), font=fb, fill=(238, 238, 240))
    if ok is not None:
        d.text((x + w - 34, y + 8), '✓' if ok else '✗', font=fb,
               fill=(90, 230, 120) if ok else (255, 90, 90))


def render(row, gtr, vqa, pipe, idx, n, tally, fonts_):
    fb, fr, fbig, ftiny = fonts_
    im = Image.open(ROOT / gtr['image']).convert('RGB')
    sx, sy = W / im.width, IH / im.height
    im = im.resize((W, IH), Image.LANCZOS)
    canvas = Image.new('RGB', (W, H), (14, 14, 17))
    canvas.paste(im, (0, HDR))
    d = ImageDraw.Draw(canvas)

    # detector boxes (cyan), thickened for the one the selector voted hardest for
    if pipe and pipe.get('boxes'):
        chosen = pipe.get('chosen', -1)
        for j, b in enumerate(pipe['boxes']):
            x1, y1, x2, y2 = b[0]*sx, b[1]*sy+HDR, b[2]*sx, b[3]*sy+HDR
            hard = (j == chosen)
            d.rectangle([x1, y1, x2, y2], outline=CYAN, width=3 if hard else 1)
            if hard:
                g = pipe['gov'][j] if j < len(pipe.get('gov', [])) else 0
                d.text((x1, y1 - 17), f"selector {g:.2f}", font=ftiny, fill=CYAN)

    # annotation: other lights thin white, the governing one in its true colour
    for b in gtr['other_boxes']:
        d.rectangle([b[0]*sx, b[1]*sy+HDR, b[2]*sx, b[3]*sy+HDR],
                    outline=(210, 210, 210), width=1)
    for b in gtr['gov_boxes']:
        d.rectangle([b[0]*sx-2, b[1]*sy+HDR-2, b[2]*sx+2, b[3]*sy+HDR+2],
                    outline=RGB[gtr['label']], width=3)

    # 8x inset of the governing lamp -- the only way to see what is being judged
    gb = max(gtr['gov_boxes'], key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    cx, cy = (gb[0]+gb[2])/2, (gb[1]+gb[3])/2
    half = max(gb[2]-gb[0], gb[3]-gb[1]) * 1.3 + 8
    src = Image.open(ROOT / gtr['image']).convert('RGB')
    ins = src.crop((int(cx-half), int(cy-half), int(cx+half), int(cy+half))).resize(
        (168, 168), Image.NEAREST)
    canvas.paste(ins, (W - 182, HDR + 14))
    d.rectangle([W-182, HDR+14, W-14, HDR+182], outline=RGB[gtr['label']], width=2)
    cap = f"governing lamp {gb[2]-gb[0]:.0f}x{gb[3]-gb[1]:.0f}px"
    d.text((W - 14 - d.textlength(cap, font=ftiny), HDR + 186), cap,
           font=ftiny, fill=(200, 200, 205))

    # header
    d.rectangle([0, 0, W, HDR], fill=(22, 22, 26))
    title = f"session {gtr['segment']}"
    d.text((14, 9), title, font=fb, fill=(240, 240, 245))
    x = 14 + d.textlength(title, font=fb) + 22          # place after the title, not over it
    d.text((x, 12), f"frame {idx+1}/{n}", font=fr, fill=(170, 170, 178))
    if gtr['discriminative']:
        d.text((x + 120, 12), "· discriminative: another visible light disagrees",
               font=fr, fill=(235, 190, 90))
    d.text((W-300, 12), f"VQA {tally['v']}/{tally['n']}   PIPELINE {tally['p']}/{tally['n']}",
           font=fr, fill=(200, 200, 208))

    # footer: ground truth, then each method
    y0 = HDR + IH
    d.rectangle([0, y0, W, H], fill=(18, 18, 22))
    d.text((16, y0 + 10), "GROUND TRUTH", font=ftiny, fill=(150, 150, 158))
    chip(d, (16, y0 + 30), 330, 46, gtr['label'], gtr['label'], None, fbig, fr)
    d.text((16, y0 + 84), "ego-lane light, from lane graph", font=ftiny, fill=(120, 120, 128))

    vp = vqa['pred'] if vqa else None
    d.text((400, y0 + 10), "VQA  (Qwen-Drive, fine-tuned, words only)", font=ftiny,
           fill=(150, 150, 158))
    chip(d, (400, y0 + 30), 400, 46, vp, vp, (vp == gtr['label']) if vqa else None, fbig, fr)

    pp = pipe['pred'] if pipe else None
    d.text((840, y0 + 10), "PIPELINE  (detector → selector → colour head)", font=ftiny,
           fill=(150, 150, 158))
    chip(d, (840, y0 + 30), 424, 46, pp, pp, (pp == gtr['label']) if pipe else None, fbig, fr)
    if pipe and pipe.get('boxes') is not None:
        d.text((840, y0 + 84), f"{len(pipe['boxes'])} lights detected, "
                               f"{len(pipe.get('voted', []))} voted", font=ftiny,
               fill=(120, 120, 128))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/tlb/demo_segments.jsonl')
    ap.add_argument('--vqa', default='outputs/tlb/hint_ft.json')
    ap.add_argument('--pipe', default='outputs/tlb/demo_e2e.json')
    ap.add_argument('--segments', nargs='+', default=None)
    ap.add_argument('--fps', type=float, default=4.0)
    ap.add_argument('--out-dir', default='outputs/tlb/demo')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    vqa = {x['key']: x for x in json.load(open(BASE / args.vqa))}
    pipe = {x['key']: x for x in json.load(open(BASE / args.pipe))}
    out = BASE / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fts = fonts()

    by = defaultdict(list)
    for r in rows:
        by[r['segment']].append(r)
    segs = args.segments or sorted(by)

    for seg in segs:
        frames = by[seg]
        tmp = out / f'_frames_{seg}'
        tmp.mkdir(exist_ok=True)
        tally = {'v': 0, 'p': 0, 'n': 0}
        for i, r in enumerate(frames):
            k = f"{r['segment']}/{r['timestamp']}"
            v, p = vqa.get(k), pipe.get(k)
            tally['n'] += 1
            tally['v'] += bool(v and v['pred'] == r['label'])
            tally['p'] += bool(p and p['pred'] == r['label'])
            render(r, r, v, p, i, len(frames), tally, fts).save(tmp / f'{i:04d}.png')
        mp4 = out / f'session_{seg}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
                        '-i', str(tmp / '%04d.png'), '-c:v', 'libx264', '-pix_fmt',
                        'yuv420p', '-crf', '20', str(mp4)], check=True)
        for f_ in tmp.glob('*.png'):
            f_.unlink()
        tmp.rmdir()
        print(f"  {mp4.name}  {len(frames)} frames   VQA {tally['v']}/{tally['n']} "
              f"({tally['v']/tally['n']:.0%})   PIPELINE {tally['p']}/{tally['n']} "
              f"({tally['p']/tally['n']:.0%})")


if __name__ == '__main__':
    main()
