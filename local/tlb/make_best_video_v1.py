#!/usr/bin/env python
"""One video, the best of each approach side by side on the same frame.

  left  PERCEPTION  end-to-end pipeline: detect -> select the ego's lights -> colour
                    (97.85% where a light exists). Draws the ego SET, not a single pick,
                    because 80% of frames genuinely have more than one governing light.
  right VQA         Qwen-Drive fine-tuned, answering in words from the image alone
                    (94.69%). No boxes, because it is given none.

Ground truth is shown once, in the middle, so neither side is scored against a different
standard. Sessions include one where perception wins by 90 points and one where it loses.
"""
from __future__ import annotations
import argparse, json, subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
ROOT = BASE / 'data/OpenLane-V2'
F = '/usr/share/fonts/truetype/dejavu/'
RGB = {'red': (255, 70, 70), 'green': (60, 235, 110), 'yellow': (255, 210, 60),
       'none': (140, 140, 150), None: (140, 140, 150), 'unknown': (120, 120, 130)}
CYAN = (0, 210, 255)
W, IH, HDR, FTR = 1280, 720, 46, 150


def fonts():
    return (ImageFont.truetype(F + 'DejaVuSans-Bold.ttf', 24),
            ImageFont.truetype(F + 'DejaVuSans.ttf', 17),
            ImageFont.truetype(F + 'DejaVuSans-Bold.ttf', 30),
            ImageFont.truetype(F + 'DejaVuSans.ttf', 13))


def chip(d, xy, w, h, colour, label, ok, fbig, ftiny, sub=None):
    x, y = xy
    d.rectangle([x, y, x + w, y + h], fill=(26, 26, 31), outline=(74, 74, 82))
    d.rectangle([x + 8, y + 8, x + 40, y + h - 8], fill=RGB.get(colour, (120, 120, 120)))
    d.text((x + 50, y + 9), str(label or 'no answer').upper(), font=fbig, fill=(238, 238, 243))
    if ok is not None:
        d.text((x + w - 30, y + 8), 'OK' if ok else 'X', font=fbig,
               fill=(90, 230, 120) if ok else (255, 90, 90))
    if sub:
        d.text((x + 6, y + h + 3), sub, font=ftiny, fill=(140, 140, 150))


def render(rec, vqa_pred, idx, n, tally, fts):
    fb, fr, fbig, ftiny = fts
    src = Image.open(ROOT / rec['image']).convert('RGB')
    sx, sy = W / src.width, IH / src.height
    canvas = Image.new('RGB', (W, HDR + IH + FTR), (14, 14, 18))
    canvas.paste(src.resize((W, IH), Image.LANCZOS), (0, HDR))
    d = ImageDraw.Draw(canvas)

    ego = set(rec.get('ego_set', []))
    # voted_colour is indexed by vote RANK, not by box index; map it back to boxes and
    # fall back to the frame's answer for any ego light that did not make the top-k
    cmap = {}
    for rank, bi in enumerate(rec.get('voted', [])):
        vc = rec.get('voted_colour', [])
        if rank < len(vc):
            cmap[bi] = vc[rank]
    for j, b in enumerate(rec.get('boxes', [])):
        x1, y1, x2, y2 = b[0]*sx, b[1]*sy+HDR, b[2]*sx, b[3]*sy+HDR
        if j in ego:
            c = RGB.get(cmap.get(j, rec['pred']), CYAN)
            d.rectangle([x1-1, y1-1, x2+1, y2+1], outline=c, width=3)
            g = rec['gov'][j] if j < len(rec.get('gov', [])) else 0
            d.text((x1, y1-16), f"ego {g:.2f}", font=ftiny, fill=c)
        else:
            d.rectangle([x1, y1, x2, y2], outline=CYAN, width=1)

    d.rectangle([0, 0, W, HDR], fill=(22, 22, 27))
    d.text((14, 11), "BEST PERCEPTION  vs  BEST VQA   ·   ego-lane traffic light",
           font=fb, fill=(240, 240, 246))
    d.text((W-300, 15), f"session {rec['segment']}   frame {idx+1}/{n}", font=fr,
           fill=(180, 180, 190))
    if rec.get('disc'):
        d.text((14, HDR+5), "another visible light disagrees with the ego's",
               font=ftiny, fill=(235, 190, 90))
    lx, ly = 14, HDR + IH - 26
    d.rectangle([lx-6, ly-6, lx+300, ly+18], fill=(0, 0, 0))
    d.line([lx, ly+6, lx+20, ly+6], fill=CYAN, width=3)
    d.text((lx+26, ly-1), "cyan = detected   coloured = the ego's light(s)",
           font=ftiny, fill=(205, 205, 215))

    y0 = HDR + IH
    d.rectangle([0, y0, W, y0+FTR], fill=(18, 18, 23))
    pp, vp, gt = rec['pred'], vqa_pred, rec['gt']
    d.text((16, y0+10), "PERCEPTION  (detect → associate → colour)", font=ftiny,
           fill=(150, 150, 160))
    chip(d, (16, y0+30), 380, 44, pp, pp, pp == gt, fbig, ftiny,
         sub=f"{len(rec.get('boxes', []))} lights detected, "
             f"{len(ego)} judged the ego's")
    d.text((452, y0+10), "GROUND TRUTH", font=ftiny, fill=(150, 150, 160))
    chip(d, (452, y0+30), 330, 44, gt, gt, None, fbig, ftiny,
         sub="from the OpenLane-V2 lane graph")
    d.text((838, y0+10), "VQA  (fine-tuned, words only, no boxes)", font=ftiny,
           fill=(150, 150, 160))
    chip(d, (838, y0+30), 420, 44, vp, vp, vp == gt, fbig, ftiny,
         sub=f"running: perception {tally['p']}/{tally['n']}   VQA {tally['v']}/{tally['n']}")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='outputs/tlb/demo_dump_best.json')
    ap.add_argument('--vqa', default='outputs/tlb/hint_ft.json')
    ap.add_argument('--fps', type=float, default=4.0)
    ap.add_argument('--gt', default='data/tlb/demo_segments.jsonl')
    ap.add_argument('--out', default='outputs/tlb/demo_v2/best_perception_vs_vqa.mp4')
    args = ap.parse_args()

    recs = json.loads((BASE / args.dump).read_text())
    vqa = {x['key']: x['pred'] for x in json.loads((BASE / args.vqa).read_text())}
    gtmap = {f"{r['segment']}/{r['timestamp']}": r
             for r in (json.loads(l) for l in open(BASE / args.gt))}
    for r in recs:
        g = gtmap.get(r['key'])
        if g:
            r['image'] = g['image']
            r['segment'] = g['segment']
    recs = [r for r in recs if r.get('image')]
    by = defaultdict(list)
    for r in recs:
        by[r['segment']].append(r)
    order = [s for s in ('11064', '11063', '11144', '11149') if s in by]
    order += [s for s in by if s not in order]
    fts = fonts()
    tmp = BASE / 'outputs/tlb/demo_v2/_best_frames'
    tmp.mkdir(parents=True, exist_ok=True)
    k = 0
    tally = {'p': 0, 'v': 0, 'n': 0}
    for seg in order:
        fr = sorted(by[seg], key=lambda r: r['key'])
        for i, rec in enumerate(fr):
            vp = vqa.get(rec['key'])
            tally['n'] += 1
            tally['p'] += int(rec['pred'] == rec['gt'])
            tally['v'] += int(vp == rec['gt'])
            render(rec, vp, i, len(fr), tally, fts).save(tmp / f'{k:05d}.png')
            k += 1
    out = BASE / args.out
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
                    '-i', str(tmp / '%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '20', str(out)], check=True)
    for f_ in tmp.glob('*.png'):
        f_.unlink()
    tmp.rmdir()
    print(f"  {out.name}  {k} frames")
    print(f"  perception {tally['p']}/{tally['n']} ({tally['p']/tally['n']:.0%})   "
          f"VQA {tally['v']}/{tally['n']} ({tally['v']/tally['n']:.0%})")


if __name__ == '__main__':
    main()
