#!/usr/bin/env python
"""v5: --box3d now also drops each light to the road beneath it.

v4 note: adds --box3d, drawing projected 3D cuboids from the predicted range.

v3 note: adds --hide-gt-boxes, for a video showing only what the system predicts.

v2 note: ground truth drawn dashed and only where predictions are shown.

v1 drew GT boxes on task1 as well, contradicting its own caption that the VLM is given
only the image, and drew GT and predictions as indistinguishable thin rectangles.

One demo video per task, over the same sessions, so the three are directly comparable.

  task1  the VLM answering in words. No boxes are drawn, because it is given none.
  task2  2D pipeline: every detected light, how strongly the selector thinks each is the
         ego's, and the colour each one reads as.
  task3  the same detections carrying range and height, with a bird's-eye panel placing
         them around the ego.

Sessions are picked to be honest rather than flattering: 11064 is where the pipeline wins
by 90 points, 11149 is where it LOSES by 40, 11144 is where both are right.
"""
from __future__ import annotations
import argparse, json, math, subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
ROOT = BASE / 'data/OpenLane-V2'
F = '/usr/share/fonts/truetype/dejavu/'
RGB = {'red': (255, 70, 70), 'green': (60, 235, 110), 'yellow': (255, 210, 60),
       'none': (150, 150, 158), None: (150, 150, 158), 'unknown': (120, 120, 128)}
CYAN = (0, 210, 255)
W, IH = 1280, 720
HDR, FTR = 46, 128


def fonts():
    return (ImageFont.truetype(F + 'DejaVuSans-Bold.ttf', 25),
            ImageFont.truetype(F + 'DejaVuSans.ttf', 18),
            ImageFont.truetype(F + 'DejaVuSans-Bold.ttf', 32),
            ImageFont.truetype(F + 'DejaVuSans.ttf', 14))


def cuboid_from_prediction(box2d, rng_m, K, depth_m=0.25):
    """8 corners of a 3D box, in camera coordinates, from predicted quantities only.

    Position comes from the predicted range plus the pinhole back-projection of the box
    centre. Width and height come from the observed 2D box scaled by that same range, so
    the cuboid's size is derived rather than assumed. Only its thickness along the optical
    axis is unobservable from one camera -- a traffic-light housing is ~0.25 m deep, and
    that constant is the single assumption in the drawing.
    """
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    u = (box2d[0] + box2d[2]) / 2.0
    v = (box2d[1] + box2d[3]) / 2.0
    Z = max(float(rng_m), 1.0)                      # optical-axis depth ~ ground range
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    W = (box2d[2] - box2d[0]) * Z / fx              # metres, from the 2D box at that range
    H = (box2d[3] - box2d[1]) * Z / fy
    D = depth_m
    pts = []
    for sz in (-1, 1):
        for sy in (-1, 1):
            for sx in (-1, 1):
                pts.append((X + sx*W/2, Y + sy*H/2, Z + sz*D/2))
    return pts, (W, H, D)


def project(pts, K):
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    out = []
    for (X, Y, Z) in pts:
        Z = max(Z, 0.1)
        out.append((fx * X / Z + cx, fy * Y / Z + cy))
    return out


CAM_H = 1.495          # CAM_FRONT height above the ego plane, from nuScenes calibration


def ground_drop(d, box2d, rng_m, height_m, K, colour, sx, sy, off):
    """Line from the light down to the road beneath it, plus a footprint ellipse.

    Camera Y points down, so a light `height_m` above the road sits (height_m - CAM_H)
    above the camera and the ground point below it is CAM_H below the camera. Both project
    through the same intrinsics, so the drop length encodes range -- a near light gets a
    long drop, a far one a short drop, which is exactly the depth cue a flat cuboid loses.
    """
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    Z = max(float(rng_m), 1.0)
    u = (box2d[0] + box2d[2]) / 2.0
    X = (u - cx) * Z / fx
    y_light = -(float(height_m) - CAM_H)
    y_ground = CAM_H
    p1 = (fx * X / Z + cx, fy * y_light / Z + cy)
    p2 = (fx * X / Z + cx, fy * y_ground / Z + cy)
    q1 = (p1[0]*sx, p1[1]*sy + off)
    q2 = (p2[0]*sx, p2[1]*sy + off)
    faint = tuple(int(c*0.6) for c in colour)
    n = 7
    for i in range(0, n, 2):                       # dashed, so it reads as a guide
        a = (q1[0] + (q2[0]-q1[0])*i/n, q1[1] + (q2[1]-q1[1])*i/n)
        b = (q1[0] + (q2[0]-q1[0])*(i+1)/n, q1[1] + (q2[1]-q1[1])*(i+1)/n)
        d.line([a, b], fill=faint, width=1)
    rw = max(3.0, 0.45 * fx / Z * sx)              # ~0.9 m footprint on the road
    d.ellipse([q2[0]-rw, q2[1]-rw*0.32, q2[0]+rw, q2[1]+rw*0.32], outline=colour, width=2)


def draw_cuboid(d, pts2d, colour, sx, sy, off, width=2, faint=None):
    """12 edges. The near face is drawn solid, the far face and the connecting edges
    thinner, so the box reads as a volume rather than two stacked rectangles."""
    P = [(x*sx, y*sy+off) for (x, y) in pts2d]
    dim = faint or colour
    for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):          # near face, solid
        d.line([P[a], P[b]], fill=colour, width=width)
    for a, b in ((4, 5), (5, 7), (7, 6), (6, 4)):
        d.line([P[a], P[b]], fill=dim, width=max(width-1, 1))
    for a, b in ((0, 4), (1, 5), (2, 6), (3, 7)):
        d.line([P[a], P[b]], fill=dim, width=max(width-1, 1))


def dash(d, x1, y1, x2, y2, colour, dash_len=6, width=2):
    """Dashed rectangle: marks ground truth so it cannot read as a prediction."""
    def seg(a, b, horiz):
        p = a
        while p < b:
            q = min(p + dash_len, b)
            if horiz:
                d.line([p, y1, q, y1], fill=colour, width=width)
                d.line([p, y2, q, y2], fill=colour, width=width)
            else:
                d.line([x1, p, x1, q], fill=colour, width=width)
                d.line([x2, p, x2, q], fill=colour, width=width)
            p += dash_len * 2
    seg(x1, x2, True); seg(y1, y2, False)


def chip(d, xy, w, h, colour, label, ok, fbig, ftiny):
    x, y = xy
    d.rectangle([x, y, x + w, y + h], fill=(26, 26, 30), outline=(72, 72, 80))
    d.rectangle([x + 8, y + 8, x + 42, y + h - 8], fill=RGB.get(colour, (120, 120, 120)))
    d.text((x + 52, y + 8), str(label or 'no answer').upper(), font=fbig, fill=(238, 238, 242))
    if ok is not None:
        d.text((x + w - 32, y + 7), 'OK' if ok else 'X', font=fbig,
               fill=(90, 230, 120) if ok else (255, 90, 90))


def bev_panel(size, dets, chosen, K, W0):
    """Plan view: lights placed by range and by lateral offset from the pinhole model."""
    S = size
    im = Image.new('RGB', (S, S), (18, 18, 22))
    d = ImageDraw.Draw(im)
    rmax = 60.0
    for rr in (15, 30, 45, 60):                      # range rings
        rad = rr / rmax * (S - 30) / 2
        d.ellipse([S/2 - rad, S - 20 - rad, S/2 + rad, S - 20 + rad],
                  outline=(48, 48, 56))
        d.text((S/2 + 3, S - 22 - rad), f"{rr}m", font=ImageFont.truetype(F+'DejaVuSans.ttf', 11),
               fill=(110, 110, 120))
    d.polygon([(S/2, S - 14), (S/2 - 7, S - 4), (S/2 + 7, S - 4)], fill=(230, 230, 240))
    fx = K[0][0]; cx = K[0][2]
    for j, (b, (rng, hgt), col) in enumerate(dets):
        u = (b[0] + b[2]) / 2
        lat = rng * (u - cx) / fx                    # metres left/right of the optical axis
        fwd = max(math.sqrt(max(rng * rng - lat * lat, 1.0)), 1.0)
        px = S/2 + (-lat) / rmax * (S - 30) / 2
        py = S - 20 - fwd / rmax * (S - 30) / 2
        r = 9 if j == chosen else 5
        d.ellipse([px-r, py-r, px+r, py+r], fill=RGB.get(col, (120, 120, 128)),
                  outline=(255, 255, 255) if j == chosen else None,
                  width=2 if j == chosen else 0)
        if j == chosen:
            d.text((px + 12, py - 8), f"{rng:.0f}m  h{hgt:.1f}m",
                   font=ImageFont.truetype(F+'DejaVuSans-Bold.ttf', 13), fill=(255, 255, 255))
    return im


def render(task, rec, vqa_pred, K, idx, n, tally, fts, hide_gt=False, box3d=False):
    fb, fr, fbig, ftiny = fts
    src = Image.open(ROOT / rec['image']).convert('RGB')
    sx, sy = W / src.width, IH / src.height
    canvas = Image.new('RGB', (W, HDR + IH + FTR), (14, 14, 17))
    canvas.paste(src.resize((W, IH), Image.LANCZOS), (0, HDR))
    d = ImageDraw.Draw(canvas)

    if task in ('task2', 'task3'):
        for j, b in enumerate(rec.get('boxes', [])):
            x1, y1, x2, y2 = b[0]*sx, b[1]*sy+HDR, b[2]*sx, b[3]*sy+HDR
            pick = (j == rec.get('chosen', -1))
            bc = RGB.get(rec['box_colour'][j], CYAN) if pick else CYAN
            if box3d and j < len(rec.get('r3', [])):
                rng3, _h3 = rec['r3'][j]
                c3, dims = cuboid_from_prediction(b, rng3, K)
                faint = tuple(int(c*0.55) for c in bc)
                draw_cuboid(d, project(c3, K), bc, sx, sy, HDR,
                            width=3 if pick else 1, faint=faint)
                ground_drop(d, b, rng3, rec['r3'][j][1], K, bc, sx, sy, HDR)
            else:
                d.rectangle([x1, y1, x2, y2], outline=bc, width=3 if pick else 1)
            g = rec['gov'][j] if j < len(rec.get('gov', [])) else 0.0
            if pick:
                d.text((x1, y1-18), f"ego {g:.2f}", font=ftiny, fill=bc)
            elif g > 0.15:
                d.text((x1, y1-15), f"{g:.2f}", font=ftiny, fill=(150, 200, 220))
            if task == 'task3' and j < len(rec.get('r3', [])):
                rng, hgt = rec['r3'][j]
                d.text((x1, y2+2), f"{rng:.0f}m h{hgt:.1f}", font=ftiny,
                       fill=bc if pick else (140, 190, 210))
    # Ground truth, drawn only where predictions are also shown. Dashed and tagged so it
    # can never be mistaken for a prediction -- in v1 both were thin coloured rectangles.
    if task in ('task2', 'task3') and not hide_gt:
        for b in rec.get('gov_boxes', []):
            x1, y1 = b[0]*sx-3, b[1]*sy+HDR-3
            x2, y2 = b[2]*sx+3, b[3]*sy+HDR+3
            dash(d, x1, y1, x2, y2, RGB[rec['gt']], dash_len=6, width=2)
            d.text((x2+3, y1-2), "GT", font=ftiny, fill=RGB[rec['gt']])

    if task == 'task3' and rec.get('boxes'):
        dets = [(rec['boxes'][j], rec['r3'][j], rec['box_colour'][j])
                for j in range(len(rec['boxes']))]
        canvas.paste(bev_panel(240, dets, rec.get('chosen', -1), K, src.width),
                     (W - 252, HDR + 12))
        d.text((W - 252, HDR + 256), "bird's-eye: predicted 3D", font=ftiny, fill=(180, 180, 190))

    TITLE = {'task1': 'TASK 1  ·  VQA  (Qwen-Drive answers in words, sees no boxes)',
             'task2': 'TASK 2  ·  2D pipeline  (detect -> select ego lane -> colour)',
             'task3': 'TASK 3  ·  3D pipeline  (same, with range and height)'}[task]
    d.rectangle([0, 0, W, HDR], fill=(22, 22, 26))
    d.text((14, 10), TITLE, font=fb, fill=(240, 240, 245))
    d.text((W-320, 14), f"session {rec['segment']}   frame {idx+1}/{n}   "
                        f"correct {tally['ok']}/{tally['n']}", font=fr, fill=(185, 185, 195))
    if rec['disc']:
        d.text((14, HDR+4), "another visible light disagrees with the ego's",
               font=ftiny, fill=(235, 190, 90))
    if task in ('task2', 'task3'):
        lx, ly = 14, HDR + IH - 42
        if hide_gt:
            d.rectangle([lx-6, ly+8, lx+330, ly+32], fill=(0, 0, 0))
            d.line([lx, ly+20, lx+22, ly+20], fill=CYAN, width=3)
            d.text((lx+28, ly+13), "every box is PREDICTED - no ground truth drawn",
                   font=ftiny, fill=(210, 210, 218))
        else:
            d.rectangle([lx-6, ly-6, lx+310, ly+32], fill=(0, 0, 0))
            d.line([lx, ly+6, lx+22, ly+6], fill=CYAN, width=3)
            d.text((lx+28, ly-1), "solid = PREDICTED detection", font=ftiny, fill=(210, 210, 218))
            dash(d, lx, ly+20, lx+22, ly+22, (235, 235, 245), dash_len=5, width=2)
            d.text((lx+28, ly+15), "dashed = GROUND TRUTH", font=ftiny, fill=(210, 210, 218))

    y0 = HDR + IH
    d.rectangle([0, y0, W, y0+FTR], fill=(18, 18, 22))
    d.text((16, y0+8), "GROUND TRUTH  (from the lane graph)", font=ftiny, fill=(150, 150, 158))
    chip(d, (16, y0+28), 320, 44, rec['gt'], rec['gt'], None, fbig, ftiny)
    pred = vqa_pred if task == 'task1' else rec['pred']
    lbl = {'task1': 'VQA ANSWER', 'task2': 'PIPELINE ANSWER',
           'task3': 'PIPELINE ANSWER'}[task]
    d.text((380, y0+8), lbl, font=ftiny, fill=(150, 150, 158))
    chip(d, (380, y0+28), 340, 44, pred, pred, pred == rec['gt'], fbig, ftiny)
    if task != 'task1':
        d.text((748, y0+8), "EVIDENCE", font=ftiny, fill=(150, 150, 158))
        nb = len(rec.get('boxes', []))
        d.text((748, y0+30), f"{nb} lights detected", font=fr, fill=(210, 210, 218))
        if rec.get('boxes') and rec.get('chosen', -1) >= 0:
            ch = rec['chosen']
            d.text((748, y0+54), f"ego light: P={rec['gov'][ch]:.2f}, reads "
                                 f"{rec['box_colour'][ch]}", font=fr, fill=(210, 210, 218))
            if task == 'task3':
                rng, hgt = rec['r3'][ch]
                d.text((748, y0+78), f"3D: {rng:.0f} m ahead, {hgt:.1f} m high",
                       font=fr, fill=(150, 210, 235))
    else:
        d.text((748, y0+8), "NOTE", font=ftiny, fill=(150, 150, 158))
        d.text((748, y0+30), "no boxes: the VLM is given only the image", font=fr,
               fill=(210, 210, 218))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='outputs/tlb/demo_dump.json')
    ap.add_argument('--vqa', default='outputs/tlb/hint_ft.json')
    ap.add_argument('--tasks', nargs='+', default=['task1', 'task2', 'task3'])
    ap.add_argument('--fps', type=float, default=4.0)
    ap.add_argument('--out-dir', default='outputs/tlb/demo')
    ap.add_argument('--hide-gt-boxes', action='store_true',
                    help='draw only predicted boxes on the image (the footer still shows '
                         'the ground-truth answer, so frames remain scoreable)')
    ap.add_argument('--suffix', default='', help='appended to each output filename')
    ap.add_argument('--box3d', action='store_true',
                    help='draw projected 3D cuboids instead of 2D rectangles')
    args = ap.parse_args()

    recs = json.loads((BASE / args.dump).read_text())
    vqa = {x['key']: x['pred'] for x in json.loads((BASE / args.vqa).read_text())}
    out = BASE / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    fts = fonts()
    by = defaultdict(list)
    for r in recs:
        by[r['segment']].append(r)
    order = ['11064', '11063', '11144', '11149']
    order = [s for s in order if s in by] + [s for s in by if s not in order]

    for task in args.tasks:
        tmp = out / f'_f_{task}'; tmp.mkdir(exist_ok=True)
        k = 0; tally = {'ok': 0, 'n': 0}
        for seg in order:
            frames = by[seg]
            for i, rec in enumerate(frames):
                info = ROOT / 'val' / rec['segment'] / 'info' / f"{rec['timestamp']}.json"
                if not info.exists():
                    info = ROOT / 'train' / rec['segment'] / 'info' / f"{rec['timestamp']}.json"
                K = (json.load(open(info))['sensor']['CAM_FRONT']['intrinsic']['K']
                     if info.exists()
                     else [[1252.8, 0, 826.6], [0, 1252.8, 470.0], [0, 0, 1]])
                p = vqa.get(rec['key']) if task == 'task1' else rec['pred']
                tally['n'] += 1; tally['ok'] += int(p == rec['gt'])
                render(task, rec, vqa.get(rec['key']), K, i, len(frames), tally, fts,
                       hide_gt=args.hide_gt_boxes, box3d=args.box3d)\
                    .save(tmp / f'{k:05d}.png'); k += 1
        mp4 = out / f'{task}{args.suffix}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
                        '-i', str(tmp / '%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '20', str(mp4)], check=True)
        for f_ in tmp.glob('*.png'):
            f_.unlink()
        tmp.rmdir()
        print(f"  {mp4.name}  {k} frames  correct {tally['ok']}/{tally['n']} "
              f"({tally['ok']/tally['n']:.0%})")


if __name__ == '__main__':
    main()
