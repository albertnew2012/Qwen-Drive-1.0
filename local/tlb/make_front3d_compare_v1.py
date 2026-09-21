#!/usr/bin/env python
"""Front view only, everything the stack predicts, with the perception-vs-VQA footer.

The layout of best_perception_vs_vqa.mp4 -- one camera, ground truth in the middle, both
methods either side -- but with the full 3D perception drawn in:

  * 3D cuboids on every detected object, projected at native 1600x900
  * the planner's 5 s trajectory, projected onto the ground plane
  * traffic lights as 3D boxes carrying predicted range and height

Runs on the three scenes that have BOTH traffic-light labels and full nuScenes sensor data,
which is what lets the repo's perception head and planner run at all.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'local'))
sys.path.insert(0, str(BASE / 'local/tlb'))

import nuscenes_session_tl_v3 as S                                   # noqa: E402
from qwen_drive_perception import geometry                           # noqa: E402
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES  # noqa: E402

F = '/usr/share/fonts/truetype/dejavu/'
RGB = {'red': (255, 70, 70), 'green': (60, 235, 110), 'yellow': (255, 210, 60),
       'none': (140, 140, 150), None: (140, 140, 150), 'unknown': (120, 120, 130)}
CYAN = (0, 210, 255)
CLS_RGB = {'vehicle': (255, 150, 30), 'pedestrian': (255, 60, 110),
           'bicycle': (60, 220, 140), 'traffic_cone': (250, 220, 60),
           'barrier': (170, 170, 180), 'czone_sign': (200, 120, 255),
           'generic_object': (150, 120, 90)}
W, IH, HDR, FTR = 1280, 720, 44, 150
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def fonts():
    return (ImageFont.truetype(F+'DejaVuSans-Bold.ttf', 23),
            ImageFont.truetype(F+'DejaVuSans.ttf', 16),
            ImageFont.truetype(F+'DejaVuSans-Bold.ttf', 29),
            ImageFont.truetype(F+'DejaVuSans.ttf', 12))


def chip(d, xy, w, h, colour, label, ok, fbig, ftiny, sub=None):
    x, y = xy
    d.rectangle([x, y, x+w, y+h], fill=(26, 26, 31), outline=(74, 74, 82))
    d.rectangle([x+8, y+8, x+38, y+h-8], fill=RGB.get(colour, (120, 120, 120)))
    d.text((x+48, y+9), str(label or 'no answer').upper(), font=fbig, fill=(238, 238, 243))
    if ok is not None:
        d.text((x+w-30, y+8), 'OK' if ok else 'X', font=fbig,
               fill=(90, 230, 120) if ok else (255, 90, 90))
    if sub:
        d.text((x+6, y+h+3), sub, font=ftiny, fill=(140, 140, 150))


def draw_objects(d, boxes, labels, l2i, sx, sy, off):
    """3D cuboids for every detection, projected with the full-resolution lidar2img."""
    if not len(boxes):
        return 0
    uv, valid = geometry.project_to_image(geometry.box_corners(boxes), l2i, 1600, 900)
    n = 0
    for i in np.where(valid)[0]:
        name = DET_CLASS_NAMES[int(labels[i])]
        c = CLS_RGB.get(name, (200, 200, 200))
        p = [(float(u)*sx, float(v)*sy+off) for u, v in uv[i]]
        for a, b in EDGES:
            d.line([p[a], p[b]], fill=c, width=2)
        n += 1
    return n


def draw_traj(d, xy, lidar2ego, l2i, sx, sy, off, colour, width=5, dots=True):
    uv, ok = S.project_ground_path(xy, lidar2ego, l2i)   # returns (uv, in-front mask)
    uv = uv[ok]
    if len(uv) < 2:
        return
    pts = [(float(u)*sx, float(v)*sy+off) for u, v in uv]
    d.line(pts, fill=colour, width=width, joint='curve')
    if dots:
        for k in range(0, len(pts), max(len(pts)//6, 1)):
            x, y = pts[k]
            d.ellipse([x-4, y-4, x+4, y+4], fill=colour, outline=(255, 255, 255))


def draw_lights(d, tl, K, sx, sy, off, ftiny):
    """3D box per light, ego's in its colour with range and height."""
    if not tl or not tl.get('boxes'):
        return
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    ego = set(tl.get('ego_set', []))
    for j, b in enumerate(tl['boxes']):
        rng, hgt = tl['r3'][j]
        pick = j in ego
        col = RGB.get(tl['colour3'][j], CYAN) if pick else CYAN
        u, v = (b[0]+b[2])/2, (b[1]+b[3])/2
        Z = max(rng, 1.0)
        X, Y = (u-cx)*Z/fx, (v-cy)*Z/fy
        Wm, Hm, D = (b[2]-b[0])*Z/fx, (b[3]-b[1])*Z/fy, 0.30
        pts = [(X+a*Wm/2, Y+e*Hm/2, Z+g*D/2)
               for g in (-1, 1) for e in (-1, 1) for a in (-1, 1)]
        p = [((fx*px/max(pz, .1)+cx)*sx, (fy*py/max(pz, .1)+cy)*sy+off) for px, py, pz in pts]
        for a, b_ in ((0,1),(1,3),(3,2),(2,0)):
            d.line([p[a], p[b_]], fill=col, width=3 if pick else 1)
        for a, b_ in ((4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)):
            d.line([p[a], p[b_]], fill=col, width=2 if pick else 1)
        tx, ty = min(q[0] for q in p), min(q[1] for q in p)
        d.text((tx, ty-15), f"{'EGO ' if pick else ''}{rng:.0f} m"
                            + (f"  h {hgt:.1f} m" if pick else ""),
               font=ftiny, fill=col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', type=int, default=[2, 5, 1])
    ap.add_argument('--vlm', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--perception', default='weights/Qwen-Drive-1.0-4B/perception')
    ap.add_argument('--planner', default='weights/Qwen-Drive-1.0-4B/planner-sft')
    ap.add_argument('--vqa', default='outputs/tlb/hint_ft.json')
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--fps', type=float, default=2.5)
    ap.add_argument('--out', default='outputs/tlb/demo_v2/front3d_perception_vs_vqa.mp4')
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from qwen_drive import InferenceMode, QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor
    from transformers import AutoTokenizer

    gt = {f"{r['segment']}/{r['timestamp']}": r
          for r in (json.loads(l) for l in open(BASE / args.gt))}
    vqa = {x['key']: x['pred'] for x in json.loads((BASE / args.vqa).read_text())}
    nusc = NuScenes(version='v1.0-mini', dataroot=str(BASE / 'data/nuscenes'), verbose=False)

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda').eval()
    head = QwenDrivePerception.from_pretrained(args.perception, dtype=torch.bfloat16).to('cuda').eval()
    pproc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, pproc)
    S.tl_init('cuda')
    print('  models up', flush=True)

    fts = fonts()
    tmp = BASE / 'outputs/tlb/_front3d'
    tmp.mkdir(parents=True, exist_ok=True)
    for f_ in tmp.glob('*.png'):
        f_.unlink()

    k = 0
    tally = {'p': 0, 'v': 0, 'n': 0}
    for si in args.scenes:
        scene = nusc.scene[si]
        samples, tok = [], scene['first_sample_token']
        while tok:
            s = nusc.get('sample', tok); samples.append(s); tok = s['next']
        track = S.ego_track(nusc, samples[0])
        cam_chains = {c: S.sensor_chain(nusc, samples[0], c) for c in S.CAM_ORDER}
        cam_ts = {c: np.asarray([x['timestamp']*1e-6 for x in cam_chains[c]]) for c in S.CAM_ORDER}
        print(f"  scene {si} {scene['name']}: {len(samples)} keyframes", flush=True)

        for s in samples:
            # OpenLane-V2 keys frames on the CAM_FRONT timestamp, not the sample (lidar)
            # timestamp -- matching on the latter finds nothing
            cam_ts_here = nusc.get('sample_data', s['data']['CAM_FRONT'])['timestamp']
            key = next((kk for kk in gt if kk.endswith(f"/{cam_ts_here}")), None)
            if key is None:                       # no traffic-light label for this frame
                continue
            frame = S.SessionFrame(nusc, s, BASE / 'data/nuscenes')
            pin, pmeta = pproc(frame, device='cuda')
            with torch.no_grad():
                res = head.infer(pin, pmeta)
            sc, gt_fut = S.build_scene_at(nusc, cam_chains, cam_ts, track,
                                          float(s['timestamp'])*1e-6, str(BASE / 'data/nuscenes'))
            with torch.no_grad():
                plan = model.run(InferenceMode.REASONING_PLANNING, scene=sc, num_samples=6)
            traj = plan.trajectories
            tl = None
            try:
                tl = S.tl_infer(model, frame._paths['CAM_FRONT'], device='cuda')
            except Exception as e:
                print(f"    [tl] {e}", flush=True)

            keep = res['scores'] >= 0.3
            pb, pl = res['boxes'][keep], res['labels'][keep]
            fi = frame.cam_order.index('CAM_FRONT')
            l2i_full = frame.img_metas(image_size=(1600, 900))['lidar2img'][fi]
            K = frame.cam_intrinsic[fi]

            src = frame.image('CAM_FRONT')
            sx, sy = W/src.width, IH/src.height
            canvas = Image.new('RGB', (W, HDR+IH+FTR), (14, 14, 18))
            canvas.paste(src.resize((W, IH), Image.LANCZOS), (0, HDR))
            d = ImageDraw.Draw(canvas)

            nobj = draw_objects(d, pb, pl, l2i_full, sx, sy, HDR)
            if gt_fut is not None:
                draw_traj(d, gt_fut[:, :2], frame.lidar2ego, l2i_full, sx, sy, HDR,
                          (255, 176, 46), width=4, dots=False)
            draw_traj(d, traj[0][:, :2], frame.lidar2ego, l2i_full, sx, sy, HDR,
                      (0, 200, 255), width=5)
            draw_lights(d, tl, K, sx, sy, HDR, fts[3])

            g = gt[key]
            pp = tl['pred'] if tl else None
            vp = vqa.get(key)
            tally['n'] += 1
            tally['p'] += int(pp == g['label'])
            tally['v'] += int(vp == g['label'])

            fb, fr, fbig, ftiny = fts
            d.rectangle([0, 0, W, HDR], fill=(22, 22, 27))
            d.text((13, 10), "3D perception + planned path + ego-lane traffic light",
                   font=fb, fill=(240, 240, 246))
            d.text((W-300, 14), f"session {g['segment']}   {nobj} objects in 3D",
                   font=fr, fill=(182, 182, 192))
            y0 = HDR + IH
            d.rectangle([0, y0, W, y0+FTR], fill=(18, 18, 23))
            d.text((16, y0+10), "PERCEPTION  (detect → associate → colour)", font=ftiny,
                   fill=(150, 150, 160))
            chip(d, (16, y0+30), 380, 44, pp, pp, pp == g['label'], fbig, ftiny,
                 sub=f"{len(tl['boxes']) if tl else 0} lights, "
                     f"{len(tl.get('ego_set', [])) if tl else 0} judged the ego's")
            d.text((452, y0+10), "GROUND TRUTH", font=ftiny, fill=(150, 150, 160))
            chip(d, (452, y0+30), 330, 44, g['label'], g['label'], None, fbig, ftiny,
                 sub="from the OpenLane-V2 lane graph")
            d.text((838, y0+10), "VQA  (fine-tuned, words only, no boxes)", font=ftiny,
                   fill=(150, 150, 160))
            chip(d, (838, y0+30), 420, 44, vp, vp, vp == g['label'], fbig, ftiny,
                 sub=f"running: perception {tally['p']}/{tally['n']}   VQA {tally['v']}/{tally['n']}")
            d.line([16, y0+FTR-16, 40, y0+FTR-16], fill=(0, 200, 255), width=4)
            d.text((46, y0+FTR-23), "predicted path (5 s)", font=ftiny, fill=(200, 200, 210))
            d.line([196, y0+FTR-16, 220, y0+FTR-16], fill=(255, 176, 46), width=4)
            d.text((226, y0+FTR-23), "actual path driven", font=ftiny, fill=(200, 200, 210))
            d.text((366, y0+FTR-23), "3D boxes: orange vehicle · pink pedestrian · "
                                     "yellow cone · grey barrier · cyan traffic light",
                   font=ftiny, fill=(160, 160, 172))
            canvas.save(tmp / f'{k:04d}.png')
            k += 1
            if k % 10 == 0:
                print(f"    {k} frames  perception {tally['p']}/{tally['n']}  "
                      f"VQA {tally['v']}/{tally['n']}", flush=True)

    out = BASE / args.out
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
                    '-i', str(tmp/'%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '20', str(out)], check=True)
    print(f"\n  {out.name}  {k} frames")
    print(f"  perception {tally['p']}/{tally['n']} ({tally['p']/max(tally['n'],1):.0%})   "
          f"VQA {tally['v']}/{tally['n']} ({tally['v']/max(tally['n'],1):.0%})")


if __name__ == '__main__':
    main()
