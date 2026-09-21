#!/usr/bin/env python
"""Task 3 pipeline: 3D traffic lights -> which one is the ego's -> its status.

    CAM_FRONT -> frozen ViT -> v2 head -> boxes + colour + P(gov) + range + height
              -> ROI pool at the detected boxes -> selector -> P(governs ego)
              -> crop -> colour head -> soft vote
              -> answer: colour, plus the 3D position of the light it came from

Two things are scored, and they are different claims:

  colour      exactly the Task 2 metric, so the 3D path can be compared like for like
  3D error    |range| and |height| against the Occ3D-refined labels, on matched lights

The 3D labels are self-derived (triangulated, snapped to Occ3D, ~0.35 m against lidar) and
exist only within 40 m, so the 3D numbers carry that caveat wherever they are quoted.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'local/tlb'))
ROOT = BASE / 'data/OpenLane-V2'

from det_model_v2 import TLDetHead3D                              # noqa: E402
from det_model import decode                                      # noqa: E402
from train_det import ViTTap                                      # noqa: E402
from selector import Selector                                     # noqa: E402
from train_colour import build as build_colour                    # noqa: E402
from extract_crops import crop_one                                # noqa: E402
from cache_roi import roi_pool, GH, GW, MAXL                      # noqa: E402


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / ua if ua > 0 else 0.0


def boot(items, stat, iters=2000, seed=0):
    by = defaultdict(list)
    for r in items:
        by[r['ep']].append(r)
    eps = list(by)
    if not eps:
        return 0., 0.
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(iters):
        p = rng.choice(len(eps), len(eps), replace=True)
        v.append(stat([x for i in p for x in by[eps[i]]]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head3d', default='outputs/tlb/det_head3d_v2.pt')
    ap.add_argument('--selector', default='outputs/tlb/selector_v2.pt')
    ap.add_argument('--colour', default='outputs/tlb/colour_head_v2.pt')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--gt3d', default='data/tlb/det3d_val.jsonl')
    ap.add_argument('--thr', type=float, default=0.30)
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='outputs/tlb/pipeline3d_val.json')
    args = ap.parse_args()

    device = 'cuda'
    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tap = ViTTap(model)

    ck = torch.load(BASE / args.head3d, map_location='cpu')
    up, patch = ck['up'], ck['patch']
    head = None
    sck = torch.load(BASE / args.selector, map_location='cpu')
    sel = Selector(d=sck['args']['d'], layers=sck['args']['layers']).to(device)
    sel.load_state_dict(sck['model']); sel.eval()
    cck = torch.load(BASE / args.colour, map_location='cpu')
    col = build_colour().to(device)
    col.load_state_dict(cck['model']); col.eval()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    if args.limit:
        rows = rows[:args.limit]
    gt3d = {f"{r['segment']}/{r['timestamp']}": r['lights3d']
            for r in (json.loads(l) for l in open(BASE / args.gt3d))}
    eid = {}; n = 0
    for s, g in groupby(rows, key=lambda r: r['segment']):
        for lb, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1

    out, err_r, err_h = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for r in rows:
            key = f"{r['segment']}/{r['timestamp']}"
            feat, gr, gc, W0, H0 = tap.features(ROOT / r['image'], device)
            if head is None:
                head = TLDetHead3D(in_dim=feat.shape[0], hid=ck['args'].get('hid', 256),
                                   up=up).to(device).float()
                head.load_state_dict(ck['head']); head.eval()
            o = head(feat.float().unsqueeze(0))
            cell = patch / up
            dets = decode(o, cell, thr=args.thr)[0]
            sx = (gc * patch) / W0; sy = (gr * patch) / H0
            boxes, r3 = [], []
            for d in dets[:MAXL]:
                bx = [d['box'][0]/sx, d['box'][1]/sy, d['box'][2]/sx, d['box'][3]/sy]
                cxg = int(np.clip((d['box'][0]+d['box'][2])/2/cell, 0, o['logrange'].shape[3]-1))
                cyg = int(np.clip((d['box'][1]+d['box'][3])/2/cell, 0, o['logrange'].shape[2]-1))
                boxes.append(bx)
                r3.append((float(o['logrange'][0, 0, cyg, cxg].exp()),
                           float(o['height'][0, 0, cyg, cxg])))
            rec = {'key': key, 'gt': r['label'], 'disc': r['discriminative'],
                   'w': r['max_gov_wh'][0], 'ep': eid[key], 'n_det': len(boxes)}
            if not boxes:
                rec['pred'] = None; out.append(rec); continue

            rf = roi_pool(feat, boxes, W0, H0).float()
            L = len(boxes)
            roi = torch.zeros(1, MAXL, rf.shape[1], device=device); roi[0, :L] = rf
            bx = torch.zeros(1, MAXL, 4, device=device)
            for j, b in enumerate(boxes):
                bx[0, j] = torch.tensor([(b[0]+b[2])/2/W0, (b[1]+b[3])/2/H0,
                                         (b[2]-b[0])/W0, (b[3]-b[1])/H0], device=device)
            ctx = F.adaptive_avg_pool2d(feat.unsqueeze(0).float(), (GH, GW))
            mask = torch.zeros(1, MAXL, dtype=torch.bool, device=device); mask[0, :L] = True
            lg = sel(roi, bx, ctx, mask).masked_fill(~mask, -1e4)
            k = min(args.topk, L)
            order = torch.argsort(lg[0, :L], descending=True)[:k].cpu().numpy()
            w = torch.sigmoid(lg[0, :L]).cpu().numpy()[order]
            im = Image.open(ROOT / r['image']).convert('RGB')
            ts = []
            for i in order:
                c = crop_one(im, boxes[int(i)])
                t = torch.from_numpy(np.asarray(c, np.uint8).copy())
                ts.append(((t.permute(2, 0, 1).float()/255.) - 0.5)/0.5)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                P = torch.softmax(col(torch.stack(ts).to(device)).float(), 1).cpu().numpy()
            rgy = (w[:, None] * P[:, 1:4]).sum(0)
            rec['pred'] = ['red', 'green', 'yellow'][int(rgy.argmax())] if rgy.sum() > 0 else None
            ch = int(order[0])
            rec['range_m'] = round(r3[ch][0], 1)
            rec['height_m'] = round(r3[ch][1], 2)

            # 3D error: match detections to 3D-labelled lights by IoU
            for g in gt3d.get(key, []):
                best, bi = 0.0, -1
                for j, b in enumerate(boxes):
                    v = iou(b, g['box'])
                    if v > best:
                        best, bi = v, j
                if best >= 0.3:
                    err_r.append(abs(r3[bi][0] - g['range_m']))
                    err_h.append(abs(r3[bi][1] - g['height_m']))
            out.append(rec)
            if len(out) % 300 == 0:
                a = sum(x['pred'] == x['gt'] for x in out)/len(out)
                print(f"    {len(out)}/{len(rows)}  {time.time()-t0:.0f}s  colour {a:.2%}",
                      flush=True)

    acc = lambda it: sum(x['pred'] == x['gt'] for x in it)/len(it) if it else 0.
    print(f"\n===== TASK 3 PIPELINE (3D detection -> ego light) =====")
    print(f"  {len(out)} frames, {len({o['ep'] for o in out})} episodes")
    for name, s in (('all', out), ('discriminative', [o for o in out if o['disc']]),
                    ('light <12px', [o for o in out if o['w'] < 12])):
        if not s:
            continue
        lo, hi = boot(s, acc)
        print(f"  colour, {name:16s} {acc(s):6.2%}  [{lo:.1%},{hi:.1%}]   n={len(s)}")
    if err_r:
        er = np.array(err_r); eh = np.array(err_h)
        print(f"\n  3D on matched lights (n={len(er)}), against self-derived labels:")
        print(f"    |range error|  median {np.median(er):5.2f} m   p90 {np.percentile(er,90):5.2f} m"
              f"   within 2 m {np.mean(er<2):.0%}")
        print(f"    |height error| median {np.median(eh):5.2f} m   p90 {np.percentile(eh,90):5.2f} m"
              f"   within 1 m {np.mean(eh<1):.0%}")
    rr = np.array([o['range_m'] for o in out if 'range_m' in o])
    if len(rr):
        print(f"  predicted range to the ego's light: median {np.median(rr):.1f} m")
    (BASE / args.out).write_text(json.dumps(out, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
