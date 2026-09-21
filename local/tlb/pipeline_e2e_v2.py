#!/usr/bin/env python
"""v2: as v1, but the selector may be an ensemble.

The deployable system: no annotation at inference, one vision-tower pass per frame.

    CAM_FRONT -> frozen Qwen-Drive ViT (one forward, shared by everything below)
              -> detector head        -> traffic-light boxes
              -> ROI pool at those boxes + coarse scene grid
              -> selector             -> P(governs the ego lane) per box
              -> crop each box        -> colour head -> P(red/green/yellow)
              -> soft vote            -> answer

Differs from pipeline.py in the one way that matters: boxes come from the detector, not
from annotation. It also replaces the detector's own colour head (93% on matched lights)
with the dedicated crop head (99.85% given a box), and votes over candidates instead of
reading the single top-scoring light.

Reports accuracy *and* precision-at-coverage, because a driving system is allowed to say
"I cannot see it" and that is not the same kind of failure as a wrong colour.
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

from det_model import TLDetHead, decode                      # noqa: E402
from train_det import ViTTap                                 # noqa: E402
from selector import Selector                                # noqa: E402
from train_colour import build as build_colour               # noqa: E402
from extract_crops import crop_one                           # noqa: E402
from cache_roi import roi_pool, GH, GW, MAXL                 # noqa: E402


def acc(it):
    return sum(r['pred'] == r['gt'] for r in it) / len(it) if it else 0.0


def prec(it):
    a = [r for r in it if r['pred'] is not None]
    return sum(r['pred'] == r['gt'] for r in a) / len(a) if a else 0.0


def cov(it):
    return sum(r['pred'] is not None for r in it) / len(it) if it else 0.0


def boot(items, stat, iters=3000, seed=0):
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
    ap.add_argument('--det', default='outputs/tlb/det_head.pt')
    ap.add_argument('--selector', nargs='+', default=['outputs/tlb/selector_v2.pt'],
                    help='one or more selector checkpoints; logits are averaged')
    ap.add_argument('--colour', default='outputs/tlb/colour_head_v2.pt')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--thr', type=float, default=0.10)
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='outputs/tlb/e2e_val.json')
    ap.add_argument('--gov-thr', type=float, default=0.0,
                    help="answer 'none' when no detected light scores above this as the "
                         "ego's. 0 disables abstention (always name a colour).")
    ap.add_argument('--dump-boxes', action='store_true',
                    help='record every detection, its P(governs) and colour, for rendering')
    args = ap.parse_args()

    device = 'cuda'
    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tap = ViTTap(model)

    dck = torch.load(BASE / args.det, map_location='cpu')
    up, patch = dck['up'], dck['patch']
    det = None

    sels = []
    for sp_ in args.selector:
        sck = torch.load(BASE / sp_, map_location='cpu')
        m_ = Selector(d=sck['args']['d'], layers=sck['args']['layers']).to(device)
        m_.load_state_dict(sck['model']); m_.eval()
        sels.append(m_)
    print(f"selector ensemble: {len(sels)} member(s)", flush=True)

    def sel_logits(roi, bx, ctx, mask):
        return torch.stack([m_(roi, bx, ctx, mask) for m_ in sels]).mean(0)

    cck = torch.load(BASE / args.colour, map_location='cpu')
    col = build_colour().to(device)
    col.load_state_dict(cck['model']); col.eval()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    if args.limit:
        rows = rows[:args.limit]
    eid = {}; n = 0
    for s, g in groupby(rows, key=lambda r: r['segment']):
        for lb, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1

    out = []
    t0 = time.time()
    with torch.no_grad():
        for r in rows:
            feat, gr, gc, W0, H0 = tap.features(ROOT / r['image'], device)
            if det is None:
                det = TLDetHead(in_dim=feat.shape[0], hid=dck['args']['hid'], up=up).to(device).float()
                det.load_state_dict(dck['head']); det.eval()
            o = det(feat.float().unsqueeze(0))
            cell = patch / up
            dets = decode(o, cell, thr=args.thr)[0]
            sx = (gc * patch) / W0
            sy = (gr * patch) / H0
            boxes = [[d['box'][0] / sx, d['box'][1] / sy, d['box'][2] / sx, d['box'][3] / sy]
                     for d in dets][:MAXL]

            key = f"{r['segment']}/{r['timestamp']}"
            rec = {'key': key, 'gt': r['label'],
                   'disc': r.get('discriminative', False),
                   'w': r.get('max_gov_wh', [0, 0])[0], 'ep': eid[key],
                   'why': r.get('why', 'governed'), 'n_det': len(boxes)}
            if not boxes:
                # nothing detected at all -> 'none' when abstention is enabled
                rec['pred'] = 'none' if args.gov_thr > 0 else None
                rec['gov_max'] = 0.0
                out.append(rec)
                continue

            # ROI features for the DETECTED boxes, same layout the selector was trained on
            rf = roi_pool(feat, boxes, W0, H0).float()
            L = len(boxes)
            roi = torch.zeros(1, MAXL, rf.shape[1], device=device)
            roi[0, :L] = rf
            bx = torch.zeros(1, MAXL, 4, device=device)
            for j, b in enumerate(boxes):
                bx[0, j] = torch.tensor([(b[0]+b[2])/2/W0, (b[1]+b[3])/2/H0,
                                         (b[2]-b[0])/W0, (b[3]-b[1])/H0], device=device)
            ctx = F.adaptive_avg_pool2d(feat.unsqueeze(0).float(), (GH, GW))
            mask = torch.zeros(1, MAXL, dtype=torch.bool, device=device)
            mask[0, :L] = True
            lg = sel_logits(roi, bx, ctx, mask).masked_fill(~mask, -1e4)

            k = min(args.topk, L)
            order = torch.argsort(lg[0, :L], descending=True)[:k].cpu().numpy()
            w = torch.sigmoid(lg[0, :L]).cpu().numpy()[order]
            im = Image.open(ROOT / r['image']).convert('RGB')
            ts = []
            for i in order:
                c = crop_one(im, boxes[int(i)])
                t = torch.from_numpy(np.asarray(c, np.uint8).copy())
                ts.append(((t.permute(2, 0, 1).float() / 255.) - 0.5) / 0.5)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                P = torch.softmax(col(torch.stack(ts).to(device)).float(), 1).cpu().numpy()
            rgy = (w[:, None] * P[:, 1:4]).sum(0)
            tot = rgy.sum()
            gmax = float(torch.sigmoid(lg[0, :L]).max())
            rec['gov_max'] = round(gmax, 4)
            if args.gov_thr > 0 and gmax < args.gov_thr:
                # lights are visible but none looks like the ego's -> 'none', not a guess
                rec['pred'] = 'none'
            else:
                rec['pred'] = ['red', 'green', 'yellow'][int(rgy.argmax())] if tot > 0 else None
            rec['conf'] = float(rgy.max() / max(tot, 1e-9))
            if args.dump_boxes:
                gov_all = torch.sigmoid(lg[0, :L]).cpu().numpy()
                rec['boxes'] = [[round(v, 1) for v in b] for b in boxes]
                rec['gov'] = [round(float(g), 3) for g in gov_all]
                rec['voted'] = [int(i) for i in order]
                rec['voted_colour'] = [['red', 'green', 'yellow'][int(pp.argmax())]
                                       for pp in P[:, 1:4]]
                rec['chosen'] = int(order[0])
            out.append(rec)
            if len(out) % 200 == 0:
                print(f"    {len(out)}/{len(rows)}  {time.time()-t0:.0f}s  "
                      f"acc {acc(out):.2%}  prec {prec(out):.2%}  cov {cov(out):.2%}", flush=True)

    print(f"\n===== END-TO-END (detector boxes, no annotation at inference) =====")
    print(f"  {len(out)} frames, {len({o['ep'] for o in out})} episodes")
    for name, s in (('all', out),
                    ('discriminative', [o for o in out if o['disc']]),
                    ('light <12px', [o for o in out if o['w'] < 12]),
                    ('light >=20px', [o for o in out if o['w'] >= 20])):
        if not s:
            continue
        alo, ahi = boot(s, acc)
        plo, phi = boot(s, prec)
        print(f"  {name:16s} n={len(s):5d} eps={len({o['ep'] for o in s}):3d}  "
              f"acc {acc(s):6.2%} [{alo:.1%},{ahi:.1%}]   "
              f"precision {prec(s):6.2%} [{plo:.1%},{phi:.1%}]   coverage {cov(s):6.2%}")
    print("  confusion:", {g: dict(Counter(o['pred'] for o in out if o['gt'] == g))
                           for g in ('red', 'green', 'yellow')})
    (BASE / args.out).write_text(json.dumps(out, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
