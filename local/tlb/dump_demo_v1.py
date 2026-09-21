#!/usr/bin/env python
"""Run the full 3D pipeline over the demo sessions and record everything for rendering.

One pass produces what all three demo videos need, so the GPU work is not repeated:
  Task 1  the VQA answer (read from a saved eval, no boxes -- the VLM never sees any)
  Task 2  detected boxes, P(governs ego) per box, per-box colour, the voted answer
  Task 3  the same, plus range and height per box, and the ego-frame position

Sessions are chosen to show both directions honestly: one where the pipeline rescues a
scene the VLM cannot read, one where the VLM wins, and one where both agree.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'local/tlb'))
ROOT = BASE / 'data/OpenLane-V2'

from det_model_v2 import TLDetHead3D            # noqa: E402
from det_model import decode                    # noqa: E402
from train_det import ViTTap                    # noqa: E402
from selector import Selector                   # noqa: E402
from train_colour import build as build_colour  # noqa: E402
from extract_crops import crop_one              # noqa: E402
from cache_roi import roi_pool, GH, GW, MAXL    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head3d', default='outputs/tlb/det_head3d_v2_unfrozen.pt')
    ap.add_argument('--selector', default='outputs/tlb/selector_v2.pt')
    ap.add_argument('--colour', default='outputs/tlb/colour_head_v2.pt')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--gt', default='data/tlb/demo_segments.jsonl')
    ap.add_argument('--thr', type=float, default=0.30)
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--out', default='outputs/tlb/demo_dump.json')
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
    out = []
    t0 = time.time()
    with torch.no_grad():
        for r in rows:
            feat, gr, gc, W0, H0 = tap.features(ROOT / r['image'], device)
            if head is None:
                head = TLDetHead3D(in_dim=feat.shape[0], hid=ck['args'].get('hid', 256),
                                   up=up).to(device).float()
                head.load_state_dict(ck['head']); head.eval()
            o = head(feat.float().unsqueeze(0))
            cell = patch / up
            dets = decode(o, cell, thr=args.thr)[0][:MAXL]
            sx = (gc * patch) / W0; sy = (gr * patch) / H0
            boxes, r3 = [], []
            for d in dets:
                bx = [d['box'][0]/sx, d['box'][1]/sy, d['box'][2]/sx, d['box'][3]/sy]
                cx = int(np.clip((d['box'][0]+d['box'][2])/2/cell, 0, o['logrange'].shape[3]-1))
                cy = int(np.clip((d['box'][1]+d['box'][3])/2/cell, 0, o['logrange'].shape[2]-1))
                boxes.append([round(v, 1) for v in bx])
                r3.append((round(float(o['logrange'][0, 0, cy, cx].exp()), 1),
                           round(float(o['height'][0, 0, cy, cx]), 2)))
            rec = {'key': f"{r['segment']}/{r['timestamp']}", 'segment': r['segment'],
                   'timestamp': r['timestamp'], 'image': r['image'], 'gt': r['label'],
                   'disc': r['discriminative'], 'gov_boxes': r['gov_boxes'],
                   'other_boxes': r['other_boxes'], 'boxes': boxes, 'r3': r3}
            if not boxes:
                rec['pred'] = None; rec['gov'] = []; rec['box_colour'] = []
                out.append(rec); continue

            rf = roi_pool(feat, boxes, W0, H0).float()
            L = len(boxes)
            roi = torch.zeros(1, MAXL, rf.shape[1], device=device); roi[0, :L] = rf
            bxt = torch.zeros(1, MAXL, 4, device=device)
            for j, b in enumerate(boxes):
                bxt[0, j] = torch.tensor([(b[0]+b[2])/2/W0, (b[1]+b[3])/2/H0,
                                          (b[2]-b[0])/W0, (b[3]-b[1])/H0], device=device)
            ctx = F.adaptive_avg_pool2d(feat.unsqueeze(0).float(), (GH, GW))
            mask = torch.zeros(1, MAXL, dtype=torch.bool, device=device); mask[0, :L] = True
            lg = sel(roi, bxt, ctx, mask).masked_fill(~mask, -1e4)
            gov = torch.sigmoid(lg[0, :L]).cpu().numpy()
            im = Image.open(ROOT / r['image']).convert('RGB')
            ts = [((torch.from_numpy(np.asarray(crop_one(im, b), np.uint8).copy())
                    .permute(2, 0, 1).float()/255.) - 0.5)/0.5 for b in boxes]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                P = torch.softmax(col(torch.stack(ts).to(device)).float(), 1).cpu().numpy()
            order = np.argsort(-gov)[:args.topk]
            rgy = (gov[order][:, None] * P[order][:, 1:4]).sum(0)
            rec['pred'] = ['red', 'green', 'yellow'][int(rgy.argmax())] if rgy.sum() > 0 else None
            rec['gov'] = [round(float(g), 3) for g in gov]
            rec['box_colour'] = [['unknown', 'red', 'green', 'yellow'][int(p.argmax())] for p in P]
            rec['chosen'] = int(order[0])
            rec['voted'] = [int(i) for i in order]
            out.append(rec)
            if len(out) % 40 == 0:
                print(f"    {len(out)}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    (BASE / args.out).write_text(json.dumps(out, indent=1))
    ok = sum(1 for o in out if o['pred'] == o['gt'])
    print(f"dumped {len(out)} frames, pipeline colour {ok}/{len(out)} = {ok/len(out):.1%}")
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
