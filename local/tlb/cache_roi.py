#!/usr/bin/env python
"""Cache per-light ROI features and a coarse scene grid from the frozen vision tower.

Association is a choice among the lights in one frame, so the selector needs (a) a vector
per candidate light and (b) enough scene context to know where the ego lane goes. Caching
just those is what makes the selector cheap to train: the full 56x100x1024 grid for 12k
frames would be ~138 GB, while 3x3 ROI pools plus a 7x12 scene grid is a few GB, so the
ViT runs once instead of once per epoch.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'local/tlb'))
ROOT = BASE / 'data/OpenLane-V2'
from train_det import ViTTap                                   # noqa: E402

COLOURS = ['unknown', 'red', 'green', 'yellow']
CIDX = {c: i for i, c in enumerate(COLOURS)}
MAXL = 16            # lights kept per frame, by box area (covers 96% of frames
                     # fully; the governing light is almost never the smallest)
ROI = 2              # ROI pool side
GH, GW = 7, 12       # coarse scene grid


@torch.no_grad()
def roi_pool(grid, boxes_xyxy, W0, H0, out=ROI):
    """grid [C,H,W] over the resized image; boxes in ORIGINAL pixels -> [N, C*out*out]."""
    C, H, W = grid.shape
    feats = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        # normalise to [-1,1] sampling coords on the feature grid
        gx1, gx2 = x1 / W0 * 2 - 1, x2 / W0 * 2 - 1
        gy1, gy2 = y1 / H0 * 2 - 1, y2 / H0 * 2 - 1
        ys = torch.linspace(gy1, gy2, out, device=grid.device)
        xs = torch.linspace(gx1, gx2, out, device=grid.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        g = torch.stack([gx, gy], -1).unsqueeze(0)
        f = F.grid_sample(grid.unsqueeze(0).float(), g, align_corners=False,
                          padding_mode='border')[0]            # [C, out, out]
        feats.append(f.reshape(-1))
    return torch.stack(feats) if feats else torch.zeros(0, C * out * out, device=grid.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out-dir', default='data/tlb/roi')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / f'data/tlb/det_{args.split}.jsonl')]
    if args.limit:
        rows = rows[:args.limit]
    outd = BASE / args.out_dir
    outd.mkdir(parents=True, exist_ok=True)

    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tap = ViTTap(model)

    N = len(rows)
    roi = np.zeros((N, MAXL, 1024 * ROI * ROI), dtype=np.float16)
    ctx = np.zeros((N, 1024, GH, GW), dtype=np.float16)
    nl = np.zeros(N, dtype=np.int16)
    box = np.zeros((N, MAXL, 4), dtype=np.float32)
    col = np.full((N, MAXL), -1, dtype=np.int8)
    gov = np.full((N, MAXL), -1, dtype=np.int8)
    keys = []
    t0 = time.time()
    for i, r in enumerate(rows):
        grid, gr, gc, W0, H0 = tap.features(ROOT / r['image'], 'cuda')
        lights = sorted(r['lights'],
                        key=lambda L: -((L['box'][2]-L['box'][0])*(L['box'][3]-L['box'][1])))[:MAXL]
        bx = [L['box'] for L in lights]
        if bx:
            f = roi_pool(grid, bx, W0, H0)
            roi[i, :len(bx)] = f.cpu().numpy().astype(np.float16)
            for j, L in enumerate(lights):
                x1, y1, x2, y2 = L['box']
                box[i, j] = [(x1+x2)/2/W0, (y1+y2)/2/H0, (x2-x1)/W0, (y2-y1)/H0]
                col[i, j] = CIDX[L['colour']]
                gov[i, j] = L['is_gov']
        nl[i] = len(bx)
        ctx[i] = F.adaptive_avg_pool2d(grid.unsqueeze(0).float(), (GH, GW))[0].cpu().numpy()
        keys.append(f"{r['segment']}/{r['timestamp']}")
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{N}  {el:.0f}s  eta {el/(i+1)*(N-i-1)/60:.1f}m", flush=True)
    p = outd / f'{args.split}.npz'
    np.savez(p, roi=roi, ctx=ctx, nl=nl, box=box, col=col, gov=gov,
             keys=np.array(keys), seg=np.array([r['segment'] for r in rows]))
    print(f"wrote {p} ({p.stat().st_size/1e9:.2f} GB, {N} frames, {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
