#!/usr/bin/env python
"""Train the v2 3D branch on top of the frozen v1 2D detector.

By default everything v1 learned is frozen and only the new 3D branch trains. That is a
deliberate choice rather than a shortcut: the 2D detector is the component whose numbers
are already reported, and 803 unique 3D lights is thin enough that joint training could
degrade detection to buy a little depth accuracy. `--unfreeze` trains the whole head at a
reduced learning rate for comparison.

Supervision is masked to cells carrying a confirmed 3D label, so frames contribute only
where the geometry was validated.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'local/tlb'))
ROOT = BASE / 'data/OpenLane-V2'

from det_model_v2 import TLDetHead3D, assign_3d, loss_3d          # noqa: E402
from train_det import ViTTap                                      # noqa: E402


def targets_for(row, rows_, cols_, W0, H0, up, patch, device):
    gh, gw = rows_ * up, cols_ * up
    cell = patch / up
    sx = (cols_ * patch) / W0
    sy = (rows_ * patch) / H0
    bs, rg, ht = [], [], []
    for L in row['lights3d']:
        x1, y1, x2, y2 = L['box']
        x1, x2 = max(0.0, x1) * sx, min(W0, x2) * sx
        y1, y2 = max(0.0, y1) * sy, min(H0, y2) * sy
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        bs.append([x1, y1, x2, y2]); rg.append(L['range_m']); ht.append(L['height_m'])
    if not bs:
        return None
    return assign_3d(torch.tensor(bs, device=device), rg, ht, gh, gw, cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/tlb/det3d_train.jsonl')
    ap.add_argument('--val', default='data/tlb/det3d_val.jsonl')
    ap.add_argument('--init', default='outputs/tlb/det_head_v2.pt',
                    help='v1 2D detector to build on')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--unfreeze', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save', default='outputs/tlb/det_head3d_v2.pt')
    ap.add_argument('--log-every', type=int, default=50)
    args = ap.parse_args()

    device = 'cuda'
    rows = [json.loads(l) for l in open(BASE / args.train)]
    vrows = [json.loads(l) for l in open(BASE / args.val)]
    if args.limit:
        rows = rows[:args.limit]; vrows = vrows[:max(args.limit // 4, 8)]
    print(f"train {len(rows)} frames ({sum(len(r['lights3d']) for r in rows)} labelled lights), "
          f"val {len(vrows)}", flush=True)

    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tap = ViTTap(model)
    patch = tap.proc.patch_size

    feat, r0, c0, W0, H0 = tap.features(ROOT / rows[0]['image'], device)
    ck = torch.load(BASE / args.init, map_location='cpu')
    up = ck['up']
    head = TLDetHead3D(in_dim=feat.shape[0], hid=ck['args']['hid'], up=up).to(device).float()
    n = head.load_v1(ck['head'])
    print(f"initialised from {args.init}: {n} tensors carried over, up={up}", flush=True)

    if args.unfreeze:
        params = list(head.parameters())
        print("training the WHOLE head", flush=True)
    else:
        for nme, p in head.named_parameters():
            p.requires_grad_(nme.startswith('d3.'))
        params = [p for p in head.parameters() if p.requires_grad]
        print(f"2D branch frozen; training the 3D branch only "
              f"({sum(p.numel() for p in params)/1e6:.2f} M params)", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    order = []
    rnd = random.Random(0)
    for _ in range(args.epochs):
        e = list(range(len(rows))); rnd.shuffle(e); order += e
    steps = max(len(order) // args.accum, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)

    @torch.no_grad()
    def evaluate():
        head.eval()
        er, eh = [], []
        for r in vrows:
            f, rr, cc, w0, h0 = tap.features(ROOT / r['image'], device)
            t = targets_for(r, rr, cc, w0, h0, up, patch, device)
            if t is None:
                continue
            lr_t, ht_t, m = t
            o = head(f.float().unsqueeze(0))
            if not m.any():
                continue
            pr = o['logrange'][0][m].float().exp().cpu().numpy()
            gt = lr_t[m].float().exp().cpu().numpy()
            er += list(np.abs(pr - gt))
            eh += list(np.abs(o['height'][0][m].float().cpu().numpy() - ht_t[m].float().cpu().numpy()))
        head.train()
        return (float(np.median(er)) if er else float('nan'),
                float(np.median(eh)) if eh else float('nan'), len(er))

    print(f"{len(order)} samples -> {steps} steps", flush=True)
    t0 = time.time(); step = 0; acc = []
    opt.zero_grad(set_to_none=True)
    for i, idx in enumerate(order):
        r = rows[idx]
        feat, rr, cc, w0, h0 = tap.features(ROOT / r['image'], device)
        t = targets_for(r, rr, cc, w0, h0, up, patch, device)
        if t is None:
            continue
        lr_t, ht_t, m = t
        out = head(feat.float().unsqueeze(0))
        L, parts = loss_3d(out, lr_t, ht_t, m)
        (L / args.accum).backward()
        acc.append(parts)
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % args.log_every == 0:
                w = acc[-args.accum * args.log_every:]
                mr = sum(x['range'] for x in w) / len(w)
                mh = sum(x['height'] for x in w) / len(w)
                el = time.time() - t0
                print(f"  step {step:5d}/{steps}  range {mr:.4f}  height {mh:.4f}  "
                      f"{el:5.0f}s eta {el/step*(steps-step)/60:5.1f}m", flush=True)
    mr, mh, nv = evaluate()
    sp = BASE / args.save
    torch.save({'head': head.state_dict(), 'args': vars(args), 'up': up, 'patch': patch,
                'from': args.init}, sp)
    print(f"\n  VAL median |range error| {mr:.2f} m   |height error| {mh:.2f} m   (n={nv} cells)")
    print(f"  saved {sp}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
