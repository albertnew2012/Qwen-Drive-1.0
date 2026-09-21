#!/usr/bin/env python
"""A colour head that only ever sees crops.

The VQA baseline's dominant error is red read as yellow or green on a lamp a dozen pixels
across inside a 1600x900 frame. Given the same lamp as an upscaled crop the problem is
nearly trivial, so this isolates it: a small ResNet over 64x64 crops, 4 classes
(unknown / red / green / yellow).

Augmentation deliberately excludes hue: shifting hue would change the label. Brightness,
contrast, small geometric jitter and horizontal flip are all label-preserving (a vertical
head's lit third stays in place under a flip).

Class counts are wildly uneven (43k unknown vs 716 yellow), so the sampler balances and
the loss is unweighted -- balancing twice over-corrects.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
COLOURS = ['unknown', 'red', 'green', 'yellow']


class Crops(Dataset):
    def __init__(self, npz, train=True, min_w=0.0, segs=None, exclude_segs=None):
        d = np.load(npz)
        keep = d['w'] >= min_w
        # Model selection must not touch val: carve the selection set out of TRAIN by
        # segment, so the epoch is chosen without ever reading the held-out split.
        if segs is not None:
            keep &= np.isin(d['seg'], list(segs))
        if exclude_segs is not None:
            keep &= ~np.isin(d['seg'], list(exclude_segs))
        self.X = d['X'][keep]
        self.y = d['y'][keep]
        self.w = d['w'][keep]
        self.gov = d['is_gov'][keep]
        self.seg = d['seg'][keep]
        self.ts = d['ts'][keep]
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].copy()).permute(2, 0, 1).float() / 255.
        if self.train:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [2])
            # brightness / contrast only: hue would change the class
            x = (x * (0.7 + 0.6 * torch.rand(1))).clamp(0, 1)
            m = x.mean()
            x = ((x - m) * (0.7 + 0.6 * torch.rand(1)) + m).clamp(0, 1)
            if torch.rand(1) < 0.7:                       # small translate/scale jitter
                s = 0.85 + 0.3 * float(torch.rand(1))
                th = torch.tensor([[s, 0, float(torch.rand(1) - .5) * .25],
                                   [0, s, float(torch.rand(1) - .5) * .25]]).unsqueeze(0)
                g = F.affine_grid(th, (1, 3, x.shape[1], x.shape[2]), align_corners=False)
                x = F.grid_sample(x.unsqueeze(0), g, align_corners=False, padding_mode='border')[0]
            x = x + torch.randn_like(x) * 0.02
        x = (x - 0.5) / 0.5
        return x, int(self.y[i])


def build(arch='resnet18'):
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    # 64x64 input: the stock 7x7 stride-2 stem plus maxpool throws away too much for a
    # 64px crop, so use a 3x3 stride-1 stem and drop the maxpool.
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, len(COLOURS))
    return m


@torch.no_grad()
def eval_gov(model, ds, device, bs=512):
    """Accuracy on *governing* lights, argmax over red/green/yellow.

    This is the number the pipeline actually consumes, so it is what the epoch is chosen
    on. Selecting on macro-recall instead rewards early epochs that over-predict the rare
    yellow class while red recall is still ~70%, which is how an earlier run saved a
    checkpoint that made 123 red->yellow errors downstream.
    """
    sel = (ds.gov == 1) & np.isin(ds.y, [1, 2, 3])
    idx = np.nonzero(sel)[0]
    if len(idx) == 0:
        return float('nan'), 0
    model.eval()
    ok = 0
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        x = torch.stack([ds[int(j)][0] for j in chunk]).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            lg = model(x).float()
        pred = lg[:, 1:4].argmax(1).cpu().numpy() + 1
        ok += int((pred == ds.y[chunk]).sum())
    model.train()
    return ok / len(idx), len(idx)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    n = ok = 0
    per = {c: [0, 0] for c in range(len(COLOURS))}
    conf = np.zeros((len(COLOURS), len(COLOURS)), dtype=int)
    for x, y in dl:
        p = model(x.to(device)).argmax(1).cpu()
        ok += int((p == y).sum()); n += len(y)
        for a, b in zip(y.tolist(), p.tolist()):
            per[a][1] += 1; per[a][0] += int(a == b); conf[a, b] += 1
    model.train()
    rec = {COLOURS[c]: (per[c][0] / per[c][1] if per[c][1] else float('nan')) for c in per}
    return ok / n, rec, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/tlb/crops/train.npz')
    ap.add_argument('--val', default='data/tlb/crops/val.npz')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--balance', action='store_true')
    ap.add_argument('--min-w', type=float, default=0.0)
    ap.add_argument('--holdout', type=float, default=0.15,
                    help='fraction of TRAIN segments held out to choose the epoch')
    ap.add_argument('--save', default='outputs/tlb/colour_head.pt')
    args = ap.parse_args()

    device = 'cuda'
    import random as _r
    all_segs = sorted(set(np.load(BASE / args.train)['seg'].tolist()))
    _r.Random(23).shuffle(all_segs)
    n_sel = max(1, int(len(all_segs) * args.holdout))
    sel_segs = set(all_segs[:n_sel])
    tr = Crops(BASE / args.train, True, args.min_w, exclude_segs=sel_segs)
    sel = Crops(BASE / args.train, False, args.min_w, segs=sel_segs)
    va = Crops(BASE / args.val, False, args.min_w)
    print(f"selection split: {len(sel_segs)} of {len(all_segs)} train segments held out "
          f"({len(sel)} crops) -- val is never used to choose the epoch")
    print(f"train {len(tr)} crops, val {len(va)}")
    print("  train colour counts:", {COLOURS[i]: int((tr.y == i).sum()) for i in range(4)})
    print("  val   colour counts:", {COLOURS[i]: int((va.y == i).sum()) for i in range(4)})

    if args.balance:
        cnt = np.bincount(tr.y, minlength=4).astype(float)
        w = (1.0 / np.maximum(cnt, 1))[tr.y]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), True)
        dtr = DataLoader(tr, batch_size=args.bs, sampler=sampler, num_workers=8,
                         pin_memory=True, drop_last=True, persistent_workers=True)
    else:
        dtr = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=8,
                         pin_memory=True, drop_last=True, persistent_workers=True)
    dsel = DataLoader(sel, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    dva = DataLoader(va, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(dtr), pct_start=0.15)
    scaler = torch.amp.GradScaler('cuda')
    best = 0.0
    t0 = time.time()
    for ep in range(args.epochs):
        tot = n = 0
        for x, y in dtr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss) * len(y); n += len(y)
        sgov, sgov_n = eval_gov(model, sel, device)       # chooses the epoch
        acc, rec, conf = evaluate(model, dva, device)     # reported only
        smacro = sgov
        macro = np.mean([rec[c] for c in ('red', 'green', 'yellow')])
        flag = ''
        if smacro > best:
            best = smacro
            torch.save({'model': model.state_dict(), 'args': vars(args),
                        'colours': COLOURS}, BASE / args.save)
            flag = '  *saved'
        print(f"  ep {ep+1:2d}/{args.epochs}  loss {tot/n:.4f}  "
              f"SEL gov-acc {smacro:.3%} (n={sgov_n}) | val acc {acc:.3%}  "
              f"macroR(rgy) {macro:.3%}  " +
              "  ".join(f"{c} {rec[c]:.1%}" for c in COLOURS) +
              f"  {time.time()-t0:.0f}s{flag}", flush=True)
    print("\nconfusion (row=gt, col=pred):", COLOURS)
    print(conf)
    print(f"best SELECTION governing-light accuracy {best:.3%} -> {args.save}")


if __name__ == '__main__':
    main()
