#!/usr/bin/env python
"""Train the light selector on cached ROI features, and score it the way it will be used.

Two metrics, and the second is the one that matters:

  gov AP / top-1 gov    did it point at a light annotated as governing
  ANSWER accuracy       does the selected light's colour equal the frame's label

They differ because several lights often share a colour: choosing a non-governing light
that happens to show the same colour still yields the right answer. Only on
`discriminative` frames does the distinction bite, so that subset is reported separately.

Epoch selection uses a held-out slice of TRAIN segments; val is read once, at the end.
"""
from __future__ import annotations
import argparse, json, random, time
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
import sys
sys.path.insert(0, str(BASE / 'local/tlb'))
from selector import Selector, select_loss                      # noqa: E402

COLOURS = ['unknown', 'red', 'green', 'yellow']


class RoiSet(Dataset):
    def __init__(self, npz, segs=None, exclude=None):
        z = np.load(npz)
        d = {k: z[k] for k in ('roi', 'ctx', 'nl', 'box', 'col', 'gov', 'seg', 'keys')}
        self.d = d
        seg = d['seg']
        keep = np.ones(len(seg), bool)
        if segs is not None:
            keep &= np.isin(seg, list(segs))
        if exclude is not None:
            keep &= ~np.isin(seg, list(exclude))
        self.idx = np.nonzero(keep)[0]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        d = self.d
        nl = int(d['nl'][j])
        roi = torch.from_numpy(np.asarray(d['roi'][j], np.float32))
        box = torch.from_numpy(np.asarray(d['box'][j], np.float32))
        ctx = torch.from_numpy(np.asarray(d['ctx'][j], np.float32))
        gov = torch.from_numpy(np.asarray(d['gov'][j], np.int64))
        col = torch.from_numpy(np.asarray(d['col'][j], np.int64))
        mask = torch.zeros(roi.shape[0], dtype=torch.bool)
        mask[:nl] = True
        return roi, box, ctx, gov, col, mask, j


@torch.no_grad()
def evaluate(model, dl, device, labels, keys, eid=None):
    """labels: key -> frame colour label (only usable frames have one)."""
    model.eval()
    ok = n = 0
    okd = nd = 0
    gov_hit = gov_n = 0
    per = []
    for roi, box, ctx, gov, col, mask, j in dl:
        lg = model(roi.to(device), box.to(device), ctx.to(device), mask.to(device))
        lg = lg.masked_fill(~mask.to(device), -1e4)
        pick = lg.argmax(1).cpu()
        for b in range(len(pick)):
            key = keys[int(j[b])]
            lab = labels.get(key)
            if lab is None:
                continue
            p = int(pick[b])
            if (gov[b] >= 0).any():
                gov_hit += int(gov[b][p] == 1); gov_n += 1
            pred = COLOURS[int(col[b][p])]
            good = (pred == lab['label'])
            ok += good; n += 1
            if lab['discriminative']:
                okd += good; nd += 1
            per.append({'key': key, 'gt': lab['label'], 'pred': pred,
                        'disc': lab['discriminative'], 'w': lab['max_gov_wh'][0],
                        'ep': (eid or {}).get(key, -1)})
    model.train()
    return (ok / max(n, 1), n, okd / max(nd, 1), nd,
            gov_hit / max(gov_n, 1), per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-roi', default='data/tlb/roi/train.npz')
    ap.add_argument('--val-roi', default='data/tlb/roi/val.npz')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--holdout', type=float, default=0.18)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--jitter', type=float, default=0.0,
                    help='box-coordinate jitter (fraction of image) applied in training')
    ap.add_argument('--feat-drop', type=float, default=0.0,
                    help='randomly zero whole light features, so the model cannot lean '
                         'on one candidate')
    ap.add_argument('--select-on', default='answer', choices=['answer', 'disc'])
    ap.add_argument('--save', default='outputs/tlb/selector.pt')
    ap.add_argument('--out', default='outputs/tlb/selector_val.json')
    args = ap.parse_args()

    device = 'cuda'
    dtr_np = np.load(BASE / args.train_roi)
    segs = sorted(set(dtr_np['seg'].tolist()))
    random.Random(31).shuffle(segs)
    hold = set(segs[:max(1, int(len(segs) * args.holdout))])
    tr = RoiSet(BASE / args.train_roi, exclude=hold)
    sel = RoiSet(BASE / args.train_roi, segs=hold)
    va = RoiSet(BASE / args.val_roi)
    print(f"train {len(tr)} frames | selection {len(sel)} ({len(hold)} segs held out) | val {len(va)}")

    def labels_for(split):
        out = {}
        for l in open(BASE / f'data/tlb/{split}.jsonl'):
            r = json.loads(l)
            out[f"{r['segment']}/{r['timestamp']}"] = r
        return out
    lab_tr, lab_va = labels_for('train'), labels_for('val')
    keys_tr = dtr_np['keys']
    keys_va = np.load(BASE / args.val_roi)['keys']

    vr = [json.loads(l) for l in open(BASE / 'data/tlb/val.jsonl')]
    vr.sort(key=lambda r: (r['segment'], r['timestamp']))
    eid = {}; n = 0
    for s, g in groupby(vr, key=lambda r: r['segment']):
        for lb, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1

    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    dl_sel = DataLoader(sel, batch_size=128, num_workers=3)
    dl_va = DataLoader(va, batch_size=128, num_workers=3)

    model = Selector(d=args.d, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=max(len(dl_tr), 1), pct_start=0.1)
    best = -1.0
    t0 = time.time()
    for ep in range(args.epochs):
        tot = c = 0
        for roi, box, ctx, gov, col, mask, _ in dl_tr:
            roi, box = roi.to(device), box.to(device)
            ctx, mask = ctx.to(device), mask.to(device)
            if args.jitter:
                box = box + torch.randn_like(box) * args.jitter
            if args.feat_drop:
                keep = (torch.rand(roi.shape[:2], device=device) > args.feat_drop).float()
                roi = roi * keep[..., None]
            lg = model(roi, box, ctx, mask)
            loss = select_loss(lg, gov.to(device), mask.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach()); c += 1
        sa, sn, sd, sdn, sgov, _ = evaluate(model, dl_sel, device, lab_tr, keys_tr)
        # Selecting on the discriminative subset alone picks epoch 1 on ~270 frames,
        # which is noise; overall answer accuracy is the deployed metric.
        crit = sa if args.select_on == 'answer' else sd
        flag = ''
        if crit > best:
            best = crit
            torch.save({'model': model.state_dict(), 'args': vars(args)}, BASE / args.save)
            flag = '  *saved'
        print(f"  ep {ep+1:2d}/{args.epochs} loss {tot/max(c,1):.4f} | SEL answer {sa:.2%} "
              f"(n={sn}) disc {sd:.2%} (n={sdn}) gov-top1 {sgov:.2%}  "
              f"{time.time()-t0:.0f}s{flag}", flush=True)

    ck = torch.load(BASE / args.save, map_location='cpu')
    model.load_state_dict(ck['model'])
    va_a, va_n, va_d, va_dn, va_gov, per = evaluate(model, dl_va, device, lab_va, keys_va, eid)
    print(f"\n===== VAL (selector picks the light; colour taken from annotation) =====")
    print(f"  answer accuracy      {va_a:.2%}  (n={va_n})")
    print(f"  discriminative       {va_d:.2%}  (n={va_dn})")
    print(f"  top-1 is governing   {va_gov:.2%}")
    (BASE / args.out).write_text(json.dumps(per, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
