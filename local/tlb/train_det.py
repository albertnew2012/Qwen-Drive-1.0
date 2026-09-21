#!/usr/bin/env python
"""Train the 2D traffic-light head on Qwen-Drive's frozen vision tower.

The tower never updates, so this is cheap: one no-grad ViT forward per image, then a
small dense head. No feature cache is needed (and would not fit -- 12k frames of
56x100x1024 bf16 is ~138 GB).

    .venv/bin/python local/tlb/train_det.py --probe        # shapes only, no training
    .venv/bin/python local/tlb/train_det.py --epochs 4
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'local/tlb'))
ROOT = BASE / 'data/OpenLane-V2'

from det_model import TLDetHead, assign, decode, focal_bce, COLOUR_IDX   # noqa: E402


class ViTTap:
    """Frozen pre-merge patch features for one image."""

    def __init__(self, model):
        self.model = model
        self.visual = model.vlm.model.visual
        self.vdtype = next(self.visual.parameters()).dtype
        self.proc = model.processor

    def patchify(self, path):
        from qwen_drive import CameraFrame
        im = Image.open(path).convert('RGB')
        frame = CameraFrame(im, target_size=im.size)
        patches, (rows, cols) = self.proc._patchify(frame, self.proc.config.current_image_pixels)
        return patches, rows, cols, im.size

    @torch.no_grad()
    def features(self, path, device):
        patches, rows, cols, (W0, H0) = self.patchify(path)
        grid = torch.tensor([[1, rows, cols]], dtype=torch.long, device=device)
        cap = {}
        h = self.visual.merger.register_forward_hook(
            lambda m, a, o=None: cap.__setitem__('p', a[0]))
        try:
            self.visual(patches.to(device, self.vdtype), grid)
        finally:
            h.remove()
        p = self.visual.merger.norm(cap['p'])                    # [rows*cols, C]
        # undo the 2x2 block ordering so the grid is row-major again
        C = p.shape[-1]
        g = (p.view(rows // 2, cols // 2, 2, 2, C).permute(0, 2, 1, 3, 4)
             .reshape(rows, cols, C))
        return g.permute(2, 0, 1).contiguous(), rows, cols, W0, H0


def make_targets(row, rows, cols, W0, H0, up, patch, device):
    """Scale GT boxes into the resized image and rasterise them onto the head grid."""
    gh, gw = rows * up, cols * up
    cell = patch / up                       # pixels per head cell, in resized coords
    sx = (cols * patch) / W0
    sy = (rows * patch) / H0
    bs, cs, gs = [], [], []
    for L in row['lights']:
        x1, y1, x2, y2 = L['box']
        x1, x2 = max(0.0, x1) * sx, min(W0, x2) * sx
        y1, y2 = max(0.0, y1) * sy, min(H0, y2) * sy
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        bs.append([x1, y1, x2, y2])
        cs.append(COLOUR_IDX[L['colour']])
        gs.append(L['is_gov'])
    if not bs:
        return None
    return assign(torch.tensor(bs, device=device), cs, gs, gh, gw, cell)


def losses(out, tgt, w_box=1.0, w_col=1.0, w_gov=1.0):
    obj_t, box_t, col_t, gov_t = tgt
    pos = obj_t[0] > 0.5
    npos = max(int(pos.sum()), 1)
    l_obj = focal_bce(out['obj'][0], obj_t) / npos
    if pos.any():
        pb = out['box'][0][:, pos]
        tb = box_t[:, pos]
        # IoU on ltrb distances, the standard FCOS regression target
        inter = (torch.min(pb[0], tb[0]) + torch.min(pb[2], tb[2])).clamp(min=0) * \
                (torch.min(pb[1], tb[1]) + torch.min(pb[3], tb[3])).clamp(min=0)
        ap = (pb[0] + pb[2]) * (pb[1] + pb[3])
        at = (tb[0] + tb[2]) * (tb[1] + tb[3])
        iou = inter / (ap + at - inter + 1e-7)
        l_box = -(iou.clamp(min=1e-7)).log().mean()
        cl = col_t[pos]
        l_col = F.cross_entropy(out['col'][0][:, pos].T.float(), cl, ignore_index=-100)
        gm = gov_t[0][pos] >= 0
        gv = out['gov'][0, 0][pos]          # [1,H,W] -> [H,W] before masking
        l_gov = (F.binary_cross_entropy_with_logits(
            gv[gm].float(), gov_t[0][pos][gm].float())
            if gm.any() else out['gov'].sum() * 0)
    else:
        l_box = l_col = l_gov = out['obj'].sum() * 0
    return l_obj + w_box * l_box + w_col * l_col + w_gov * l_gov, \
        {'obj': float(l_obj), 'box': float(l_box), 'col': float(l_col), 'gov': float(l_gov)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/tlb/det_train.jsonl')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--up', type=int, default=2)
    ap.add_argument('--hid', type=int, default=256)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--save', default='outputs/tlb/det_head.pt')
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda'
    torch.manual_seed(args.seed)
    rows = [json.loads(l) for l in open(BASE / args.train)]
    if args.limit:
        rows = rows[:args.limit]

    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tap = ViTTap(model)
    patch = tap.proc.patch_size

    feat, r, c, W0, H0 = tap.features(ROOT / rows[0]['image'], device)
    print(f"image {W0}x{H0} -> grid {r}x{c} (patch {patch}px), feature {tuple(feat.shape)}", flush=True)
    head = TLDetHead(in_dim=feat.shape[0], hid=args.hid, up=args.up).to(device).float()
    print(f"head params: {sum(p.numel() for p in head.parameters())/1e6:.2f} M", flush=True)
    if args.probe:
        out = head(feat.float().unsqueeze(0))
        print({k: tuple(v.shape) for k, v in out.items()})
        t = make_targets(rows[0], r, c, W0, H0, args.up, patch, device)
        print("targets:", None if t is None else [tuple(x.shape) for x in t])
        print("n lights:", len(rows[0]['lights']), rows[0]['lights'][:2])
        if t is not None:
            L, parts = losses(out, t)
            print("loss:", float(L), parts)
        return

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    order = []
    rnd = random.Random(args.seed)
    for _ in range(args.epochs):
        e = list(range(len(rows))); rnd.shuffle(e); order += e
    steps = len(order) // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1),
                                                pct_start=0.05)
    print(f"{len(rows)} frames x {args.epochs} epochs = {len(order)} samples -> {steps} steps",
          flush=True)
    t0 = time.time(); acc = []; step = 0
    opt.zero_grad(set_to_none=True)
    for i, idx in enumerate(order):
        row = rows[idx]
        feat, r, c, W0, H0 = tap.features(ROOT / row['image'], device)
        tgt = make_targets(row, r, c, W0, H0, args.up, patch, device)
        if tgt is None:
            continue
        out = head(feat.float().unsqueeze(0))
        L, parts = losses(out, tgt)
        (L / args.accum).backward()
        acc.append(parts | {'total': float(L)})
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % args.log_every == 0:
                w = acc[-args.accum * args.log_every:]
                m = {k: sum(x[k] for x in w) / len(w) for k in w[0]}
                el = time.time() - t0
                print(f"  step {step:5d}/{steps}  total {m['total']:.3f}  obj {m['obj']:.3f}  "
                      f"box {m['box']:.3f}  col {m['col']:.3f}  gov {m['gov']:.3f}  "
                      f"{el:5.0f}s eta {el/step*(steps-step)/60:5.1f}m", flush=True)
    sp = BASE / args.save
    sp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'head': head.state_dict(), 'args': vars(args),
                'up': args.up, 'patch': patch}, sp)
    print(f"saved {sp} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
