#!/usr/bin/env python
"""A 2D traffic-light head on Qwen-Drive's frozen vision tower.

The repo's existing BEV head reads the ViT *before* the 2x2 merge (perception tap 1)
because those patches still carry geometry. The same tap works for 2D detection, and it
is cheap: the tower is frozen, so only this head trains.

Grid: CAM_FRONT at native 1600x900 snaps to 1600x896 -> a 56x100 patch grid at 16 px per
cell. A median traffic light is 18x27 px, barely one cell wide, so the head upsamples 2x
to 8 px cells before predicting.

Per cell it predicts, FCOS-style and anchor-free:
    obj     1   is a light centred here
    box     4   l,t,r,b distances to the box edges, in cells, via exp()
    colour  4   unknown / red / green / yellow
    gov     1   does this light govern the ego lane

`gov` is the piece that makes the downstream rule trivial: no geometric reasoning at
inference, just read the light the head already marked as the ego's.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

COLOURS = ['unknown', 'red', 'green', 'yellow']
COLOUR_IDX = {c: i for i, c in enumerate(COLOURS)}


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, 3, padding=1)
        self.n = nn.GroupNorm(32, cout)

    def forward(self, x):
        return F.gelu(self.n(self.c(x)))


class TLDetHead(nn.Module):
    """Dense predictor over an upsampled patch grid."""

    def __init__(self, in_dim=1024, hid=256, blocks=3, up=2):
        super().__init__()
        self.up = up
        self.stem = nn.Sequential(nn.Conv2d(in_dim, hid, 1), nn.GroupNorm(32, hid), nn.GELU(),
                                  *[ConvBlock(hid, hid) for _ in range(blocks)])
        self.post = nn.Sequential(ConvBlock(hid, hid), ConvBlock(hid, hid))
        self.obj = nn.Conv2d(hid, 1, 3, padding=1)
        self.box = nn.Conv2d(hid, 4, 3, padding=1)
        self.col = nn.Conv2d(hid, len(COLOURS), 3, padding=1)
        self.gov = nn.Conv2d(hid, 1, 3, padding=1)
        # start with a low objectness prior: most cells are background
        nn.init.constant_(self.obj.bias, -4.6)   # sigmoid(-4.6) ~ 0.01
        nn.init.constant_(self.gov.bias, 0.0)

    def forward(self, feat):
        """feat: [B, C, H, W] pre-merge ViT grid -> dict of [B, *, H*up, W*up]."""
        x = self.stem(feat)
        if self.up != 1:
            x = F.interpolate(x, scale_factor=self.up, mode='bilinear', align_corners=False)
        x = self.post(x)
        return {'obj': self.obj(x), 'box': self.box(x).clamp(-8, 8).exp(),
                'col': self.col(x), 'gov': self.gov(x)}


def focal_bce(logits, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """Sigmoid focal loss (RetinaNet)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = w * ((1 - pt) ** gamma) * ce
    return loss.sum() if reduction == 'sum' else loss.mean()


def assign(boxes, colours, govs, gh, gw, cell, radius=1.5):
    """Build dense targets. boxes are xyxy in *resized image* pixels.

    A cell is positive when its centre lies inside the box and within `radius` cells of
    the box centre -- centre sampling, which keeps tiny boxes from being swamped by the
    ambiguous cells at their edges.
    """
    dev = boxes.device if isinstance(boxes, torch.Tensor) else 'cpu'
    obj = torch.zeros(1, gh, gw, device=dev)
    box_t = torch.zeros(4, gh, gw, device=dev)
    col_t = torch.full((gh, gw), -100, dtype=torch.long, device=dev)
    gov_t = torch.full((1, gh, gw), -1.0, device=dev)
    if len(boxes) == 0:
        return obj, box_t, col_t, gov_t
    ys = (torch.arange(gh, device=dev).float() + 0.5) * cell
    xs = (torch.arange(gw, device=dev).float() + 0.5) * cell
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = torch.argsort(areas, descending=True)      # small boxes win overlaps
    for i in order.tolist():
        x1, y1, x2, y2 = boxes[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inside = (gx > x1) & (gx < x2) & (gy > y1) & (gy < y2)
        near = (gx - cx).abs() <= radius * cell
        near &= (gy - cy).abs() <= radius * cell
        m = inside & near
        if not m.any():                                 # box smaller than a cell
            j = ((gy - cy) ** 2 + (gx - cx) ** 2).argmin()
            m = torch.zeros_like(inside).view(-1)
            m[j] = True
            m = m.view(gh, gw)
        obj[0][m] = 1.0
        box_t[0][m] = ((gx - x1) / cell)[m]
        box_t[1][m] = ((gy - y1) / cell)[m]
        box_t[2][m] = ((x2 - gx) / cell)[m]
        box_t[3][m] = ((y2 - gy) / cell)[m]
        col_t[m] = int(colours[i])
        gov_t[0][m] = float(govs[i])
    return obj, box_t, col_t, gov_t


def decode(out, cell, topk=50, thr=0.3):
    """Dense maps -> boxes. Returns list of dicts per batch item."""
    res = []
    B, _, H, W = out['obj'].shape
    prob = torch.sigmoid(out['obj'])[:, 0]
    # 3x3 max-pool NMS: cheap and enough for objects this small
    keep = (F.max_pool2d(prob[:, None], 3, 1, 1)[:, 0] == prob) & (prob > thr)
    for b in range(B):
        idx = keep[b].nonzero(as_tuple=False)
        if len(idx) > topk:
            sc = prob[b][idx[:, 0], idx[:, 1]]
            idx = idx[sc.topk(topk).indices]
        dets = []
        for yy, xx in idx.tolist():
            l, t, r, d = out['box'][b, :, yy, xx].tolist()
            cx, cy = (xx + 0.5) * cell, (yy + 0.5) * cell
            dets.append({
                'box': [cx - l * cell, cy - t * cell, cx + r * cell, cy + d * cell],
                'score': float(prob[b, yy, xx]),
                'colour': COLOURS[int(out['col'][b, :, yy, xx].argmax())],
                'colour_probs': torch.softmax(out['col'][b, :, yy, xx], 0).tolist(),
                'gov': float(torch.sigmoid(out['gov'][b, 0, yy, xx])),
            })
        res.append(sorted(dets, key=lambda d: -d['score']))
    return res
