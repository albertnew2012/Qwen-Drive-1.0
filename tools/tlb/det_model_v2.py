#!/usr/bin/env python
"""v2 of the traffic-light head: adds monocular 3D to the v1 2D outputs.

v1 (det_model.py) predicts obj / box / colour / gov on an upsampled patch grid. v2 keeps
those unchanged and adds two regression channels per cell:

    logrange   log of the ground-plane distance, so relative error is penalised evenly
               across 15-40 m rather than the head spending capacity on the far tail
    height     metres above the ego plane, regressed directly (it spans only ~2-5 m)

Supervision comes from triangulated-then-Occ3D-refined labels (~0.35 m, validated against
lidar), which exist only within 40 m -- beyond ~55 m triangulation carries no signal at
all, so the head is deliberately near-range and the loss is masked where no label exists.

v1's weights load into v2 unchanged; only the new 3D branch starts fresh.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from det_model import ConvBlock, COLOURS, COLOUR_IDX, focal_bce, assign   # noqa: F401,E402


class TLDetHead3D(nn.Module):
    """v1 head plus a 3D branch. `load_v1` brings the 2D weights across."""

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
        # 3D branch: its own two blocks, so learning depth cannot disturb the 2D outputs
        self.d3 = nn.Sequential(ConvBlock(hid, hid), nn.Conv2d(hid, 2, 3, padding=1))
        nn.init.constant_(self.obj.bias, -4.6)
        nn.init.constant_(self.gov.bias, 0.0)
        # start at the label medians: log(28 m) and 3.2 m
        nn.init.zeros_(self.d3[-1].weight)
        with torch.no_grad():
            self.d3[-1].bias.copy_(torch.tensor([3.33, 3.2]))

    def forward(self, feat):
        x = self.stem(feat)
        if self.up != 1:
            x = F.interpolate(x, scale_factor=self.up, mode='bilinear', align_corners=False)
        x = self.post(x)
        d3 = self.d3(x)
        return {'obj': self.obj(x), 'box': self.box(x).clamp(-8, 8).exp(),
                'col': self.col(x), 'gov': self.gov(x),
                'logrange': d3[:, 0:1], 'height': d3[:, 1:2]}

    @torch.no_grad()
    def load_v1(self, state):
        """Copy the v1 2D weights; the 3D branch keeps its initialisation."""
        own = self.state_dict()
        taken = 0
        for k, v in state.items():
            if k in own and own[k].shape == v.shape and not k.startswith('d3.'):
                own[k].copy_(v)
                taken += 1
        self.load_state_dict(own)
        return taken


def assign_3d(boxes, rng_m, h_m, gh, gw, cell, radius=1.5):
    """Dense 3D targets on the same cells the 2D assigner marks positive.

    Returns (logrange, height, mask). Cells with no 3D-labelled light are masked out --
    most lights have no confirmed 3D position, and inventing one would be worse than
    training on fewer.
    """
    dev = boxes.device if isinstance(boxes, torch.Tensor) else 'cpu'
    lr = torch.zeros(1, gh, gw, device=dev)
    ht = torch.zeros(1, gh, gw, device=dev)
    m = torch.zeros(1, gh, gw, dtype=torch.bool, device=dev)
    if len(boxes) == 0:
        return lr, ht, m
    ys = (torch.arange(gh, device=dev).float() + 0.5) * cell
    xs = (torch.arange(gw, device=dev).float() + 0.5) * cell
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in torch.argsort(areas, descending=True).tolist():
        x1, y1, x2, y2 = boxes[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sel = (gx > x1) & (gx < x2) & (gy > y1) & (gy < y2)
        sel &= ((gx - cx).abs() <= radius * cell) & ((gy - cy).abs() <= radius * cell)
        if not sel.any():
            j = ((gy - cy) ** 2 + (gx - cx) ** 2).argmin()
            sel = torch.zeros_like(sel).view(-1)
            sel[j] = True
            sel = sel.view(gh, gw)
        lr[0][sel] = float(torch.log(torch.as_tensor(max(rng_m[i], 1.0))))
        ht[0][sel] = float(h_m[i])
        m[0][sel] = True
    return lr, ht, m


def loss_3d(out, lr_t, ht_t, mask, w_range=1.0, w_height=0.5):
    """Smooth-L1 on log-range and height, only where a 3D label exists."""
    if not mask.any():
        z = out['logrange'].sum() * 0
        return z, {'range': 0.0, 'height': 0.0}
    lr_p = out['logrange'][0][mask]
    ht_p = out['height'][0][mask]
    l_r = F.smooth_l1_loss(lr_p.float(), lr_t[mask].float(), beta=0.1)
    l_h = F.smooth_l1_loss(ht_p.float(), ht_t[mask].float(), beta=0.25)
    return w_range * l_r + w_height * l_h, {'range': float(l_r), 'height': float(l_h)}
