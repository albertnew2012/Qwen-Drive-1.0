#!/usr/bin/env python
"""Which of the visible traffic lights governs the ego's lane?

Colour is solved (99.8% from a correct crop), so this is the whole remaining problem.
It is a choice among candidates, not an independent judgement per light, so the model
attends across the lights in a frame rather than scoring each alone: whether a light is
"mine" depends on where the others are and where the road goes.

Inputs per frame, all cached from the frozen vision tower:
    roi   [L, C*ROI*ROI]  appearance of each light
    box   [L, 4]          centre and size, normalised -- where it sits in the image
    ctx   [C, GH, GW]     coarse scene grid, which is what carries the lane layout

Output: one logit per light. Trained with BCE against is_gov (a frame can have several
governing lights); at inference the highest-scoring light is taken as the ego's.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


class Selector(nn.Module):
    def __init__(self, roi_dim=4096, ctx_dim=1024, d=256, heads=8, layers=3,
                 n_ctx=84, dropout=0.1):
        super().__init__()
        self.light = nn.Sequential(nn.LayerNorm(roi_dim), nn.Linear(roi_dim, d), nn.GELU(),
                                   nn.Linear(d, d))
        # geometry gets its own embedding: a light's position in the frame is the single
        # strongest cue for whether it is over the ego's lane
        self.geo = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.Sequential(nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.ctx_pos = nn.Parameter(torch.zeros(n_ctx, d))
        nn.init.normal_(self.ctx_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=dropout,
                                           batch_first=True, norm_first=True,
                                           activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, roi, box, ctx, mask):
        """roi [B,L,R]  box [B,L,4]  ctx [B,C,GH,GW]  mask [B,L] True where a light exists."""
        B, L, _ = roi.shape
        x = self.light(roi) + self.geo(box)
        c = ctx.flatten(2).transpose(1, 2)                  # [B, GH*GW, C]
        c = self.ctx(c) + self.ctx_pos[None, :c.shape[1]]
        seq = torch.cat([x, c], 1)
        pad = torch.cat([~mask, torch.zeros(B, c.shape[1], dtype=torch.bool,
                                            device=mask.device)], 1)
        out = self.enc(seq, src_key_padding_mask=pad)
        return self.head(out[:, :L]).squeeze(-1)             # [B, L] logits


def select_loss(logits, gov, mask):
    """BCE over real lights only; `gov` of -1 (ego lane unknown) is ignored."""
    valid = mask & (gov >= 0)
    if not valid.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(
        logits[valid].float(), gov[valid].float().clamp(0, 1))
