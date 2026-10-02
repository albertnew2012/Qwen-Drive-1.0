"""Temporal BEV fusion: warp the previous keyframe's BEV into the current ego frame.

WHY
The student's remaining detection gap is recall at 10-30 m (student 63-76% vs teacher
74-91% on the same frames), which is mid-range depth ambiguity from a single view. Motion
parallax across keyframes resolves it, and every camera-only line since BEVDet4D takes its
largest single gain from exactly this (+5 to +10 mAP). Sweeps and keyframe linkage are
present; `data/distill/temporal_index.json` holds each frame's previous cached keyframe
and its ego-to-global pose (median ego motion 2.69 m per 0.50 s keyframe gap).

HOW, AND WHY IT EXPORTS
`T = inv(E_curr) @ E_prev` maps previous-ego coordinates into current-ego coordinates.
The warp is the inverse: for every CURRENT cell, look up where it was in the PREVIOUS BEV.
That lookup is one 4-D `grid_sample`, whose CUDA kernel at opset 20 was verified in this
repo (`Resize` and `GridSample` both placed on CUDA under ORT 1.25.1). The sampling grid
is built on the host from T -- it is 2x3 of affine arithmetic -- and passed to the graph as
an input, so the graph stays static and the interface is:

    inputs : image, bev_index, valid, ego, prev_bev (B,C,H,W), warp_grid (B,H,W,2)
    outputs: cls, box, occ, seg, trajectory, bev_state (B,C,H,W)   <- feed back next step

`bev_state` is the current frame's UNFUSED lift-splat BEV -- exactly what `bev_from`
produces for the previous frame at training time -- so the deployment loop reproduces
training; a fused (recurrent) state would be a different, untrained model.

At training time the previous frame's BEV is computed from its own six images under
no_grad (BEVDet4D detaches history); at inference the deployment loop feeds `bev_state`
back as `prev_bev`, so the backbone runs once per frame, not twice.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["warp_grid_from_T", "TemporalFuse", "load_temporal_index", "prev_bev_from_batch", "warp_from_batch"]


def load_temporal_index(path="data/distill/temporal_index.json"):
    import json
    return json.load(open(path))


def relative_T(idx: dict, tok: str):
    """prev-ego -> curr-ego 4x4 for a cached frame, or None if it has no cached prev."""
    v = idx.get(tok)
    if not v or not v["prev"]:
        return None
    Ec = np.array(v["ego2global"], dtype=np.float64).reshape(4, 4)
    Ep = np.array(idx[v["prev"]]["ego2global"], dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(Ec) @ Ep


def warp_grid_from_T(T: np.ndarray, cfg, n: int | None = None) -> np.ndarray:
    """(H, W, 2) normalised sampling grid for `grid_sample`, from prev->curr T.

    Every current cell centre (x_c, y_c) is mapped back into the previous ego frame with
    inv(T); that location, normalised to [-1, 1] over the BEV extent, is where
    `grid_sample` reads the previous BEV. Cells that fall outside the previous BEV read
    zeros (padding_mode='zeros'), which is the honest value for "no evidence".
    BEV layout matches geometry.py: dim -2 is y, dim -1 is x, cell 0 at pc_range min.
    """
    n = n or cfg.bev_size
    x0, y0, _, x1, y1, _ = cfg.pc_range
    xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    ys = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
    gy, gx = np.meshgrid(ys, xs, indexing="ij")                      # (H, W)
    P = np.stack([gx, gy, np.zeros_like(gx), np.ones_like(gx)], -1)  # (H, W, 4), z = 0
    Tinv = np.linalg.inv(np.asarray(T, dtype=np.float64))
    Q = P.reshape(-1, 4) @ Tinv.T                                    # curr -> prev ego
    px, py = Q[:, 0], Q[:, 1]
    u = 2.0 * (px - x0) / (x1 - x0) - 1.0                            # normalise to [-1,1]
    v = 2.0 * (py - y0) / (y1 - y0) - 1.0
    return np.stack([u, v], -1).reshape(n, n, 2).astype(np.float32)


class TemporalFuse(nn.Module):
    """Warp the previous BEV(s), concatenate with the current one, fuse back to C channels.

    history=1: prev_bev (B, C, H, W), warp_grid (B, H, W, 2) -- the original interface.
    history=K: prev_bev (B, K, C, H, W), warp_grid (B, K, H, W, 2); each of the K previous
    keyframes (0.5 s, 1.0 s, ... back) is warped with its own ego-motion grid and all K+1
    BEVs are concatenated (BEVDet4D -> SOLOFusion-style long history).
    """

    def __init__(self, channels: int, history: int = 1):
        super().__init__()
        self.history = history
        self.fuse = nn.Sequential(
            nn.Conv2d((1 + history) * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        # Zero-init the last conv's contribution from the previous frame? No: init is the
        # default so the fused BEV starts as a mix; the model learns how much history to
        # trust. What IS controlled is the warm start -- the current-frame path keeps its
        # pretrained weights and only `fuse` is new.

    def forward(self, bev, prev_bev, warp_grid):
        # grid_sample expects (B, H, W, 2) with (x, y) in [-1, 1]; align_corners=False
        # pairs with the cell-centre convention used to build the grid.
        if prev_bev.dim() == 4:
            warped = [F.grid_sample(prev_bev, warp_grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)]
        else:
            warped = [F.grid_sample(prev_bev[:, k], warp_grid[:, k], mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
                      for k in range(prev_bev.shape[1])]
        return self.fuse(torch.cat([bev] + warped, 1))


def prev_bev_from_batch(model, b, batched: bool):
    """The previous-frame BEV(s) the model expects, from a loader item or a collated batch.

    history=1 items carry prev_image (N, 3, H, W); history=K items carry (K, N, 3, H, W).
    Collated batches add the leading B. Returns (B, C, H, W) or (B, K, C, H, W) on cuda.
    """
    pi, idx, val = b["prev_image"], b["prev_bev_index"], b["prev_valid"]
    if not batched:
        pi, idx, val = pi[None], idx[None], val[None]
    if pi.dim() == 5:                                            # (B, N, 3, H, W): K = 1
        return model.bev_from(pi.cuda(non_blocking=True), idx.cuda(), val.cuda())
    B, K = pi.shape[:2]                                          # (B, K, N, 3, H, W)
    out = model.bev_from(pi.reshape(B * K, *pi.shape[2:]).cuda(non_blocking=True),
                         idx.reshape(B * K, -1).cuda(), val.reshape(B * K, -1).cuda())
    return out.reshape(B, K, *out.shape[1:])


def warp_from_batch(b, batched: bool):
    g = b["warp_grid"]
    return (g if batched else g[None]).cuda()
