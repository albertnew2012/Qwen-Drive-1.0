"""Dense CenterPoint-style BEV detection head, and its loss.

WHY THIS REPLACES THE QUERY DECODER
Six experiments with the DETR-style head (900 queries + Hungarian matching) produced
0.0, 0.0, 1.1, 2.1 and 1.5 percent F1. The reasons were real and stacked:

    * DETR needs 50-500 epochs to converge; the budget here is ~12.
    * Box supervision only ever reached the ~22 matched or teacher-asserted queries, so
      878 of 900 had unconstrained boxes. Whenever classification fired on one of those,
      the box was garbage -- measured at 96-97% hallucinated in E3 and E4.
    * Matching is unstable without per-query spatial anchoring, so a query's target
      object changes between frames and it learns the average of several.

Meanwhile a LINEAR probe on the student's own BEV feature separates object cells from
empty cells at ROC-AUC 0.844 (AP 0.629 against a 0.249 chance rate). The features carry
the objects; the head was the bottleneck.

A dense head removes all three failure modes at once: every cell is supervised every
step, each object owns exactly one cell so there is no assignment problem, and duplicate
suppression is a 3x3 max-pool. It also reads the BEV at its native 0.512 m/cell instead
of the 2.56 m/cell the decoder's 40x40 pooling left it with -- against a measured centre
error of 0.83 m, that pooling was itself a floor.

This is the CenterNet/CenterPoint construction: a per-class Gaussian heatmap plus dense
regression, trained with the Gaussian-focal loss, decoded by top-k over local maxima.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CenterHead", "center_targets", "center_loss"]

REG_DIMS = 8          # dx, dy (sub-cell offset), z, log w, log l, log h, sin yaw, cos yaw
REG_DIMS_V = 10       # + vx, vy (ego frame, m/s): cfg.velocity, for the official NDS


class CenterHead(nn.Module):
    """(B, C_in, H, W) -> per-class heatmap and dense regression at full BEV resolution."""

    def __init__(self, in_ch: int, num_classes: int, hidden: int = 64, blocks: int = 2,
                 reg_dims: int = REG_DIMS, upsample: int = 1, dilation: int = 1):
        super().__init__()
        self.upsample = int(upsample)
        # dilation == upsample makes a 2x-grid head an EXACT re-expression of the 1x head it is
        # warm-started from: a 3x3 conv with dilation 2 on a nearest-upsampled map equals the
        # upsampled 3x3 conv of the original, so the fine-tune starts from the trained function
        # and only has to learn sub-cell structure through the transposed conv.
        self.dilation = int(dilation)
        # blocks=2, hidden=64 is the E5c head (1x1 then one 3x3). At 120k it separated
        # objects from background weakly -- recall 55.7% needed thr 0.05, precision 10%
        # -- so E6 widens/deepens it. A 3x3 at 200x200x128 is ~1.5 GFLOP, ~0.1 ms.
        if self.upsample > 1:
            # learned 2x (or 4x) upsampling instead of the 1x1 entry conv: the head then
            # predicts on a grid finer than the BEV, which is what separates two
            # pedestrians 0.6 m apart or localises a 0.3 m cone (see StudentConfig).
            layers = [nn.ConvTranspose2d(in_ch, hidden, self.upsample, stride=self.upsample, bias=False),
                      nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        else:
            layers = [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden),
                      nn.ReLU(inplace=True)]
        for _ in range(max(1, blocks - 1)):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=self.dilation, dilation=self.dilation, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        self.shared = nn.Sequential(*layers)
        self.hm = nn.Conv2d(hidden, num_classes, 1)
        self.reg = nn.Conv2d(hidden, reg_dims, 1)
        # Same prior-probability trick focal loss needs everywhere: start every cell at
        # p=0.01 so the ~40,000 background cells do not swamp the handful of positives.
        nn.init.constant_(self.hm.bias, -float(np.log((1 - 0.01) / 0.01)))

    def forward(self, bev):
        f = self.shared(bev)
        return self.hm(f), self.reg(f)


def center_targets(gt_boxes, gt_labels, cfg, device):
    """Gaussian heatmap + regression targets for one frame.

    gt_boxes is (N, 9) [x, y, z, w, l, h, yaw, _, _] in metres, ego frame.
    """
    n, C = cfg.bev_size * int(getattr(cfg, "head_upsample", 1)), cfg.num_classes   # head grid, not BEV grid
    nd = REG_DIMS_V if getattr(cfg, "velocity", False) else REG_DIMS
    hm = torch.zeros(C, n, n, device=device)
    reg = torch.zeros(nd, n, n, device=device)
    mask = torch.zeros(n, n, device=device)
    if len(gt_boxes) == 0:
        return hm, reg, mask
    x0, y0, _, x1, y1, _ = cfg.pc_range
    cell_x, cell_y = (x1 - x0) / n, (y1 - y0) / n
    fx = (gt_boxes[:, 0] - x0) / cell_x
    fy = (gt_boxes[:, 1] - y0) / cell_y
    for k in range(len(gt_boxes)):
        cx, cy = float(fx[k]), float(fy[k])
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < n and 0 <= iy < n):
            continue
        c = int(gt_labels[k])
        # radius from the object footprint, the CenterNet heuristic, floored at 1 cell
        w, l = float(gt_boxes[k, 3]), float(gt_boxes[k, 4])
        # Floor from cfg: a pedestrian (0.7 m) rounds to ONE positive cell in 40,000 at
        # 0.512 m/cell, and pedestrian recall was the worst class. CenterPoint floors at 2.
        r = max(int(getattr(cfg, 'center_min_radius', 1)),
                int(round(0.5 * max(w / cell_x, l / cell_y))))
        ys, xs = torch.meshgrid(torch.arange(max(0, iy - r), min(n, iy + r + 1), device=device),
                                torch.arange(max(0, ix - r), min(n, ix + r + 1), device=device),
                                indexing="ij")
        # Centred on the INTEGER cell, not the fractional (cx, cy). exp(0) == 1.0 exactly,
        # so the peak cell is a positive under `hm_tgt.eq(1.0)`; the sub-cell part goes
        # into reg[0:2] below. Centring on (cx, cy) instead meant the peak was 1.0 only
        # when an object sat exactly on a cell centre -- i.e. almost never -- so the
        # loss saw NO positives, pushed every cell down, hit exactly 0.0000, and the
        # model produced zero detections at every threshold after 36k steps.
        g = torch.exp(-(((xs.float() - ix) ** 2 + (ys.float() - iy) ** 2)
                        / (2 * (r / 3 + 1e-6) ** 2)))
        hm[c, ys, xs] = torch.maximum(hm[c, ys, xs], g)
        reg[0, iy, ix] = cx - ix                       # sub-cell offset, recovers the
        reg[1, iy, ix] = cy - iy                       # 0.512 m quantisation
        reg[2, iy, ix] = float(gt_boxes[k, 2])
        reg[3, iy, ix] = float(np.log(max(w, 1e-3)))
        reg[4, iy, ix] = float(np.log(max(l, 1e-3)))
        reg[5, iy, ix] = float(np.log(max(float(gt_boxes[k, 5]), 1e-3)))
        reg[6, iy, ix] = float(np.sin(gt_boxes[k, 6]))
        reg[7, iy, ix] = float(np.cos(gt_boxes[k, 6]))
        if nd > REG_DIMS:                              # gt_boxes10.npz carries vx, vy
            reg[8, iy, ix] = float(gt_boxes[k, 7])
            reg[9, iy, ix] = float(gt_boxes[k, 8])
        mask[iy, ix] = 1.0
    return hm, reg, mask


CLASS_WEIGHT = None   # optional (C,) per-class heatmap loss weight, set by the trainer's --hm-class-weight


def center_loss(hm_pred, reg_pred, hm_tgt, reg_tgt, mask, reg_weight=1.0, class_weight=None):
    """Gaussian-focal on the heatmap, L1 on regression at positive cells only.

    ``class_weight`` (or the module-level CLASS_WEIGHT) scales each class's heatmap loss: the
    object-count-weighted F1 is dominated by cars, and the student trails the teacher only on
    pedestrians, motorcycles, bicycles and cones, so a fine-tune can lean on those classes
    without touching the rest of the recipe.
    """
    p = torch.clamp(hm_pred.sigmoid(), 1e-4, 1 - 1e-4)
    pos = hm_tgt.eq(1.0).float()
    neg = 1.0 - pos
    # negatives are down-weighted by (1 - gaussian)^4, so cells near a true centre are
    # not punished for being warm -- the CenterNet formulation
    neg_w = torch.pow(1.0 - hm_tgt, 4)
    cw = class_weight if class_weight is not None else CLASS_WEIGHT
    w = cw.to(hm_pred.device, p.dtype).view(1, -1, 1, 1) if cw is not None else 1.0
    l_pos = -(torch.log(p) * torch.pow(1 - p, 2) * pos * w).sum()
    l_neg = -(torch.log(1 - p) * torch.pow(p, 2) * neg_w * neg * w).sum()
    n_pos = pos.sum().clamp_min(1.0)
    l_hm = (l_pos + l_neg) / n_pos

    m = mask.unsqueeze(1)                                   # (B, 1, H, W)
    l_reg = ((reg_pred - reg_tgt).abs() * m).sum() / (m.sum() * hm_pred.shape[1]).clamp_min(1.0)
    return l_hm, reg_weight * l_reg


def decode_centers(hm, reg, cfg, k=900):
    """Top-k local maxima -> the (cls, box) pair the rest of the pipeline expects.

    The 3x3 max-pool is the duplicate suppression that the query head never had; it is a
    single exportable op rather than host-side NMS.
    """
    B, C, H, W = hm.shape
    p = hm.sigmoid()
    keep = (F.max_pool2d(p, 3, stride=1, padding=1) == p).float()
    p = p * keep
    flat = p.reshape(B, C, -1)
    score, idx = flat.reshape(B, -1).topk(k, dim=1)         # over class*cell
    cls_i = idx // (H * W)
    cell = idx % (H * W)
    iy, ix = cell // W, cell % W
    r = reg.reshape(B, reg.shape[1], -1)
    g = cell.unsqueeze(1).expand(-1, reg.shape[1], -1)
    rv = torch.gather(r, 2, g)                              # (B, REG, k)
    x0, y0, _, x1, y1, _ = cfg.pc_range
    cx = (ix.float() + rv[:, 0]) * (x1 - x0) / W + x0
    cy = (iy.float() + rv[:, 1]) * (y1 - y0) / H + y0
    vx = rv[:, 8] if rv.shape[1] > 8 else torch.zeros_like(cx)
    vy = rv[:, 9] if rv.shape[1] > 9 else torch.zeros_like(cx)
    box = torch.stack([cx, cy, rv[:, 2], rv[:, 3], rv[:, 4], rv[:, 5],
                       rv[:, 6], rv[:, 7], vx, vy], -1)
    # one-hot-ish logits so downstream argmax/max keep working unchanged
    logits = torch.full((B, k, C), -10.0, device=hm.device, dtype=box.dtype)
    logits.scatter_(2, cls_i.unsqueeze(-1), torch.logit(score.clamp(1e-4, 1 - 1e-4)).unsqueeze(-1))
    return logits, box
