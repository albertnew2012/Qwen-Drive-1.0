"""Loss functions for Qwen-Drive-1.0, as specified in the technical report.

    L_perc = L_det + L_occ + L_map

    L_det  = sum_{l=1..L} ( 2 * L_focal^(l) + 0.75 * L_l1^(l) )     (L = 6 layers)
    L_occ  = 100 * L_focal + L_geo + L_sem + L_lov
    L_map  = 100 * L_focal + L_lov
    L_plan = L_fm + 2e-4 * L_d1 + 2e-5 * L_d2

arXiv:2609.00111. The report gives these coefficients but no optimiser
hyperparameters; those live in ``config.py`` and are flagged as our choice.

``L_geo`` / ``L_sem`` are the geometric and semantic scaling losses from
MonoScene, and ``L_lov`` is the Lovasz-softmax surrogate for IoU - the standard
pairing for semantic occupancy, and the only reading of those symbols that fits
a 10-class voxel grid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:                                    # only needed for detection
    from scipy.optimize import linear_sum_assignment
except ImportError:                     # pragma: no cover
    linear_sum_assignment = None

__all__ = [
    "sigmoid_focal_loss", "normalize_bbox", "HungarianMatcher3D",
    "detection_loss", "occupancy_loss", "map_loss", "planning_loss",
]

# DETR3D box code weights: velocity terms are down-weighted, as in BEVFormer.
CODE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2)


# ─────────────────────────────────────────────────────────────── shared pieces

def sigmoid_focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2.0,
                       reduction: str = "sum"):
    """Focal loss on sigmoid logits (Lin et al.). ``targets`` is one-hot."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    return loss.sum() if reduction == "sum" else loss.mean() if reduction == "mean" else loss


def normalize_bbox(boxes):
    """GT ``[cx, cy, cz, w, l, h, rot, vx, vy]`` -> the head's 10-dim encoding.

    Exact inverse of ``heads.denormalize_bbox``:
    ``[cx, cy, log w, log l, cz, log h, sin rot, cos rot, vx, vy]``.
    """
    cx, cy, cz = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3]
    w, l, h = boxes[..., 3:4].log(), boxes[..., 4:5].log(), boxes[..., 5:6].log()
    rot = boxes[..., 6:7]
    if boxes.shape[-1] > 7:
        vx, vy = boxes[..., 7:8], boxes[..., 8:9]
    else:
        vx = vy = torch.zeros_like(cx)
    return torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy], dim=-1)


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels, ignore_index: int | None = None):
    """Lovasz-softmax: a convex surrogate for IoU. ``probs`` is [N, C]."""
    if probs.numel() == 0:
        return probs * 0.0
    losses = []
    for c in range(probs.shape[1]):
        if ignore_index is not None and c == ignore_index:
            continue
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def geo_scal_loss(probs, target, empty_label: int):
    """MonoScene geometric scaling: precision/recall/specificity of OCCUPANCY."""
    nonempty_p = 1.0 - probs[:, empty_label]
    empty_p = probs[:, empty_label]
    nonempty_t = (target != empty_label).float()
    eps = 1e-6
    inter = (nonempty_t * nonempty_p).sum()
    precision = inter / (nonempty_p.sum() + eps)
    recall = inter / (nonempty_t.sum() + eps)
    spec = ((1 - nonempty_t) * empty_p).sum() / ((1 - nonempty_t).sum() + eps)
    return -(torch.log(precision.clamp_min(eps))
             + torch.log(recall.clamp_min(eps))
             + torch.log(spec.clamp_min(eps)))


def sem_scal_loss(probs, target):
    """MonoScene semantic scaling: the same three quantities, per class."""
    eps, losses = 1e-6, []
    for c in range(probs.shape[1]):
        p = probs[:, c]
        t = (target == c).float()
        if t.sum() == 0:
            continue
        nom = (p * t).sum()
        li = 0.0
        if p.sum() > 0:
            li = li - torch.log((nom / (p.sum() + eps)).clamp_min(eps))
        li = li - torch.log((nom / (t.sum() + eps)).clamp_min(eps))
        if (1 - t).sum() > 0:
            spec = ((1 - p) * (1 - t)).sum() / ((1 - t).sum() + eps)
            li = li - torch.log(spec.clamp_min(eps))
        losses.append(li)
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


# ──────────────────────────────────────────────────────────────── 3D detection

class HungarianMatcher3D(nn.Module):
    """One-to-one assignment of the 900 queries to ground-truth boxes.

    This is what makes the model NMS-free: distinct queries are taught not to
    cover the same object. The report says only "the Hungarian algorithm matches
    object queries to ground-truth boxes" and gives no cost weights, so we use
    the DETR3D/BEVFormer values, which also match this repo's loss weights.
    """

    def __init__(self, cls_cost: float = 2.0, reg_cost: float = 0.25):
        super().__init__()
        self.cls_cost, self.reg_cost = cls_cost, reg_cost

    @torch.no_grad()
    def forward(self, cls_scores, bbox_preds, gt_labels, gt_boxes_norm):
        """``cls_scores`` [Q, C], ``bbox_preds`` [Q, 10]. Returns (query_idx, gt_idx)."""
        n_gt = gt_labels.numel()
        if n_gt == 0:
            empty = torch.zeros(0, dtype=torch.long, device=cls_scores.device)
            return empty, empty
        # focal-style classification cost, as in DETR3D
        alpha, gamma = 0.25, 2.0
        p = cls_scores.sigmoid()
        neg = (1 - alpha) * (p ** gamma) * (-(1 - p).clamp_min(1e-8).log())
        pos = alpha * ((1 - p) ** gamma) * (-p.clamp_min(1e-8).log())
        cls_cost = (pos[:, gt_labels] - neg[:, gt_labels]) * self.cls_cost
        # L1 over the 8 geometric dims only (velocity is excluded from matching)
        w = torch.tensor(CODE_WEIGHTS, device=cls_scores.device, dtype=bbox_preds.dtype)
        reg_cost = torch.cdist(bbox_preds[:, :8] * w[:8],
                               gt_boxes_norm[:, :8] * w[:8], p=1) * self.reg_cost
        cost = (cls_cost + reg_cost).nan_to_num(nan=1e4, posinf=1e4, neginf=-1e4)
        if linear_sum_assignment is None:
            raise RuntimeError("scipy is required for Hungarian matching")
        qi, gi = linear_sum_assignment(cost.float().cpu().numpy())
        return (torch.as_tensor(qi, dtype=torch.long, device=cls_scores.device),
                torch.as_tensor(gi, dtype=torch.long, device=cls_scores.device))


def detection_loss(all_cls_scores, all_bbox_preds, gt_boxes, gt_labels, matcher,
                   num_classes: int = 7, cls_weight: float = 2.0,
                   reg_weight: float = 0.75):
    """L_det = sum over the 6 decoder layers of (2 * focal + 0.75 * L1).

    Deep supervision: every decoder layer is matched and supervised separately,
    which is what teaches the iterative refinement to converge.
    """
    device = all_cls_scores.device
    n_layers, bs = all_cls_scores.shape[0], all_cls_scores.shape[1]
    w = torch.tensor(CODE_WEIGHTS, device=device, dtype=all_bbox_preds.dtype)
    total_cls = all_cls_scores.new_zeros(())
    total_reg = all_cls_scores.new_zeros(())
    n_matched_total = 0

    for lvl in range(n_layers):
        for b in range(bs):
            cls_s, box_p = all_cls_scores[lvl, b], all_bbox_preds[lvl, b]
            gtb, gtl = gt_boxes[b], gt_labels[b]
            gtn = normalize_bbox(gtb) if gtb.numel() else gtb.new_zeros((0, 10))
            qi, gi = matcher(cls_s, box_p, gtl, gtn)

            target = torch.zeros_like(cls_s)
            if qi.numel():
                target[qi, gtl[gi]] = 1.0
            n_pos = max(qi.numel(), 1)
            total_cls = total_cls + sigmoid_focal_loss(cls_s, target) / n_pos

            if qi.numel():
                pred = box_p[qi] * w
                tgt = gtn[gi] * w
                valid = torch.isfinite(tgt).all(-1)
                if valid.any():
                    total_reg = total_reg + F.l1_loss(
                        pred[valid], tgt[valid], reduction="sum") / n_pos
                n_matched_total += qi.numel()

    denom = n_layers * bs
    return {
        "det_cls": cls_weight * total_cls / denom,
        "det_reg": reg_weight * total_reg / denom,
        "det_matched": torch.tensor(float(n_matched_total) / denom, device=device),
    }


# ──────────────────────────────────────────────────────────────────── occupancy

def occupancy_loss(occ_pred, gt_occ, empty_label: int, focal_weight: float = 100.0,
                   ignore_index: int = 255, max_points: int = 400_000):
    """L_occ = 100 * focal + geo + sem + lovasz.

    The grid is 200x200x16 = 640k voxels and overwhelmingly ``empty``, so the
    scaling losses are subsampled - full Lovasz sorting over 640k x 10 does not
    fit alongside the backward graph.
    """
    logits = occ_pred.reshape(-1, occ_pred.shape[-1]).float()
    target = gt_occ.reshape(-1)
    keep = target != ignore_index
    logits, target = logits[keep], target[keep]

    onehot = F.one_hot(target.clamp_min(0), logits.shape[-1]).to(logits.dtype)
    focal = sigmoid_focal_loss(logits, onehot) / max(target.numel(), 1)

    if target.numel() > max_points:                 # stratified-ish subsample
        idx = torch.randperm(target.numel(), device=target.device)[:max_points]
        logits_s, target_s = logits[idx], target[idx]
    else:
        logits_s, target_s = logits, target
    probs = logits_s.softmax(-1)
    return {
        "occ_focal": focal_weight * focal,
        "occ_geo": geo_scal_loss(probs, target_s, empty_label),
        "occ_sem": sem_scal_loss(probs, target_s),
        "occ_lov": lovasz_softmax_flat(probs, target_s),
    }


# ────────────────────────────────────────────────────────────── map segmentation

def map_loss(seg_preds, gt_map, focal_weight: float = 100.0, ignore_index: int = 255,
             max_points: int = 400_000):
    """L_map = 100 * focal + lovasz.  ``seg_preds`` is [B, C, H, W]."""
    bs, c = seg_preds.shape[0], seg_preds.shape[1]
    logits = seg_preds.permute(0, 2, 3, 1).reshape(-1, c).float()
    target = gt_map.reshape(-1)
    keep = target != ignore_index
    logits, target = logits[keep], target[keep]

    onehot = F.one_hot(target.clamp_min(0), c).to(logits.dtype)
    focal = sigmoid_focal_loss(logits, onehot) / max(target.numel(), 1)

    if target.numel() > max_points:
        idx = torch.randperm(target.numel(), device=target.device)[:max_points]
        logits, target = logits[idx], target[idx]
    return {
        "map_focal": focal_weight * focal,
        "map_lov": lovasz_softmax_flat(logits.softmax(-1), target),
    }


# ───────────────────────────────────────────────────────────────────── planning

def planning_loss(pred_x1, target, t=None, d1_weight: float = 2e-4,
                  d2_weight: float = 2e-5, valid_mask=None):
    """L_plan = L_fm + 2e-4 * L_d1 + 2e-5 * L_d2.

    The expert is trained with a CLEAN-ENDPOINT (x1) flow-matching objective -
    ``PlanningExpert.sample`` reconstructs velocity as ``(x1_hat - x_t)/(1-t)``,
    so the network's own output is x1 and the flow-matching term is a regression
    onto the ground-truth waypoints.

    The two extra terms penalise the first and second differences along the
    trajectory: jerk/acceleration smoothness. Their tiny weights say they are
    regularisers, not objectives.
    """
    if valid_mask is not None:
        m = valid_mask.to(pred_x1.dtype).unsqueeze(-1)
        fm = ((pred_x1 - target).square() * m).sum() / m.sum().clamp_min(1.0) / pred_x1.shape[-1]
    else:
        fm = F.mse_loss(pred_x1, target)

    d1_p = pred_x1[..., 1:, :] - pred_x1[..., :-1, :]
    d1_t = target[..., 1:, :] - target[..., :-1, :]
    d2_p = d1_p[..., 1:, :] - d1_p[..., :-1, :]
    d2_t = d1_t[..., 1:, :] - d1_t[..., :-1, :]
    return {
        "plan_fm": fm,
        "plan_d1": d1_weight * F.mse_loss(d1_p, d1_t),
        "plan_d2": d2_weight * F.mse_loss(d2_p, d2_t),
    }
