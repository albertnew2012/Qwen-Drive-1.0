"""Hungarian-matched detection loss against ground truth -- the E1 objective.

WHY THIS REPLACES PER-QUERY DISTILLATION
``HANDOVER.md`` chose per-query distillation to avoid matching and ground truth: query i
of the student trains against query i of the teacher. Elegant, and it is the direct cause
of the detection failure. Measured on the 78k-step student:

    55.9 detections/frame against the teacher's ~22
      52% within 2 m of a real object   -> duplicates, nothing enforces one query per object
      48% isolated                      -> hallucinations, the soft teacher target never
                                           says "there is nothing here"
    F1 34.5% against the teacher's 75.1%

Three consequences, all fixed here:

1. **No one-to-one assignment.** DETR-family models get duplicate suppression for free
   from Hungarian matching. Inference NMS recovers some of it (+4.8 F1, measured) but
   training one-to-one is the structural fix.
2. **No hard background.** The target was the teacher's *soft* sigmoid, so background
   queries were trained toward small-but-nonzero scores. Focal loss over hard labels
   supervises background explicitly.
3. **A ceiling at the teacher's own 75.1% F1.** Every teacher mistake was a target.
   Ground truth has no such ceiling.

BOX ENCODING, VERIFIED NOT ASSUMED
The teacher's 10-dim box is ``[x, y, z, log w, log l, log h, sin yaw, cos yaw, vx, vy]``.
Checked over 8,922 asserted boxes: ``exp(dim3)`` has median 1.91 m against ground-truth
width 1.90 m, ``exp(dim5)`` 1.71 m against height 1.69 m, and ``hypot(dim6, dim7)`` has
median 0.984 -- i.e. a unit vector, to bfloat16. Ground truth is ``[x,y,z,w,l,h,yaw,0,0]``
in metres, so it converts in directly.

Velocity (dims 8, 9) is **not** supervised from ground truth: ``nusc_gt_boxes.py`` writes
zeros there. Supervising it from those zeros would actively teach the student that
nothing moves.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

__all__ = ["gt_to_teacher_encoding", "hungarian_detection_loss"]

# dims 0..7 come from ground truth; 8,9 (velocity) are absent from GT and stay unsupervised
GT_SUPERVISED_DIMS = 8


def gt_to_teacher_encoding(boxes: np.ndarray) -> np.ndarray:
    """(N,9) [x,y,z,w,l,h,yaw,_,_] metres -> (N,10) the teacher's parameterisation."""
    if len(boxes) == 0:
        return np.zeros((0, 10), np.float32)
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    w, l, h = np.maximum(boxes[:, 3], 1e-3), np.maximum(boxes[:, 4], 1e-3), np.maximum(boxes[:, 5], 1e-3)
    yaw = boxes[:, 6]
    return np.stack([x, y, z, np.log(w), np.log(l), np.log(h),
                     np.sin(yaw), np.cos(yaw),
                     np.zeros_like(x), np.zeros_like(x)], -1).astype(np.float32)


def hungarian_detection_loss(cls_logits, box_pred, gt_boxes, gt_labels, box_std,
                             focal_alpha=0.25, focal_gamma=2.0,
                             cost_class=2.0, cost_box=5.0, w_box=5.0,
                             bev_extent=102.4):
    """One-to-one matched detection loss.

    cls_logits (B, Q, C) · box_pred (B, Q, 10) in teacher units · gt_boxes list of (N,10)
    tensors already in teacher encoding · gt_labels list of (N,) · box_std (10,).

    Sigmoid focal loss over all queries -- the multi-label formulation DETR3D and
    BEVFormer use, not softmax-with-background. Unmatched queries are all-zeros targets,
    which is the explicit "nothing here" signal the distillation objective never gave.
    """
    B, Q, C = cls_logits.shape
    device = cls_logits.device
    targets = torch.zeros_like(cls_logits)
    box_terms, n_matched = [], 0

    with torch.no_grad():
        prob = cls_logits.sigmoid()

    for b in range(B):
        g_box, g_lab = gt_boxes[b], gt_labels[b]
        if g_box.numel() == 0:
            continue                      # no objects: every query is background
        with torch.no_grad():
            # classification cost: how unwilling this query is to call it that class
            c_cls = -prob[b][:, g_lab]                                   # (Q, N)
            # box cost on the centre only -- the metric is centre distance at 2 m, so
            # matching on centres matches on what is scored.
            #
            # NORMALISED by the BEV extent, and that is not cosmetic. In raw metres this
            # term is 5 x tens of metres while the class term is at most 2, so matching
            # became purely geometric: confident queries were never selected, their
            # confidence was then charged as a false positive, and the only way down was
            # to suppress everything. Measured with raw metres, the loss was INVERTED --
            # the F1 34.5% model scored l_cls 1.293 and the F1 1.1% model scored 0.537.
            # DETR normalises box coordinates to [0,1] for exactly this reason.
            c_box = torch.cdist(box_pred[b][:, :2] / bev_extent,
                                g_box[:, :2] / bev_extent, p=1)          # (Q, N)
            cost = (cost_class * c_cls + cost_box * c_box).cpu().numpy()
        qi, gi = linear_sum_assignment(cost)
        qi = torch.as_tensor(qi, device=device, dtype=torch.long)
        gi = torch.as_tensor(gi, device=device, dtype=torch.long)
        targets[b, qi, g_lab[gi]] = 1.0
        # standardised, or x and y (std 24.6 and 16.9 m against 0.34-2.3 for the rest)
        # would be 94% of the gradient -- the same trap box_stats fixes elsewhere
        d = (box_pred[b][qi, :GT_SUPERVISED_DIMS] - g_box[gi, :GT_SUPERVISED_DIMS])
        # mean over dims, not sum: summing 8 standardised dims puts the box term at ~40
        # against a focal classification term of ~0.7, and box would own the gradient.
        box_terms.append((d.abs() / box_std[:GT_SUPERVISED_DIMS]).mean(-1).sum())
        n_matched += len(qi)

    p = cls_logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(cls_logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = focal_alpha * targets + (1 - focal_alpha) * (1 - targets)
    l_cls = (alpha_t * (1 - p_t) ** focal_gamma * ce).sum() / max(n_matched, 1)

    l_box = (torch.stack(box_terms).sum() / max(n_matched, 1)
             if box_terms else cls_logits.sum() * 0.0)
    return l_cls, w_box * l_box, n_matched
