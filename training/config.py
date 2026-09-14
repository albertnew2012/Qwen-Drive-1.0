"""Training hyperparameters.

The technical report (arXiv:2609.00111) specifies the STAGES and every LOSS
COEFFICIENT, but gives no optimiser table. Everything marked (*) below is
therefore our choice, taken from BEVFormer/DETR3D, on which this head is built.
One anchor is real: the report states the BEV head trains at "20x the learning
rate of the VLM", and 20 x 1e-5 = 2e-4 is exactly the BEVFormer default, so the
two are consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerceptionTrainConfig:
    """Stage 1: BEV perception head only, VLM frozen."""
    # optimiser (*)
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    grad_clip_norm: float = 35.0            # BEVFormer's value (*)
    # schedule (*)
    epochs: int = 24
    warmup_iters: int = 500
    warmup_ratio: float = 1.0 / 3.0
    lr_min_ratio: float = 1e-3              # cosine floor
    # loss coefficients - THESE ARE FROM THE PAPER, do not tune casually
    det_cls_weight: float = 2.0             # L_det = sum_l (2*focal + 0.75*L1)
    det_reg_weight: float = 0.75
    occ_focal_weight: float = 100.0         # L_occ = 100*focal + geo + sem + lov
    map_focal_weight: float = 100.0         # L_map = 100*focal + lov
    # matcher (*) - DETR3D defaults
    match_cls_cost: float = 2.0
    match_reg_cost: float = 0.25
    # runtime
    batch_size: int = 1                     # 19 GiB peak per sample on one 3090
    amp_dtype: str = "bfloat16"
    occ_max_points: int = 400_000
    map_max_points: int = 400_000
    seed: int = 0


@dataclass
class JointTrainConfig(PerceptionTrainConfig):
    """Stage 2: perception + VQA, everything trainable.

    The report: "the BEV perception head uses a learning rate 20x that of the
    VLM". Full fine-tuning of a 4.5 B VLM needs far more than one 24 GB card, so
    ``train_joint.py`` defaults to LoRA on the VLM; ``--full`` attempts the real
    thing and will require model/optimiser sharding.
    """
    vlm_lr: float = 1e-5
    head_lr_multiplier: float = 20.0        # from the paper
    lora_rank: int = 16
    lora_alpha: int = 32
    vqa_loss_weight: float = 1.0            # L_ntp
    perception_sample_ratio: float = 0.5


@dataclass
class PlannerTrainConfig:
    """Stage 3: planning expert only, VLM frozen."""
    lr: float = 1e-4                        # (*)
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    epochs: int = 50
    warmup_iters: int = 100
    # from the paper: L_plan = L_fm + 2e-4 L_d1 + 2e-5 L_d2
    d1_weight: float = 2e-4
    d2_weight: float = 2e-5
    num_waypoints: int = 50
    batch_size: int = 1
    amp_dtype: str = "bfloat16"
    seed: int = 0
