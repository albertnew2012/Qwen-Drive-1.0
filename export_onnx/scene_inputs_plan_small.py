"""Planning scene inputs with the side views shrunk.

Planning is 1,685 ms in ONNX -- 539 of vision over 12 images and 886 of decoder over
3,385 tokens -- and it is the half that cannot be pruned: the Planning Expert
cross-attends to all eight full-attention caches, and dropping even twelve
linear-attention layers doubles ADE. Tokens are the only lever, and they go as the
square of each frame's ``target_size``.

``CAMERA_VIEWS`` is FRONT, FRONT LEFT, FRONT RIGHT and ``DrivingScene`` refuses a
scene missing any of them, so the side views are shrunk rather than dropped.
"""
from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["planning_inputs_small"]


def planning_inputs_small(vlm_dir: str, planner_dir: str, scenes: str, image_root: str,
                          image_archive: str, side: float = 0.5, dtype=torch.float32):
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.images import ImageArchive
    from local.prune.optimise_planning_v1 import shrink_side_views

    holder = QwenDriveForPlanning.from_pretrained(
        vlm_dir, planner=planner_dir, dtype=dtype, attn_implementation="eager")
    vlm = holder.vlm.eval()
    archive = ImageArchive.open(image_archive) if image_archive else None
    sample = next(iter(read_scene_file(
        scenes, image_archive=archive,
        num_history_points=holder.config.num_history_points, limit=1)))
    scene = shrink_side_views(sample.scene, side)
    builder = holder._prompt_builder if hasattr(holder, "_prompt_builder") else None
    prompt = holder.build_planning_prompt(scene) if hasattr(
        holder, "build_planning_prompt") else None
    return dict(holder=holder, vlm=vlm, sample=sample, scene=scene,
                side=side, prompt=prompt, builder=builder)
