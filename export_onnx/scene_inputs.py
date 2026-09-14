"""Build model inputs for either task, so the exporters share one code path.

The two tasks have genuinely different shapes, and every exported graph is
frozen to the shape it was traced at:

    perception   6 cameras, one timestamp, 896x512   ->  2744 tokens
    planning     3 cameras x 4 timestamps, mixed     ->  3385 tokens

so each task needs its own vision graph and its own set of layer graphs. That is
not a limitation of this exporter - it follows from the Gated-DeltaNet chunk loop
unrolling at trace time.
"""
from __future__ import annotations

from pathlib import Path

import torch


def perception_inputs(vlm_dir: str, perception_dir: str, frames: str, frame: str,
                      dtype=torch.float32):
    from transformers import AutoTokenizer
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor

    holder = QwenDriveForPlanning.from_pretrained(
        vlm_dir, dtype=dtype, attn_implementation="eager")
    vlm = holder.vlm.eval()
    head = QwenDrivePerception.from_pretrained(perception_dir, dtype=dtype).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(vlm_dir))
    head.attach(vlm, proc)
    fr = PerceptionFrame(Path(frames) / frame)
    inputs, metas = proc(fr, device="cpu")
    pos = holder._rope_positions(inputs["input_ids"], inputs["image_grid_thw"])
    return dict(holder=holder, vlm=vlm, head=head, inputs=inputs, metas=metas,
                position_ids=pos, token=fr.token)


def planning_inputs(vlm_dir: str, planner_dir: str, scenes: str, image_root: str,
                    image_archive: str | None, index: int = 0, dtype=torch.float32):
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.images import ImageArchive

    model = QwenDriveForPlanning.from_pretrained(
        vlm_dir, planner=planner_dir, dtype=dtype,
        attn_implementation="eager").eval()
    archive = ImageArchive.open(image_archive) if image_archive else None
    samples = list(read_scene_file(Path(scenes), image_root=image_root,
                                   image_archive=archive))
    sample = samples[index]
    inputs = model.processor(sample.scene, with_reasoning=False, device="cpu")
    pos = model._rope_positions(inputs["input_ids"], inputs["image_grid_thw"])
    return dict(holder=model, vlm=model.vlm, head=None, inputs=inputs,
                position_ids=pos, sample=sample, token=sample.token)
