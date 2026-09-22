"""Scene inputs for a reduced camera rig.

``scene_inputs.perception_inputs`` always builds the full six-camera prompt. A
front-camera export needs the same structures with the rig subset applied
consistently -- prompt content, ``cam_order`` and the calibration arrays -- because
the head's frozen geometry is derived from exactly those.

Perception at one camera is 468 tokens against 2744, and the voxel volume it writes
falls from 48.5% occupied to 8.7%.
"""
from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["perception_inputs_cams"]


def perception_inputs_cams(vlm_dir: str, perception_dir: str, frames: str, frame: str,
                           cams: str, dtype=torch.float32):
    from transformers import AutoTokenizer
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
    from local.prune.eval_camera_subset_v1 import subset_frame

    holder = QwenDriveForPlanning.from_pretrained(
        vlm_dir, dtype=dtype, attn_implementation="eager")
    vlm = holder.vlm.eval()
    head = QwenDrivePerception.from_pretrained(perception_dir, dtype=dtype).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(vlm_dir))
    head.attach(vlm, proc)
    fr = PerceptionFrame(Path(frames) / frame)
    # 'front' rather than a name, because the demo set mixes nuScenes (CAM_FRONT)
    # and nuPlan (CAM_F0) conventions; cam_order starts with the forward camera in both
    wanted = ([fr.cam_order[0]] if cams == "front"
              else [c for c in cams.split(",") if c])
    fr = subset_frame(fr, wanted)
    inputs, metas = proc(fr, device="cpu")
    pos = holder._rope_positions(inputs["input_ids"], inputs["image_grid_thw"])
    return dict(holder=holder, vlm=vlm, head=head, inputs=inputs, metas=metas,
                position_ids=pos, token=fr.token, cam_order=fr.cam_order)
