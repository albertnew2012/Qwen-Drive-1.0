"""Export the vision tower for a one-camera batch.

The shipped ``vlm_vision/vision.onnx`` has ``pixel_values[10752, 1536]`` baked in --
six cameras' patches. A front-camera run supplies 1792, so the graph has to be
re-traced at that shape. Nothing else about it changes.
"""
from __future__ import annotations

import os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch import nn


class VisionONNX(nn.Module):
    """Vision tower plus the pre-merge tap the BEV head reads."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values, grid_thw):
        tap = {}
        handle = self.visual.merger.register_forward_hook(
            lambda m, a, o=None: tap.__setitem__("p", a[0]))
        try:
            out = self.visual(pixel_values, grid_thw=grid_thw)
        finally:
            handle.remove()
        last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        pre = self.visual.merger.norm(tap["p"])
        merged = self.visual.merger(tap["p"])
        merged = merged[0] if isinstance(merged, tuple) else merged
        return last, pre, merged


def main() -> int:
    os.chdir(_ROOT)
    from export_onnx.scene_inputs_front import perception_inputs_cams
    ctx = perception_inputs_cams("weights/Qwen-Drive-1.0-4B",
                                 "weights/Qwen-Drive-1.0-4B/perception",
                                 "data/demo/perception",
                                 "90162f90eceb4ada9e595bc1adb71b5f", "front")
    vlm, inputs = ctx["vlm"], ctx["inputs"]
    px = inputs["pixel_values"]
    grid = inputs["image_grid_thw"]
    out_dir = _ROOT / "outputs" / "onnx" / "vlm_vision_front"
    out_dir.mkdir(parents=True, exist_ok=True)
    module = VisionONNX(vlm.model.visual).eval()
    print(f"  tracing vision tower at pixel_values {tuple(px.shape)}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(module, (px, grid), str(out_dir / "vision.onnx"),
                          input_names=["pixel_values", "grid_thw"],
                          output_names=["last_hidden", "pre_merge", "merged"],
                          opset_version=20, do_constant_folding=True, dynamo=False)
    import onnx
    n = len(onnx.load(str(out_dir / "vision.onnx"),
                      load_external_data=False).graph.node)
    print(f"  exported {n} nodes in {time.time()-t0:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
