"""Export the planning vision tower at a reduced patch count.

The shipped ``vlm_vision_plan/vision.onnx`` bakes ``pixel_values[12216, 1536]`` -- three
views over four timesteps at full size. Shrinking the side views changes that number,
so the tower has to be re-traced or it will not accept the input.
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import torch

from export_onnx._export_front_vision import VisionONNX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from export_onnx.export_vlm_layers_v2 import _install_side_shrink
    from export_onnx.scene_inputs import planning_inputs
    if args.side != 1.0:
        _install_side_shrink(args.side)
    ctx = planning_inputs("weights/Qwen-Drive-1.0-4B",
                          "weights/Qwen-Drive-1.0-4B/planner-sft",
                          "data/demo/planning_scenes.jsonl",
                          "data/demo", "data/demo/frames.parquet")
    vlm, inputs = ctx["vlm"], ctx["inputs"]
    px, grid = inputs["pixel_values"], inputs["image_grid_thw"]
    tag = f"s{str(args.side).replace('.','')}"
    out_dir = Path(args.out or f"outputs/onnx/vlm_vision_plan_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  tracing planning vision at pixel_values {tuple(px.shape)}", flush=True)
    module = VisionONNX(vlm.model.visual).eval()
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(module, (px, grid), str(out_dir / "vision.onnx"),
                          input_names=["pixel_values", "grid_thw"],
                          output_names=["last_hidden", "pre_merge", "merged"],
                          opset_version=20, do_constant_folding=True, dynamo=False)
    import onnx
    n = len(onnx.load(str(out_dir/"vision.onnx"), load_external_data=False).graph.node)
    print(f"  exported {n} nodes in {time.time()-t0:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
