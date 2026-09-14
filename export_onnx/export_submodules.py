"""Export and verify the perception head PIECE BY PIECE.

Two purposes. It produces deployable graphs for the parts that matter, and it
bisects the full-head export: if a piece verifies here but the whole head does
not, the fault is in the composition, not the piece.

    python export_onnx/export_submodules.py
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive_perception import QwenDrivePerception
from training.differentiable import enable_training_ops


def try_export(name, module, example, out_dir, opset, verify=True):
    out = out_dir / f"{name}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval()
    with torch.no_grad():
        ref = module(*example)
    refs = (ref,) if torch.is_tensor(ref) else tuple(ref)
    names = [f"in_{i}" for i in range(len(example))]
    onames = [f"out_{i}" for i in range(len(refs))]
    t0 = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(module, example, str(out), input_names=names,
                              output_names=onames, opset_version=opset,
                              do_constant_folding=True, dynamo=False)
    except Exception as exc:
        print(f"  {name:22s} EXPORT FAILED: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:90]}")
        return None
    dt = time.time() - t0
    size = out.stat().st_size / 2**20
    if not verify:
        print(f"  {name:22s} exported {dt:5.1f}s  {size:7.1f} MiB  (not verified)")
        return 0.0
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = sess.run(None, {n: t.cpu().numpy() for n, t in zip(names, example)})
        worst = 0.0
        for a, b in zip(got, refs):
            b = b.cpu().numpy()
            worst = max(worst, float(np.abs(a - b).max()) /
                        max(float(np.abs(b).max()), 1e-9))
        flag = "PASS" if worst < 1e-3 else "FAIL"
        print(f"  {name:22s} exported {dt:5.1f}s  {size:7.1f} MiB   "
              f"rel diff {worst:.2e}  {flag}")
        return worst
    except Exception as exc:
        print(f"  {name:22s} VERIFY FAILED: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:80]}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--out", default="outputs/onnx/submodules")
    ap.add_argument("--opset", type=int, default=20)
    args = ap.parse_args()

    enable_training_ops()
    head = QwenDrivePerception.from_pretrained(args.model, dtype=torch.float32).eval()
    bev = head.bev_modeling
    out_dir = Path(args.out)
    print(f"exporting submodules at opset {args.opset} -> {out_dir}\n")

    n_cam = 6
    vit = torch.randn(n_cam, 1024, 32, 56)
    llm = torch.randn(n_cam, 2560, 16, 28)

    # 1. the two feature adaptors
    try_export("vit_neck", bev.vit_neck, (vit,), out_dir, args.opset)
    try_export("adaptor", bev.adaptor, (llm,), out_dir, args.opset)

    # 2. DepthNet - the heart of the push lift
    with torch.no_grad():
        neck_out = bev.vit_neck(vit)
    feat = neck_out[0] if isinstance(neck_out, (list, tuple)) else neck_out
    try_export("depth_net", bev.depth_net, (feat,), out_dir, args.opset)

    # 3. the occupancy refiner (3D U-Net) - where the 5D GridSample lives
    if hasattr(bev, "occ_refiner"):
        vol = torch.randn(1, bev.config.occ_dim if hasattr(bev, "config") else 32,
                          16, 50, 50)
        try_export("occ_refiner", bev.occ_refiner, (vol,), out_dir, args.opset)

    # 4. the map segmentation decoder
    seg = getattr(getattr(bev, "head", None), "seg_head", None) or \
          getattr(bev, "map_seg", None)
    if seg is not None:
        try_export("map_seg", seg, (torch.randn(1, 256, 200, 200),),
                   out_dir, args.opset)

    print("\nA piece that passes here but fails inside the whole-head export tells "
          "you the fault is in the composition, not the piece.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
