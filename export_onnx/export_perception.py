"""Export the BEV perception head to ONNX, and verify it numerically.

The released package cannot be exported as shipped: it calls two custom CUDA
kernels that ONNX has no way to represent. ``training.differentiable`` already
routes both to pure-PyTorch equivalents, which is exactly what a tracer needs,
so the same switch that enables training also enables export.

The VLM is NOT exported here. Its hybrid stack is 24 Gated-DeltaNet recurrent
layers plus 8 softmax layers; that belongs in a dedicated LLM export path. This
covers the 125 M perception head, whose inputs are the VLM's two feature taps.

    python export_onnx/export_perception.py --out outputs/onnx/perception.onnx
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive_perception import QwenDrivePerception

from export_onnx.geometry_freeze import (FrozenGeometry, FrozenVoxelIndices,
                                         capture_geometry, capture_voxel_indices,
                                         frozen_geometry, frozen_voxel_pool)
from training.differentiable import enable_training_ops

OUTPUT_NAMES = ["all_cls_scores", "all_bbox_preds", "occ_pred", "seg_preds"]


class PerceptionONNX(nn.Module):
    """Tensors in, tensors out - no dicts, no numpy, no calibration arguments.

    The calibration is baked in by ``frozen_geometry``; see geometry_freeze.py
    for why that is the right call rather than a shortcut.
    """

    def __init__(self, bev, img_metas):
        super().__init__()
        self.bev = bev
        self._metas = img_metas

    def forward(self, img_vit_feats, img_llm_feats):
        o = self.bev(img_vit_feats=img_vit_feats, img_llm_feats=img_llm_feats,
                     img_metas=[self._metas])
        return (o["all_cls_scores"], o["all_bbox_preds"], o["occ_pred"], o["seg_preds"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--cache", default="data/train_cache")
    ap.add_argument("--record", default="90162f90eceb4ada9e595bc1adb71b5f.pt")
    ap.add_argument("--out", default="outputs/onnx/perception.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dynamo", action="store_true", help="use the TorchDynamo exporter")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--no-cuda-gridsample", action="store_true",
                    help="keep stock opset-20 GridSample, which ORT runs on CPU")
    ap.add_argument("--no-fold", action="store_true",
                    help="disable constant folding (keeps big zeros as "
                         "ConstantOfShape instead of a materialised tensor)")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    enable_training_ops()
    print("portable ops enabled (the shipped CUDA kernels are not representable in ONNX)")

    head = QwenDrivePerception.from_pretrained(
        args.model, dtype=torch.float32).to(args.device).eval()
    bev = head.bev_modeling
    rec = torch.load(Path(args.cache) / args.record, weights_only=False)
    vit = rec["img_vit_feats"].to(args.device, torch.float32)
    llm = rec["img_llm_feats"].to(args.device, torch.float32)
    metas = rec["img_metas"]
    print(f"example inputs: vit {tuple(vit.shape)}  llm {tuple(llm.shape)}")

    store, vidx = FrozenGeometry(), FrozenVoxelIndices()
    print("\ncapturing calibration geometry (one eager forward)...")
    t0 = time.time()
    with torch.no_grad(), capture_geometry(store), capture_voxel_indices(vidx):
        ref_out = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
    print(f"  {time.time()-t0:.0f}s   {store.summary()}")
    print(f"  voxel scatter: {vidx.ranks.numel()} points -> {vidx.n_rows} rows")
    ref = tuple(ref_out[k].detach() for k in OUTPUT_NAMES)

    wrapper = PerceptionONNX(bev, metas).eval()
    print(f"\nexporting (opset {args.opset}, "
          f"{'dynamo' if args.dynamo else 'torchscript'})...")
    t0 = time.time()
    with torch.no_grad(), frozen_geometry(store, device=args.device), \
         frozen_voxel_pool(vidx, device=args.device):
        if args.dynamo:
            prog = torch.onnx.export(wrapper, (vit, llm), dynamo=True,
                                     input_names=["img_vit_feats", "img_llm_feats"],
                                     output_names=OUTPUT_NAMES, opset_version=args.opset)
            prog.save(str(out))
        else:
            torch.onnx.export(
                wrapper, (vit, llm), str(out),
                input_names=["img_vit_feats", "img_llm_feats"],
                output_names=OUTPUT_NAMES, opset_version=args.opset,
                do_constant_folding=not args.no_fold, dynamo=False)
    print(f"  exported in {time.time()-t0:.0f}s -> {out}  "
          f"({out.stat().st_size/2**20:.1f} MiB)")

    import onnx
    onnx.checker.check_model(str(out), full_check=False)
    model = onnx.load(str(out))
    ops = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  checker OK.  {len(model.graph.node)} nodes, {len(ops)} distinct ops")
    print("  most common:", ", ".join(f"{k}x{v}" for k, v in
                                      sorted(ops.items(), key=lambda x: -x[1])[:8]))

    if args.skip_verify:
        if not args.no_cuda_gridsample:
            n, total = _gridsample_to_cuda_contrib(str(out))
            print(f"\nGridSample -> com.microsoft: {n}/{total} nodes now run on CUDA")
        return 0
    print("\nverifying against PyTorch with onnxruntime...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    t0 = time.time()
    got = sess.run(None, {"img_vit_feats": vit.cpu().numpy(),
                          "img_llm_feats": llm.cpu().numpy()})
    print(f"  onnxruntime forward {time.time()-t0:.0f}s")
    worst = 0.0
    for name, a, b in zip(OUTPUT_NAMES, got, ref):
        b = b.cpu().numpy()
        d = float(np.abs(a - b).max())
        rel = d / max(float(np.abs(b).max()), 1e-9)
        worst = max(worst, rel)
        print(f"  {name:16s} {str(tuple(a.shape)):26s} max abs diff {d:.3e}  rel {rel:.3e}")
    ok = worst < 1e-3
    print(f"\nVERIFY: {'PASS' if ok else 'FAIL'}  (worst relative diff {worst:.2e})")

    if not args.no_cuda_gridsample:
        n, total = _gridsample_to_cuda_contrib(str(out))
        print(f"\nGridSample -> com.microsoft: {n}/{total} nodes now run on CUDA "
              f"({total - n} are 5-D and stay on CPU)")
    return 0 if ok else 1


def _gridsample_to_cuda_contrib(path: str):
    """Route 4-D GridSample at ORT's CUDA kernel instead of letting it hit CPU.

    ORT registers a CUDA GridSample only in the ``com.microsoft`` domain, and it
    accepts the opset-16 spelling ``bilinear``. Opset 20 renamed that mode to
    ``linear``, so a stock opset-20 export silently runs every GridSample on the
    CPU - 80% of this graph's runtime, plus the device round-trips it forces.
    The 5-D nodes have no CUDA kernel at all and are left alone, which is why the
    model has to stay at opset 20.
    """
    import onnx
    meta = onnx.load(path, load_external_data=False)
    ranks = {}
    for vi in (list(onnx.shape_inference.infer_shapes(meta).graph.value_info)
               + list(meta.graph.input)):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            ranks[vi.name] = len(vi.type.tensor_type.shape.dim)

    model = onnx.load(path)
    converted = total = 0
    for node in model.graph.node:
        if node.op_type != "GridSample" or node.domain not in ("", "ai.onnx"):
            continue
        total += 1
        if ranks.get(node.input[0]) != 4:
            continue
        node.domain = "com.microsoft"
        for attr in node.attribute:
            if attr.name == "mode" and attr.s == b"linear":
                attr.s = b"bilinear"
        converted += 1
    if converted:
        model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
        before = {p for p in Path(path).parent.iterdir() if p.suffix != ".onnx"}
        onnx.save(model, path, save_as_external_data=True,
                  all_tensors_to_one_file=False, size_threshold=1024)
        # Re-saving renames the external tensor files; the originals are now
        # unreferenced and would otherwise double the directory on disk.
        keep = {kv.value for t in model.graph.initializer
                for kv in t.external_data if kv.key == "location"}
        for stale in before:
            if stale.name not in keep:
                stale.unlink()
    return converted, total


if __name__ == "__main__":
    raise SystemExit(main())
