"""Export the BEV head with only the detection outputs.

Profiling the front head under ORT 1.25.1 (710 ms wall, 660 ms of kernels, no CPU
fallback) puts the cost in the two auxiliary decoders rather than in the lift-splat:

    seg_decoder/layer2  conv 51.6 + 47.9, add 49.8 + 45.9   ~195 ms
    occ_decoder         conv 26.9 + 20.7                     ~48 ms
    view_trans/conv_layer.0                                   15.8 ms

Requesting a subset of outputs at run time does not help -- ORT partitions the whole
graph regardless, measured 696 ms against 704 ms -- so the outputs have to be dropped
at export. This writes a detection-only head (``all_cls_scores``, ``all_bbox_preds``)
so the map-segmentation and occupancy branches are never traced.

That is a change in what the model produces, not just how fast: occupancy and map
segmentation are real perception outputs. It is exported alongside the full head, not
in place of it, so the two can be compared and the full one kept if the outputs are
needed every frame rather than at a lower rate.
"""
from __future__ import annotations

import os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import torch
from torch import nn

DET_OUTPUTS = ["all_cls_scores", "all_bbox_preds"]

# The DETR decoder costs 233.9 ms of the head's 444 ms of kernel time in ONNX against
# 7 ms in PyTorch: its reference-point update is a chain of tiny Log/Clip/ScatterND
# ops on [1, 900, 3] tensors, 2,008 kernels averaging 221 us. Each layer carries its
# own trained cls/reg branch, so stopping early yields a trained detector rather than
# an intermediate -- measured in PyTorch at 32/37 detections kept and 0.023 m centre
# error for layer 3. Free there, worth ~117 ms here.
DECODER_LAYERS = int(os.environ.get("DET_DECODER_LAYERS", "0"))  # 0 = keep all


class DetOnlyONNX(nn.Module):
    def __init__(self, bev, img_metas):
        super().__init__()
        self.bev = bev
        self._metas = img_metas

    def forward(self, img_vit_feats, img_llm_feats):
        o = self.bev(img_vit_feats=img_vit_feats, img_llm_feats=img_llm_feats,
                     img_metas=[self._metas])
        return o["all_cls_scores"], o["all_bbox_preds"]


def main() -> int:
    os.chdir(_ROOT)
    from qwen_drive_perception import QwenDrivePerception
    from export_onnx.export_perception import (
        FrozenGeometry, FrozenVoxelIndices, capture_geometry, capture_voxel_indices,
        frozen_geometry, frozen_voxel_pool, _gridsample_to_cuda_contrib)
    from training.differentiable import enable_training_ops

    enable_training_ops()
    suffix = f"_d{DECODER_LAYERS}" if DECODER_LAYERS else ""
    out = _ROOT / "outputs" / "onnx" / f"perception_front_det{suffix}" / "perception.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    head = QwenDrivePerception.from_pretrained(
        "weights/Qwen-Drive-1.0-4B/perception", dtype=torch.float32).to("cuda").eval()
    bev = head.bev_modeling
    rec = torch.load(_ROOT/"data"/"train_cache_front"/"front.pt", weights_only=False)
    vit = rec["img_vit_feats"].to("cuda", torch.float32)
    llm = rec["img_llm_feats"].to("cuda", torch.float32)
    metas = rec["img_metas"]
    print(f"  inputs vit {tuple(vit.shape)} llm {tuple(llm.shape)}", flush=True)

    store, vidx = FrozenGeometry(), FrozenVoxelIndices()
    with torch.no_grad(), capture_geometry(store), capture_voxel_indices(vidx):
        bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
    print(f"  geometry frozen: {vidx.ranks.numel()} points -> {vidx.n_rows} rows",
          flush=True)

    if DECODER_LAYERS:
        import torch.nn as nn
        dec = bev.head.transformer.decoder
        kept = list(dec.layers)[:DECODER_LAYERS]
        dec.layers = nn.ModuleList(kept)
        if hasattr(dec, "num_layers"):
            dec.num_layers = len(kept)
        print(f"  detection decoder truncated to {len(kept)} layers", flush=True)

    wrapper = DetOnlyONNX(bev, metas).eval()
    t0 = time.time()
    with torch.no_grad(), frozen_geometry(store, device="cuda"), \
         frozen_voxel_pool(vidx, device="cuda"):
        torch.onnx.export(wrapper, (vit, llm), str(out),
                          input_names=["img_vit_feats", "img_llm_feats"],
                          output_names=DET_OUTPUTS, opset_version=20,
                          do_constant_folding=True, dynamo=False)
    import onnx
    n = len(onnx.load(str(out), load_external_data=False).graph.node)
    print(f"  exported {n} nodes in {time.time()-t0:.0f}s", flush=True)
    c, tot = _gridsample_to_cuda_contrib(str(out))
    print(f"  GridSample -> com.microsoft: {c}/{tot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
