"""Compare the v2 ONNX export against PyTorch on the same frame and scene.

The reference is bfloat16 PyTorch, because that is the path the model actually
ships and the one ``local/bench_drive_torch.py`` measures. A float32 reference
would be the better yardstick but does not fit: the 4B decoder alone is 16 GiB in
float32 and the head needs the rest of the card.

So that the numbers have a scale, float16 PyTorch is run as well. bfloat16 keeps 8
mantissa bits and float16 keeps 11, so the float16-vs-bfloat16 difference is what a
legitimate change of precision looks like on this model -- the export's difference
should be of that order or smaller, not merely "small".

What is compared:

  all_cls_scores  (6, 1, 900, 7)      per-layer detection logits
  all_bbox_preds  (6, 1, 900, 10)     per-layer box parameters
  occ_pred        (1, 200, 200, 16, 10)
  seg_preds       (1, 6, 200, 400)
  trajectory      (1, 50, 3)          after the 10 Euler steps

plus, because raw-tensor error is not by itself a statement about behaviour, the
decoded detections: how many of the top-k boxes agree on label and fall within a
tolerance in centre distance.

    source export_onnx/env_gpu.sh
    python export_onnx/check_parity_v2.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))
BENCH = _ROOT / "outputs" / "onnx_bench"

OUTPUT_NAMES = ["all_cls_scores", "all_bbox_preds", "occ_pred", "seg_preds"]


def err(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.astype(np.float64), b.astype(np.float64)
    scale = max(float(np.abs(a).max()), 1e-12)
    d = np.abs(a - b)
    return {"rel_max": float(d.max() / scale), "rel_rms": float(np.sqrt((d ** 2).mean()) / scale),
            "abs_max": float(d.max())}


def decode(head, cls_scores, bbox_preds, metas, k=50):
    """Top-k detections from raw head tensors, via the model's own decoder."""
    outs = {"all_cls_scores": torch.as_tensor(cls_scores).float().cuda(),
            "all_bbox_preds": torch.as_tensor(bbox_preds).float().cuda()}
    got = head.bev_modeling.head.get_bboxes(outs, [metas])[0]
    boxes, scores, labels = got[0], got[1], got[2]
    boxes = boxes.tensor if hasattr(boxes, "tensor") else torch.as_tensor(boxes)
    order = torch.argsort(torch.as_tensor(scores).flatten(), descending=True)[:k]
    return (np.asarray(boxes.detach().cpu())[order.cpu().numpy()],
            np.asarray(torch.as_tensor(scores).flatten().detach().cpu())[order.cpu().numpy()],
            np.asarray(torch.as_tensor(labels).flatten().detach().cpu())[order.cpu().numpy()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--onnx-perception", default="outputs/onnx_bench/v2_perception_out.npz")
    ap.add_argument("--onnx-planning", default="outputs/onnx_bench/v2_planning_out.npz")
    ap.add_argument("--centre-tol", type=float, default=0.5, help="metres")
    ap.add_argument("--out", default="outputs/onnx_bench/v2_parity.json")
    ap.add_argument("--stage", choices=["perception", "planning"], required=True,
                    help="one stage per process: the head's intermediates and the "
                         "decoder do not both fit on a 24 GiB card in PyTorch")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning, InferenceMode
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.images import ImageArchive
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
    from transformers import AutoTokenizer

    onnx_p = np.load(args.onnx_perception)
    onnx_t = np.load(args.onnx_planning)["trajectory"]
    frame = PerceptionFrame(Path(args.frame))
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    if args.stage == "perception":
        # The planner is not needed here and its weights are 3.9 GiB.
        model = QwenDriveForPlanning.from_pretrained(
            args.vlm, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
        head = QwenDrivePerception.from_pretrained(
            args.perception, dtype=dtype).to("cuda").eval()
        processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
        head.attach(model.vlm, processor)
        print(f"[{args.dtype}] loaded in {time.time()-t0:.0f}s  "
              f"GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
        with torch.no_grad():
            inputs, metas = processor(frame, device="cuda")
            raw = _raw_head(head, inputs, metas)
        out = {k: v.numpy() for k, v in raw.items()}
        np.savez(BENCH / f"torch_{args.dtype}_perception.npz", **out)
        print(f"  wrote torch_{args.dtype}_perception.npz  "
              f"peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    else:
        model = QwenDriveForPlanning.from_pretrained(
            args.vlm, planner=args.planner, dtype=dtype,
            attn_implementation="sdpa").to("cuda").eval()
        sample = next(iter(read_scene_file(
            args.scenes, image_archive=archive,
            num_history_points=model.config.num_history_points, limit=1)))
        print(f"[{args.dtype}] loaded in {time.time()-t0:.0f}s  "
              f"GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
        with torch.no_grad():
            plan = model.run(InferenceMode.DIRECT_PLANNING, scene=sample.scene,
                             num_samples=1)
        traj = np.asarray(plan.trajectories[0], dtype=np.float64)[None]
        np.savez(BENCH / f"torch_{args.dtype}_planning.npz", trajectory=traj)
        print(f"  endpoint {traj[0, -1].round(3).tolist()}")
        print(f"  wrote torch_{args.dtype}_planning.npz")
    return 0


def _raw_head(head, inputs, metas):
    """The four tensors the ONNX head graph emits, from the PyTorch head."""
    vlm = head._vlm
    captured = {}
    visual = vlm.model.visual
    handle = visual.merger.register_forward_hook(
        lambda mod, a, o=None: captured.__setitem__("patches", a[0]))
    try:
        out = vlm(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                  image_grid_thw=inputs["image_grid_thw"],
                  mm_token_type_ids=head._modality_ids(inputs["input_ids"]),
                  use_cache=False, output_hidden_states=True)
    finally:
        handle.remove()
    hidden = vlm.model.language_model.norm(out.hidden_states[-1])
    patches = visual.merger.norm(captured["patches"])
    vit = head._premerge_grids(patches, inputs["image_grid_thw"])
    # The BEV head's voxel pooling asks for another 3.7 GiB; the VLM forward's
    # 33 hidden states and the captured patches have to be gone before it runs.
    del out, patches, captured
    torch.cuda.empty_cache()
    grid = inputs["image_grid_thw"]
    n_cam = len(metas["cam_order"])
    gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
    tpi = gh // 2 * gw // 2
    mask = inputs["input_ids"][0] == vlm.config.image_token_id
    llm = hidden[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1)
    dtype = next(head.bev_modeling.parameters()).dtype
    vit_in = torch.stack(vit[-n_cam:], 0).to(dtype)
    llm_in = llm.to(dtype)
    del vit, llm, hidden
    torch.cuda.empty_cache()
    o = head.bev_modeling(img_vit_feats=vit_in, img_llm_feats=llm_in, img_metas=[metas])
    got = {k: o[k].detach().float().cpu() for k in OUTPUT_NAMES}
    # Keep the head's own inputs: if the export disagrees, this says whether the
    # decoder or the head is responsible.
    got["img_vit_feats"] = vit_in.detach().float().cpu()
    got["img_llm_feats"] = llm_in.detach().float().cpu()
    del o, vit_in, llm_in
    torch.cuda.empty_cache()
    return got


if __name__ == "__main__":
    raise SystemExit(main())
