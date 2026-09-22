"""Parity for the front-camera ONNX configuration, against PyTorch doing the same thing.

Comparing the pruned front-camera ONNX export against the unmodified six-camera
PyTorch model would measure the workload reduction, not the export. So PyTorch is run
in exactly the configuration the export encodes -- one camera, the same 20 layers
skipped, the same voxel crop -- and the detections are compared.

Saves the PyTorch side; ``compare`` then reads the ONNX side from the runner's npz.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

RANK_LEAST_FIRST = [13, 12, 16, 17, 4, 9, 15, 14, 20, 3, 8, 1, 21, 2, 18, 10,
                    25, 11, 29, 24, 5, 28, 30, 22, 26, 7, 6, 23, 19, 27, 31, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=20)
    ap.add_argument("--crop", action="store_true", default=True)
    ap.add_argument("--onnx", default="outputs/onnx_bench/v2_perception_out.npz")
    ap.add_argument("--out", default="outputs/prune/parity_front_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame
    from local.prune.eval_camera_subset_v1 import subset_frame
    from local.prune.measure_layer_influence_v1 import run_stack
    from local.prune.voxel_crop_v1 import install

    # float32 so the comparison is against arithmetic, not against bfloat16 rounding
    model = QwenDriveForPlanning.from_pretrained(
        "weights/Qwen-Drive-1.0-4B", dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    head = QwenDrivePerception.from_pretrained(
        "weights/Qwen-Drive-1.0-4B/perception", dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained("weights/Qwen-Drive-1.0-4B"))
    head.attach(model.vlm, proc)
    vlm, lm = model.vlm, model.vlm.model.language_model
    types = list(vlm.config.text_config.layer_types)
    bev = head.bev_modeling
    dt = next(bev.parameters()).dtype

    frame = PerceptionFrame(Path("data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f"))
    front = subset_frame(frame, [frame.cam_order[0]])
    restore = install(bev.view_trans) if args.crop else (lambda: None)
    try:
        with torch.no_grad():
            inputs, metas = proc(front, device="cuda")
            grid = inputs["image_grid_thw"]
            n_cam = len(metas["cam_order"])
            gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
            tpi = gh // 2 * gw // 2
            emb = vlm.model.get_input_embeddings()(inputs["input_ids"])
            cap = {}
            hk = vlm.model.visual.merger.register_forward_hook(
                lambda m, a, o=None: cap.__setitem__("p", a[0]))
            vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
            hk.remove()
            vit = torch.stack(head._premerge_grids(
                vlm.model.visual.merger.norm(cap["p"]), grid)[-n_cam:], 0).to(dt)
            mt = vlm.model.visual.merger(cap["p"])
            mt = mt[0] if isinstance(mt, tuple) else mt
            mask = inputs["input_ids"][0] == vlm.config.image_token_id
            x = emb.clone()
            x[0, mask] = mt[-int(mask.sum()):].to(emb.dtype)
            pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
            hidden, _ = run_stack(lm, types, x, pos,
                                  skip=set(RANK_LEAST_FIRST[:args.skip]))
            llm = hidden[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1).to(dt)
            o = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
        cls = o["all_cls_scores"].float().cpu().numpy()
        box = o["all_bbox_preds"].float().cpu().numpy()
    finally:
        restore()

    np.savez(_ROOT / f"outputs/onnx_bench/torch_front_skip{args.skip}.npz",
             cls=cls, box=box)
    print(f"  PyTorch front+skip{args.skip}: cls {cls.shape} box {box.shape}")

    onnx_path = Path(args.onnx)
    rep = {"skip": args.skip, "crop": bool(args.crop)}
    if onnx_path.exists():
        d = np.load(onnx_path)
        if "cls" in d and d["cls"].shape[0] <= cls.shape[0]:
            n = d["cls"].shape[0]
            for name, a, b in (("cls", cls[:n], d["cls"]), ("box", box[:n], d["box"])):
                s = max(float(np.abs(a).max()), 1e-9)
                dd = np.abs(a - b)
                rep[name] = {"rel_max": float(dd.max() / s),
                             "rel_rms": float(np.sqrt((dd ** 2).mean()) / s)}
                print(f"  {name}: rel_max {rep[name]['rel_max']:.2e}  "
                      f"rel_rms {rep[name]['rel_rms']:.2e}")
            a, b = cls[n - 1, 0], d["cls"][-1, 0]
            agree = float((a.argmax(-1) == b.argmax(-1)).mean())
            rep["argmax_agreement"] = agree
            pa = 1 / (1 + np.exp(-a.max(-1)))
            pb = 1 / (1 + np.exp(-b.max(-1)))
            ka, kb = pa >= 0.3, pb >= 0.3
            rep["detections"] = {"torch": int(ka.sum()), "onnx": int(kb.sum()),
                                 "both": int((ka & kb).sum())}
            print(f"  argmax agreement {agree:.2%}   detections torch "
                  f"{int(ka.sum())} onnx {int(kb.sum())} both {int((ka & kb).sum())}")
        else:
            print(f"  ONNX npz shape {d['cls'].shape if 'cls' in d else '?'} "
                  f"does not line up; skipping comparison")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
