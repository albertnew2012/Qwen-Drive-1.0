"""Measure the cropped lift-splat: exactness first, then speed.

Runs front-camera-only perception with and without ``voxel_crop_v1``, verifies the
cropped conv output against the uncropped one, and reports the frame time.
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

LEAST16 = [1, 2, 3, 4, 8, 9, 10, 12, 13, 14, 15, 16, 17, 20, 21, 25]


def dets(cls, box, thr=0.3):
    p = 1 / (1 + np.exp(-cls.max(-1)))
    return p >= thr, cls.argmax(-1), box[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/prune/voxel_crop_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame
    from local.prune.eval_camera_subset_v1 import subset_frame
    from local.prune.measure_layer_influence_v1 import run_stack
    from local.prune.voxel_crop_v1 import install

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    head = QwenDrivePerception.from_pretrained(
        args.perception, dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, proc)
    vlm, lm = model.vlm, model.vlm.model.language_model
    types = list(vlm.config.text_config.layer_types)
    bev = head.bev_modeling
    dt = next(bev.parameters()).dtype
    base_frame = PerceptionFrame(Path(args.frame))

    def prep(cams):
        f = subset_frame(base_frame, cams)
        with torch.no_grad():
            inputs, metas = proc(f, device="cuda")
            grid = inputs["image_grid_thw"]
            n_cam = len(metas["cam_order"])
            gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
            tpi = gh // 2 * gw // 2
            emb = vlm.model.get_input_embeddings()(inputs["input_ids"])
            cap = {}
            hk = vlm.model.visual.merger.register_forward_hook(
                lambda m, a, o=None: cap.__setitem__("p", a[0]))
            torch.cuda.synchronize(); t = time.perf_counter()
            vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
            torch.cuda.synchronize(); vis = (time.perf_counter() - t) * 1e3
            hk.remove()
            vit = torch.stack(head._premerge_grids(
                vlm.model.visual.merger.norm(cap["p"]), grid)[-n_cam:], 0).to(dt)
            mt = vlm.model.visual.merger(cap["p"])
            mt = mt[0] if isinstance(mt, tuple) else mt
            mask = inputs["input_ids"][0] == vlm.config.image_token_id
            x = emb.clone()
            x[0, mask] = mt[-int(mask.sum()):].to(emb.dtype)
            pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
        return x, pos, mask, vit, metas, n_cam, gh, gw, tpi, vis, int(inputs["input_ids"].shape[1])

    def once(state, skip):
        x, pos, mask, vit, metas, n_cam, gh, gw, tpi, vis, _ = state
        with torch.no_grad():
            torch.cuda.synchronize(); t = time.perf_counter()
            hidden, _ = run_stack(lm, types, x, pos, skip=set(skip))
            torch.cuda.synchronize(); dec = (time.perf_counter() - t) * 1e3
            llm = hidden[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1).to(dt)
            torch.cuda.synchronize(); t = time.perf_counter()
            o = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
            torch.cuda.synchronize(); hd = (time.perf_counter() - t) * 1e3
        return (o["all_cls_scores"][-1, 0].float().cpu().numpy(),
                o["all_bbox_preds"][-1, 0].float().cpu().numpy(), dec, hd, vis)

    # reference: all six cameras, no pruning
    st6 = prep(None)
    c6, b6, dec6, hd6, vis6 = once(st6, [])
    ref = dets(c6, b6, args.thr)
    front = ref[2][:, 0] > 0.0
    print(f"  reference all6: {vis6+dec6+hd6:.0f} ms, {int(ref[0].sum())} dets "
          f"({int((ref[0] & front).sum())} ahead)\n")
    del st6
    torch.cuda.empty_cache()

    st1 = prep(["CAM_FRONT"])
    print(f"  front camera only, {st1[10]} tokens")
    rows = {}
    print(f"\n  verifying the crop is exact")
    restore = install(bev.view_trans, verify=True)
    once(st1, [])
    restore()

    for cname, crop in (("uncropped", False), ("cropped", True)):
        for skname, skip in (("full", []), ("skip16", LEAST16)):
            restore = install(bev.view_trans) if crop else (lambda: None)
            try:
                once(st1, skip)
                d, h, v = [], [], None
                for _ in range(3):
                    cls, box, dec, hd, vis = once(st1, skip)
                    d.append(dec); h.append(hd); v = vis
                dec, hd = float(np.median(d)), float(np.median(h))
                c = dets(cls, box, args.thr)
                fb = ref[0] & front
                both = fb & c[0]
                total = v + dec + hd
                name = f"front1+{cname}+{skname}"
                rows[name] = {"vision_ms": v, "decoder_ms": dec, "head_ms": hd,
                              "total_ms": total, "hz": 1000.0 / total,
                              "kept_front": int(both.sum()), "of_front": int(fb.sum())}
                r = rows[name]
                print(f"  {name:28s} vis {v:5.0f} dec {dec:6.0f} head {hd:6.0f} "
                      f"total {total:6.0f} -> {r['hz']:5.2f} Hz   "
                      f"front {r['kept_front']}/{r['of_front']}", flush=True)
            finally:
                restore()
            torch.cuda.empty_cache()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    best = min(rows.values(), key=lambda r: r["total_ms"])
    print(f"\n  best {best['total_ms']:.0f} ms -> {best['hz']:.2f} Hz  (3 Hz = 333 ms)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
