"""Fewer cameras: what it buys, and what it cannot buy.

Every other lever left the BEV head at 420 ms in all configurations, and 228 ms of
that is three Conv3d over the 16x200x200 voxel volume -- running at 29.8 TFLOPS, so
near peak, and destroying detection if any one is removed. Dropping cameras is the
first lever that changes the *shape* of the problem rather than the depth of a stack:

  decoder      tokens fall almost linearly (448 image tokens per camera of 2744)
  vision       linear in cameras
  view_trans   coord_preparing and feat_sampling are linear in cameras,
               but feat_encoding is over the voxel grid and does NOT scale

So this measures where the floor actually lands. It also reports quality restricted
to the observable region: with a front camera only, the rear of the BEV is not
merely wrong, it is unseen, and counting rear detections against a front-only model
measures the camera rig rather than the model.

    python local/prune/eval_camera_subset_v1.py
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

LEAST16 = [1, 2, 3, 4, 8, 9, 10, 12, 13, 14, 15, 16, 17, 20, 21, 25]

SUBSETS = {
    "front1": ["CAM_FRONT"],
    "front3": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
    "all6": None,
}


def subset_frame(frame, cams):
    """A copy of the frame restricted to ``cams``, calibration included."""
    if cams is None:
        return frame
    f = copy.copy(frame)
    keep = [frame.cam_order.index(c) for c in cams if c in frame.cam_order]
    f.cam_order = [frame.cam_order[i] for i in keep]
    f.cam_intrinsic = frame.cam_intrinsic[keep]
    f.sensor2lidar_rotation = frame.sensor2lidar_rotation[keep]
    f.sensor2lidar_translation = frame.sensor2lidar_translation[keep]
    # content is <view tag>, <image> per camera, then the instruction
    content = []
    for item in frame.content:
        if "image" in item:
            if item["image"] in f.cam_order:
                content.append(item)
        elif content and len(content) % 2 == 0 or "image" not in item:
            content.append(item)
    # rebuild strictly as tag/image pairs followed by the trailing instruction
    pairs, i = [], 0
    src = frame.content
    while i < len(src):
        if i + 1 < len(src) and "text" in src[i] and "image" in src[i + 1]:
            if src[i + 1]["image"] in f.cam_order:
                pairs += [src[i], src[i + 1]]
            i += 2
        else:
            pairs.append(src[i]); i += 1
    f.content = pairs
    return f


def dets(cls, box, thr=0.3):
    p = 1 / (1 + np.exp(-cls.max(-1)))
    return p >= thr, cls.argmax(-1), box[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/prune/camera_subset_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame
    from local.prune.measure_layer_influence_v1 import run_stack

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

    def stage(cams, skip):
        f = subset_frame(base_frame, cams)
        with torch.no_grad():
            inputs, metas = proc(f, device="cuda")
            grid = inputs["image_grid_thw"]
            n_cam = len(metas["cam_order"])
            gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
            tpi = gh // 2 * gw // 2
            embeds = vlm.model.get_input_embeddings()(inputs["input_ids"])
            cap = {}
            hk = vlm.model.visual.merger.register_forward_hook(
                lambda m, a, o=None: cap.__setitem__("p", a[0]))
            torch.cuda.synchronize(); t = time.perf_counter()
            out = vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
            torch.cuda.synchronize(); vis_ms = (time.perf_counter() - t) * 1e3
            hk.remove()
            merged = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            vit = torch.stack(head._premerge_grids(
                vlm.model.visual.merger.norm(cap["p"]), grid)[-n_cam:], 0).to(dt)
            # splice the merged vision tokens the way infer() does
            mask = inputs["input_ids"][0] == vlm.config.image_token_id
            x = embeds.clone()
            mm = bev  # placeholder to keep names clear
            merged_tok = vlm.model.visual.merger(cap["p"])
            merged_tok = merged_tok[0] if isinstance(merged_tok, tuple) else merged_tok
            x[0, mask] = merged_tok[-int(mask.sum()):].to(embeds.dtype)
            pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
            torch.cuda.synchronize(); t = time.perf_counter()
            hidden, _ = run_stack(lm, types, x, pos, skip=set(skip))
            torch.cuda.synchronize(); dec_ms = (time.perf_counter() - t) * 1e3
            llm = hidden[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1).to(dt)
            torch.cuda.synchronize(); t = time.perf_counter()
            o = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
            torch.cuda.synchronize(); head_ms = (time.perf_counter() - t) * 1e3
        del cap, x, hidden, vit
        return (o["all_cls_scores"][-1, 0].float().cpu().numpy(),
                o["all_bbox_preds"][-1, 0].float().cpu().numpy(),
                {"tokens": int(inputs["input_ids"].shape[1]), "n_cam": n_cam,
                 "vision_ms": vis_ms, "decoder_ms": dec_ms, "head_ms": head_ms})

    cls0, box0, m0 = stage(None, [])
    base = dets(cls0, box0, args.thr)
    front = base[2][:, 0] > 0.0
    print(f"  baseline: {m0['n_cam']} cams, {m0['tokens']} tokens, "
          f"{int(base[0].sum())} detections ({int((base[0] & front).sum())} ahead of ego)")
    print(f"  vision {m0['vision_ms']:.0f}  decoder {m0['decoder_ms']:.0f}  "
          f"head {m0['head_ms']:.0f} ms\n")
    print(f"  {'config':18s} {'cams':>4s} {'tok':>5s} {'vis':>6s} {'dec':>7s} {'head':>7s} "
          f"{'total':>8s} {'Hz':>6s} {'kept(front)':>12s} {'spur':>5s}")
    rows = {}
    for sname, cams in SUBSETS.items():
        for skname, skip in (("full", []), ("skip16", LEAST16)):
            cls, box, m = stage(cams, skip)
            c = dets(cls, box, args.thr)
            fb = base[0] & front
            both = fb & c[0]
            total = m["vision_ms"] + m["decoder_ms"] + m["head_ms"]
            name = f"{sname}+{skname}"
            rows[name] = {**m, "total_ms": total, "hz": 1000.0 / total,
                          "kept_front": int(both.sum()), "of_front": int(fb.sum()),
                          "spurious": int((c[0] & ~base[0]).sum())}
            r = rows[name]
            print(f"  {name:18s} {m['n_cam']:4d} {m['tokens']:5d} {m['vision_ms']:6.0f} "
                  f"{m['decoder_ms']:7.0f} {m['head_ms']:7.0f} {total:8.0f} {r['hz']:6.2f} "
                  f"{r['kept_front']:5d}/{r['of_front']:<5d} {r['spurious']:5d}", flush=True)
            torch.cuda.empty_cache()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"baseline": m0, "rows": rows}, indent=1))
    best = min(rows.values(), key=lambda r: r["total_ms"])
    print(f"\n  fastest perception {best['total_ms']:.0f} ms -> {best['hz']:.2f} Hz "
          f"(3 Hz needs 333 ms)")
    print(f"  head floor in that config: {best['head_ms']:.0f} ms")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
