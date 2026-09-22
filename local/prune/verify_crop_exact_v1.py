"""Is the cropped conv stack exact, or only close?

The first measurement reported a 3.8e-01 max absolute difference against the
uncropped result, which needs explaining before the crop can be trusted. Two
candidate causes:

  bf16 rounding    the cropped and uncropped tensors have different shapes, so
                   cuDNN picks different tilings and accumulates in a different
                   order. bfloat16 carries 8 mantissa bits, so on activations of
                   O(50) a difference of O(0.4) is ordinary.
  a real bug       the occupied box mapped to the wrong axes, or the constant fill
                   is wrong, in which case float32 will show it too.

So this runs both precisions and reports the difference relative to the tensor's own
magnitude, separately inside the pasted region and in the constant-filled region.
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame
    from local.prune.eval_camera_subset_v1 import subset_frame
    from local.prune.measure_layer_influence_v1 import run_stack
    from local.prune import voxel_crop_v1 as vc

    for dtype in (torch.bfloat16, torch.float32):
        tag = str(dtype).split(".")[-1]
        model = QwenDriveForPlanning.from_pretrained(
            args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        head = QwenDrivePerception.from_pretrained(
            args.perception, dtype=dtype).to("cuda").eval()
        proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
        head.attach(model.vlm, proc)
        vlm, lm = model.vlm, model.vlm.model.language_model
        types = list(vlm.config.text_config.layer_types)
        bev = head.bev_modeling
        f = subset_frame(PerceptionFrame(Path(args.frame)), ["CAM_FRONT"])
        with torch.no_grad():
            inputs, metas = proc(f, device="cuda")
            grid = inputs["image_grid_thw"]; n_cam = len(metas["cam_order"])
            gh, gw = int(grid[-1, 1]), int(grid[-1, 2]); tpi = gh//2*gw//2
            emb = vlm.model.get_input_embeddings()(inputs["input_ids"])
            cap = {}
            hk = vlm.model.visual.merger.register_forward_hook(
                lambda m, a, o=None: cap.__setitem__("p", a[0]))
            vlm.model.visual(inputs["pixel_values"], grid_thw=grid); hk.remove()
            vit = torch.stack(head._premerge_grids(
                vlm.model.visual.merger.norm(cap["p"]), grid)[-n_cam:], 0).to(dtype)
            mt = vlm.model.visual.merger(cap["p"])
            mt = mt[0] if isinstance(mt, tuple) else mt
            mask = inputs["input_ids"][0] == vlm.config.image_token_id
            x = emb.clone(); x[0, mask] = mt[-int(mask.sum()):].to(emb.dtype)
            pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
            hidden, _ = run_stack(lm, types, x, pos)
            llm = hidden[0][mask][-n_cam*tpi:].view(n_cam, gh//2, gw//2, -1).to(dtype)

        vt = bev.view_trans
        grab = {}
        orig_fe = vt.feat_encoding
        orig_cp = vt.coord_preparing
        boxes = {}

        def cp(img_metas):
            co, mk = orig_cp(img_metas)
            sel = co[mk]
            boxes["w"] = (int(sel[:, 1].min()), int(sel[:, 1].max()) + 1)
            boxes["h"] = (int(sel[:, 2].min()), int(sel[:, 2].max()) + 1)
            return co, mk

        def fe(v):
            grab["ref"] = orig_fe(v)
            return grab["ref"]

        vt.coord_preparing = cp; vt.feat_encoding = fe
        with torch.no_grad():
            bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
        ref = grab["ref"].float().clone()
        vt.coord_preparing = orig_cp; vt.feat_encoding = orig_fe

        restore = vc.install(vt)
        grab2 = {}
        inner_fe = vt.feat_encoding
        def fe2(v):
            grab2["got"] = inner_fe(v)
            return grab2["got"]
        vt.feat_encoding = fe2
        with torch.no_grad():
            bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
        got = grab2["got"].float().clone()
        restore()

        h0, h1 = boxes["h"]; w0, w1 = boxes["w"]
        scale = max(float(ref.abs().max()), 1e-9)
        d = (ref - got).abs()
        inside = d[..., max(0, h0-3):h1+3, max(0, w0-3):w1+3]
        out_mask = torch.ones_like(d, dtype=torch.bool)
        out_mask[..., max(0, h0-3):h1+3, max(0, w0-3):w1+3] = False
        print(f"\n  {tag}:  |ref|max {scale:.3f}   occupied box h {h0}..{h1}  w {w0}..{w1}")
        print(f"    pasted region : max abs {inside.max().item():.3e}  "
              f"rel {inside.max().item()/scale:.3e}")
        print(f"    constant region: max abs {d[out_mask].max().item():.3e}  "
              f"rel {d[out_mask].max().item()/scale:.3e}")
        print(f"    overall        : rel {d.max().item()/scale:.3e}", flush=True)
        del model, head, ref, got
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
