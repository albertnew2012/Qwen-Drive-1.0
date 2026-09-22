"""Find a front-camera configuration under 333 ms, and report what it costs.

Standing at 352 ms (2.84 Hz) with: front camera only, 16 of 32 decoder layers
skipped, and the bit-exact voxel crop. 19 ms short of 3 Hz. This sweeps the
remaining levers and reports detections honestly against the full six-camera model,
restricted to the sector a front camera can see.

  skip        how many decoder layers are passed through (ranked least-influential
              first, from measure_layer_influence_v1.py)
  exit        which detection decoder layer's prediction is used; each carries its
              own trained cls/reg branch, so an early exit is a trained detector
  scale       front image resize; tokens fall as the square

Everything here except the crop costs accuracy, which is what distillation would
have to buy back.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

RANK_LEAST_FIRST = [13, 12, 16, 17, 4, 9, 15, 14, 20, 3, 8, 1, 21, 2, 18, 10,
                    25, 11, 29, 24, 5, 28, 30, 22, 26, 7, 6, 23, 19, 27, 31, 0]


def dets(cls, box, thr=0.3):
    p = 1 / (1 + np.exp(-cls.max(-1)))
    return p >= thr, cls.argmax(-1), box[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/prune/final_push_v1.json")
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

    def prep(cams, scale):
        f = subset_frame(base_frame, cams)
        if scale != 1.0:
            w, h = proc.image_size
            proc.image_size = (int(round(w * scale / proc.factor)) * proc.factor,
                               int(round(h * scale / proc.factor)) * proc.factor)
        try:
            with torch.no_grad():
                inputs, metas = proc(f, device="cuda")
        finally:
            if scale != 1.0:
                proc.image_size = (896, 512)
        with torch.no_grad():
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
        return dict(x=x, pos=pos, mask=mask, vit=vit, metas=metas, n_cam=n_cam,
                    gh=gh, gw=gw, tpi=tpi, vis=vis,
                    tokens=int(inputs["input_ids"].shape[1]))

    def once(s, skip, exit_layer):
        with torch.no_grad():
            torch.cuda.synchronize(); t = time.perf_counter()
            hidden, _ = run_stack(lm, types, s["x"], s["pos"], skip=set(skip))
            torch.cuda.synchronize(); dec = (time.perf_counter() - t) * 1e3
            llm = hidden[0][s["mask"]][-s["n_cam"]*s["tpi"]:].view(
                s["n_cam"], s["gh"]//2, s["gw"]//2, -1).to(dt)
            torch.cuda.synchronize(); t = time.perf_counter()
            o = bev(img_vit_feats=s["vit"], img_llm_feats=llm, img_metas=[s["metas"]])
            torch.cuda.synchronize(); hd = (time.perf_counter() - t) * 1e3
        return (o["all_cls_scores"][exit_layer, 0].float().cpu().numpy(),
                o["all_bbox_preds"][exit_layer, 0].float().cpu().numpy(), dec, hd)

    s6 = prep(None, 1.0)
    c6, b6, _, _ = once(s6, [], 5)
    ref = dets(c6, b6, args.thr)
    front = ref[2][:, 0] > 0.0
    fb = ref[0] & front
    print(f"  reference: 6 cams, {int(ref[0].sum())} dets, {int(fb.sum())} ahead of ego\n")
    del s6
    torch.cuda.empty_cache()

    restore = install(bev.view_trans)
    rows = {}
    print(f"  all rows: front camera only + bit-exact voxel crop")
    print(f"  {'skip':>5s} {'exit':>4s} {'scale':>6s} {'tok':>5s} {'vis':>5s} {'dec':>6s} "
          f"{'head':>6s} {'total':>7s} {'Hz':>6s} {'front':>8s} {'spur':>5s}")
    try:
        for scale in (1.0, 0.9):
            s = prep(["CAM_FRONT"], scale)
            for nskip in (16, 20, 24):
                skip = RANK_LEAST_FIRST[:nskip]
                for exit_layer in (5, 3):
                    once(s, skip, exit_layer)
                    d, h = [], []
                    for _ in range(3):
                        cls, box, dec, hd = once(s, skip, exit_layer)
                        d.append(dec); h.append(hd)
                    dec, hd = float(np.median(d)), float(np.median(h))
                    c = dets(cls, box, args.thr)
                    both = fb & c[0]
                    total = s["vis"] + dec + hd
                    key = f"skip{nskip}_exit{exit_layer}_scale{scale}"
                    rows[key] = {"skip": nskip, "exit": exit_layer, "scale": scale,
                                 "tokens": s["tokens"], "vision_ms": s["vis"],
                                 "decoder_ms": dec, "head_ms": hd, "total_ms": total,
                                 "hz": 1000.0 / total, "front_kept": int(both.sum()),
                                 "front_of": int(fb.sum()),
                                 "spurious": int((c[0] & ~ref[0]).sum())}
                    r = rows[key]
                    flag = "  <- under 333 ms" if total < 333 else ""
                    print(f"  {nskip:5d} {exit_layer:4d} {scale:6.2f} {s['tokens']:5d} "
                          f"{s['vis']:5.0f} {dec:6.0f} {hd:6.0f} {total:7.0f} "
                          f"{r['hz']:6.2f} {r['front_kept']:3d}/{r['front_of']:<4d} "
                          f"{r['spurious']:5d}{flag}", flush=True)
            del s
            torch.cuda.empty_cache()
    finally:
        restore()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows.values() if r["total_ms"] < 333]
    if ok:
        best = max(ok, key=lambda r: r["front_kept"])
        print(f"\n  3 Hz reached: {best['total_ms']:.0f} ms = {best['hz']:.2f} Hz, "
              f"keeping {best['front_kept']}/{best['front_of']} front detections")
    else:
        b = min(rows.values(), key=lambda r: r["total_ms"])
        print(f"\n  closest {b['total_ms']:.0f} ms = {b['hz']:.2f} Hz")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
