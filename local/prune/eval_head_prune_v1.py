"""Where the 429 ms of BEV head goes, and what can be taken out of it.

The head is 125M parameters: a view transform onto a 200x200 BEV, a 6-layer BEV
encoder, and a 6-layer DETR-style detection decoder over 900 queries. Two levers,
and the first costs nothing at all.

**Early exit.** ``all_cls_scores`` has shape (6, 1, 900, 7) -- one prediction per
decoder layer, because each layer carries its own ``cls_branches`` and
``reg_branches`` and was trained with its own auxiliary loss. Layer k's output is
therefore already a trained detector, not an intermediate. Taking it and skipping
the rest needs no retraining whatsoever; this measures what it costs in quality.

**Encoder depth.** Same treatment as the language decoder: skip a layer by passing
its input through, and see what the detections do.

Both are measured against the full head on the same frame, in detections rather
than in tensor norms, because the norm of a BEV feature map is not the deliverable.

    python local/prune/eval_head_prune_v1.py
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


def dets(cls, box, thr=0.3):
    p = 1 / (1 + np.exp(-cls.max(-1)))
    return p >= thr, cls.argmax(-1), box[:, :3]


def compare(ref, got, thr=0.3):
    k0, l0, c0 = ref
    k1, l1, c1 = got
    both = k0 & k1
    n = int(both.sum())
    return {"kept": n, "of": int(k0.sum()), "spurious": int((k1 & ~k0).sum()),
            "centre_median_m": (float(np.median(np.linalg.norm(c0[both]-c1[both], axis=-1)))
                                if n else float("nan")),
            "argmax_agreement": float((l0 == l1).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/prune/head_prune_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor
    from local.prune.measure_layer_influence_v1 import build, run_stack

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    head = QwenDrivePerception.from_pretrained(
        args.perception, dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, proc)
    vlm, lm = model.vlm, model.vlm.model.language_model
    types = list(vlm.config.text_config.layer_types)

    x, inputs, metas, mask = build(vlm, args.frame, proc)
    pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
    grid = inputs["image_grid_thw"]
    n_cam = len(metas["cam_order"]); gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
    tpi = gh // 2 * gw // 2
    dt = next(head.bev_modeling.parameters()).dtype
    captured = {}
    hk = vlm.model.visual.merger.register_forward_hook(
        lambda m, a, o=None: captured.__setitem__("p", a[0]))
    with torch.no_grad():
        vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
    hk.remove()
    with torch.no_grad():
        vit = torch.stack(head._premerge_grids(
            vlm.model.visual.merger.norm(captured["p"]), grid)[-n_cam:], 0).to(dt)
        hidden, _ = run_stack(lm, types, x, pos)
        llm = hidden[0][mask][-n_cam*tpi:].view(n_cam, gh//2, gw//2, -1).to(dt)
    del captured, x, hidden
    torch.cuda.empty_cache()

    bev = head.bev_modeling
    enc = bev.head.transformer.encoder.layers
    dec = bev.head.transformer.decoder.layers

    def run_head():
        with torch.no_grad():
            return bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])

    def timed(fn, n=3):
        fn(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

    o = run_head()
    cls_all = o["all_cls_scores"].float().cpu().numpy()
    box_all = o["all_bbox_preds"].float().cpu().numpy()
    base_ms = timed(run_head)
    ref = dets(cls_all[-1, 0], box_all[-1, 0], args.thr)
    print(f"  head {base_ms:.1f} ms   {int(ref[0].sum())} detections above {args.thr}")
    print(f"  encoder layers {len(enc)}  decoder layers {len(dec)}\n")

    report = {"head_ms": base_ms, "n_det": int(ref[0].sum()), "early_exit": {},
              "encoder_skip": {}}

    print("  EARLY EXIT -- use decoder layer k's own prediction (free, no retraining)")
    print(f"  {'layer':>5s} {'kept':>9s} {'spurious':>9s} {'centre_med':>11s} {'argmax':>8s}")
    for k in range(cls_all.shape[0]):
        c = compare(ref, dets(cls_all[k, 0], box_all[k, 0], args.thr), args.thr)
        report["early_exit"][k] = c
        print(f"  {k:5d} {c['kept']:4d}/{c['of']:<4d} {c['spurious']:9d} "
              f"{c['centre_median_m']:11.3f} {c['argmax_agreement']:7.1%}")

    # how much time each decoder layer costs, by truncating the ModuleList
    print(f"\n  DECODER TRUNCATION -- cost of the layers an early exit skips")
    print(f"  {'keep':>5s} {'head ms':>9s} {'saved':>8s}")
    full_dec = list(dec)
    for keep in range(1, len(full_dec) + 1):
        dec._modules = {str(i): m for i, m in enumerate(full_dec[:keep])}
        ms = timed(run_head)
        report.setdefault("decoder_truncate", {})[keep] = {"head_ms": ms,
                                                           "saved_ms": base_ms - ms}
        print(f"  {keep:5d} {ms:9.1f} {base_ms-ms:8.1f}")
    dec._modules = {str(i): m for i, m in enumerate(full_dec)}

    print(f"\n  ENCODER TRUNCATION")
    print(f"  {'keep':>5s} {'head ms':>9s} {'saved':>8s} {'kept':>9s} "
          f"{'spurious':>9s} {'centre_med':>11s}")
    full_enc = list(enc)
    for keep in range(1, len(full_enc) + 1):
        enc._modules = {str(i): m for i, m in enumerate(full_enc[:keep])}
        ms = timed(run_head)
        oo = run_head()
        c = compare(ref, dets(oo["all_cls_scores"][-1, 0].float().cpu().numpy(),
                              oo["all_bbox_preds"][-1, 0].float().cpu().numpy(),
                              args.thr), args.thr)
        report["encoder_skip"][keep] = {"head_ms": ms, "saved_ms": base_ms - ms, **c}
        print(f"  {keep:5d} {ms:9.1f} {base_ms-ms:8.1f} {c['kept']:4d}/{c['of']:<4d} "
              f"{c['spurious']:9d} {c['centre_median_m']:11.3f}")
    enc._modules = {str(i): m for i, m in enumerate(full_enc)}

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
