"""Stage 1 - train the BEV perception head with the VLM frozen.

    L_perc = L_det + L_occ + L_map                          (arXiv:2609.00111)

The VLM's two taps are pre-extracted by ``cache_features.py``, so the 4.5 B VLM
is never loaded here. Only the 125 M head is trained.

    python training/cache_features.py
    python training/train_perception.py --steps 60 --overfit

``--overfit`` trains on a single frame: if the loss does not collapse, the
pipeline is broken. That is the test that matters.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.configuration_perception import OCC_EMPTY_LABEL

from training.config import PerceptionTrainConfig
from training.data import CachedPerceptionDataset
from training.differentiable import enable_training_ops
from training.losses import (HungarianMatcher3D, detection_loss, map_loss,
                             occupancy_loss)


def lr_at(step, total, cfg):
    """Linear warmup then cosine decay - the BEVFormer schedule."""
    if step < cfg.warmup_iters:
        a = step / max(cfg.warmup_iters, 1)
        return cfg.lr * (cfg.warmup_ratio + (1 - cfg.warmup_ratio) * a)
    p = (step - cfg.warmup_iters) / max(total - cfg.warmup_iters, 1)
    cos = 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return cfg.lr * (cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * cos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--cache", default="data/train_cache")
    ap.add_argument("--out", default="outputs/train_perception")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--overfit", action="store_true",
                    help="single frame - the loss MUST collapse")
    ap.add_argument("--scratch", action="store_true",
                    help="randomly re-init the head instead of fine-tuning")
    ap.add_argument("--no-det", action="store_true")
    ap.add_argument("--no-occ", action="store_true")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = PerceptionTrainConfig()
    if args.lr is not None:
        cfg.lr = args.lr
    torch.manual_seed(cfg.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    enable_training_ops()
    print("differentiable ops enabled (the shipped kernels have no backward)")

    dtype = getattr(torch, cfg.amp_dtype)
    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(args.device)
    bev = head.bev_modeling.train()
    if args.scratch:
        for m in bev.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        print("head RE-INITIALISED from scratch")

    ds = CachedPerceptionDataset(args.cache)
    n_params = sum(p.numel() for p in bev.parameters() if p.requires_grad)
    print(f"{len(ds)} cached frames   {n_params/1e6:.3f} M trainable parameters")

    opt = torch.optim.AdamW(bev.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=cfg.betas)
    matcher = HungarianMatcher3D(cfg.match_cls_cost, cfg.match_reg_cost)

    history, t_start = [], time.time()
    for step in range(args.steps):
        rec = ds[0] if args.overfit else ds[step % len(ds)]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, cfg)

        vit = rec["img_vit_feats"].to(args.device, dtype)
        llm = rec["img_llm_feats"].to(args.device, dtype)
        outs = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[rec["img_metas"]])

        parts, total = {}, torch.zeros((), device=args.device)
        if not args.no_det:
            d = detection_loss(
                outs["all_cls_scores"].float(), outs["all_bbox_preds"].float(),
                [rec["gt_boxes"].to(args.device)], [rec["gt_labels"].to(args.device)],
                matcher, cls_weight=cfg.det_cls_weight, reg_weight=cfg.det_reg_weight)
            matched = d.pop("det_matched")
            parts.update(d); parts["n_match"] = matched
            total = total + d["det_cls"] + d["det_reg"]
        if not args.no_occ:
            o = occupancy_loss(outs["occ_pred"], rec["gt_occ"].to(args.device),
                               empty_label=OCC_EMPTY_LABEL,
                               focal_weight=cfg.occ_focal_weight,
                               max_points=cfg.occ_max_points)
            parts.update(o); total = total + sum(o.values())
        if not args.no_map:
            m = map_loss(outs["seg_preds"], rec["gt_map"].to(args.device),
                         focal_weight=cfg.map_focal_weight,
                         max_points=cfg.map_max_points)
            parts.update(m); total = total + sum(m.values())

        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(bev.parameters(), cfg.grad_clip_norm)
        opt.step()

        row = {"step": step, "loss": float(total),
               "lr": opt.param_groups[0]["lr"], "grad_norm": float(gn),
               **{k: float(v) for k, v in parts.items()}}
        history.append(row)
        if step % 5 == 0 or step == args.steps - 1:
            det = f"det {row.get('det_cls',0):6.3f}/{row.get('det_reg',0):6.3f}"
            occ = f"occ {row.get('occ_focal',0):6.3f}+{row.get('occ_geo',0):5.2f}+{row.get('occ_sem',0):5.2f}+{row.get('occ_lov',0):5.3f}"
            mp = f"map {row.get('map_focal',0):6.3f}+{row.get('map_lov',0):5.3f}"
            print(f"  step {step:4d}  loss {float(total):9.4f}   {det}  {occ}  {mp}"
                  f"   |g| {float(gn):7.1f}")

    dt = time.time() - t_start
    (out / "history.json").write_text(json.dumps(history, indent=2))
    first, last = history[0]["loss"], history[-1]["loss"]
    print(f"\n{args.steps} steps in {dt:.0f}s ({dt/args.steps:.1f}s/step)"
          f"   peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    print(f"loss {first:.4f} -> {last:.4f}   ({100*(first-last)/abs(first):+.1f}%)")
    if args.overfit:
        ok = last < first * 0.5
        print("OVERFIT TEST:", "PASS - the loss collapsed" if ok else
              "FAIL - loss did not halve; the pipeline is not learning")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
