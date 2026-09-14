"""Stage 1 on BOTH GPUs with DistributedDataParallel.

    torchrun --nproc_per_node=2 training/train_perception_ddp.py --steps 60

Each rank keeps a full copy of the 125 M head and processes its own frame; the
gradients are all-reduced every step, so the effective batch is `world_size`.
The head peaks near 20.7 GiB for a single sample, which is why the batch stays
at one PER RANK - two samples on one card does not fit, two cards do.

NCCL NOTE. These are consumer 3090s with no NVLink, and
``can_device_access_peer`` is False. NCCL must be told so explicitly
(``NCCL_P2P_DISABLE=1``) or collectives can hang; the launcher script sets it.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.configuration_perception import OCC_EMPTY_LABEL

from training.config import PerceptionTrainConfig
from training.data import CachedPerceptionDataset
from training.differentiable import enable_training_ops
from training.losses import (HungarianMatcher3D, detection_loss, map_loss,
                             occupancy_loss)
from training.train_perception import lr_at


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--cache", default="data/train_cache")
    ap.add_argument("--out", default="outputs/train_ddp")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = f"cuda:{local}"
    is_main = rank == 0

    cfg = PerceptionTrainConfig()
    if args.lr is not None:
        cfg.lr = args.lr
    torch.manual_seed(cfg.seed + rank)
    enable_training_ops()

    dtype = getattr(torch, cfg.amp_dtype)
    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(dev)
    bev = head.bev_modeling.train()
    # Camera count differs between nuScenes (6) and nuPlan (8) frames, so the
    # graph is not identical every step; DDP must not assume a fixed set of
    # participating parameters.
    ddp = DDP(bev, device_ids=[local], find_unused_parameters=True,
              broadcast_buffers=False)

    ds = CachedPerceptionDataset(args.cache)
    if is_main:
        n = sum(p.numel() for p in bev.parameters())
        print(f"world_size {world}   {len(ds)} frames   {n/1e6:.3f} M params/rank",
              flush=True)

    opt = torch.optim.AdamW(ddp.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=cfg.betas)
    matcher = HungarianMatcher3D(cfg.match_cls_cost, cfg.match_reg_cost)
    history, t0 = [], time.time()

    for step in range(args.steps):
        rec = ds[(step * world + rank) % len(ds)]        # disjoint shards
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, cfg)

        vit = rec["img_vit_feats"].to(dev, dtype)
        llm = rec["img_llm_feats"].to(dev, dtype)
        outs = ddp(img_vit_feats=vit, img_llm_feats=llm,
                   img_metas=[rec["img_metas"]])

        d = detection_loss(outs["all_cls_scores"].float(),
                           outs["all_bbox_preds"].float(),
                           [rec["gt_boxes"].to(dev)], [rec["gt_labels"].to(dev)],
                           matcher, cls_weight=cfg.det_cls_weight,
                           reg_weight=cfg.det_reg_weight)
        d.pop("det_matched")
        o = occupancy_loss(outs["occ_pred"], rec["gt_occ"].to(dev),
                           empty_label=OCC_EMPTY_LABEL,
                           focal_weight=cfg.occ_focal_weight,
                           max_points=cfg.occ_max_points)
        m = map_loss(outs["seg_preds"], rec["gt_map"].to(dev),
                     focal_weight=cfg.map_focal_weight,
                     max_points=cfg.map_max_points)
        total = sum(d.values()) + sum(o.values()) + sum(m.values())

        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(ddp.parameters(), cfg.grad_clip_norm)
        opt.step()

        # report the mean loss across ranks, not just this one
        shared = total.detach().clone()
        dist.all_reduce(shared, op=dist.ReduceOp.SUM)
        shared /= world
        if is_main:
            row = {"step": step, "loss": float(shared),
                   "lr": opt.param_groups[0]["lr"], "grad_norm": float(gn)}
            history.append(row)
            if step % 5 == 0 or step == args.steps - 1:
                print(f"  step {step:4d}  loss(mean over {world}) {float(shared):8.4f}"
                      f"   |g| {float(gn):7.1f}   "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
                      flush=True)

    dist.barrier()
    if is_main:
        dt = time.time() - t0
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        k = max(1, len(history) // 10)
        first = sum(r["loss"] for r in history[:k]) / k
        last = sum(r["loss"] for r in history[-k:]) / k
        print(f"\n{args.steps} steps in {dt:.0f}s ({dt/args.steps:.2f}s/step)"
              f"   {world} GPUs   peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        print(f"loss {first:.4f} -> {last:.4f}   ({100*(first-last)/abs(first):+.1f}%)")
        print(f"samples/s = {args.steps*world/dt:.3f}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
