"""Stage 3 on both GPUs with DDP.

    bash training/run_ddp.sh training/train_planner.py ...   # single-GPU version
    bash training/run_ddp.sh training/train_planner_ddp.py --steps 200 --scratch

The planning expert is 1.04 B, and with bf16 AdamW states its persistent
footprint is ~8 GiB - it fits on one card with room to spare, so the win here is
throughput, not capacity. Each rank takes a different scene and a different flow
time, which also makes every step a larger, better-conditioned sample of the
flow-matching objective.
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

from qwen_drive import QwenDriveForPlanning

from training.config import PlannerTrainConfig
from training.losses import planning_loss


class EndpointWrapper(torch.nn.Module):
    """DDP calls ``forward``; ``PlanningExpert`` only defines ``predict_endpoint``.

    Wrapping keeps the expert untouched and gives DDP the entry point it needs.
    ``ddp.module.expert`` still reaches the real model for ``encode_history``.
    """

    def __init__(self, expert):
        super().__init__()
        self.expert = expert

    def forward(self, waypoints, flow_time, history_queries, scene_cache,
                position_anchor, nav_command, ego_status):
        return self.expert.predict_endpoint(
            waypoints, flow_time, history_queries, scene_cache,
            position_anchor, nav_command, ego_status)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--cache", default="data/train_cache_plan")
    ap.add_argument("--out", default="outputs/train_planner_ddp")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = f"cuda:{local}"
    is_main = rank == 0

    cfg = PlannerTrainConfig()
    torch.manual_seed(cfg.seed + rank)
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa")
    expert = holder.planning_expert.to(dev).train()
    del holder.vlm
    torch.cuda.empty_cache()
    if args.scratch:
        n = 0
        for m in expert.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters(); n += 1
        if is_main:
            print(f"expert RE-INITIALISED ({n} modules)", flush=True)

    ddp = DDP(EndpointWrapper(expert), device_ids=[local],
              find_unused_parameters=False)
    recs = sorted(Path(args.cache).glob("*.pt"))
    if is_main:
        print(f"world_size {world}   {len(recs)} scenes   "
              f"{sum(p.numel() for p in expert.parameters())/1e9:.4f} B/rank", flush=True)

    opt = torch.optim.AdamW(ddp.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    history, t0 = [], time.time()
    for step in range(args.steps):
        idx = 0 if args.overfit else (step * world + rank) % len(recs)
        rec = torch.load(recs[idx], weights_only=False)
        cache = [(k.to(dev), v.to(dev)) for k, v in rec["scene_cache"]]
        x1 = rec["target_normalized"].to(dev).float()
        valid = (rec["future_valid"].to(dev).reshape(1, -1)
                 if rec["future_valid"].numel() else None)

        if step < cfg.warmup_iters:
            lr = cfg.lr * (step + 1) / cfg.warmup_iters
        else:
            p = (step - cfg.warmup_iters) / max(args.steps - cfg.warmup_iters, 1)
            lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr

        t = torch.rand(x1.shape[0], device=dev)
        noise = torch.randn_like(x1)
        xt = (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * x1
        hq = ddp.module.expert.encode_history(
            rec["history"].to(dev).float(), rec["nav_command"].to(dev),
            rec["history_velocity"].to(dev).float(),
            rec["history_acceleration"].to(dev).float())
        pred = ddp(xt, t, hq, cache, rec["anchor"].to(dev),
                   rec["nav_command"].to(dev), rec["ego_status"].to(dev).float())
        parts = planning_loss(pred, x1, valid_mask=valid,
                              d1_weight=cfg.d1_weight, d2_weight=cfg.d2_weight)
        total = sum(parts.values())

        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(ddp.parameters(), cfg.grad_clip_norm)
        opt.step()

        shared = total.detach().clone()
        dist.all_reduce(shared, op=dist.ReduceOp.SUM); shared /= world
        if is_main:
            history.append({"step": step, "loss": float(shared), "lr": lr,
                            "grad_norm": float(gn)})
            if step % 20 == 0 or step == args.steps - 1:
                print(f"  step {step:4d}  loss(mean over {world}) {float(shared):9.5f}"
                      f"   |g| {float(gn):6.2f}", flush=True)

    dist.barrier()
    if is_main:
        dt = time.time() - t0
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        k = max(1, len(history) // 10)
        first = sum(r["loss"] for r in history[:k]) / k
        last = sum(r["loss"] for r in history[-k:]) / k
        print(f"\n{args.steps} steps in {dt:.0f}s ({dt/args.steps:.3f}s/step)  "
              f"{world} GPUs   peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        print(f"loss {first:.5f} -> {last:.5f}   ({100*(first-last)/abs(first):+.1f}%)")
        ok = last < first * 0.5
        print("DDP PLANNER TEST:", "PASS" if ok else "FAIL")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
