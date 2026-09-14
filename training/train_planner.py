"""Stage 3 - train the Planning Expert with flow matching, VLM frozen.

    L_plan = L_fm + 2e-4 * L_d1 + 2e-5 * L_d2               (arXiv:2609.00111)

The expert uses a CLEAN-ENDPOINT parameterisation: ``predict_endpoint`` returns
x1 directly, and ``sample()`` turns that into a velocity by dividing by the
remaining time. So training is a regression onto the ground-truth waypoints at
a random point along the noise->trajectory path:

    t ~ U(0, 1)
    x_t = (1 - t) * noise + t * x1
    loss = || predict_endpoint(x_t, t) - x1 ||^2  + smoothness terms

    python training/cache_planner_features.py
    python training/train_planner.py --steps 200 --overfit
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive import QwenDriveForPlanning

from training.config import PlannerTrainConfig
from training.losses import planning_loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--cache", default="data/train_cache_plan")
    ap.add_argument("--out", default="outputs/train_planner")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--scratch", action="store_true",
                    help="re-initialise the expert. REQUIRED for a meaningful overfit "
                         "test: the released planner-sft already fits the demo scenes "
                         "to ~4e-5 MSE, so from those weights there is nothing to learn.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = PlannerTrainConfig()
    if args.lr is not None:
        cfg.lr = args.lr
    torch.manual_seed(cfg.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Load ONLY the expert - the VLM's contribution is the cached scene attention.
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16, attn_implementation="sdpa")
    expert = holder.planning_expert.to(args.device).train()
    del holder.vlm
    torch.cuda.empty_cache()

    if args.scratch:
        n_reset = 0
        for m in expert.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters(); n_reset += 1
        print(f"expert RE-INITIALISED ({n_reset} modules) - loss must now start high")

    recs = sorted(Path(args.cache).glob("*.pt"))
    if not recs:
        raise FileNotFoundError("run training/cache_planner_features.py first")
    n = sum(p.numel() for p in expert.parameters())
    print(f"{len(recs)} cached scenes   planning expert {n/1e9:.4f} B parameters")

    opt = torch.optim.AdamW(expert.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    dev = args.device
    history, t0 = [], time.time()
    for step in range(args.steps):
        rec = torch.load(recs[0] if args.overfit else recs[step % len(recs)],
                         weights_only=False)
        cache = [(k.to(dev), v.to(dev)) for k, v in rec["scene_cache"]]
        x1 = rec["target_normalized"].to(dev).float()
        valid = rec["future_valid"].to(dev).reshape(1, -1) if rec["future_valid"].numel() else None

        # warmup then cosine
        if step < cfg.warmup_iters:
            lr = cfg.lr * (step + 1) / cfg.warmup_iters
        else:
            p = (step - cfg.warmup_iters) / max(args.steps - cfg.warmup_iters, 1)
            lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr

        # flow-matching path: x_t between noise and the ground-truth trajectory
        t = torch.rand(x1.shape[0], device=dev)
        noise = torch.randn_like(x1)
        xt = (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * x1

        hq = expert.encode_history(
            rec["history"].to(dev).float(), rec["nav_command"].to(dev),
            rec["history_velocity"].to(dev).float(),
            rec["history_acceleration"].to(dev).float())
        pred = expert.predict_endpoint(
            xt, t, hq, cache, rec["anchor"].to(dev),
            rec["nav_command"].to(dev), rec["ego_status"].to(dev).float())

        parts = planning_loss(pred, x1, valid_mask=valid,
                              d1_weight=cfg.d1_weight, d2_weight=cfg.d2_weight)
        total = sum(parts.values())

        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(expert.parameters(), cfg.grad_clip_norm)
        opt.step()

        row = {"step": step, "loss": float(total.detach()), "lr": lr,
               "grad_norm": float(gn), "t": float(t.mean()),
               **{k: float(v.detach()) for k, v in parts.items()}}
        history.append(row)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {row['loss']:9.5f}   fm {row['plan_fm']:8.5f}"
                  f"   d1 {row['plan_d1']:.2e}  d2 {row['plan_d2']:.2e}"
                  f"   t {row['t']:.2f}  |g| {row['grad_norm']:7.2f}")

    dt = time.time() - t0
    (out / "history.json").write_text(json.dumps(history, indent=2))
    # compare the first and last 10% rather than single steps: t is random, so a
    # single step's loss is noisy by construction
    k = max(1, args.steps // 10)
    first = sum(r["loss"] for r in history[:k]) / k
    last = sum(r["loss"] for r in history[-k:]) / k
    print(f"\n{args.steps} steps in {dt:.0f}s ({dt/args.steps:.2f}s/step)"
          f"   peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    print(f"mean loss over first {k} steps {first:.5f} -> last {k} steps {last:.5f}"
          f"   ({100*(first-last)/abs(first):+.1f}%)")
    if args.overfit:
        ok = last < first * 0.5
        print("OVERFIT TEST:", "PASS - the loss collapsed" if ok
              else "FAIL - loss did not halve")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
