"""Stage 3b - fine-tune the Planning Expert so its trajectory answers to perception.

The released recipe optimises L_plan = L_fm + 2e-4 L_d1 + 2e-5 L_d2, which regresses
onto the driven path and never consults the perception heads. Nothing then forces the
trajectory and the occupancy/map predictions to agree; they merely correlate because
both read the same frozen VLM.

This adds two terms that score the planned waypoints against what perception predicted
for the same frame:

    L = L_fm + 2e-4 L_d1 + 2e-5 L_d2
        + w_col  * mean P(object occupies the cell under each waypoint)
        + w_road * mean (1 - P(driveable under each waypoint))

The VLM and the perception head stay frozen - both fields are cached constants - so only
the 1.04 B expert trains, and the gradient reaches the waypoints through ``grid_sample``.

Run the control and the grounded run with the same seed to isolate the effect:

    python training/train_planner_grounded.py --collision-weight 0 --offroad-weight 0 \
        --out outputs/plan_control
    python training/train_planner_grounded.py --out outputs/plan_grounded
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive import QwenDriveForPlanning
from qwen_drive.trajectory import denormalize_trajectory

from training.config import PlannerTrainConfig
from training.losses import (grounded_planning_loss, occupancy_collision_loss,
                             offroad_loss, smooth_field)

from qwen_drive_perception.configuration_perception import MAP_XBOUND, MAP_YBOUND

# MAP_CLASS_NAMES = (background, driveable_surface, road_line, road_edge, crosswalk, walkway).
# A car drives over painted lines and crosswalks too, so counting only class 1 scores the
# human's own driven path at ~0.19 offroad.
DRIVABLE_CLASSES = [1, 2, 4]

# Cell sizes of the two rasters, needed to express a blur radius in metres.
OCC_CELL_M = 0.4     # 200 cells over the 80 m occupancy grid
MAP_CELL_M = 0.15    # MAP_XBOUND / MAP_YBOUND stride


def load_record(path, dev, occ_sigma_m: float = 0.0, map_sigma_m: float = 0.0):
    """One cached frame. ``*_soft`` are the blurred fields the loss uses.

    The sharp rasters stay under ``occ``/``drv`` and are what the metric scores, so
    smoothing the cost can never flatter the reported numbers.
    """
    rec = torch.load(path, weights_only=False)
    occ = rec["occ_risk"].to(dev).float().unsqueeze(1)          # [1,1,200,200]
    drv = rec["map_probs"].to(dev).float()[:, DRIVABLE_CLASSES].sum(1, keepdim=True)
    return {
        "cache": [(k.to(dev, non_blocking=True), v.to(dev, non_blocking=True))
                  for k, v in rec["scene_cache"]],
        "anchor": rec["anchor"].to(dev),
        "history": rec["history"].to(dev).float(),
        "history_velocity": rec["history_velocity"].to(dev).float(),
        "history_acceleration": rec["history_acceleration"].to(dev).float(),
        "nav_command": rec["nav_command"].to(dev),
        "ego_status": rec["ego_status"].to(dev).float(),
        "x1": rec["target_normalized"].to(dev).float(),
        "valid": rec["future_valid"].to(dev).reshape(1, -1),
        "occ": occ,
        "drv": drv,
        "occ_soft": smooth_field(occ, occ_sigma_m / OCC_CELL_M),
        "drv_soft": smooth_field(drv, map_sigma_m / MAP_CELL_M),
        "pc_range": tuple(rec["pc_range"].tolist()),
        "token": rec["token"],
    }


def rollout_endpoint(expert, r, hq, noise, num_steps, min_one_minus_t):
    """The trajectory the sampler would emit, differentiable through its last step.

    This exists because applying the grounded terms to ``predict_endpoint(xt, t)``
    at a random ``t`` trains the wrong object. At high ``t`` the input is already
    mostly the target, so the prediction matches it trivially and the terms read
    zero; at low ``t`` the prediction is a guess from noise that never becomes the
    deployed trajectory. Neither is what gets driven.

    ``sample()`` is Euler on the endpoint prediction, so replaying it under
    ``no_grad`` and keeping the graph on only the final call gives exactly the
    deployed waypoints for one extra backward path rather than ``num_steps`` of
    them.
    """
    waypoints = noise.float()
    step = 1.0 / num_steps

    def advance(w, index, grad: bool):
        ft = torch.full((w.shape[0],), index * step, device=w.device, dtype=torch.float32)
        with torch.set_grad_enabled(grad):
            endpoint = expert.predict_endpoint(w, ft, hq, r["cache"], r["anchor"],
                                               r["nav_command"], r["ego_status"])
        remaining = max(1.0 - index * step, min_one_minus_t)
        return w + (endpoint - w) / remaining * step

    with torch.no_grad():
        for index in range(num_steps - 1):
            waypoints = advance(waypoints, index, grad=False)
    return advance(waypoints, num_steps - 1, grad=True)


@torch.no_grad()
def evaluate(expert, cfg_model, records, scale, dev, num_steps, max_n=None,
             col_h=None, off_h=None):
    """Sampler-based metrics on the held-out frames.

    ``collision``/``offroad`` are the raw agreement rates; ``excess_*`` hinge them
    against the driven path and are the ones to read, because the raw rates charge
    the plan for occupancy that is merely stale (see training/losses.py).
    """
    expert.eval()
    acc, n = {}, 0
    for path in (records if max_n is None else records[:max_n]):
        r = load_record(path, dev)
        gen = torch.Generator(device=dev).manual_seed(cfg_model.noise_seed)
        noise = cfg_model.noise_init_std * torch.randn(
            1, cfg_model.num_future_points, cfg_model.trajectory_point_dim,
            generator=gen, device=dev, dtype=torch.float32)
        pred = expert.sample(
            scene_cache=r["cache"], position_anchor=r["anchor"],
            history=r["history"], history_velocity=r["history_velocity"],
            history_acceleration=r["history_acceleration"],
            nav_command=r["nav_command"], ego_status=r["ego_status"],
            noise=noise, num_steps=num_steps,
            min_one_minus_t=cfg_model.min_one_minus_t)
        traj = denormalize_trajectory(pred.float(), scale)
        gt = denormalize_trajectory(r["x1"], scale)
        err = torch.linalg.norm(traj[..., :2] - gt[..., :2], dim=-1)
        row = {
            "ade_m": float(err.mean()),
            "fde_m": float(err[..., -1].mean()),
            "collision": float(occupancy_collision_loss(traj, r["occ"], r["pc_range"], horizon=col_h)),
            "offroad": float(offroad_loss(traj, r["drv"], MAP_XBOUND, MAP_YBOUND, horizon=off_h)),
            "excess_collision": float(occupancy_collision_loss(
                traj, r["occ"], r["pc_range"], horizon=col_h, reference=gt)),
            "excess_offroad": float(offroad_loss(
                traj, r["drv"], MAP_XBOUND, MAP_YBOUND, horizon=off_h, reference=gt)),
        }
        for k, v in row.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
        del r
    expert.train()
    return {**{k: v / n for k, v in acc.items()}, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--cache", default="data/train_cache_grounded")
    ap.add_argument("--out", default="outputs/train_planner_grounded")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=4, help="records per optimiser step")
    ap.add_argument("--collision-weight", type=float, default=0.007,
                    help="set so the term contributes ~40%% of imitation's median "
                         "gradient; see study/11 section 2.6")
    ap.add_argument("--offroad-weight", type=float, default=0.004)
    ap.add_argument("--collision-horizon", type=int, default=None,
                    help="waypoints the occupancy term scores; default the full 5 s plan")
    ap.add_argument("--offroad-horizon", type=int, default=None,
                    help="waypoints the map term scores; default the full 5 s plan")
    ap.add_argument("--rollout-steps", type=int, default=10,
                    help="sampler steps replayed so the grounded terms score the "
                         "deployed trajectory; must be <= 1/min_one_minus_t")
    ap.add_argument("--occ-sigma", type=float, default=2.0,
                    help="metres of Gaussian blur on the occupancy cost field (loss only)")
    ap.add_argument("--map-sigma", type=float, default=2.0,
                    help="metres of Gaussian blur on the drivable cost field (loss only)")
    ap.add_argument("--save-every-eval", action="store_true",
                    help="keep a checkpoint at every eval, so one run gives the whole trend")
    ap.add_argument("--val-scenes", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--num-steps", type=int, default=10, help="Euler steps at eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = PlannerTrainConfig()
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16, attn_implementation="sdpa")
    cfg_model = holder.config
    scale = holder.trajectory_scale(args.device)
    expert = holder.planning_expert.to(args.device).train()
    del holder.vlm
    torch.cuda.empty_cache()

    recs = sorted(Path(args.cache).glob("*.pt"))
    if not recs:
        raise FileNotFoundError("run training/cache_grounded_features.py first")
    # Split by SCENE, not by frame, so no scene appears in both halves.
    scenes = sorted({p.stem.rsplit("_", 1)[0] for p in recs})
    val_scenes = set(scenes[-args.val_scenes:])
    train = [p for p in recs if p.stem.rsplit("_", 1)[0] not in val_scenes]
    val = [p for p in recs if p.stem.rsplit("_", 1)[0] in val_scenes]
    n_par = sum(p.numel() for p in expert.parameters())
    print(f"{len(recs)} records from {len(scenes)} scenes -> "
          f"{len(train)} train / {len(val)} val   expert {n_par/1e9:.4f} B params")
    print(f"weights: collision {args.collision_weight} (H={args.collision_horizon})"
          f"  offroad {args.offroad_weight} (H={args.offroad_horizon})"
          f"   blur occ {args.occ_sigma} m / map {args.map_sigma} m"
          f"   lr {args.lr}  accum {args.accum}  steps {args.steps}")

    opt = torch.optim.AdamW(expert.parameters(), lr=args.lr,
                            weight_decay=cfg.weight_decay)
    dev = args.device
    order = np.random.default_rng(args.seed).permutation(len(train))
    grounded_on = bool(args.collision_weight or args.offroad_weight)
    # Its own stream, so the imitation half draws exactly what the control run drew.
    roll_gen = torch.Generator(device=dev).manual_seed(args.seed + 1)

    base = evaluate(expert, cfg_model, val, scale, dev, args.num_steps, args.eval_n,
                    args.collision_horizon, args.offroad_horizon)
    print(f"  BEFORE  ADE {base['ade_m']:.3f} m   excess collision {base['excess_collision']:.4f}"
          f"   excess offroad {base['excess_offroad']:.4f}   (n={base['n']})", flush=True)

    history, t0 = [{"step": -1, "eval": base}], time.time()
    for step in range(args.steps):
        if step < cfg.warmup_iters:
            lr = args.lr * (step + 1) / cfg.warmup_iters
        else:
            p = (step - cfg.warmup_iters) / max(args.steps - cfg.warmup_iters, 1)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        agg = {}
        for k in range(args.accum):
            r = load_record(train[order[(step * args.accum + k) % len(train)]], dev,
                            args.occ_sigma, args.map_sigma)
            x1 = r["x1"]
            gt_metres = denormalize_trajectory(x1, scale)
            t = torch.rand(x1.shape[0], device=dev)
            noise = torch.randn_like(x1)
            xt = (1 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * x1
            hq = expert.encode_history(r["history"], r["nav_command"],
                                       r["history_velocity"], r["history_acceleration"])
            pred = expert.predict_endpoint(xt, t, hq, r["cache"], r["anchor"],
                                           r["nav_command"], r["ego_status"])
            if grounded_on:
                rnoise = cfg_model.noise_init_std * torch.randn(
                    tuple(x1.shape), generator=roll_gen, device=dev, dtype=torch.float32)
                traj = rollout_endpoint(expert, r, hq, rnoise, args.rollout_steps,
                                        cfg_model.min_one_minus_t)
            else:
                traj = pred
            parts = grounded_planning_loss(
                pred, x1, denormalize_trajectory(traj.float(), scale),
                r["occ_soft"], r["drv_soft"], r["pc_range"], MAP_XBOUND, MAP_YBOUND,
                valid_mask=r["valid"], d1_weight=cfg.d1_weight, d2_weight=cfg.d2_weight,
                collision_weight=args.collision_weight, offroad_weight=args.offroad_weight,
                collision_horizon=args.collision_horizon,
                offroad_horizon=args.offroad_horizon,
                reference_metres=gt_metres)
            (sum(parts.values()) / args.accum).backward()
            for key, v in parts.items():
                agg[key] = agg.get(key, 0.0) + float(v.detach()) / args.accum
            del r
        gn = torch.nn.utils.clip_grad_norm_(expert.parameters(), cfg.grad_clip_norm)
        opt.step()

        row = {"step": step, "lr": lr, "grad_norm": float(gn),
               "loss": sum(agg.values()), **agg}
        history.append(row)
        if step % 25 == 0 or step == args.steps - 1:
            extra = "".join(f"  {k.replace('plan_','')} {agg[k]:.4f}"
                            for k in ("plan_collision", "plan_offroad") if k in agg)
            print(f"  step {step:4d}  loss {row['loss']:8.5f}  fm {agg['plan_fm']:8.5f}"
                  f"{extra}  |g| {float(gn):6.2f}  {(time.time()-t0)/(step+1):.2f}s/step",
                  flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            ev = evaluate(expert, cfg_model, val, scale, dev, args.num_steps, args.eval_n,
                          args.collision_horizon, args.offroad_horizon)
            history.append({"step": step, "eval": ev})
            print(f"  EVAL @{step:4d}  ADE {ev['ade_m']:.3f} m   "
                  f"excess collision {ev['excess_collision']:.4f}   "
                  f"excess offroad {ev['excess_offroad']:.4f}", flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
            if args.save_every_eval:
                torch.save({"planning_expert." + k: v for k, v in expert.state_dict().items()},
                           out / f"ckpt_step{step + 1}.pt")

    final = evaluate(expert, cfg_model, val, scale, dev, args.num_steps, args.eval_n,
                     args.collision_horizon, args.offroad_horizon)
    history.append({"step": args.steps, "eval": final, "final": True})
    (out / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"planning_expert." + k: v for k, v in expert.state_dict().items()},
               out / "planning_expert.pt")
    print(f"\n  BEFORE  ADE {base['ade_m']:.3f}  excess collision {base['excess_collision']:.4f}  "
          f"excess offroad {base['excess_offroad']:.4f}")
    print(f"  AFTER   ADE {final['ade_m']:.3f}  excess collision {final['excess_collision']:.4f}  "
          f"excess offroad {final['excess_offroad']:.4f}")
    print(f"  wrote {out}/planning_expert.pt   ({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
