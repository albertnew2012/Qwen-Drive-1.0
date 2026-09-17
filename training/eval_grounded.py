"""Score several planning experts on the same held-out frames with the same metrics.

Training prints its own numbers, but a control run and a grounded run only compare
if both are measured the same way, on the same frames, with the same sampler seed.
This does that in one process, and adds the row that makes the numbers readable:
the human-driven path's own score, which is the floor the metrics can reach.

    python training/eval_grounded.py \
        --ckpt sft=pretrained \
        --ckpt control=outputs/plan_control/planning_expert.pt \
        --ckpt grounded=outputs/plan_grounded/planning_expert.pt
"""
from __future__ import annotations

import argparse, json, os, random, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive import QwenDriveForPlanning
from qwen_drive.trajectory import denormalize_trajectory

from training.losses import occupancy_collision_loss, offroad_loss
from training.train_planner_grounded import load_record

from qwen_drive_perception.configuration_perception import MAP_XBOUND, MAP_YBOUND


def metrics(traj, r, scale, col_h, off_h):
    gt = denormalize_trajectory(r["x1"], scale)
    err = torch.linalg.norm(traj[..., :2] - gt[..., :2], dim=-1)
    return {
        "ade_m": float(err.mean()),
        "fde_m": float(err[..., -1].mean()),
        "collision": float(occupancy_collision_loss(traj, r["occ"], r["pc_range"], horizon=col_h)),
        "offroad": float(offroad_loss(traj, r["drv"], MAP_XBOUND, MAP_YBOUND, horizon=off_h)),
        "excess_collision": float(occupancy_collision_loss(
            traj, r["occ"], r["pc_range"], horizon=col_h, reference=gt)),
        "excess_offroad": float(offroad_loss(
            traj, r["drv"], MAP_XBOUND, MAP_YBOUND, horizon=off_h, reference=gt)),
    }


def paired_ci(a: list[float], b: list[float], rounds: int = 20000, seed: int = 0):
    """95%% bootstrap interval on mean(a) - mean(b), resampling *frames* jointly.

    Pairing matters: the frames differ enormously in difficulty, and that variance
    is common to both models. Comparing unpaired means buries a real 3%% effect
    under it.
    """
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        means.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (sum(d) / n, means[int(0.025 * rounds)], means[int(0.975 * rounds)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--cache", default="data/train_cache_grounded")
    ap.add_argument("--ckpt", action="append", default=[],
                    help="NAME=PATH, or NAME=pretrained for the released planner-sft")
    ap.add_argument("--val-scenes", type=int, default=80)
    ap.add_argument("--split", choices=("val", "train"), default="val",
                    help="'train' scores frames the runs were fitted on, which "
                         "separates 'the loss does nothing' from 'it does not generalise'")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N frames of the split")
    ap.add_argument("--collision-horizon", type=int, default=None)
    ap.add_argument("--offroad-horizon", type=int, default=None)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--out", default="outputs/plan_eval.json")
    ap.add_argument("--paired-against", default=None,
                    help="model NAME to report paired bootstrap deltas against")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    recs = sorted(Path(args.cache).glob("*.pt"))
    scenes = sorted({p.stem.rsplit("_", 1)[0] for p in recs})
    val_scenes = set(scenes[-args.val_scenes:])
    keep = val_scenes if args.split == "val" else set(scenes) - val_scenes
    val = [p for p in recs if p.stem.rsplit("_", 1)[0] in keep]
    if args.limit:
        val = val[: args.limit]
    print(f"{len(val)} {args.split} frames from {len(keep)} scenes")
    print("excess-col / excess-off are hinged against the driven path and are the")
    print("primary numbers; the raw rates charge a plan for stale occupancy too.\n")

    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16, attn_implementation="sdpa")
    cfg_model = holder.config
    scale = holder.trajectory_scale(args.device)
    expert = holder.planning_expert.to(args.device).eval()
    del holder.vlm
    torch.cuda.empty_cache()
    pretrained = {k: v.detach().clone() for k, v in expert.state_dict().items()}

    dev = args.device
    results: dict[str, dict] = {}
    per_frame: dict[str, dict[str, list[float]]] = {}

    # The driven path itself: what the two agreement metrics read when the plan is
    # by construction correct. Any model score must be judged against this, not zero.
    acc: dict[str, list[float]] = {}
    for path in val:
        r = load_record(path, dev)
        m = metrics(denormalize_trajectory(r["x1"], scale), r, scale,
                    args.collision_horizon, args.offroad_horizon)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
        del r
    per_frame["human (driven path)"] = acc
    results["human (driven path)"] = {k: sum(v) / len(v) for k, v in acc.items()}

    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        if path == "pretrained":
            expert.load_state_dict(pretrained)
        else:
            sd = torch.load(path, map_location=dev, weights_only=False)
            sd = {k.removeprefix("planning_expert."): v for k, v in sd.items()}
            expert.load_state_dict(sd)
        expert.eval()

        acc, n = {}, 0
        with torch.no_grad():
            for p in val:
                r = load_record(p, dev)
                gen = torch.Generator(device=dev).manual_seed(cfg_model.noise_seed)
                noise = cfg_model.noise_init_std * torch.randn(
                    1, cfg_model.num_future_points, cfg_model.trajectory_point_dim,
                    generator=gen, device=dev, dtype=torch.float32)
                pred = expert.sample(
                    scene_cache=r["cache"], position_anchor=r["anchor"],
                    history=r["history"], history_velocity=r["history_velocity"],
                    history_acceleration=r["history_acceleration"],
                    nav_command=r["nav_command"], ego_status=r["ego_status"],
                    noise=noise, num_steps=args.num_steps,
                    min_one_minus_t=cfg_model.min_one_minus_t)
                m = metrics(denormalize_trajectory(pred.float(), scale), r, scale,
                            args.collision_horizon, args.offroad_horizon)
                for k, v in m.items():
                    acc.setdefault(k, []).append(v)
                n += 1
                del r
        per_frame[name] = acc
        results[name] = {k: sum(v) / len(v) for k, v in acc.items()}
        print(f"  {name:24s} done ({n} frames)", flush=True)

    print(f"\n  {'model':24s} {'ADE (m)':>9s} {'FDE (m)':>9s} {'excess-col':>11s} "
          f"{'excess-off':>11s} {'collision':>10s} {'offroad':>9s}")
    print("  " + "-" * 88)
    for name, m in results.items():
        print(f"  {name:24s} {m['ade_m']:9.3f} {m['fde_m']:9.3f} "
              f"{m['excess_collision']:11.4f} {m['excess_offroad']:11.4f} "
              f"{m['collision']:10.4f} {m['offroad']:9.4f}")

    if args.paired_against and args.paired_against in per_frame:
        base = per_frame[args.paired_against]
        print(f"\n  paired bootstrap, 95% CI on (model - {args.paired_against}), "
              f"n={len(val)} frames")
        print(f"  a CI that straddles 0 means the difference is not resolved here\n")
        print(f"  {'model':24s} {'d ADE (m)':>24s} {'d excess-col':>26s}")
        print("  " + "-" * 76)
        for name, m in per_frame.items():
            if name == args.paired_against or name == "human (driven path)":
                continue
            da, la, ha = paired_ci(m["ade_m"], base["ade_m"])
            dc, lc, hc = paired_ci(m["excess_collision"], base["excess_collision"])
            print(f"  {name:24s} {da:+8.3f} [{la:+.3f},{ha:+.3f}] "
                  f"{dc:+9.5f} [{lc:+.5f},{hc:+.5f}]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"frames": len(val), "collision_horizon": args.collision_horizon,
         "offroad_horizon": args.offroad_horizon, "results": results,
         "per_frame": per_frame}, indent=2))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
