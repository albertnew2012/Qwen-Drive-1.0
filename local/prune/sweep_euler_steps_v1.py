"""How many Euler steps does the trajectory actually need?

The Planning Expert integrates a flow-matching field, and the shipped setting is 10
steps costing 392 ms in the ONNX path and a comparable share in PyTorch. Each step
is a full pass over the expert, so the cost is linear in the count. Flow-matching
samplers are usually far more forgiving than that, and unlike every other lever
here this one needs no retraining and no export change -- it is one config field.

Reported against the ground-truth future trajectory, so the question is not "does it
match the 10-step answer" but "is it as good a trajectory", which is the thing that
actually matters and which a lower step count is allowed to win on.

    python local/prune/sweep_euler_steps_v1.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--scenes-limit", type=int, default=4)
    ap.add_argument("--steps", default="1,2,3,4,5,6,8,10")
    ap.add_argument("--out", default="outputs/prune/euler_steps_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning, InferenceMode
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.images import ImageArchive

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    samples = list(read_scene_file(args.scenes, image_archive=archive,
                                   num_history_points=model.config.num_history_points,
                                   limit=args.scenes_limit))
    shipped = int(model.config.num_inference_steps)
    print(f"  {len(samples)} scenes, shipped step count {shipped}", flush=True)

    steps = [int(s) for s in args.steps.split(",")]
    rows = {}
    ref = {}
    for n in sorted(steps, reverse=True):
        model.config.num_inference_steps = n
        ades, fdes, wall = [], [], []
        for si, sample in enumerate(samples):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                plan = model.run(InferenceMode.DIRECT_PLANNING, scene=sample.scene,
                                 num_samples=1)
            torch.cuda.synchronize(); wall.append((time.perf_counter() - t) * 1e3)
            traj = np.asarray(plan.trajectories[0], dtype=np.float64)
            if n == max(steps):
                ref[si] = traj
            gt = sample.future_trajectory
            if gt is not None:
                m = min(len(traj), len(gt))
                d = np.linalg.norm(traj[:m, :2] - np.asarray(gt)[:m, :2], axis=-1)
                ades.append(float(d.mean())); fdes.append(float(d[-1]))
        drift = float(np.mean([
            np.linalg.norm(ref[si][:, :2] -
                           np.asarray(np.asarray(t2, dtype=np.float64))[:, :2], axis=-1).mean()
            for si, t2 in [(i, None)] ])) if False else None
        rows[n] = {"steps": n, "ade_vs_gt_m": float(np.mean(ades)) if ades else None,
                   "fde_vs_gt_m": float(np.mean(fdes)) if fdes else None,
                   "wall_ms": float(np.median(wall))}
        print(f"    steps {n:2d}   ADE {rows[n]['ade_vs_gt_m']:.4f} m   "
              f"FDE {rows[n]['fde_vs_gt_m']:.4f} m   {rows[n]['wall_ms']:7.1f} ms",
              flush=True)

    base = rows[max(steps)]
    print(f"\n  {'steps':>5s} {'ADE m':>8s} {'dADE':>8s} {'wall ms':>9s} {'saved ms':>9s}")
    for n in sorted(rows):
        r = rows[n]
        print(f"  {n:5d} {r['ade_vs_gt_m']:8.4f} "
              f"{r['ade_vs_gt_m']-base['ade_vs_gt_m']:+8.4f} {r['wall_ms']:9.1f} "
              f"{base['wall_ms']-r['wall_ms']:9.1f}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"shipped_steps": shipped, "rows": rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
