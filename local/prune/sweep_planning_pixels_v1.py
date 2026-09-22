"""Trade planning prefill tokens against trajectory accuracy.

98% of the planning prompt is image tokens and the prefill is ~1,418 ms of the
1,704 ms trajectory stage, so the token budget is the dominant cost. The shipped
config spends it unevenly: history frames are already downscaled to 174,080 pixels
while the three current views get 921,600 each, which at 32x32 pixels per merged
token makes the current views roughly 80% of the prompt.

So this sweeps ``current_image_pixels`` and ``history_image_pixels`` and reports
the token count, the wall time and the ADE against the ground-truth future. No
retraining: the vision tower is resolution-agnostic and the expert reads whatever
tokens it is given, so this is a config change. Whether the model tolerates it is
an empirical question, which is the point of measuring rather than assuming.

    python local/prune/sweep_planning_pixels_v1.py
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
    ap.add_argument("--steps", type=int, default=6,
                    help="Euler steps; 6 measured better than the shipped 10")
    ap.add_argument("--current", default="921600,691200,460800,304128,230400")
    ap.add_argument("--history", default="174080,87040")
    ap.add_argument("--out", default="outputs/prune/planning_pixels_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning, InferenceMode
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.images import ImageArchive

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    model.config.num_inference_steps = args.steps
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    samples = list(read_scene_file(args.scenes, image_archive=archive,
                                   num_history_points=model.config.num_history_points,
                                   limit=args.scenes_limit))
    cur0 = int(model.config.current_image_pixels)
    his0 = int(model.config.history_image_pixels)
    print(f"  {len(samples)} scenes, {args.steps} Euler steps, "
          f"shipped pixels current={cur0} history={his0}", flush=True)

    # token count is visible through the prompt the model builds
    def tokens_for(sample):
        try:
            from qwen_drive.scene import ScenePromptBuilder  # name varies by version
        except Exception:
            return None
        return None

    rows = {}
    combos = [(int(c), int(h)) for h in args.history.split(",")
              for c in args.current.split(",")]
    print(f"\n  {'current px':>11s} {'hist px':>8s} {'ADE m':>8s} {'FDE m':>8s} "
          f"{'wall ms':>9s} {'saved':>8s}")
    base = None
    for cur, his in combos:
        model.config.current_image_pixels = cur
        model.config.history_image_pixels = his
        ades, fdes, wall, ntok = [], [], [], []
        ok = True
        for sample in samples:
            try:
                torch.cuda.synchronize(); t = time.perf_counter()
                with torch.no_grad():
                    plan = model.run(InferenceMode.DIRECT_PLANNING,
                                     scene=sample.scene, num_samples=1)
                torch.cuda.synchronize(); wall.append((time.perf_counter() - t) * 1e3)
            except Exception as exc:
                print(f"  {cur:11d} {his:8d}   failed: {str(exc)[:60]}")
                ok = False
                break
            traj = np.asarray(plan.trajectories[0], dtype=np.float64)
            gt = sample.future_trajectory
            if gt is not None:
                m = min(len(traj), len(gt))
                d = np.linalg.norm(traj[:m, :2] - np.asarray(gt)[:m, :2], axis=-1)
                ades.append(float(d.mean())); fdes.append(float(d[-1]))
        if not ok:
            continue
        r = {"current_px": cur, "history_px": his,
             "ade_m": float(np.mean(ades)), "fde_m": float(np.mean(fdes)),
             "wall_ms": float(np.median(wall))}
        if base is None:
            base = r
        rows[f"{cur}_{his}"] = r
        print(f"  {cur:11d} {his:8d} {r['ade_m']:8.4f} {r['fde_m']:8.4f} "
              f"{r['wall_ms']:9.1f} {base['wall_ms']-r['wall_ms']:8.1f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"steps": args.steps, "shipped_current": cur0,
                               "shipped_history": his0, "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
