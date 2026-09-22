"""Trade planning prefill tokens against trajectory accuracy, via target_size.

v1 swept ``current_image_pixels`` / ``history_image_pixels`` and every setting gave
byte-identical ADE. The reason is in ``_patchify``: the budget applies only to
frames *without* an explicit ``target_size``, and the released benchmark metadata
carries one for every frame -- which is exactly what keeps history at ~320p and the
current frame at ~720p. So the config fields are dead on this data and the real
knob is the per-frame ``target_size``.

98% of the planning prompt is image tokens, and the prefill is ~1,418 ms of the
1,704 ms trajectory stage. Tokens go as the square of the scale factor, so a 0.707x
resize halves them.

Reported against the ground-truth future trajectory, over whichever scenes the demo
file carries. ``--scope`` chooses whether the resize hits every frame or only the
current one per view, since the current frames are ~80% of the budget.

    python local/prune/sweep_planning_pixels_v2.py --scope current
"""
from __future__ import annotations

import argparse, dataclasses, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch


def rescale(scene, factor: float, scope: str):
    """A copy of the scene with frame target sizes scaled by ``factor``."""
    from qwen_drive.scene import CameraFrame
    views = {}
    for view, frames in scene.views.items():
        out = []
        for i, f in enumerate(frames):
            is_current = (i == len(frames) - 1)
            touch = scope == "all" or (scope == "current" and is_current) \
                or (scope == "history" and not is_current)
            if touch and f.target_size is not None and factor != 1.0:
                w, h = f.target_size
                # snap to the 32-pixel merged-token grid so the resize is exact
                nw = max(64, int(round(w * factor / 32)) * 32)
                nh = max(64, int(round(h * factor / 32)) * 32)
                out.append(CameraFrame(image=f.image, target_size=(nw, nh)))
            else:
                out.append(f)
        views[view] = out
    return dataclasses.replace(scene, views=views)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--scenes-limit", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--scope", default="current", choices=["all", "current", "history"])
    ap.add_argument("--factors", default="1.0,0.85,0.707,0.6,0.5,0.4")
    ap.add_argument("--out", default="")
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
    s0 = samples[0].scene
    sizes = {v: [f.target_size for f in fr] for v, fr in list(s0.views.items())[:1]}
    print(f"  {len(samples)} scenes, {args.steps} Euler steps, scope={args.scope}")
    print(f"  frame target sizes (one view): {list(sizes.values())[0]}", flush=True)

    rows, base = {}, None
    print(f"\n  {'factor':>7s} {'ADE m':>8s} {'dADE':>8s} {'FDE m':>8s} "
          f"{'wall ms':>9s} {'saved':>8s}")
    for factor in [float(f) for f in args.factors.split(",")]:
        ades, fdes, wall = [], [], []
        for sample in samples:
            sc = rescale(sample.scene, factor, args.scope)
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                plan = model.run(InferenceMode.DIRECT_PLANNING, scene=sc, num_samples=1)
            torch.cuda.synchronize(); wall.append((time.perf_counter() - t) * 1e3)
            traj = np.asarray(plan.trajectories[0], dtype=np.float64)
            gt = sample.future_trajectory
            if gt is not None:
                m = min(len(traj), len(gt))
                d = np.linalg.norm(traj[:m, :2] - np.asarray(gt)[:m, :2], axis=-1)
                ades.append(float(d.mean())); fdes.append(float(d[-1]))
        r = {"factor": factor, "ade_m": float(np.mean(ades)),
             "fde_m": float(np.mean(fdes)), "wall_ms": float(np.median(wall))}
        if base is None:
            base = r
        rows[f"{factor}"] = r
        print(f"  {factor:7.3f} {r['ade_m']:8.4f} {r['ade_m']-base['ade_m']:+8.4f} "
              f"{r['fde_m']:8.4f} {r['wall_ms']:9.1f} {base['wall_ms']-r['wall_ms']:8.1f}")

    out = Path(args.out or f"outputs/prune/planning_pixels_v2_{args.scope}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"steps": args.steps, "scope": args.scope,
                               "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
