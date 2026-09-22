"""Bring the trajectory stage toward 3 Hz, the way perception was brought there.

Perception now runs at 322 ms (3.10 Hz) with a forward camera, a bit-exact voxel
crop and 20 of 32 decoder layers skipped. Planning is 1,546 ms (0.65 Hz) and is
therefore the binding constraint on a whole frame.

Planning already uses only forward views -- ``CAMERA_VIEWS`` is FRONT, FRONT LEFT,
FRONT RIGHT -- and ``DrivingScene`` refuses a scene missing any of them, so the
side views cannot simply be dropped. They can be shrunk: tokens go as the square of
``target_size``, so taking the two side views to a quarter scale removes most of
their cost while leaving the prompt structurally intact.

Levers swept here:

  side        resize applied to the two side views only, front left alone
  skip        decoder layers passed through, least-influential first
  steps       Euler steps; 6 was measured better than the shipped 10

Quality is ADE against the ground-truth future, so a configuration is allowed to
win rather than merely lose slowly.

    python local/prune/optimise_planning_v1.py
"""
from __future__ import annotations

import argparse, dataclasses, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

RANK_LEAST_FIRST = [13, 12, 16, 17, 4, 9, 15, 14, 20, 3, 8, 1, 21, 2, 18, 10,
                    25, 11, 29, 24, 5, 28, 30, 22, 26, 7, 6, 23, 19, 27, 31, 0]


def shrink_side_views(scene, factor: float):
    """Scale the two side views' target sizes, leaving the forward view alone."""
    from qwen_drive.scene import CameraFrame, CAMERA_VIEWS
    if factor == 1.0:
        return scene
    keep = CAMERA_VIEWS[0]
    views = {}
    for view, frames in scene.views.items():
        if view == keep:
            views[view] = list(frames)
            continue
        out = []
        for f in frames:
            if f.target_size is None:
                out.append(f); continue
            w, h = f.target_size
            nw = max(32, int(round(w * factor / 32)) * 32)
            nh = max(32, int(round(h * factor / 32)) * 32)
            out.append(CameraFrame(image=f.image, target_size=(nw, nh)))
        views[view] = out
    return dataclasses.replace(scene, views=views)


def prune_layers(lm, skip):
    """Drop layers from the stack, keeping the config consistent with it.

    The model's loop is ``for i, layer in enumerate(self.layers[:num_hidden_layers])``
    and it looks up ``config.layer_types[i]`` *by position* to choose the attention
    mask and the cache slot. Removing layers without rewriting that list hands a
    linear-attention block a full-attention cache entry, which fails with
    ``'LinearAttentionLayer' object has no attribute 'update'``. So the type list and
    the layer count move together with the ModuleList, and each block's ``layer_idx``
    is renumbered so the cache stays dense.
    """
    if not skip:
        return lambda: None
    skip = set(skip)
    original_layers = list(lm.layers)
    cfg = lm.config
    original_types = list(cfg.layer_types)
    original_n = cfg.num_hidden_layers
    original_idx = [(m, m.layer_idx)
                    for layer in original_layers
                    for attr in ("self_attn", "linear_attn")
                    for m in [getattr(layer, attr, None)]
                    if m is not None and hasattr(m, "layer_idx")]

    kept, kept_types = [], []
    for i, layer in enumerate(original_layers):
        if i not in skip:
            kept.append(layer)
            kept_types.append(original_types[i])
    for new_i, layer in enumerate(kept):
        for attr in ("self_attn", "linear_attn"):
            mod = getattr(layer, attr, None)
            if mod is not None and hasattr(mod, "layer_idx"):
                mod.layer_idx = new_i
    lm.layers = nn.ModuleList(kept)
    cfg.layer_types = kept_types
    cfg.num_hidden_layers = len(kept)

    def restore():
        lm.layers = nn.ModuleList(original_layers)
        cfg.layer_types = original_types
        cfg.num_hidden_layers = original_n
        for mod, idx in original_idx:
            mod.layer_idx = idx
    return restore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--scenes-limit", type=int, default=4)
    ap.add_argument("--sides", default="1.0,0.5,0.25")
    ap.add_argument("--skips", default="0,16,20")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", default="outputs/prune/planning_optimised_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning, InferenceMode
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.images import ImageArchive

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    model.config.num_inference_steps = args.steps
    lm = model.vlm.model.language_model
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    samples = list(read_scene_file(args.scenes, image_archive=archive,
                                   num_history_points=model.config.num_history_points,
                                   limit=args.scenes_limit))
    # The Planning Expert cross-attends to scene_k_0..7 / scene_v_0..7, i.e. the
    # caches of all eight full-attention layers. Removing one leaves the expert
    # asking for a cache entry that no longer exists ("list index out of range"),
    # so only the 24 linear-attention blocks are available to prune here. Perception
    # has no such constraint: it reads the final hidden state, not the cache.
    types = list(lm.config.layer_types)
    prunable = [i for i in RANK_LEAST_FIRST if types[i] == "linear_attention"]
    print(f"  {len(samples)} scenes, {args.steps} Euler steps, {len(lm.layers)} layers "
          f"({len(prunable)} linear-attention prunable, "
          f"{len(types)-len(prunable)} full-attention required by the expert)\n",
          flush=True)
    print(f"  {'side':>5s} {'skip':>5s} {'wall ms':>8s} {'Hz':>6s} {'ADE m':>8s} "
          f"{'FDE m':>8s} {'dADE':>8s}")
    rows, base_ade = {}, None
    for side in [float(s) for s in args.sides.split(",")]:
        for nskip in [int(s) for s in args.skips.split(",")]:
            restore = prune_layers(lm, prunable[:nskip])
            try:
                ades, fdes, wall = [], [], []
                for sample in samples:
                    sc = shrink_side_views(sample.scene, side)
                    torch.cuda.synchronize(); t = time.perf_counter()
                    with torch.no_grad():
                        plan = model.run(InferenceMode.DIRECT_PLANNING, scene=sc,
                                         num_samples=1)
                    torch.cuda.synchronize()
                    wall.append((time.perf_counter() - t) * 1e3)
                    traj = np.asarray(plan.trajectories[0], dtype=np.float64)
                    gt = sample.future_trajectory
                    if gt is not None:
                        m = min(len(traj), len(gt))
                        d = np.linalg.norm(traj[:m, :2] - np.asarray(gt)[:m, :2], axis=-1)
                        ades.append(float(d.mean())); fdes.append(float(d[-1]))
                ms = float(np.median(wall))
                ade = float(np.mean(ades)); fde = float(np.mean(fdes))
                if base_ade is None:
                    base_ade = ade
                key = f"side{side}_skip{nskip}"
                rows[key] = {"side": side, "skip": nskip, "wall_ms": ms,
                             "hz": 1000.0 / ms, "ade_m": ade, "fde_m": fde}
                flag = "  <- under 333 ms" if ms < 333 else ""
                print(f"  {side:5.2f} {nskip:5d} {ms:8.0f} {1000.0/ms:6.2f} "
                      f"{ade:8.4f} {fde:8.4f} {ade-base_ade:+8.4f}{flag}", flush=True)
            except Exception as exc:
                print(f"  {side:5.2f} {nskip:5d}   failed: {str(exc)[:70]}", flush=True)
            finally:
                restore()
            torch.cuda.empty_cache()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"steps": args.steps, "rows": rows}, indent=1))
    if rows:
        best = min(rows.values(), key=lambda r: r["wall_ms"])
        print(f"\n  fastest planning {best['wall_ms']:.0f} ms = {best['hz']:.2f} Hz "
              f"(ADE {best['ade_m']:.4f} m)")
        print(f"  with perception at 322 ms, a frame on two cards is "
              f"{max(322, best['wall_ms']):.0f} ms = "
              f"{1000.0/max(322, best['wall_ms']):.2f} Hz")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
