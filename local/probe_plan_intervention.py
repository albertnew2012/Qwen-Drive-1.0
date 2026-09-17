#!/usr/bin/env python
"""Does the *planner* respond to the ego's lamp, or only the VQA head?

The VQA intervention showed the text answers track the ego's lamp and ignore a
non-ego lamp. That is the vision-language head talking. The trajectory is produced by
the planning expert, which never sees text - it reads the KV cache of eight attention
layers plus ego history, velocity, acceleration and nav command. Those non-visual
inputs are identical across variants here, so any change in the planned path is
attributable to the edited pixels alone.

Only the current CAM_FRONT frame is swapped; the three history frames keep the original
red lamp, which is what a genuine red-to-green transition looks like.

    .venv/bin/python local/probe_plan_intervention.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

VARIANTS = [
    ("baseline", "0882_baseline_UNEDITED.jpg", "ego=red   other=red"),
    ("A_ego", "0882_A_ego_MADE-GREEN.jpg", "ego=GREEN other=red"),
    ("B_other", "0882_B_other_MADE-GREEN.jpg", "ego=red   other=GREEN"),
    ("C_decoy", "0882_C_decoy_MADE-GREEN.jpg", "ego=red   other=red (sky)"),
    ("D_both", "0882_D_both_MADE-GREEN.jpg", "ego=GREEN other=GREEN"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/zhengzhiliu/Documents/nuscenes/trainval_root")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--scene", type=int, default=668)
    ap.add_argument("--kf", type=int, default=16, help="keyframe the variants were cut from")
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--variants", default="outputs/traffic_light/intervention")
    ap.add_argument("--out", default="outputs/traffic_light/intervention/plan_results.json")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "local")
    from nuscenes.nuscenes import NuScenes
    from qwen_drive import CameraFrame, InferenceMode, QwenDriveForPlanning
    from nuscenes_session import CAM_ORDER, build_scene_at, ego_track, sensor_chain

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    sc = nusc.scene[args.scene]
    toks, t = [], sc["first_sample_token"]
    while t:
        toks.append(t)
        t = nusc.get("sample", t)["next"]
    ts = [nusc.get("sample", k)["timestamp"] * 1e-6 for k in toks]
    t0 = float(ts[args.kf])
    print(f"  {sc['name']}  kf{args.kf}  t+{t0 - ts[0]:.1f}s")

    track = ego_track(nusc, nusc.get("sample", toks[0]))
    cam_chains = {c: sensor_chain(nusc, nusc.get("sample", toks[0]), c) for c in CAM_ORDER}
    cam_ts = {c: np.asarray([x["timestamp"] * 1e-6 for x in cam_chains[c]]) for c in CAM_ORDER}
    scene, gt = build_scene_at(nusc, cam_chains, cam_ts, track, t0, args.dataroot)
    speed = float(np.linalg.norm(scene.ego_velocity))
    print(f"  ego speed {speed * 3.6:.1f} km/h, nav_command={scene.nav_command} "
          f"(0=straight 1=left 2=right)  - identical for every variant\n")

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    vdir = Path(args.variants)
    rows = []
    for name, fname, desc in VARIANTS:
        views = {k: list(v) for k, v in scene.views.items()}
        views["<FRONT VIEW>"][-1] = CameraFrame(str(vdir / fname))   # current frame only
        variant_scene = dataclasses.replace(scene, views=views)
        with torch.no_grad():
            plan = model.run(InferenceMode.REASONING_PLANNING, scene=variant_scene,
                             num_samples=args.num_samples, seed=3407)
        traj = plan.trajectories[0]
        dist = float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum())
        end = traj[-1]
        print(f"  {name:9s} [{desc:24s}] planned {dist:5.2f} m / 5 s   "
              f"endpoint ({end[0]:6.2f}, {end[1]:6.2f})")
        print(f"            CoT: {plan.reasoning}", flush=True)
        rows.append({"variant": name, "edit": desc, "planned_m": round(dist, 2),
                     "endpoint": [round(float(x), 3) for x in end[:3]],
                     "reasoning": plan.reasoning})

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
