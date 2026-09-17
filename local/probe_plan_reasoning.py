#!/usr/bin/env python
"""Does the PLANNING chain-of-thought bind the traffic light to the ego's own action?

The VQA probe shows the model can read a light. This asks the harder question: when
the planning expert produces a trajectory, does the reasoning it is conditioned on
mention the light, and does the trajectory agree with it?

Runs REASONING_PLANNING at chosen keyframes of a scene and prints, side by side, the
ego's true speed, how far the model plans to travel in 5 s, and the rationale. At a
red light with the ego stopped the planned path should be near zero.

    .venv/bin/python local/probe_plan_reasoning.py --scene 4 --frames 6,12,20,26,32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nuscenes_session import CAM_ORDER, build_scene_at, ego_track, sensor_chain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", type=int, default=4, help="index into nusc.scene")
    ap.add_argument("--frames", default="", help="comma list of keyframe indices")
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from qwen_drive import InferenceMode, QwenDriveForPlanning

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    samples, tok = [], scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        samples.append(s)
        tok = s["next"]
    print(f"scene {args.scene}: {scene['name']}  {len(samples)} keyframes")
    print(f"  {scene['description']}\n")

    sample_ts = np.asarray([s["timestamp"] * 1e-6 for s in samples])
    track = ego_track(nusc, samples[0])
    cam_chains = {c: sensor_chain(nusc, samples[0], c) for c in CAM_ORDER}
    cam_ts = {c: np.asarray([x["timestamp"] * 1e-6 for x in cam_chains[c]]) for c in CAM_ORDER}

    # a scene needs 1.5 s of history behind it and 5 s of future ahead
    lo = int(np.searchsorted(sample_ts, sample_ts[0] + 1.5)) + 1
    hi = int(np.searchsorted(sample_ts, sample_ts[-1] - 5.0)) - 1
    picks = ([int(x) for x in args.frames.split(",")] if args.frames
             else list(range(lo, hi, max(1, (hi - lo) // 8))))
    picks = [i for i in picks if lo <= i <= hi]
    print(f"  usable keyframes {lo}..{hi}; probing {picks}\n")

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    rows = []
    for i in picks:
        t = float(sample_ts[i])
        sc, gt_fut = build_scene_at(nusc, cam_chains, cam_ts, track, t, args.dataroot)
        with torch.no_grad():
            plan = model.run(InferenceMode.REASONING_PLANNING, scene=sc,
                             num_samples=args.num_samples)
        traj = plan.trajectories[0]
        planned = float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum())
        actual = float(np.linalg.norm(np.diff(gt_fut[:, :2], axis=0), axis=1).sum())
        speed = float(np.linalg.norm(sc.ego_velocity)) * 3.6
        print(f"keyframe {i:2d}  speed {speed:5.1f} km/h  "
              f"planned {planned:5.1f} m / 5 s   actual {actual:5.1f} m")
        print(f"            {plan.reasoning}\n", flush=True)
        rows.append({"keyframe": i, "speed_kmh": round(speed, 1),
                     "planned_m": round(planned, 1), "actual_m": round(actual, 1),
                     "reasoning": plan.reasoning})

    out = Path(args.out or f"outputs/traffic_light/plan_scene{args.scene}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scene": scene["name"],
                               "description": scene["description"],
                               "frames": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
