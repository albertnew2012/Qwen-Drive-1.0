#!/usr/bin/env python
"""Why does the planner move on a cross-traffic green? Change-detection, or any green?

In the first planning intervention only the current frame was edited, so the cross-
traffic lamp went red -> green across the history window and the model saw a transition
event. This repeats it with the lamp edited in all four FRONT frames, so the green is
simply present and never changes.

    persistent B still drives  -> the planner reacts to any green, wherever it is
    persistent B now stops     -> the trigger was the temporal change, not the lamp
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EGO_LAMP, EGO_R = (236, 101), 20
OTHER_LAMP, OTHER_R = (1021, 224), 9


def recolour(img, centre, radius, to="green"):
    a = np.asarray(img.convert("RGB")).astype(float).copy()
    cx, cy = centre
    y0, y1 = max(0, cy - radius), min(a.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(a.shape[1], cx + radius + 1)
    patch = a[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    lum = patch.max(axis=2, keepdims=True)
    falloff = np.clip(1.0 - d / radius, 0, 1)[..., None] ** 0.6
    tint = {"red": np.array([1.0, 0.13, 0.10]), "green": np.array([0.16, 1.0, 0.45])}[to]
    a[y0:y1, x0:x1] = patch * (1 - falloff) + (lum * tint) * falloff
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/zhengzhiliu/Documents/nuscenes/trainval_root")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--scene", type=int, default=668)
    ap.add_argument("--kf", type=int, default=16)
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--out", default="outputs/traffic_light/intervention/plan_persistent.json")
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
        toks.append(t); t = nusc.get("sample", t)["next"]
    ts = [nusc.get("sample", k)["timestamp"] * 1e-6 for k in toks]
    t0 = float(ts[args.kf])
    track = ego_track(nusc, nusc.get("sample", toks[0]))
    cam_chains = {c: sensor_chain(nusc, nusc.get("sample", toks[0]), c) for c in CAM_ORDER}
    cam_ts = {c: np.asarray([x["timestamp"] * 1e-6 for x in cam_chains[c]]) for c in CAM_ORDER}
    scene, _ = build_scene_at(nusc, cam_chains, cam_ts, track, t0, args.dataroot)

    front = list(scene.views["<FRONT VIEW>"])
    print(f"  FRONT view has {len(front)} frames (-1.5s, -1.0s, -0.5s, current)")
    tmp = Path(tempfile.mkdtemp(prefix="persist_"))

    def edited_front(which, persist):
        """which: 'ego'|'other'|None. persist: edit all frames, else only the current."""
        out = []
        for i, cf in enumerate(front):
            is_cur = i == len(front) - 1
            if which is None or (not persist and not is_cur):
                out.append(cf); continue
            c, r = (EGO_LAMP, EGO_R) if which == "ego" else (OTHER_LAMP, OTHER_R)
            im = recolour(cf.load(), c, r, "green")
            p = tmp / f"{which}_{persist}_{i}.jpg"
            im.save(p, quality=96)
            out.append(CameraFrame(str(p)))
        return out

    cases = [
        ("baseline", None, False, "no edit"),
        ("A_ego_current", "ego", False, "ego lamp green in CURRENT frame only"),
        ("A_ego_persist", "ego", True, "ego lamp green in ALL 4 frames"),
        ("B_other_current", "other", False, "cross lamp green in CURRENT frame only"),
        ("B_other_persist", "other", True, "cross lamp green in ALL 4 frames"),
    ]

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    rows = []
    for name, which, persist, desc in cases:
        views = {k: list(v) for k, v in scene.views.items()}
        views["<FRONT VIEW>"] = edited_front(which, persist)
        with torch.no_grad():
            plan = model.run(InferenceMode.REASONING_PLANNING,
                             scene=dataclasses.replace(scene, views=views),
                             num_samples=args.num_samples, seed=3407)
        traj = plan.trajectories[0]
        dist = float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum())
        verdict = "DRIVES" if dist > 4.0 else "stays"
        print(f"  {name:17s} {desc:38s} {dist:5.2f} m  {verdict}")
        print(f"        CoT: {plan.reasoning}", flush=True)
        rows.append({"case": name, "desc": desc, "planned_m": round(dist, 2),
                     "reasoning": plan.reasoning})

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
