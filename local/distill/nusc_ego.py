"""Ego history and future trajectory per keyframe, in the ego frame.

The planning head needs a target. Distilling the teacher's trajectory would mean
converting nuScenes into the 40-field WOD_E2E scene format the planner consumes, which
is a large detour -- and unnecessary, because for planning the ground truth *is* the
objective. The teacher's own ADE against it is 0.335 m, so a student trained directly on
it is aiming at the same thing the teacher was.

``ego_pose`` has a row per ``sample_data``, i.e. ~50 Hz rather than the 2 Hz of keyframes,
so the 10 Hz trajectory the model emits is interpolated from real poses rather than from
keyframes. Everything is expressed relative to the pose at the keyframe, which is what
makes it a prediction rather than a global path.

    python local/distill/nusc_ego.py --version v1.0-mini
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from local.distill.nusc_frames import quat_to_R

HORIZON_S = 5.0
HZ = 10.0
N_FUTURE = 50
N_HISTORY = 16
HIST_S = 1.5


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="data/distill/ego")
    args = ap.parse_args()
    os.chdir(_ROOT)

    meta = Path(args.root) / args.version
    if not meta.is_dir():
        print(f"  no metadata at {meta}")
        return 1
    sample = json.loads((meta / "sample.json").read_text())
    sdata = json.loads((meta / "sample_data.json").read_text())
    epose = {r["token"]: r for r in json.loads((meta / "ego_pose.json").read_text())}

    # every pose with a timestamp, grouped by scene through the sample it belongs to
    samp_scene = {s["token"]: s["scene_token"] for s in sample}
    per_scene: dict[str, list] = {}
    for r in sdata:
        sc = samp_scene.get(r["sample_token"])
        if sc is None:
            continue
        p = epose.get(r["ego_pose_token"])
        if p is None:
            continue
        per_scene.setdefault(sc, []).append((p["timestamp"], p))
    for sc in per_scene:
        seen = {}
        for ts, p in per_scene[sc]:
            seen[ts] = p
        per_scene[sc] = sorted(seen.items())

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    made = short = skip = 0
    have = {p.stem for p in out.glob("*.npz")}
    # Per-scene pose tracks, built once. Building them inside the sample loop meant
    # 34k samples x ~2k poses of quaternion conversion -- about 68M -- for data that is
    # identical for every sample in a scene.
    tracks = {}
    for sc, track in per_scene.items():
        times = np.array([t for t, _ in track], dtype=np.float64)
        xy = np.array([[q["translation"][0], q["translation"][1]] for _, q in track])
        # unwrap before interpolating or a +-pi crossing invents a spin
        yaw = np.unwrap(np.array([yaw_of(quat_to_R(q["rotation"])) for _, q in track]))
        tracks[sc] = (times, xy, yaw)
    print(f"  {len(tracks)} scene pose tracks built", flush=True)

    for s in sample:
        if s["token"] in have:
            skip += 1
            continue
        sc = s["scene_token"]
        if sc not in tracks:
            continue
        times, xy, yaw = tracks[sc]
        t0 = s["timestamp"]

        def at(ts_us):
            ts_us = np.clip(ts_us, times[0], times[-1])
            x = np.interp(ts_us, times, xy[:, 0])
            y = np.interp(ts_us, times, xy[:, 1])
            h = np.interp(ts_us, times, yaw)
            return np.stack([x, y, h], axis=-1)

        fut_t = t0 + (np.arange(1, N_FUTURE + 1) / HZ) * 1e6
        his_t = t0 - (np.arange(N_HISTORY)[::-1] / (N_HISTORY / HIST_S)) * 1e6
        if fut_t[-1] > times[-1] + 1e5:
            short += 1
            continue
        here = at(np.array([float(t0)]))[0]
        R = np.array([[np.cos(-here[2]), -np.sin(-here[2])],
                      [np.sin(-here[2]),  np.cos(-here[2])]])

        def to_ego(pts):
            d = pts[:, :2] - here[:2]
            local = d @ R.T
            return np.concatenate([local, (pts[:, 2:3] - here[2])], axis=-1)

        future = to_ego(at(fut_t)).astype(np.float32)
        history = to_ego(at(his_t)).astype(np.float32)
        dt = 1.0 / (N_HISTORY / HIST_S)
        vel = np.gradient(history[:, :2], dt, axis=0).astype(np.float32)
        acc = np.gradient(vel, dt, axis=0).astype(np.float32)
        # nav intent from where the path actually goes, the way the model's three
        # commands are defined: lateral offset at the horizon
        lat = float(future[-1, 1])
        nav = 0 if abs(lat) < 4.0 else (1 if lat > 0 else 2)
        np.savez(out / f"{s['token']}.npz", future=future, history=history,
                 velocity=vel, acceleration=acc,
                 nav=np.int64(nav),
                 speed=np.float32(np.linalg.norm(vel[-1])))
        made += 1
    print(f"  wrote {made} ego records to {out}  ({skip} already present, "
          f"{short} keyframes too close to scene end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
