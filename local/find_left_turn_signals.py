#!/usr/bin/env python
"""Find nuScenes scenes where the ego waits at a red light and then turns left.

The pattern has a clean signature in the ego poses and needs no annotations: the ego
sits still for a few seconds (red), then accelerates while its heading sweeps through
roughly +90 degrees (green, protected left). Scenes that turn left *without* stopping
first are permissive greens and are filtered out - those cannot show a red-to-green
transition, which is the thing worth watching.

    python3 local/find_left_turn_signals.py --top 15
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="/home/zhengzhiliu/Documents/nuscenes/blobs/"
                                      "v1.0-trainval_meta/v1.0-trainval")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-turn", type=float, default=60.0, help="degrees of left sweep")
    ap.add_argument("--stop-speed", type=float, default=0.5, help="m/s counted as stopped")
    ap.add_argument("--min-stop", type=int, default=4, help="keyframes stopped before the turn")
    ap.add_argument("--out", default="outputs/traffic_light/left_turn_candidates.json")
    args = ap.parse_args()

    meta = Path(args.meta)
    load = lambda n: json.loads((meta / n).read_text())
    print("  loading metadata (sample_data.json is ~1.3 GB) ...", flush=True)
    scenes = load("scene.json")
    logs = {l["token"]: l for l in load("log.json")}
    samples = {s["token"]: s for s in load("sample.json")}
    sensors = {s["token"]: s for s in load("sensor.json")}
    calib = {c["token"]: c for c in load("calibrated_sensor.json")}
    poses = {e["token"]: e for e in load("ego_pose.json")}

    front = {}
    for d in load("sample_data.json"):
        if not d["is_key_frame"]:
            continue
        if sensors[calib[d["calibrated_sensor_token"]]["sensor_token"]]["channel"] != "CAM_FRONT":
            continue
        front[d["sample_token"]] = d
    print(f"  {len(scenes)} scenes, {len(front)} CAM_FRONT keyframes\n", flush=True)

    rows = []
    for sc in scenes:
        toks, t = [], sc["first_sample_token"]
        while t:
            toks.append(t)
            t = samples[t]["next"]
        track = []
        for tk in toks:
            d = front.get(tk)
            if d is None:
                break
            p = poses[d["ego_pose_token"]]
            track.append((samples[tk]["timestamp"] * 1e-6, p["translation"], yaw(p["rotation"])))
        if len(track) < 12:
            continue

        speeds, dyaw = [0.0], [0.0]
        for i in range(1, len(track)):
            dt = track[i][0] - track[i - 1][0] or 1e-3
            speeds.append(math.dist(track[i][1][:2], track[i - 1][1][:2]) / dt)
            dyaw.append(math.degrees(wrap(track[i][2] - track[i - 1][2])))

        # the turn is the longest run of sustained left rotation
        best = (0, 0, 0.0)                       # start, end, degrees
        i = 0
        while i < len(dyaw):
            if dyaw[i] > 1.0:
                j, total = i, 0.0
                while j < len(dyaw) and dyaw[j] > -0.5:
                    total += dyaw[j]
                    j += 1
                if total > best[2]:
                    best = (i, j, total)
                i = j
            else:
                i += 1
        start, end, degrees = best
        if degrees < args.min_turn:
            continue

        # a red light means a stop immediately before the sweep begins
        pre = speeds[max(0, start - 12):start]
        stopped = sum(1 for v in pre if v < args.stop_speed)
        if stopped < args.min_stop:
            continue
        after = speeds[end:end + 6]
        rows.append({
            "scene": sc["name"], "token": sc["token"],
            "location": logs[sc["log_token"]]["location"],
            "logfile": logs[sc["log_token"]]["logfile"],
            "description": sc["description"],
            "turn_deg": round(degrees, 1),
            "turn_start_kf": start, "turn_end_kf": end,
            "stopped_kf_before": stopped,
            "speed_at_turn_start": round(speeds[start], 2),
            "speed_after": round(sum(after) / max(1, len(after)), 2),
            "turn_start_time": round(track[start][0], 6),
            "keyframes": len(track),
        })

    # prefer a long stop then a decisive sweep - that is the clearest red-to-green
    rows.sort(key=lambda r: (-r["stopped_kf_before"], -r["turn_deg"]))
    print(f"=== {len(rows)} scenes: stopped >= {args.min_stop} keyframes, then left turn "
          f">= {args.min_turn:g} deg ===\n")
    for r in rows[:args.top]:
        print(f"  {r['scene']}  [{r['location'][:20]:20s}] stop {r['stopped_kf_before']:2d} kf, "
              f"turn {r['turn_deg']:5.1f}deg at kf {r['turn_start_kf']:2d}-{r['turn_end_kf']:2d}, "
              f"then {r['speed_after']:4.1f} m/s")
        print(f"      {r['description'][:96]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {out}  ({len(rows)} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
