"""Find a left turn where the ego *arrives* at a red light, rather than starting stopped.

The approach is the part that carries evidence. A model that says "slow down for the red
light" while still rolling toward it has read the signal; one that says "stopped at a red
light" after already standing still for ten seconds may only be describing its own state.
scene-0882 fails this test - it begins at 0.00 m/s and never moves until the light turns.

Required pattern, in order: rolling at >= APPROACH m/s, braking to a stop of >= MIN_STOP
keyframes, pulling away again, then a left sweep of >= MIN_TURN degrees.
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
    ap.add_argument("--approach", type=float, default=3.0, help="m/s before braking")
    ap.add_argument("--stop-speed", type=float, default=0.5)
    ap.add_argument("--min-stop", type=int, default=3, help="keyframes at a standstill")
    ap.add_argument("--min-turn", type=float, default=45.0)
    ap.add_argument("--out", default="outputs/traffic_light/approach_left_turn.json")
    args = ap.parse_args()

    meta = Path(args.meta)
    load = lambda n: json.loads((meta / n).read_text())
    print("  loading metadata ...", flush=True)
    scenes = load("scene.json")
    logs = {l["token"]: l for l in load("log.json")}
    samples = {s["token"]: s for s in load("sample.json")}
    sensors = {s["token"]: s for s in load("sensor.json")}
    calib = {c["token"]: c for c in load("calibrated_sensor.json")}
    poses = {e["token"]: e for e in load("ego_pose.json")}
    front = {}
    for d in load("sample_data.json"):
        if d["is_key_frame"] and sensors[calib[d["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "CAM_FRONT":
            front[d["sample_token"]] = d
    print(f"  {len(scenes)} scenes\n", flush=True)

    rows = []
    for sc in scenes:
        toks, t = [], sc["first_sample_token"]
        while t:
            toks.append(t); t = samples[t]["next"]
        track = []
        ok = True
        for tk in toks:
            d = front.get(tk)
            if d is None:
                ok = False; break
            p = poses[d["ego_pose_token"]]
            track.append((samples[tk]["timestamp"] * 1e-6, p["translation"], yaw(p["rotation"])))
        if not ok or len(track) < 16:
            continue

        v, dy = [0.0], [0.0]
        for i in range(1, len(track)):
            dt = track[i][0] - track[i - 1][0] or 1e-3
            v.append(math.dist(track[i][1][:2], track[i - 1][1][:2]) / dt)
            dy.append(math.degrees(wrap(track[i][2] - track[i - 1][2])))

        # every standstill run, then the first one that has motion before and after it
        runs, i = [], 0
        while i < len(v):
            if v[i] < args.stop_speed:
                j = i
                while j < len(v) and v[j] < args.stop_speed:
                    j += 1
                if j - i >= args.min_stop:
                    runs.append((i, j))
                i = j
            else:
                i += 1

        for s0, s1 in runs:
            if max(v[:s0] or [0]) < args.approach:       # must have been rolling first
                continue
            if max(v[s1:] or [0]) < 2.0:                 # and must pull away after
                continue
            sweep = sum(d for d in dy[s1:] if d > -0.5)
            if sweep < args.min_turn:
                continue
            brake = [k for k in range(max(0, s0 - 8), s0) if v[k] >= args.stop_speed]
            rows.append({
                "scene": sc["name"], "token": sc["token"],
                "location": logs[sc["log_token"]]["location"],
                "logfile": logs[sc["log_token"]]["logfile"],
                "description": sc["description"],
                "approach_peak": round(max(v[:s0]), 1),
                "brake_kf": [brake[0], s0] if brake else None,
                "stop_kf": [s0, s1], "stop_len": s1 - s0,
                "turn_deg": round(sweep, 1),
                "keyframes": len(track),
                "t0": round(track[0][0], 6),
                "stop_start_time": round(track[s0][0], 6),
                "go_time": round(track[min(s1, len(track) - 1)][0], 6),
            })
            break

    rows.sort(key=lambda r: (-r["approach_peak"], -r["turn_deg"]))
    print(f"=== {len(rows)} scenes: roll >= {args.approach:g} m/s -> stop >= {args.min_stop} kf "
          f"-> pull away -> left >= {args.min_turn:g} deg ===\n")
    for r in rows[:20]:
        b = f"brake kf{r['brake_kf'][0]}-{r['brake_kf'][1]}" if r["brake_kf"] else "brake ?"
        print(f"  {r['scene']:12s} [{r['location'][:20]:20s}] peak {r['approach_peak']:4.1f} m/s, "
              f"{b}, stop kf{r['stop_kf'][0]}-{r['stop_kf'][1]}, turn {r['turn_deg']:5.1f}deg")
        print(f"      {r['description'][:94]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {out}  ({len(rows)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
