#!/usr/bin/env python
"""Encode a rendered session at the correct playback speed.

nuscenes_session.py writes encode.json alongside the frames, recording the SOURCE rate
(the sensor cadence the frames were rendered at) and the requested speed. Reading it here
means the playback rate can never drift from what was actually rendered.

    python local/encode_session.py outputs/sweep_s0 outputs/nuscenes_session_0.75x.mp4
    python local/encode_session.py outputs/sweep_s0 out.mp4 --speed 1.0
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--speed", type=float, default=None,
                    help="override the speed recorded at render time")
    ap.add_argument("--container-fps", type=int, default=30,
                    help="frames duplicated up to this for smooth players")
    a = ap.parse_args()

    info = json.loads((a.session / "encode.json").read_text())
    n = len(list((a.session / "frames").glob("*.png")))
    speed = a.speed if a.speed is not None else info["speed"]
    src = info["source_hz"]
    fps = src * speed

    print(f"  {n} frames rendered at {src:.2f} Hz (sensor cadence)")
    print(f"  {speed}x real speed  ->  {fps:.3f} fps")
    print(f"  {n/src:.1f}s of real driving  ->  {n/fps:.1f}s of video")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.6f}",
           "-i", str(a.session / "frames" / "%04d.png"),
           "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps={a.container_fps}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", str(a.output)]
    subprocess.run(cmd, check=True)

    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(a.output)],
                         capture_output=True, text=True).stdout.strip()
    real = n / src
    print(f"  wrote {a.output}  ({float(dur):.1f}s of video for {real:.1f}s of driving"
          f"  =  {real/float(dur):.2f}x real speed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
