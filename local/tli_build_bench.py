#!/usr/bin/env python
"""Build a movement-conditioned traffic-light benchmark from gravity_tli_data.

Inferring the ego's lane from road markings turned out to be unreliable - the arrows are
labelled across the whole visible road (median 61 m ahead, lateral spread to 147 m), so
only a handful of frames have an unambiguous ego-lane arrow.

This instead tests the binding that actually matters and that the dataset labels exactly:
given a frame, what colour is the signal for a *specific movement*. A head showing
Red_Left is the left-turn signal; Green_Solid is the through signal. Asking for each
separately measures whether the model can bind a colour to a movement rather than
reporting whichever lamp is brightest.

The interesting frames are the CONFLICT ones, where the left-turn signal and the through
signal show different colours. There, "report any light" and "bind the right light" give
different answers - which is the whole question.

    python local/tli_build_bench.py --batches 120 --out outputs/tli_eval/bench.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/perception_data/gravity_tli_data")

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-")

CLASSES = ["Red_Solid", "Red_Left", "Red_Right", "Red_Straight", "Red_LeftDiagonal",
           "Red_RightDiagonal", "Yellow_Solid", "Yellow_Left", "Yellow_Right",
           "Yellow_Straight", "Yellow_LeftDiagonal", "Yellow_RightDiagonal",
           "Green_Solid", "Green_Left", "Green_Right", "Green_Straight",
           "Green_LeftDiagonal", "Green_RightDiagonal"]
CLASS_OFFSET = 2

LEFT_SHAPES = {"Left", "LeftDiagonal"}
THROUGH_SHAPES = {"Solid", "Straight"}
RIGHT_SHAPES = {"Right", "RightDiagonal"}


def parse_row(line: str):
    f = [x for x in line.split() if not UUID_RE.match(x)]     # some batches append a UUID
    if len(f) < 14 or f[0] != "traffic_light":
        return None
    try:
        tail = [int(float(v)) for v in f[13:]]
    except ValueError:
        return None
    lit = [CLASSES[i] for i in range(len(CLASSES))
           if CLASS_OFFSET + i < len(tail) and tail[CLASS_OFFSET + i] > 0]
    if not lit:
        return None
    x1, y1, x2, y2 = (float(v) for v in f[2:6])
    return {"bbox": [x1, y1, x2, y2], "area": (x2 - x1) * (y2 - y1),
            "z": float(f[11]), "lit": lit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=120)
    ap.add_argument("--cam", default="cam-02")
    ap.add_argument("--min-box", type=float, default=18.0, help="min bbox side in px")
    ap.add_argument("--out", default="outputs/tli_eval/bench.json")
    args = ap.parse_args()

    batches = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    rows, stats = [], Counter()

    for b in batches[:args.batches]:
        lab = b / "labels" / b.name
        tl_root = lab / "traffic_lights" / "KITTI_CAM_FRAME"
        if not tl_root.is_dir():
            stats["no_labels"] += 1
            continue
        for seq in tl_root.iterdir():
            if not seq.is_dir():
                continue
            cams = [c for c in seq.iterdir() if c.is_dir() and args.cam in c.name]
            if not cams:
                continue
            cam = cams[0]
            png_dir = b / "data" / seq.name / cam.name / "png_files"
            for lf in sorted(cam.glob("*.txt")):
                idx = int(lf.stem.split("-")[-1])
                heads = [h for h in (parse_row(l) for l in lf.read_text().splitlines()) if h]
                if not heads:
                    stats["no_lit_heads"] += 1
                    continue
                # a head must be big enough to be legible at all
                heads = [h for h in heads
                         if min(h["bbox"][2] - h["bbox"][0], h["bbox"][3] - h["bbox"][1]) >= args.min_box]
                if not heads:
                    stats["all_heads_tiny"] += 1
                    continue

                def colours(shapes):
                    out = set()
                    for h in heads:
                        for s in h["lit"]:
                            if s.split("_")[1] in shapes:
                                out.add(s.split("_")[0].lower())
                    return out

                left, through, right = colours(LEFT_SHAPES), colours(THROUGH_SHAPES), colours(RIGHT_SHAPES)
                entry = {
                    "batch": b.name, "seq": seq.name, "cam": cam.name, "frame": idx,
                    "png": str(png_dir / f"{cam.name}-{idx:06d}.png"),
                    "n_heads": len(heads),
                    "nearest_m": round(min(h["z"] for h in heads), 1),
                    "largest_box_px": round(max(max(h["bbox"][2] - h["bbox"][0],
                                                    h["bbox"][3] - h["bbox"][1]) for h in heads), 1),
                    "left": sorted(left), "through": sorted(through), "right": sorted(right),
                    "all_states": sorted({s for h in heads for s in h["lit"]}),
                }
                entry["left_gt"] = left.pop() if len(left) == 1 else None
                entry["through_gt"] = through.pop() if len(through) == 1 else None
                entry["conflict"] = bool(entry["left_gt"] and entry["through_gt"]
                                         and entry["left_gt"] != entry["through_gt"])
                if entry["left_gt"] or entry["through_gt"]:
                    rows.append(entry)
                    stats["usable"] += 1
                    if entry["conflict"]:
                        stats["CONFLICT"] += 1
                else:
                    stats["no_unique_movement_colour"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"  scanned {args.batches} batches")
    for k, v in stats.most_common():
        print(f"    {k:28s} {v}")
    if rows:
        conf = [r for r in rows if r["conflict"]]
        print(f"\n  usable frames     {len(rows)}")
        print(f"  CONFLICT frames   {len(conf)}   <- left-turn and through signals differ")
        print(f"  left_gt           {Counter(r['left_gt'] for r in rows if r['left_gt']).most_common()}")
        print(f"  through_gt        {Counter(r['through_gt'] for r in rows if r['through_gt']).most_common()}")
        if conf:
            print(f"  conflict pairs    {Counter((r['left_gt'], r['through_gt']) for r in conf).most_common()}")
            print(f"  conflict distance m: min {min(r['nearest_m'] for r in conf):.0f} "
                  f"median {sorted(r['nearest_m'] for r in conf)[len(conf)//2]:.0f} "
                  f"max {max(r['nearest_m'] for r in conf):.0f}")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
