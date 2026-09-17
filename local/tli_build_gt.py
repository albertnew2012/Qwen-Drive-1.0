#!/usr/bin/env python
"""Build traffic-light / ego-lane association ground truth from gravity_tli_data.

The dataset labels every signal head with a multi-hot state over 19 classes
(Red_Solid, Red_Left, ... Green_RightDiagonal) and every road marking with an arrow
type. Neither on its own says which signal governs the ego. The association is
recoverable by combining them:

    ego lane arrow  ->  required movement  ->  the signal head carrying that movement

A frame is only usable if the ego's own lane marking is identifiable (an arrow close
ahead and nearly straight in front) and at least one visible signal carries the
matching movement. Frames where a *different* signal shows a *different* colour are
marked `discriminative`: those are the only ones where association and "report any
light" give different answers, and they are what the model actually has to be tested on.

    python local/tli_build_gt.py --batches 40 --out outputs/tli_eval/gt.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/perception_data/gravity_tli_data")

# verified against label_statistics.json: tail column 2 is Red_Solid, and so on
CLASSES = ["Red_Solid", "Red_Left", "Red_Right", "Red_Straight", "Red_LeftDiagonal",
           "Red_RightDiagonal", "Yellow_Solid", "Yellow_Left", "Yellow_Right",
           "Yellow_Straight", "Yellow_LeftDiagonal", "Yellow_RightDiagonal",
           "Green_Solid", "Green_Left", "Green_Right", "Green_Straight",
           "Green_LeftDiagonal", "Green_RightDiagonal"]
CLASS_OFFSET = 2          # tail[2] == CLASSES[0]

MOVEMENT_OF_SHAPE = {"Solid": "any", "Straight": "straight", "Left": "left",
                     "Right": "right", "LeftDiagonal": "left", "RightDiagonal": "right"}


def parse_state(tail: list[int]) -> list[str]:
    """Class names of every lit bulb on one head."""
    out = []
    for i, name in enumerate(CLASSES):
        j = CLASS_OFFSET + i
        if j < len(tail) and tail[j] > 0:
            out.append(name)
    return out


def cam_frame_rows(path: Path):
    """KITTI_CAM_FRAME: class alpha x1 y1 x2 y2 h w l X Y Z ry <tail...>"""
    rows = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 14 or f[0] != "traffic_light":
            continue
        tail = [int(float(v)) for v in f[13:]]
        rows.append({"bbox": [float(v) for v in f[2:6]],
                     "xyz": [float(v) for v in f[9:12]],
                     "state": parse_state(tail)})
    return rows


def sensorfusion_rows(path: Path):
    """SENSORFUSION: ts frame track class <10 floats> <tail...>; floats[4:7] are x,y,z."""
    by_frame = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 15:
            continue
        frame = int(f[1])
        nums = [float(v) for v in f[4:14]]
        by_frame.setdefault(frame, []).append({"cls": f[3], "x": nums[4], "y": nums[5],
                                               "z": nums[6]})
    return by_frame


def ego_movement(markings, max_ahead=40.0, max_lat=2.2):
    """The arrow painted in the ego's own lane, i.e. close ahead and nearly straight on."""
    best = None
    for m in markings:
        if not (0.0 < m["x"] < max_ahead and abs(m["y"]) < max_lat):
            continue
        if best is None or m["x"] < best["x"]:
            best = m
    if best is None:
        return None, None
    parts = set(best["cls"].split("."))
    if parts == {"left_arrow"}:
        return "left", best["cls"]
    if parts == {"straight_arrow"}:
        return "straight", best["cls"]
    if parts == {"right_arrow"}:
        return "right", best["cls"]
    return None, best["cls"]          # combinations are ambiguous, skip them


def colour_of(state_name: str) -> str:
    return state_name.split("_")[0].lower()


def shape_of(state_name: str) -> str:
    return state_name.split("_")[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--cam", default="cam-02")
    ap.add_argument("--out", default="outputs/tli_eval/gt.json")
    args = ap.parse_args()

    batches = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    samples, stats = [], Counter()

    for b in batches[:args.batches]:
        lab = b / "labels" / b.name
        if not lab.is_dir():
            stats["no_labels"] += 1
            continue
        tl_root = lab / "traffic_lights" / "KITTI_CAM_FRAME"
        rm_root = lab / "road_markings" / "KITTI_SENSORFUSION"
        if not tl_root.is_dir() or not rm_root.is_dir():
            stats["missing_dirs"] += 1
            continue
        for seq in tl_root.iterdir():
            if not seq.is_dir():
                continue
            cams = [c for c in seq.iterdir() if c.is_dir() and args.cam in c.name]
            if not cams:
                continue
            cam = cams[0]
            rm_file = next(rm_root.glob(f"{seq.name}*.txt"), None)
            if rm_file is None:
                stats["no_road_markings"] += 1
                continue
            markings = sensorfusion_rows(rm_file)
            png_dir = b / "data" / seq.name / cam.name / "png_files"
            for lf in sorted(cam.glob("*.txt")):
                idx = int(lf.stem.split("-")[-1])
                lights = cam_frame_rows(lf)
                if not lights:
                    stats["frame_no_lights"] += 1
                    continue
                move, arrow = ego_movement(markings.get(idx, []))
                if move is None:
                    stats["frame_no_ego_arrow"] += 1
                    continue
                # which heads carry the movement the ego needs
                gov, other = [], []
                for L in lights:
                    if not L["state"]:
                        continue
                    hit = False
                    for s in L["state"]:
                        mv = MOVEMENT_OF_SHAPE.get(shape_of(s), "any")
                        if mv == move or (mv == "any" and move == "straight"):
                            gov.append((s, L)); hit = True; break
                    if not hit:
                        other.append((L["state"][0], L))
                if not gov:
                    stats["frame_no_matching_signal"] += 1
                    continue
                gt_colours = {colour_of(s) for s, _ in gov}
                if len(gt_colours) != 1:
                    stats["frame_ambiguous_gov"] += 1
                    continue
                gt = gt_colours.pop()
                other_colours = {colour_of(s) for s, _ in other}
                png = png_dir / f"{cam.name}-{idx:06d}.png"
                samples.append({
                    "batch": b.name, "seq": seq.name, "cam": cam.name, "frame": idx,
                    "png": str(png), "png_exists": png.exists(),
                    "ego_arrow": arrow, "ego_movement": move,
                    "gt_colour": gt,
                    "gov_states": sorted({s for s, _ in gov}),
                    "gov_boxes": [L["bbox"] for _, L in gov],
                    "gov_dist_m": round(min(L["xyz"][2] for _, L in gov), 1),
                    "n_lights": len(lights),
                    "other_colours": sorted(other_colours),
                    "discriminative": bool(other_colours - {gt}),
                })
                stats["usable"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=1))
    print(f"  scanned {args.batches} batches")
    for k, v in stats.most_common():
        print(f"    {k:26s} {v}")
    if samples:
        disc = [s for s in samples if s["discriminative"]]
        print(f"\n  usable frames      {len(samples)}")
        print(f"  discriminative     {len(disc)}  (a differently-coloured signal is also visible)")
        print(f"  png present        {sum(s['png_exists'] for s in samples)}")
        print(f"  gt colour          {Counter(s['gt_colour'] for s in samples).most_common()}")
        print(f"  ego movement       {Counter(s['ego_movement'] for s in samples).most_common()}")
        print(f"  gov states         {Counter(x for s in samples for x in s['gov_states']).most_common(8)}")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
