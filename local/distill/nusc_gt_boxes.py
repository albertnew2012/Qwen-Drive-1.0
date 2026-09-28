"""Populate real 3-D box ground truth for the cached distillation frames.

WHY THIS EXISTS
``nusc_frames.py`` deliberately writes an empty ``gt.npz``: distillation supervises from
the teacher, so training never needs ground truth. But *evaluation* does, because
"performance degradation under 10%" has two readings and only one is measurable without it:

    replication   the student reproduces >= 90% of what the teacher outputs. Harsh: it
                  demands the student copy a 4 B model's mistakes too.
    task quality  the student scores within 10% of what the TEACHER scores against ground
                  truth. This is the ordinary meaning, and it needs GT.

Writes ``gt_boxes.npz`` beside the frame rather than overwriting ``gt.npz``, so the
training loader keeps seeing what it expects and the running pipeline is untouched.

Boxes land in the EGO frame, matching the teacher's own box output (verified: 99.9% of
teacher detections fall inside the +/-51.2 m pc_range when read that way).

    python local/distill/nusc_gt_boxes.py --root data/nuscenes_trainval --version v1.0-trainval
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np

CLASSES = ("vehicle", "czone_sign", "bicycle", "generic_object",
           "pedestrian", "traffic_cone", "barrier")
IDX = {c: i for i, c in enumerate(CLASSES)}

# nuScenes category -> those 7. The prefix rule covers the pedestrian subtree, which the
# mapping in local/nuscenes_session.py omits entirely.
EXACT = {
    "vehicle.car": "vehicle", "vehicle.truck": "vehicle", "vehicle.bus.bendy": "vehicle",
    "vehicle.bus.rigid": "vehicle", "vehicle.trailer": "vehicle",
    "vehicle.construction": "vehicle", "vehicle.emergency.ambulance": "vehicle",
    "vehicle.emergency.police": "vehicle",
    "vehicle.bicycle": "bicycle", "vehicle.motorcycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone", "movable_object.barrier": "barrier",
    "movable_object.debris": "generic_object",
    "movable_object.pushable_pullable": "generic_object",
    "static_object.bicycle_rack": "generic_object",
}


def to_class(name: str):
    if name in EXACT:
        return IDX[EXACT[name]]
    if name.startswith("human.pedestrian"):
        return IDX["pedestrian"]
    return None


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=np.float64)


def rt(R, t):
    m = np.eye(4); m[:3, :3] = R; m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes_trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--taxonomy", default="teacher7", choices=["teacher7", "nuscenes10"],
                    help="teacher7: the 7 teacher classes above -> gt_boxes.npz; nuscenes10: "
                         "the 10 official detection classes -> gt_boxes10.npz, with per-box "
                         "num_pts, visibility and ego-frame velocity for the official metric")
    args = ap.parse_args()
    out_name = "gt_boxes.npz" if args.taxonomy == "teacher7" else "gt_boxes10.npz"
    if args.taxonomy == "nuscenes10":
        from nuscenes.eval.detection.utils import category_to_detection_name
        from nuscenes.eval.detection.constants import DETECTION_NAMES
        det_idx = {n: i for i, n in enumerate(DETECTION_NAMES)}
    os.chdir(_ROOT)

    meta = Path(args.root) / args.version

    def load(name):
        return json.loads((meta / f"{name}.json").read_text())

    print("  reading annotation tables", flush=True)
    sample_ts = {s["token"]: s["timestamp"] for s in load("sample")}
    samples = set(sample_ts)
    anns = load("sample_annotation")
    ann_by_tok = {a["token"]: a for a in anns}

    def velocity(a):
        """nusc.box_velocity re-done on the raw tables: central difference over the
        neighbouring annotations of the same instance, zero when there is none."""
        first = ann_by_tok[a["prev"]] if a["prev"] else a
        last = ann_by_tok[a["next"]] if a["next"] else a
        if first is last:
            return np.zeros(3)
        dt = (sample_ts[last["sample_token"]] - sample_ts[first["sample_token"]]) * 1e-6
        if not (0 < dt <= 1.5):          # the devkit's max_time_diff
            return np.zeros(3)
        return (np.asarray(last["translation"]) - np.asarray(first["translation"])) / dt
    insts = {i["token"]: i for i in load("instance")}
    cats = {c["token"]: c["name"] for c in load("category")}
    ep = {e["token"]: e for e in load("ego_pose")}

    by_sample = {}
    for a in anns:
        by_sample.setdefault(a["sample_token"], []).append(a)
    # sample_data carries no "channel" field; nusc_frames.py derives it from the
    # filename path ("samples/LIDAR_TOP/....pcd.bin"), so do the same here.
    lidar_of = {}
    for d in load("sample_data"):
        if not d.get("is_key_frame"):
            continue
        parts = Path(d["filename"]).parts
        if len(parts) > 1 and parts[1] == "LIDAR_TOP":
            lidar_of[d["sample_token"]] = d
    print(f"  {len(anns)} annotations over {len(by_sample)} samples", flush=True)

    frames = sorted(p for p in Path(args.frames).iterdir() if p.is_dir())
    if args.limit:
        frames = frames[:args.limit]

    made = skipped = empty = 0
    for d in frames:
        tok = d.name
        if (d / out_name).exists():
            made += 1
            continue
        if tok not in samples or tok not in lidar_of:
            skipped += 1
            continue
        lid = lidar_of[tok]
        ep_lid = ep[lid["ego_pose_token"]]
        # global -> ego at the lidar timestamp: the frame the teacher's boxes live in
        glob2ego = np.linalg.inv(rt(quat_to_R(ep_lid["rotation"]), ep_lid["translation"]))

        boxes, labels, npts, vis, vels = [], [], [], [], []
        for a in by_sample.get(tok, []):
            name = cats[insts[a["instance_token"]]["category_token"]]
            if args.taxonomy == "teacher7":
                c = to_class(name)
            else:
                dn = category_to_detection_name(name)
                c = det_idx[dn] if dn is not None else None
            if c is None:
                continue
            xyz = (glob2ego @ np.array([*a["translation"], 1.0]))[:3]
            w, l, h = a["size"]
            # Yaw in the EGO frame (rotate the box orientation with the position). The
            # teacher7 files written before 2026-09-24 stored the GLOBAL yaw next to an
            # ego-frame position, which only the sin/cos regression targets ever saw.
            R = glob2ego[:3, :3] @ quat_to_R(a["rotation"])
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            v = glob2ego[:3, :3] @ velocity(a)
            boxes.append([*xyz, w, l, h, yaw, float(v[0]), float(v[1])])
            labels.append(c)
            npts.append(int(a.get("num_lidar_pts", 0)) + int(a.get("num_radar_pts", 0)))
            vis.append(int(a.get("visibility_token") or 0))
            vels.append(v[:2])
        arr = np.asarray(boxes, np.float32) if boxes else np.zeros((0, 9), np.float32)
        np.savez(d / out_name, boxes=arr, labels=np.asarray(labels, np.int64),
                 num_pts=np.asarray(npts, np.int64), visibility=np.asarray(vis, np.int64),
                 velocity=(np.asarray(vels, np.float32) if vels else np.zeros((0, 2), np.float32)))
        empty += (0 if boxes else 1)
        made += 1
        if made % 4000 == 0:
            print(f"  {made}/{len(frames)}", flush=True)

    print(f"  wrote {made} frames ({empty} with no in-taxonomy object, "
          f"{skipped} skipped)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
