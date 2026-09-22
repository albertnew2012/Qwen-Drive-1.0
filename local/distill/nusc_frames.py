"""Turn nuScenes samples into the packed frame layout the perception model reads.

``PerceptionFrame`` expects a directory holding ``frame.json`` (the prompt content and
camera order), ``calib.npz`` (intrinsics and the camera-to-lidar extrinsics) and an
``images/`` folder. Producing that from nuScenes directly means everything downstream --
the processor, the head, the teacher -- works unchanged on as many frames as there are
samples, rather than on the six demo frames.

Images are symlinked, not copied: trainval is ~350 GB and the blobs are still landing.

The extrinsics are composed the full way round rather than assumed, because a camera
and the lidar are sampled at slightly different timestamps and therefore at different
ego poses:

    camera -> ego(t_cam) -> global -> ego(t_lidar) -> lidar

Skipping the two ego poses is the usual shortcut and it puts boxes tens of centimetres
out at speed, which would show up as distillation error that is really a calibration bug.

    python local/distill/nusc_frames.py --version v1.0-mini --limit 0
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]

CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
VIEW_TAG = {"CAM_FRONT": "<FRONT VIEW>", "CAM_FRONT_RIGHT": "<FRONT RIGHT VIEW>",
            "CAM_BACK_RIGHT": "<BACK RIGHT VIEW>", "CAM_BACK": "<BACK VIEW>",
            "CAM_BACK_LEFT": "<BACK LEFT VIEW>", "CAM_FRONT_LEFT": "<FRONT LEFT VIEW>"}


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]], dtype=np.float64)


def rt(R, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, dtype=np.float64)
    return M


def load_tables(meta: Path) -> dict:
    out = {}
    for name in ("sample", "sample_data", "calibrated_sensor", "ego_pose", "scene"):
        out[name] = json.loads((meta / f"{name}.json").read_text())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes",
                    help="directory holding samples/ and the metadata folder")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="data/distill/frames")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-images", action="store_true", default=True)
    args = ap.parse_args()
    os.chdir(_ROOT)

    root = Path(args.root)
    meta = root / args.version
    if not meta.is_dir():
        print(f"  no metadata at {meta}")
        return 1
    t = load_tables(meta)
    by_token = {r["token"]: r for r in t["sample_data"]}
    cs = {r["token"]: r for r in t["calibrated_sensor"]}
    ep = {r["token"]: r for r in t["ego_pose"]}

    # sample_data rows that are keyframes, indexed by (sample, channel)
    chan = {}
    for r in t["sample_data"]:
        if not r.get("is_key_frame"):
            continue
        name = Path(r["filename"]).parts[1] if len(Path(r["filename"]).parts) > 1 else ""
        chan.setdefault(r["sample_token"], {})[name] = r

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    made = skipped = 0
    samples = t["sample"]
    if args.limit:
        samples = samples[:args.limit]

    for s in samples:
        tok = s["token"]
        rows = chan.get(tok, {})
        if not all(c in rows for c in CAM_ORDER) or "LIDAR_TOP" not in rows:
            skipped += 1
            continue
        paths = {c: root / rows[c]["filename"] for c in CAM_ORDER}
        if args.require_images and not all(p.exists() for p in paths.values()):
            skipped += 1
            continue

        lid = rows["LIDAR_TOP"]
        cs_lid = cs[lid["calibrated_sensor_token"]]
        ep_lid = ep[lid["ego_pose_token"]]
        lidar2ego = rt(quat_to_R(cs_lid["rotation"]), cs_lid["translation"])
        ego_lid2glob = rt(quat_to_R(ep_lid["rotation"]), ep_lid["translation"])
        glob2lidar = np.linalg.inv(ego_lid2glob @ lidar2ego)

        K, R_s2l, t_s2l = [], [], []
        for c in CAM_ORDER:
            r = rows[c]
            cs_cam = cs[r["calibrated_sensor_token"]]
            ep_cam = ep[r["ego_pose_token"]]
            cam2ego = rt(quat_to_R(cs_cam["rotation"]), cs_cam["translation"])
            ego2glob = rt(quat_to_R(ep_cam["rotation"]), ep_cam["translation"])
            cam2lidar = glob2lidar @ ego2glob @ cam2ego
            K.append(np.asarray(cs_cam["camera_intrinsic"], dtype=np.float64))
            R_s2l.append(cam2lidar[:3, :3])
            t_s2l.append(cam2lidar[:3, 3])

        d = out_root / tok
        (d / "images").mkdir(parents=True, exist_ok=True)
        for c in CAM_ORDER:
            link = d / "images" / f"{c}.jpg"
            if not link.exists():
                link.symlink_to(paths[c].resolve())
        content = []
        for c in CAM_ORDER:
            content += [{"text": VIEW_TAG[c]}, {"image": c}]
        content.append({"text": "Analyze the scene."})
        (d / "frame.json").write_text(json.dumps(
            {"dataset_type": "nuscenes", "cam_order": CAM_ORDER, "content": content}))
        np.savez(d / "calib.npz",
                 cam_intrinsic=np.stack(K).astype(np.float32),
                 sensor2lidar_rotation=np.stack(R_s2l).astype(np.float32),
                 sensor2lidar_translation=np.stack(t_s2l).astype(np.float32),
                 lidar2ego=lidar2ego.astype(np.float32))
        # the loader reads gt.npz; distillation supervises from the teacher, so it is
        # written empty rather than populated from sample_annotation
        if not (d / "gt.npz").exists():
            np.savez(d / "gt.npz", boxes=np.zeros((0, 9), np.float32),
                     labels=np.zeros((0,), np.int64))
        made += 1
        if made % 200 == 0:
            print(f"  {made} frames written", flush=True)

    print(f"  wrote {made} frames to {out_root}  (skipped {skipped} without all images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
