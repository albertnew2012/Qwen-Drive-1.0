"""Build data/distill/temporal_index.json for EVERY keyframe.

    {sample_token: {"prev": previous keyframe token or null, "ego2global": 16 floats, "ts": us}}

`ego2global` is the LIDAR_TOP keyframe's ego pose (the frame the teacher's boxes and the
student's BEV live in); `prev` is the immediate previous keyframe (0.5 s), which always has
images and calibration in data/distill/frames even when the teacher was never run on it.
The first version of this index (built inline on 2026-09-24) covered only the 25,599 cached
frames and linked each to its previous CACHED keyframe, sometimes 1.0 s back.

    python local/distill/build_temporal_index.py --root data/nuscenes
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default="data/distill/temporal_index.json")
    args = ap.parse_args()
    os.chdir(_ROOT)
    from local.distill.nusc_gt_boxes import quat_to_R, rt
    meta = Path(args.root) / args.version
    load = lambda n: json.loads((meta / f"{n}.json").read_text())
    ep = {e["token"]: e for e in load("ego_pose")}
    lidar_of = {}
    for d in load("sample_data"):
        if d.get("is_key_frame") and len(Path(d["filename"]).parts) > 1 \
                and Path(d["filename"]).parts[1] == "LIDAR_TOP":
            lidar_of[d["sample_token"]] = d
    frames = set(p.name for p in Path("data/distill/frames").iterdir() if p.is_dir())
    idx = {}
    for s in load("sample"):
        if s["token"] not in lidar_of or s["token"] not in frames:
            continue
        e = ep[lidar_of[s["token"]]["ego_pose_token"]]
        prev = s["prev"] if s["prev"] and s["prev"] in frames else None
        idx[s["token"]] = {"prev": prev, "ts": s["timestamp"],
                           "ego2global": rt(quat_to_R(e["rotation"]), e["translation"]).reshape(-1).tolist()}
    old_p = Path(args.out)
    if old_p.exists():
        old = json.load(open(old_p))
        same_pose = sum(np.allclose(old[t]["ego2global"], idx[t]["ego2global"], atol=1e-5) for t in old if t in idx)
        same_prev = sum(1 for t in old if t in idx and old[t]["prev"] == idx[t]["prev"])
        print(f"  consistency with the old index: {same_pose}/{len(old)} poses identical, "
              f"{same_prev}/{len(old)} same prev (the rest now link 0.5 s back instead of 1.0 s)")
        assert same_pose == len(old), "ego pose convention changed -- not writing"
    tmp = old_p.with_suffix(".tmp"); json.dump(idx, open(tmp, "w")); tmp.replace(old_p)
    n_prev = sum(1 for v in idx.values() if v["prev"])
    print(f"  wrote {len(idx)} frames ({n_prev} with a previous keyframe) -> {old_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
