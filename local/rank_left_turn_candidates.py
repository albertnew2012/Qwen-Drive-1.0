"""Rank left-turn candidates by whether a lit signal head is actually visible while stopped.

Scores the keyframes just before the turn begins, which is when a red arrow would still
be lit. Only the top 45% of the image is considered: brake lights and tail lights sit
below that line and otherwise dominate the red channel.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path("/home/zhengzhiliu/Documents/nuscenes")
META = ROOT / "blobs/v1.0-trainval_meta/v1.0-trainval"

index = {}
for blob in sorted(ROOT.glob("blobs/v1.0-trainval*_blobs")):
    d = blob / "samples/CAM_FRONT"
    if d.is_dir():
        for f in d.iterdir():
            index[f.name] = f

load = lambda n: json.loads((META / n).read_text())
scenes = {s["name"]: s for s in load("scene.json")}
samples = {s["token"]: s for s in load("sample.json")}
sensors = {s["token"]: s for s in load("sensor.json")}
calib = {c["token"]: c for c in load("calibrated_sensor.json")}
front = {}
for d in load("sample_data.json"):
    if d["is_key_frame"] and sensors[calib[d["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "CAM_FRONT":
        front[d["sample_token"]] = d["filename"]


def lamps(img, frac=0.45):
    a = np.asarray(img.convert("HSV")).astype(int)
    H, S, V = a[..., 0], a[..., 1], a[..., 2]
    masks = {"red": ((H < 8) | (H > 245)) & (S > 90) & (V > 110),
             "green": (H > 95) & (H < 135) & (S > 80) & (V > 110)}
    hits = {}
    for c, m in masks.items():
        m = m.copy()
        m[int(img.height * frac):, :] = False
        lab, _ = ndimage.label(m)
        n = 0
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            area = int((lab[sl] == i).sum())
            if not 5 <= area <= 300:
                continue
            ys, xs = sl
            w, h = xs.stop - xs.start, ys.stop - ys.start
            if max(w, h) > 2.5 * max(1, min(w, h)):
                continue
            n += 1
        hits[c] = n
    return hits


cands = json.loads(Path("outputs/traffic_light/left_turn_candidates.json").read_text())
print(f"{'scene':12s} {'loc':22s} {'turn':>6s} {'red(pre)':>9s} {'green(post)':>12s}  description")
rows = []
for c in cands:
    name = c["scene"]
    toks, t = [], scenes[name]["first_sample_token"]
    while t:
        toks.append(t); t = samples[t]["next"]
    st = c["turn_start_kf"]
    pre = [k for k in range(max(0, st - 8), st) if k < len(toks)]
    post = [k for k in range(st, min(len(toks), st + 6))]
    red = green = 0
    for k in pre:
        p = index.get(Path(front[toks[k]]).name)
        if p: red += lamps(Image.open(p))["red"]
    for k in post:
        p = index.get(Path(front[toks[k]]).name)
        if p: green += lamps(Image.open(p))["green"]
    rows.append((red, green, c))
    print(f"{name:12s} {c['location'][:22]:22s} {c['turn_deg']:5.0f}째 {red:9d} {green:12d}  {c['description'][:44]}")

print("\n=== ranked by red-while-stopped, then green-while-turning ===")
for red, green, c in sorted(rows, key=lambda r: (-r[0], -r[1]))[:8]:
    print(f"  {c['scene']:12s} red {red:3d}  green {green:3d}  turn starts kf{c['turn_start_kf']}  "
          f"{c['location'][:20]}")
    print(f"      {c['description'][:92]}")
