"""Find a left turn that begins when a red signal turns green.

Two changes over the first pass. The candidate net is wider (shorter stops, shallower
turns), and the lamp detector now requires a dark surround: a signal lamp sits in a
black housing, whereas brake lights sit on a bright car body and street furniture
reflects its background. That one test removes almost all the false reds.

Prints, per scene, the per-keyframe lamp colour alongside ego speed, so a
red-while-stopped -> green-then-moving transition is visible directly.
"""
import json
from pathlib import Path
import math
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


def yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def signal_lamps(path):
    """Lit lamps that sit inside a dark housing."""
    img = Image.open(path)
    a = np.asarray(img.convert("HSV")).astype(int)
    H, S, V = a[..., 0], a[..., 1], a[..., 2]
    cut = int(img.height * 0.45)
    out = {"red": 0, "green": 0, "amber": 0}
    masks = {"red": ((H < 8) | (H > 245)) & (S > 90) & (V > 110),
             "amber": (H > 12) & (H < 32) & (S > 130) & (V > 150),
             "green": (H > 95) & (H < 135) & (S > 80) & (V > 110)}
    Vf = V.astype(float)
    for colour, m in masks.items():
        m = m.copy()
        m[cut:, :] = False
        lab, _ = ndimage.label(m)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            area = int((lab[sl] == i).sum())
            if not 5 <= area <= 250:
                continue
            ys, xs = sl
            w, h = xs.stop - xs.start, ys.stop - ys.start
            if max(w, h) > 2.2 * max(1, min(w, h)):
                continue
            # the housing: a band around the lamp must be markedly darker
            pad = max(4, 2 * max(w, h))
            y0, y1 = max(0, ys.start - pad), min(V.shape[0], ys.stop + pad)
            x0, x1 = max(0, xs.start - pad), min(V.shape[1], xs.stop + pad)
            patch = Vf[y0:y1, x0:x1].copy()
            inner = Vf[ys.start:ys.stop, xs.start:xs.stop].mean()
            patch[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = np.nan
            surround = np.nanmedian(patch)
            if surround < 95 and inner > surround + 45:
                out[colour] += 1
    return out


cands = json.loads(Path("outputs/traffic_light/left_turn_candidates.json").read_text())
print(f"scanning {len(cands)} candidates with the housing test\n")
best = []
for c in cands:
    sc = next(s for s in scenes if s["name"] == c["scene"])
    toks, t = [], sc["first_sample_token"]
    while t:
        toks.append(t); t = samples[t]["next"]
    track = []
    for tk in toks:
        d = front.get(tk)
        p = poses[d["ego_pose_token"]]
        track.append((samples[tk]["timestamp"] * 1e-6, p["translation"], yaw(p["rotation"])))
    seq = []
    for k, tk in enumerate(toks):
        v = 0.0
        if k:
            dt = track[k][0] - track[k - 1][0] or 1e-3
            v = math.dist(track[k][1][:2], track[k - 1][1][:2]) / dt
        p = index.get(Path(front[tk]["filename"]).name)
        lamp = signal_lamps(p) if p else {"red": 0, "green": 0, "amber": 0}
        seq.append((k, v, lamp))
    red_stop = [k for k, v, l in seq if l["red"] > 0 and v < 0.6]
    grn_go = [k for k, v, l in seq if l["green"] > 0 and v > 1.0]
    trans = bool(red_stop and grn_go and min(grn_go) > min(red_stop))
    score = (len(red_stop), len(grn_go))
    best.append((trans, score, c, seq))
    flag = "  <== RED-while-stopped THEN GREEN-while-moving" if trans else ""
    print(f"  {c['scene']:12s} red@stop {len(red_stop):2d} kf, green@go {len(grn_go):2d} kf{flag}")

print("\n=== scenes showing the transition ===")
for trans, score, c, seq in sorted(best, key=lambda r: (-r[0], -r[1][0] - r[1][1])):
    if not trans:
        continue
    print(f"\n  {c['scene']}  [{c['location']}]  turn {c['turn_deg']:.0f} deg from kf{c['turn_start_kf']}")
    print(f"    {c['description']}")
    print(f"    kf : speed : lamps")
    for k, v, l in seq:
        if l["red"] or l["green"] or l["amber"]:
            bits = " ".join(f"{n}x{col}" for col, n in l.items() if n)
            print(f"    {k:2d} : {v:5.2f} : {bits}")
