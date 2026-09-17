"""Pull CAM_FRONT frames for the left-turn candidates so the signal heads can be read.

The trainval images are spread over ten blob directories, so build one filename -> path
index first rather than merging 400 GB.
"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("/home/zhengzhiliu/Documents/nuscenes")
META = ROOT / "blobs/v1.0-trainval_meta/v1.0-trainval"
OUT = Path("outputs/traffic_light/candidates")
OUT.mkdir(parents=True, exist_ok=True)

print("  indexing blobs ...", flush=True)
index = {}
for blob in sorted(ROOT.glob("blobs/v1.0-trainval*_blobs")):
    d = blob / "samples/CAM_FRONT"
    if d.is_dir():
        for f in d.iterdir():
            index[f.name] = f
print(f"  {len(index)} CAM_FRONT keyframes indexed\n", flush=True)

load = lambda n: json.loads((META / n).read_text())
scenes = {s["name"]: s for s in load("scene.json")}
samples = {s["token"]: s for s in load("sample.json")}
sensors = {s["token"]: s for s in load("sensor.json")}
calib = {c["token"]: c for c in load("calibrated_sensor.json")}
front = {}
for d in load("sample_data.json"):
    if d["is_key_frame"] and sensors[calib[d["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "CAM_FRONT":
        front[d["sample_token"]] = d["filename"]

cands = json.loads(Path("outputs/traffic_light/left_turn_candidates.json").read_text())
want = sys.argv[1].split(",") if len(sys.argv) > 1 else [c["scene"] for c in cands[:8]]

for name in want:
    c = next((x for x in cands if x["scene"] == name), None)
    if c is None:
        print(f"  {name}: not a candidate"); continue
    toks, t = [], scenes[name]["first_sample_token"]
    while t:
        toks.append(t); t = samples[t]["next"]
    start = c["turn_start_kf"]
    # a few keyframes before the sweep is where a red arrow would still be lit
    picks = [max(0, start - 8), max(0, start - 4), max(0, start - 1), start, min(len(toks) - 1, start + 4)]
    tiles = []
    for kf in picks:
        fn = Path(front[toks[kf]]).name
        p = index.get(fn)
        if p is None:
            print(f"    {name} kf{kf}: {fn} NOT FOUND"); continue
        im = Image.open(p).crop((0, 150, 1600, 620))
        im = im.resize((im.width * 3 // 2, im.height * 3 // 2), Image.LANCZOS)
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width - 1, 22], fill=(0, 0, 0))
        dr.text((5, 6), f"{name} kf{kf}  (turn starts kf{start})", fill=(255, 255, 0))
        tiles.append(im)
    if not tiles:
        continue
    W = max(t.width for t in tiles); H = sum(t.height for t in tiles)
    m = Image.new("RGB", (W, H), (20, 20, 20)); y = 0
    for t in tiles:
        m.paste(t, (0, y)); y += t.height
    q = OUT / f"{name}.jpg"
    m.save(q, quality=88)
    print(f"  {q}  {m.size}")
