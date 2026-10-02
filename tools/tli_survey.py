"""Survey gravity_tli_data: which cameras carry labels, image sizes, and frame counts.

Run before building an evaluation set, so the harness is written against the real layout
rather than assumptions.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/perception_data/gravity_tli_data")
batches = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("batch_"))
print(f"  {len(batches)} batches\n")

def survey(b: Path):
    out = {"batch": b.name}
    lab = b / "labels" / b.name
    if not lab.is_dir():
        cands = [p for p in (b / "labels").iterdir()] if (b / "labels").is_dir() else []
        lab = cands[0] if cands else None
    if lab is None:
        return {**out, "error": "no labels dir"}
    tl = lab / "traffic_lights" / "KITTI_CAM_FRAME"
    if not tl.is_dir():
        return {**out, "error": "no KITTI_CAM_FRAME"}
    seqs = [p for p in tl.iterdir() if p.is_dir()]
    out["sequences"] = len(seqs)
    cams = {}
    for s in seqs:
        for c in s.iterdir():
            if c.is_dir():
                cams[c.name] = len(list(c.glob("*.txt")))
    out["labelled_cams"] = cams
    out["has_road_markings"] = (lab / "road_markings").is_dir()
    out["has_calib"] = (lab / "calib").is_dir()
    if seqs:
        s = seqs[0].name
        d = b / "data" / s
        if d.is_dir():
            png = {}
            for c in d.iterdir():
                p = c / "png_files"
                if p.is_dir():
                    png[c.name] = len(list(p.glob("*.png")))
            out["png_dirs"] = png
    return out

for b in batches[:6]:
    info = survey(b)
    print(f"  {info['batch']}")
    if "error" in info:
        print(f"      {info['error']}")
        continue
    print(f"      sequences={info['sequences']}  road_markings={info['has_road_markings']}  calib={info['has_calib']}")
    for c, n in list(info.get("labelled_cams", {}).items())[:4]:
        print(f"      labelled  {c}: {n} frames")
    for c, n in list(info.get("png_dirs", {}).items())[:4]:
        print(f"      png       {c}: {n} images")
    print()
