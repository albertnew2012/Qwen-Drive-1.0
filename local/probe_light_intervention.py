#!/usr/bin/env python
"""Does the model bind a traffic light to the ego's lane, or just report any light it sees?

Recognition and association look identical whenever the ego's light is also the nearest
and brightest one, which it is in every nuScenes frame checked so far. This separates
them by editing one lamp at a time in a single frame and holding everything else fixed.

scene-0796 kf21 shows two green heads: one on the mast arm over the ego's lane, one on
the right belonging to a different approach. Turning the ego's lamp red should change
the decision; turning the other lamp red should not. A model that merely scans for red
fails the second case, and that is the whole test.

    .venv/bin/python local/probe_light_intervention.py --make      # build the variants
    .venv/bin/python local/probe_light_intervention.py --ask       # query the model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("outputs/traffic_light/intervention")

# scene-0796 is a moving shot: traffic flows through the junction at 12 m/s, so a single
# recoloured lamp contradicts every other cue and the model can reasonably discount it.
# scene-0882 kf16 removes that confound - the ego is stopped on an empty wet junction,
# nothing is moving, and the lamps are the only evidence of right of way either way.
FRAMES = {
    "0796": {"src": Path("outputs/traffic_light/frames/scene-0796_f21_CAM_FRONT.jpg"),
             "ego": (882, 356), "other": (1267, 477), "decoy": (300, 250),
             "ego_r": 11, "other_r": 11, "flip": "red", "moving": True},
    "0882": {"src": Path("outputs/traffic_light/intervention/0882_kf16_full.jpg"),
             "ego": (236, 101), "other": (1021, 224), "decoy": (700, 120),
             "ego_r": 20, "other_r": 9, "flip": "green", "moving": False},
}


def questions(moving: bool):
    lead = ("What should the ego vehicle do at this intersection right now, and why?"
            if moving else
            "The ego vehicle is stopped at this intersection. Should it stay stopped or "
            "proceed, and why?")
    return [
        ("action", lead + " Answer in one sentence."),
        ("ego_colour", "What colour is the traffic light that governs the ego vehicle's lane?"),
        ("all_lights", "What colour is each traffic light visible in this image?"),
    ]


def recolour(img: Image.Image, centre, radius: int = 11, to: str = "red") -> Image.Image:
    """Repaint a lamp, keeping its glow profile so the result still looks like a lamp."""
    a = np.asarray(img).astype(float).copy()
    cx, cy = centre
    y0, y1 = max(0, cy - radius), min(a.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(a.shape[1], cx + radius + 1)
    patch = a[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # luminance carries the lamp's shape; only the hue is swapped
    lum = patch.max(axis=2, keepdims=True)
    falloff = np.clip(1.0 - d / radius, 0, 1)[..., None] ** 0.6
    tint = {"red": np.array([1.0, 0.13, 0.10]),
            "green": np.array([0.16, 1.0, 0.45])}[to]
    a[y0:y1, x0:x1] = patch * (1 - falloff) + (lum * tint) * falloff
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def build(key: str):
    cfg = FRAMES[key]
    OUT.mkdir(parents=True, exist_ok=True)
    base = Image.open(cfg["src"]).convert("RGB")
    to = cfg["flip"]
    ego, other, decoy = cfg["ego"], cfg["other"], cfg["decoy"]
    er, orr = cfg["ego_r"], cfg["other_r"]
    variants = {
        "baseline": base,
        "A_ego": recolour(base, ego, er, to),
        "B_other": recolour(base, other, orr, to),
        "C_decoy": recolour(base, decoy, er, to),
        "D_both": recolour(recolour(base, ego, er, to), other, orr, to),
    }
    for name, im in variants.items():
        im.save(OUT / f"{key}_{name}.jpg", quality=96)
    from PIL import ImageDraw
    tiles = []
    for name, im in variants.items():
        strip = Image.new("RGB", (760, 200), (18, 18, 18))
        for i, (c, r) in enumerate(((ego, er), (other, orr))):
            pad = max(45, 3 * r)
            z = im.crop((c[0] - pad, c[1] - pad, c[0] + pad, c[1] + pad)).resize((180, 180), Image.LANCZOS)
            strip.paste(z, (20 + i * 200, 12))
        d = ImageDraw.Draw(strip)
        d.text((420, 60), f"{key}  {name}  (-> {to})", fill=(255, 255, 0))
        d.text((420, 84), "left: EGO lamp", fill=(180, 180, 180))
        d.text((420, 104), "right: OTHER lamp", fill=(180, 180, 180))
        tiles.append(strip)
    m = Image.new("RGB", (760, 200 * len(tiles)), (18, 18, 18))
    for i, t in enumerate(tiles):
        m.paste(t, (0, i * 200))
    m.save(OUT / f"_{key}_variants_zoom.jpg", quality=95)
    print(f"  wrote {len(variants)} variants + _{key}_variants_zoom.jpg to {OUT}/")


def ask(args, key: str):
    import torch
    from qwen_drive import QwenDriveForPlanning
    qs = questions(FRAMES[key]["moving"])
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rows = []
    for name in ("baseline", "A_ego", "B_other", "C_decoy", "D_both"):
        p = OUT / f"{key}_{name}.jpg"
        print("=" * 84)
        print(f"  {key}  {name}")
        print("=" * 84, flush=True)
        for kind, q in qs:
            r = model.generate_text([str(p)], q, max_new_tokens=120, repetition_penalty=1.05)
            print(f"  [{kind}] {r.text}", flush=True)
            rows.append({"frame": key, "variant": name, "kind": kind,
                         "question": q, "answer": r.text})
        print(flush=True)
    (OUT / f"results_{key}.json").write_text(json.dumps(rows, indent=2))
    print(f"  wrote {OUT}/results_{key}.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--frame", default="0882", choices=sorted(FRAMES))
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--ask", action="store_true")
    args = ap.parse_args()
    if args.make or not args.ask:
        build(args.frame)
    if args.ask:
        ask(args, args.frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
