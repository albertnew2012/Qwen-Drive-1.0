#!/usr/bin/env python
"""Measure whether Qwen-Drive binds a traffic-light colour to a specific movement.

Two questions per frame - the colour of the left-turn signal, and the colour of the
through signal - scored against labels derived from gravity_tli_data. On CONFLICT frames
those two answers differ, so a model that reports whichever lamp is most salient cannot
score above chance on both at once. That separation is the measurement.

Two input conditions:
  full  the 4K frame as the model would normally receive it
  crop  a tight crop around the labelled signal heads, upscaled

The processor resizes any input to ~921k pixels, so on a 4K frame a head at 25 m is only
a few pixels across. If crop beats full, the limit is spatial resolution rather than
reasoning, and a detect-then-zoom front end is the fix.

    .venv/bin/python local/tli_eval_run.py --condition full --limit 405
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

COLOUR_RE = re.compile(r"\b(red|green|yellow|amber)\b", re.I)

QUESTIONS = {
    "left": "What colour is the left-turn arrow traffic signal? "
            "Answer with one word: red, yellow, or green.",
    "through": "What colour is the traffic signal that controls traffic going straight "
               "ahead? Answer with one word: red, yellow, or green.",
}


def first_colour(text: str):
    m = COLOUR_RE.search(text or "")
    if not m:
        return None
    c = m.group(1).lower()
    return "yellow" if c == "amber" else c


def crop_around(path: str, boxes, pad_frac=0.6, min_px=640):
    """Tight crop around every labelled head, with context, upscaled if small."""
    im = Image.open(path).convert("RGB")
    if not boxes:
        return im
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    w, h = x2 - x1, y2 - y1
    px, py = max(w * pad_frac, 120), max(h * pad_frac, 120)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(im.width, x2 + px), min(im.height, y2 + py)
    c = im.crop((int(x1), int(y1), int(x2), int(y2)))
    if max(c.size) < min_px:
        s = min_px / max(c.size)
        c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    return c


def history_paths(r: dict, n: int = 4, step: int = 18):
    """The current frame preceded by n-1 earlier ones, oldest first.

    Labelled frames sit every 9 indices of a 30 Hz stream, so step=18 is 0.6 s and a
    4-frame window covers 1.8 s - long enough to show the approach without the geometry
    changing beyond recognition.
    """
    src = Path(r["png"])
    out = []
    for k in range(n - 1, 0, -1):
        idx = r["frame"] - k * step
        if idx < 0:
            continue
        p = src.with_name(f"{r['cam']}-{idx:06d}.png")
        if p.exists():
            out.append(str(p))
    out.append(r["local"])          # current frame from the local copy
    return out


def build_input(cond: str, r: dict):
    """What actually reaches the processor for one frame.

    `full` leaves the path alone, so the processor applies current_image_pixels
    (921600) and a 4K frame is shrunk ~9x. `hires` sets target_size, which switches the
    cap to grid_pixel_limit (13.1 MP) and keeps the frame at native size. `crop` keeps
    native pixels but sends only the region containing signals. `temporal` sends a short
    history at the default budget.
    """
    from qwen_drive import CameraFrame
    if cond == "full":
        return [r["local"]]
    if cond == "hires":
        im = Image.open(r["local"]).convert("RGB")
        return [CameraFrame(im, target_size=im.size)]
    if cond == "temporal":
        return history_paths(r)
    if cond == "temporal_crop":
        boxes = r["gov_boxes_all"]
        out = []
        for p in history_paths(r):
            c = crop_around(p, boxes)
            out.append(CameraFrame(c, target_size=c.size))
        return out
    c = crop_around(r["local"], r["gov_boxes_all"])
    return [CameraFrame(c, target_size=c.size)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="outputs/tli_eval/eval_set.json")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--condition", default="full",
                    choices=["full", "hires", "crop", "temporal", "temporal_crop"])
    ap.add_argument("--only", default="all", choices=["all", "conflict", "agree"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = json.loads(Path(args.eval_set).read_text())
    if args.only == "conflict":
        rows = [r for r in rows if r["conflict"]]
    elif args.only == "agree":
        rows = [r for r in rows if not r["conflict"]]
    if args.limit:
        rows = rows[:args.limit]
    out_path = Path(args.out or f"outputs/tli_eval/results_{args.condition}_{args.only}.json")
    done = {}
    if out_path.exists():
        done = {d["name"]: d for d in json.loads(out_path.read_text())}
        print(f"  resuming, {len(done)} already scored")

    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    results = list(done.values())
    t0 = time.time()
    for i, r in enumerate(rows):
        if r["name"] in done:
            continue
        img = build_input(args.condition, r)
        rec = {"name": r["name"], "conflict": r["conflict"], "nearest_m": r["nearest_m"],
               "largest_box_px": r["largest_box_px"], "n_heads": r["n_heads"],
               "left_gt": r["left_gt"], "through_gt": r["through_gt"],
               "condition": args.condition, "n_images": len(img)}
        for kind, q in QUESTIONS.items():
            gt = r[f"{kind}_gt"]
            if gt is None:
                rec[f"{kind}_pred"] = None
                continue
            ans = model.generate_text(img, q,
                                      max_new_tokens=args.max_new_tokens,
                                      repetition_penalty=1.05).text
            rec[f"{kind}_pred"] = first_colour(ans)
            rec[f"{kind}_raw"] = ans.strip()[:120]
        results.append(rec)
        if (i + 1) % 20 == 0:
            n = len(results)
            def acc(kind, sel):
                v = [x for x in results if x.get(f"{kind}_gt") and sel(x)]
                return (sum(x[f"{kind}_pred"] == x[f"{kind}_gt"] for x in v) / len(v), len(v)) if v else (0, 0)
            lc, ln = acc("left", lambda x: x["conflict"])
            tc, tn = acc("through", lambda x: x["conflict"])
            print(f"    {n:4d} done  {time.time()-t0:5.0f}s | conflict left {lc:.0%} (n={ln}) "
                  f"through {tc:.0%} (n={tn})", flush=True)
            out_path.write_text(json.dumps(results, indent=1))

    out_path.write_text(json.dumps(results, indent=1))
    print(f"\n  wrote {out_path}  ({len(results)} frames, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
