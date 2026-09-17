#!/usr/bin/env python
"""Ask the same questions about every image in a directory, loading the model once.

    .venv/bin/python local/run_vqa_dir.py --dir outputs/traffic_light_lane_association
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from qwen_drive import QwenDriveForPlanning

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

QUESTIONS = [
    ("lane", "Which lane is the ego vehicle in? Describe its position among the lanes "
             "and any marking painted on that lane."),
    ("ego_light", "What is the traffic light status for the ego vehicle's lane?"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--glob", default="*")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--max-new-tokens", type=int, default=220)
    # 1.0 is the released default but makes short factual answers loop
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    args = ap.parse_args()

    d = Path(args.dir)
    images = sorted(p for p in d.glob(args.glob)
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    if not images:
        raise SystemExit(f"no images under {d}")
    print(f"  {len(images)} images in {d}\n")

    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=DTYPES[args.dtype], attn_implementation="sdpa"
    ).to(args.device).eval()

    rows = []
    for p in images:
        print("=" * 86)
        print(f"  {p.name}")
        print("=" * 86, flush=True)
        for kind, q in QUESTIONS:
            t = time.time()
            r = model.generate_text([str(p)], q, max_new_tokens=args.max_new_tokens,
                                    repetition_penalty=args.repetition_penalty)
            print(f"  [{kind}] ({time.time() - t:.1f}s) {r.text}\n", flush=True)
            rows.append({"image": p.name, "kind": kind, "question": q, "answer": r.text})

    out = Path(args.out or d / "vqa_answers.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
