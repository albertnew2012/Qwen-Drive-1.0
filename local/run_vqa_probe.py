#!/usr/bin/env python
"""Ask several questions about single demo frames, loading the model once.

Single-image prompts are ~550 image tokens instead of the 3054 a full planning
scene needs, so each answer costs a fraction of a planning pass on CPU. The point
is to show both halves of the release's claim on the same model: driving-specific
understanding AND the retained general vision-language ability (OCR, counting,
open description) that Alpamayo lost.

    PYTHONPATH=src python local/run_vqa_probe.py --device cpu
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from qwen_drive import QwenDriveForPlanning
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.images import ImageArchive

# (view, question, kind) - view is picked from the scene's current frame
PROBES = [
    ("<FRONT VIEW>",       "What is the state of the traffic light ahead, and what should the ego vehicle do?", "driving"),
    ("<FRONT RIGHT VIEW>", "Read every piece of text visible in this image, including street signs and road markings.", "general / OCR"),
    ("<FRONT VIEW>",       "How many lanes are visible, and which lane is the ego vehicle travelling in?", "spatial"),
    ("<FRONT LEFT VIEW>",  "Describe this scene in one sentence. Is it day or night, and how do you know?", "general"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--scene-index", type=int, default=0)
    ap.add_argument("--output", default="outputs/vqa_probe.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="the released default is 1.0 (pure greedy with top_k=1), which "
                         "loops on open-ended enumeration; 1.05 fixes it")
    ap.add_argument("--only-kind", default=None, help="run just one probe kind")
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    t0 = time.time()
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation=args.attn
    ).to(args.device).eval()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    samples = list(read_scene_file(args.scenes, image_archive=archive,
                                   limit=args.scene_index + 1))
    scene = samples[args.scene_index].scene
    token = samples[args.scene_index].token
    print(f"[scene] {token}", flush=True)

    out = {"token": token, "answers": []}
    for view, question, kind in PROBES:
        if view not in scene.views:
            continue
        if args.only_kind and args.only_kind not in kind:
            continue
        image = scene.views[view][-1].load()          # the current frame only
        t = time.time()
        res = model.generate_text([image], question, max_new_tokens=args.max_new_tokens,
                                  repetition_penalty=args.repetition_penalty)
        dt = time.time() - t
        print(f"\n[{kind}] ({view}, {dt:.1f}s)\n  Q: {question}\n  A: {res.text}", flush=True)
        out["answers"].append({"view": view, "kind": kind, "question": question,
                               "answer": res.text, "seconds": round(dt, 1),
                               "repetition_penalty": args.repetition_penalty})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
