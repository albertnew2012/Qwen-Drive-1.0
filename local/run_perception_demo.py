#!/usr/bin/env python
"""Device-flexible perception inference over the bundled demo frames.

A replacement for ``scripts/run_perception.py``, which defaults to CUDA +
flash-attention. On CPU this relies on two torch fallbacks:
  * ``multi_scale_deformable_attn_pytorch``  (already in the upstream repo)
  * ``_voxel_pool_depth_torch``              (added in ops/__init__.py)

    PYTHONPATH=src python local/run_perception_demo.py --device cpu --attn sdpa
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwen_drive import QwenDriveForPlanning
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", type=Path, default=Path("data/demo/perception"))
    ap.add_argument("--output", type=Path, default=Path("outputs/perception_demo"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--only", default=None, help="comma-separated frame tokens")
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        if args.dtype == "bfloat16":
            # the CPU fallbacks are cleaner in fp32, and the head ships as fp32
            print("[cpu] switching dtype to float32", flush=True)
            args.dtype = "float32"
    dtype = getattr(torch, args.dtype)

    t0 = time.time()
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=dtype, attn_implementation=args.attn
    )
    vlm = holder.vlm
    del holder.planning_expert
    model = QwenDrivePerception.from_pretrained(args.model, dtype=dtype)
    model.to(args.device).eval()

    from transformers import AutoTokenizer
    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    model.attach(vlm.to(args.device).eval(), processor)
    print(f"[load] {time.time()-t0:.1f}s  device={args.device} dtype={args.dtype} attn={args.attn}",
          flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in args.frames.iterdir() if p.is_dir())
    if args.only:
        wanted = set(args.only.split(","))
        dirs = [p for p in dirs if p.name in wanted]

    for i, frame_dir in enumerate(dirs):
        frame = PerceptionFrame(frame_dir)
        t = time.time()
        inputs, img_metas = processor(frame, device=args.device)
        n_tok = int(inputs["input_ids"].shape[1])
        result = model.infer(inputs, img_metas)
        np.savez(args.output / f"{frame.token}.npz", **result)
        keep = (result["scores"] > 0.3).sum() if len(result["scores"]) else 0
        print(f"[{i+1}/{len(dirs)}] {frame.token} ({frame.dataset_type}, "
              f"{len(frame.cam_order)} cams, {n_tok} tokens) "
              f"{len(result['boxes'])} boxes ({keep} over 0.3)  "
              f"occ{tuple(result['occ'].shape)} map{tuple(result['map'].shape)}  "
              f"{time.time()-t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
