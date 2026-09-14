"""Stage 3 freezes the VLM, so prefill each scene ONCE and cache its attention cache.

The expert reads the VLM's post-rotary keys and values directly. With the VLM
frozen those are constant per scene, so the 4.5 B model never has to be loaded
during planner training.

    python training/cache_planner_features.py --out data/train_cache_plan
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive import QwenDriveForPlanning
from qwen_drive.images import ImageArchive
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.trajectory import normalize_history, normalize_trajectory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-root", default="data/demo")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--out", default="data/train_cache_plan")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(args.device).eval()
    proc = model.processor          # lazily built from the tokenizer + config

    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    samples = list(read_scene_file(Path(args.scenes), image_root=args.image_root,
                                   image_archive=archive))
    print(f"caching {len(samples)} planning scenes -> {out}")
    scale = model.trajectory_scale(args.device)
    manifest = []
    for s in samples:
        inputs = proc(s.scene, with_reasoning=False, device=args.device)
        scene_cache, anchor = model._prefill(inputs)

        fut = torch.as_tensor(np.asarray(s.future_trajectory), dtype=torch.float32,
                              device=args.device).unsqueeze(0)
        target = normalize_trajectory(fut, scale)
        rec = out / f"{s.token.replace('/', '_')}.pt"
        torch.save({
            "scene_cache": [(k.cpu(), v.cpu()) for k, v in scene_cache],
            "anchor": anchor.cpu(),
            # normalize_history drops the first point and rescales, exactly as
            # _sample_trajectories does at inference - the expert expects 16, not 17.
            "history": normalize_history(inputs["history"].float(), scale).cpu(),
            "history_velocity": inputs["history_velocity"].cpu(),
            "history_acceleration": inputs["history_acceleration"].cpu(),
            "nav_command": inputs["nav_command"].cpu(),
            "ego_status": inputs["ego_status"].cpu(),
            "target_normalized": target.cpu(),
            "future_valid": torch.as_tensor(np.asarray(s.future_valid)).cpu(),
            "token": s.token,
        }, rec)
        kv = sum(k.numel() + v.numel() for k, v in scene_cache) * 2 / 2**20
        print(f"  {s.token[:34]:34s}  cache {len(scene_cache)} groups, {kv:6.1f} MiB"
              f"   target {tuple(target.shape)}   {rec.stat().st_size/2**20:6.1f} MiB")
        manifest.append(rec.name)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
