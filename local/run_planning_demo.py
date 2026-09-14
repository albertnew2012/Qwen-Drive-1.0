#!/usr/bin/env python
"""Run the three Qwen-Drive inference modes over the bundled demo scenes.

A device/dtype-flexible replacement for ``scripts/demo.py``, which hardcodes
flash-attention (CUDA only). Results are cached to an .npz per scene so the
video renderer can run without re-doing inference.

    PYTHONPATH=src python local/run_planning_demo.py --device cpu --attn sdpa
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwen_drive import InferenceMode, QwenDriveForPlanning
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.images import ImageArchive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-rl")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--output", default="outputs/planning_demo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                    help="sdpa needs nothing extra; flash_attention_2 needs flash-attn installed")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--threads", type=int, default=12, help="CPU only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip scenes whose output already has every requested mode")
    ap.add_argument("--modes", default="direct,reasoning,vqa")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="cap on VQA generation; the default 32768 can stall for hours on CPU")
    ap.add_argument("--question", default="Describe the traffic scene and the safest action.")
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    modes = set(args.modes.split(","))
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = QwenDriveForPlanning.from_pretrained(
        args.model, planner=args.planner, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to(args.device).eval()
    print(f"[load] {time.time()-t0:.1f}s  device={args.device} attn={args.attn}", flush=True)

    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    samples = list(read_scene_file(
        args.scenes, image_archive=archive,
        num_history_points=model.config.num_history_points, limit=args.limit,
    ))
    print(f"[scenes] {len(samples)}", flush=True)

    for i, sample in enumerate(samples):
        scene = sample.scene
        existing = out_dir / f"{sample.token}.npz"
        if args.skip_existing and existing.exists():
            have = set(np.load(existing, allow_pickle=True).files)
            meta = out_dir / f"{sample.token}.json"
            if meta.exists():
                have |= {k.split("_")[0] for k in json.loads(meta.read_text())}
            wanted = {m for m in modes if m in ("direct", "reasoning")}
            if wanted <= have and ("vqa" not in modes or "vqa_answer" in
                                   (json.loads(meta.read_text()) if meta.exists() else {})):
                print(f"\n=== [{i+1}/{len(samples)}] {sample.token} - already done, skipping ===",
                      flush=True)
                continue
        record: dict = {"token": sample.token, "nav_command": int(scene.nav_command)}
        print(f"\n=== [{i+1}/{len(samples)}] {sample.token} "
              f"({scene.num_camera_frames} frames x {len(scene.views)} views) ===", flush=True)

        record["history"] = scene.history
        if sample.future_trajectory is not None:
            record["ground_truth"] = sample.future_trajectory

        def flush(rec=record, tok=sample.token):
            """Write after every mode, so a long run is never all-or-nothing."""
            np.savez(out_dir / f"{tok}.npz",
                     **{k: v for k, v in rec.items() if not isinstance(v, str)})
            (out_dir / f"{tok}.json").write_text(json.dumps(
                {k: v for k, v in rec.items() if isinstance(v, (str, int, float))}, indent=2))

        def score(name):
            if sample.future_trajectory is None or name not in record:
                return
            traj = record[name][0]
            err = np.linalg.norm(
                traj[:, :2] - sample.future_trajectory[: len(traj), :2], axis=-1)
            record[f"{name}_ade"] = float(err.mean())
            record[f"{name}_fde"] = float(err[-1])
            print(f"  {name}: ADE {err.mean():.3f} m  FDE {err[-1]:.3f} m", flush=True)

        # Trajectories first: they are the cheapest and carry the video.
        if "direct" in modes:
            t = time.time()
            d = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=args.num_samples)
            record["direct"] = d.trajectories
            print(f"[direct {time.time()-t:.1f}s] {d.trajectories.shape} "
                  f"endpoint={np.round(d.trajectory[-1],3).tolist()}", flush=True)
            score("direct"); flush()

        if "reasoning" in modes:
            t = time.time()
            r = model.run(InferenceMode.REASONING_PLANNING, scene=scene, num_samples=args.num_samples)
            record["reasoning"] = r.trajectories
            record["reasoning_text"] = r.reasoning
            print(f"[reasoning {time.time()-t:.1f}s] {r.reasoning}", flush=True)
            print(f"  endpoint={np.round(r.trajectory[-1],3).tolist()}", flush=True)
            score("reasoning"); flush()

        if "vqa" in modes:
            t = time.time()
            ans = model.run(InferenceMode.VQA, scene=scene, question=args.question,
                            max_new_tokens=args.max_new_tokens)
            record["vqa_question"] = args.question
            record["vqa_answer"] = ans.text
            print(f"[vqa {time.time()-t:.1f}s] {ans.text}", flush=True)
            flush()
        flush()
        print(f"  -> wrote {out_dir}/{sample.token}.npz", flush=True)


if __name__ == "__main__":
    main()
