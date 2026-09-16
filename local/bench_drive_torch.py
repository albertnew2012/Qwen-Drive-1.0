"""Time the PyTorch GPU path for perception + trajectory only - no CoT, no VQA.

The ONNX export of this model is launch-bound: the Gated-DeltaNet recurrence is
unrolled into 14,161 nodes per layer, so the decoder spends its time launching
kernels rather than running them.  PyTorch keeps the recurrence as a loop over
fused kernels, which is why it wins here by a wide margin.

Both tasks share one VLM.  Perception and planning still need two separate
prefills - perception sees 6 cameras at one timestamp, planning sees 3 views at
four timestamps with a different prompt, so neither token sequence is a prefix
of the other - but the weights are loaded once instead of twice.

    PYTHONPATH=src python local/bench_drive_torch.py
    PYTHONPATH=src python local/bench_drive_torch.py --compile
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwen_drive import InferenceMode, QwenDriveForPlanning
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.images import ImageArchive
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor


def timed(fn, reps: int, warmup: int = 1):
    """Median wall time of ``fn`` in ms, with the GPU flushed around each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times)), out


def _profile(head, model, inputs, img_metas, sample, args) -> None:
    """Compare GPU busy time against wall time to tell launch-bound from compute-bound."""
    from torch.profiler import ProfilerActivity, profile

    for tag, fn in (("perception", lambda: head.infer(inputs, img_metas)),
                    ("trajectory", lambda: model.run(InferenceMode.DIRECT_PLANNING,
                                                     scene=sample.scene,
                                                     num_samples=args.num_samples))):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                fn()
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t) * 1e3

        events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
        busy = sum(e.self_device_time_total for e in events) / 1e3
        launches = sum(e.count for e in events)
        print(f"\n  === {tag} ===")
        print(f"  wall {wall:.0f} ms | GPU busy {busy:.0f} ms "
              f"({100 * busy / wall:.0f}%) | {launches} kernel launches")
        print(f"  {'kernel':44s} {'ms':>8s} {'calls':>7s}")
        for e in sorted(events, key=lambda e: -e.self_device_time_total)[:10]:
            print(f"  {e.key[:44]:44s} {e.self_device_time_total/1e3:8.1f} {e.count:7d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-rl")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the VLM with CUDA graphs")
    ap.add_argument("--profile", action="store_true",
                    help="report GPU busy time vs wall time and the top kernels")
    ap.add_argument("--out", default="outputs/onnx_bench/latency_torch_cuda.json")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    dev = args.device

    t0 = time.time()
    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=dtype, attn_implementation=args.attn,
    ).to(dev).eval()
    head = QwenDrivePerception.from_pretrained(args.perception, dtype=dtype).to(dev).eval()
    from transformers import AutoTokenizer
    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, processor)
    load_s = time.time() - t0

    if args.compile:
        # reduce-overhead captures CUDA graphs, which is the whole point on a
        # launch-bound model; the shapes are static so replay should be valid.
        model.vlm = torch.compile(model.vlm, mode="reduce-overhead")

    print(f"[load] {load_s:.1f}s  device={dev} dtype={args.dtype} attn={args.attn} "
          f"compile={args.compile}  GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    frame = PerceptionFrame(Path(args.frame))
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    sample = next(iter(read_scene_file(
        args.scenes, image_archive=archive,
        num_history_points=model.config.num_history_points, limit=1)))

    with torch.no_grad():
        pre_ms, (inputs, img_metas) = timed(
            lambda: processor(frame, device=dev), args.reps)
        perc_ms, result = timed(
            lambda: head.infer(inputs, img_metas), args.reps)
        plan_ms, plan = timed(
            lambda: model.run(InferenceMode.DIRECT_PLANNING, scene=sample.scene,
                              num_samples=args.num_samples), args.reps)

    if args.profile:
        _profile(head, model, inputs, img_metas, sample, args)

    n_tok = int(inputs["input_ids"].shape[1])
    total = pre_ms + perc_ms + plan_ms
    print(f"\n  frame {frame.token}  ({len(frame.cam_order)} cams, {n_tok} tokens)")
    print(f"  scene {sample.token}  ({sample.scene.num_camera_frames} frames x "
          f"{len(sample.scene.views)} views)\n")
    print(f"  {'stage':28s} {'ms':>9s}")
    print(f"  {'perception preprocess (cpu)':28s} {pre_ms:9.1f}")
    print(f"  {'perception forward':28s} {perc_ms:9.1f}")
    print(f"  {'trajectory (10 euler steps)':28s} {plan_ms:9.1f}")
    print(f"  {'-'*38}")
    print(f"  {'TOTAL':28s} {total:9.1f}   ->  {1000/total:.2f} FPS")
    print(f"\n  boxes {len(result['boxes'])}  occ {tuple(result['occ'].shape)} "
          f"map {tuple(result['map'].shape)}")
    print(f"  endpoint {np.round(plan.trajectory[-1], 3).tolist()}")
    if sample.future_trajectory is not None:
        traj = plan.trajectories[0]
        err = np.linalg.norm(
            traj[:, :2] - sample.future_trajectory[: len(traj), :2], axis=-1)
        print(f"  ADE {err.mean():.3f} m   FDE {err[-1]:.3f} m")
    print(f"  peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dtype": args.dtype, "attn": args.attn, "compile": args.compile,
        "load_s": load_s, "total_ms": total, "fps": 1000 / total,
        "stages_ms": {"perception_preprocess": pre_ms,
                      "perception_forward": perc_ms,
                      "trajectory": plan_ms},
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
