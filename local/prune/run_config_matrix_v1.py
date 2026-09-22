"""Measure the stacked configurations end to end, unattended.

Every lever has been measured in isolation. This runs the combinations that matter
and reports achieved Hz plus what each costs in accuracy, so the speed/quality
curve is one table rather than seven.

Levers, and where their numbers came from:

  euler_steps      10 -> 6 saves 143 ms and *improves* ADE by 0.011 m
                   (sweep_euler_steps_v1.py). Free.
  early_exit       detection decoder layer 3 instead of 5: 32/37 kept, 0.023 m,
                   but only 3 ms -- kept in the matrix to confirm it is not worth it.
  decoder_skip     12 of 32 language-model layers: 1.65x on the decoder, 30/37 kept
                   with no retraining (eval_depth_prune_v1.py).
  image_scale      planning frames resized; tokens go as the square. 0.707x saves
                   249 ms for +0.106 m ADE, which distillation would have to recover.

Perception quality is reported against the unmodified model on the same frame;
planning is reported against the ground-truth future, because there a lower step
count can legitimately win.

    python local/prune/run_config_matrix_v1.py
"""
from __future__ import annotations

import argparse, dataclasses, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

LEAST12 = [1, 3, 4, 8, 9, 12, 13, 14, 15, 16, 17, 20]

PERCEPTION_CONFIGS = [
    {"name": "shipped", "skip": [], "exit": 5},
    {"name": "exit3", "skip": [], "exit": 3},
    {"name": "skip12", "skip": LEAST12, "exit": 5},
    {"name": "skip12+exit3", "skip": LEAST12, "exit": 3},
    {"name": "skip16", "skip": LEAST12 + [2, 10, 21, 25], "exit": 3},
]
PLANNING_CONFIGS = [
    {"name": "shipped", "steps": 10, "scale": 1.0},
    {"name": "euler6", "steps": 6, "scale": 1.0},
    {"name": "euler6+img.85", "steps": 6, "scale": 0.85},
    {"name": "euler6+img.707", "steps": 6, "scale": 0.707},
    {"name": "euler4+img.707", "steps": 4, "scale": 0.707},
]


def dets(cls, box, thr=0.3):
    p = 1 / (1 + np.exp(-cls.max(-1)))
    return p >= thr, cls.argmax(-1), box[:, :3]


def perception_matrix(args, report):
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor
    from local.prune.measure_layer_influence_v1 import build, run_stack

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    head = QwenDrivePerception.from_pretrained(
        args.perception, dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, proc)
    vlm, lm = model.vlm, model.vlm.model.language_model
    types = list(vlm.config.text_config.layer_types)

    x, inputs, metas, mask = build(vlm, args.frame, proc)
    pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
    grid = inputs["image_grid_thw"]
    n_cam = len(metas["cam_order"]); gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
    tpi = gh // 2 * gw // 2
    dt = next(head.bev_modeling.parameters()).dtype
    cap = {}
    hk = vlm.model.visual.merger.register_forward_hook(
        lambda m, a, o=None: cap.__setitem__("p", a[0]))
    with torch.no_grad():
        t = time.perf_counter()
        vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
        torch.cuda.synchronize()
        vision_ms = (time.perf_counter() - t) * 1e3
    hk.remove()
    with torch.no_grad():
        vit = torch.stack(head._premerge_grids(
            vlm.model.visual.merger.norm(cap["p"]), grid)[-n_cam:], 0).to(dt)
    del cap
    torch.cuda.empty_cache()
    bev = head.bev_modeling

    def once(skip, exit_layer):
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            hidden, _ = run_stack(lm, types, x, pos, skip=set(skip))
            torch.cuda.synchronize(); dec = (time.perf_counter() - t0) * 1e3
            llm = hidden[0][mask][-n_cam*tpi:].view(n_cam, gh//2, gw//2, -1).to(dt)
            t0 = time.perf_counter()
            o = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
            torch.cuda.synchronize(); hd = (time.perf_counter() - t0) * 1e3
        return (o["all_cls_scores"][exit_layer, 0].float().cpu().numpy(),
                o["all_bbox_preds"][exit_layer, 0].float().cpu().numpy(), dec, hd)

    print(f"  vision tower {vision_ms:.1f} ms  (shared by every configuration)\n")
    c0, b0, _, _ = once([], 5)
    base = dets(c0, b0, args.thr)
    print(f"  {'config':16s} {'dec ms':>7s} {'head ms':>8s} {'total':>8s} {'Hz':>6s} "
          f"{'kept':>9s} {'spur':>5s} {'centre_med':>11s}")
    rows = {}
    for cfg in PERCEPTION_CONFIGS:
        for _ in range(1):
            once(cfg["skip"], cfg["exit"])
        dec_l, head_l = [], []
        for _ in range(3):
            cls, box, dec, hd = once(cfg["skip"], cfg["exit"])
            dec_l.append(dec); head_l.append(hd)
        dec, hd = float(np.median(dec_l)), float(np.median(head_l))
        c = dets(cls, box, args.thr)
        both = base[0] & c[0]
        cd = (float(np.median(np.linalg.norm(base[2][both] - c[2][both], axis=-1)))
              if both.any() else float("nan"))
        total = vision_ms + dec + hd
        rows[cfg["name"]] = {"skip": cfg["skip"], "exit": cfg["exit"],
                             "vision_ms": vision_ms, "decoder_ms": dec, "head_ms": hd,
                             "total_ms": total, "hz": 1000.0 / total,
                             "kept": int(both.sum()), "of": int(base[0].sum()),
                             "spurious": int((c[0] & ~base[0]).sum()),
                             "centre_median_m": cd}
        r = rows[cfg["name"]]
        print(f"  {cfg['name']:16s} {dec:7.1f} {hd:8.1f} {total:8.1f} {r['hz']:6.2f} "
              f"{r['kept']:4d}/{r['of']:<4d} {r['spurious']:5d} {cd:11.3f}", flush=True)
    report["perception"] = rows
    del model, head, x, vit
    torch.cuda.empty_cache()


def planning_matrix(args, report):
    from qwen_drive import QwenDriveForPlanning, InferenceMode
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.images import ImageArchive
    from local.prune.sweep_planning_pixels_v2 import rescale

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    samples = list(read_scene_file(args.scenes, image_archive=archive,
                                   num_history_points=model.config.num_history_points,
                                   limit=args.scenes_limit))
    print(f"\n  {'config':16s} {'wall ms':>8s} {'Hz':>6s} {'ADE m':>8s} {'FDE m':>8s}")
    rows = {}
    for cfg in PLANNING_CONFIGS:
        model.config.num_inference_steps = cfg["steps"]
        ades, fdes, wall = [], [], []
        for sample in samples:
            sc = rescale(sample.scene, cfg["scale"], "current")
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                plan = model.run(InferenceMode.DIRECT_PLANNING, scene=sc, num_samples=1)
            torch.cuda.synchronize(); wall.append((time.perf_counter() - t) * 1e3)
            traj = np.asarray(plan.trajectories[0], dtype=np.float64)
            gt = sample.future_trajectory
            if gt is not None:
                m = min(len(traj), len(gt))
                d = np.linalg.norm(traj[:m, :2] - np.asarray(gt)[:m, :2], axis=-1)
                ades.append(float(d.mean())); fdes.append(float(d[-1]))
        ms = float(np.median(wall))
        rows[cfg["name"]] = {"steps": cfg["steps"], "scale": cfg["scale"],
                             "wall_ms": ms, "hz": 1000.0 / ms,
                             "ade_m": float(np.mean(ades)), "fde_m": float(np.mean(fdes))}
        r = rows[cfg["name"]]
        print(f"  {cfg['name']:16s} {ms:8.1f} {r['hz']:6.2f} {r['ade_m']:8.4f} "
              f"{r['fde_m']:8.4f}", flush=True)
    report["planning"] = rows
    del model
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--scenes-limit", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/prune/config_matrix_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    report = {}
    perception_matrix(args, report)
    planning_matrix(args, report)

    # the frame rate if the two run on separate cards
    best_p = min(report["perception"].values(), key=lambda r: r["total_ms"])
    best_l = min(report["planning"].values(), key=lambda r: r["wall_ms"])
    frame = max(best_p["total_ms"], best_l["wall_ms"])
    report["parallel_two_gpu"] = {"perception_ms": best_p["total_ms"],
                                  "planning_ms": best_l["wall_ms"],
                                  "frame_ms": frame, "hz": 1000.0 / frame}
    print(f"\n  fastest perception {best_p['total_ms']:.0f} ms, "
          f"fastest planning {best_l['wall_ms']:.0f} ms")
    print(f"  two cards in parallel: {frame:.0f} ms -> {1000.0/frame:.2f} Hz "
          f"(target 3 Hz = 333 ms)")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
