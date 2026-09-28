"""The teacher's planned trajectory on the student's evaluation frames.

The teacher's ADE of 1.393 m was measured on the 250 frame-split frames; the student is now
scored on 250 frames from the held-out val scenes, a more dynamic set (constant-velocity
anchor 2.26 m vs 2.01 m). For a like-for-like "retained %" on trajectory the planner has to be
run on those same frames. Scenes are built exactly as in local/nuscenes_session.py (four
front keyframes over 1.5 s, ego history from the dense pose track, nav from the future).

    python local/distill/teacher_traj.py --gpu 3
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "local"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--tokens", default="scene")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--mode", default="direct", choices=["direct", "reasoning"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="teacher-traj-valscenes")
    args = ap.parse_args()
    os.chdir(_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from nuscenes.nuscenes import NuScenes
    from nuscenes_session import ego_track, build_scene
    from qwen_drive import InferenceMode, QwenDriveForPlanning
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
    toks = json.load(open("data/distill/scene_split.json"))["eval250"][:args.limit] if args.tokens == "scene" else \
        sorted(p.stem for p in Path("data/distill/teacher").glob("*.npz"))[:args.limit]
    t0 = time.time()
    model = QwenDriveForPlanning.from_pretrained(args.vlm, planner=args.planner, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    print(f"  planner loaded in {time.time()-t0:.0f}s; {len(toks)} frames", flush=True)
    mode = InferenceMode.DIRECT_PLANNING if args.mode == "direct" else InferenceMode.REASONING_PLANNING

    scene_cache = {}
    ades, fdes, cv_ades, ego_diff, out = [], [], [], [], {}
    skipped = 0
    for i, tok in enumerate(toks):
        s = nusc.get("sample", tok)
        st = s["scene_token"]
        if st not in scene_cache:
            samples, cur = [], nusc.get("scene", st)["first_sample_token"]
            while cur:
                samples.append(nusc.get("sample", cur)); cur = samples[-1]["next"]
            scene_cache[st] = (samples, ego_track(nusc, samples[0]))
        samples, track = scene_cache[st]
        idx = next(k for k, x in enumerate(samples) if x["token"] == tok)
        if idx < 3:
            skipped += 1; continue                       # needs 1.5 s of history behind it
        sc, fut = build_scene(nusc, samples, idx, track)
        with torch.no_grad():
            plan = model.run(mode, scene=sc, num_samples=1)
        traj = np.asarray(plan.trajectories[0], dtype=np.float32)       # (50, 3) x, y, heading
        n = min(len(traj), len(fut))
        err = np.linalg.norm(traj[:n, :2] - fut[:n, :2], axis=-1)
        ades.append(float(err.mean())); fdes.append(float(err[-1]))
        # constant-velocity anchor from the last history velocity, the student's own anchor
        v = sc.history_velocity[-1]; t = np.arange(1, n + 1) / 10.0
        cv = np.stack([v[0] * t, v[1] * t], -1)
        cv_ades.append(float(np.linalg.norm(cv - fut[:n, :2], axis=-1).mean()))
        e = Path("data/distill/ego") / f"{tok}.npz"
        if e.exists():                                   # convention check against the student's GT
            ego_diff.append(float(np.abs(np.load(e)["future"][:n, :2] - fut[:n, :2]).mean()))
        out[tok] = traj
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(toks)}  running ADE {np.mean(ades):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    rec = {"tag": args.tag, "teacher": True, "mode": args.mode, "eval_tokens": args.tokens, "frames": len(ades),
           "skipped_no_history": skipped, "trajectory_ade_m": float(np.mean(ades)), "trajectory_fde_m": float(np.mean(fdes)),
           "cv_anchor_ade_m": float(np.mean(cv_ades)),
           "gt_convention_mean_abs_diff_m": (float(np.mean(ego_diff)) if ego_diff else None),
           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path("outputs/distill").mkdir(parents=True, exist_ok=True)
    np.savez(f"outputs/distill/{args.tag}.npz", **{k: v for k, v in out.items()})
    nb = json.load(open("outputs/distill/lab_notebook.json")); nb.append(rec)
    json.dump(nb, open("outputs/distill/lab_notebook.json", "w"), indent=1)
    print(f"\n=== TEACHER trajectory ({args.mode}) on {len(ades)} {args.tokens} frames: ADE {rec['trajectory_ade_m']:.3f} m, "
          f"FDE {rec['trajectory_fde_m']:.3f} m; constant-velocity anchor {rec['cv_anchor_ade_m']:.3f} m; "
          f"GT convention diff vs ego cache {rec['gt_convention_mean_abs_diff_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
