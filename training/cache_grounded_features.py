"""Cache planner inputs AND the perception fields for the SAME nuScenes keyframe.

Stage 3 as released cannot ground the trajectory in perception, for a data reason
rather than a modelling one: the planning scenes are WOD-E2E with three camera
views, the perception head needs a full six- or eight-camera ring, and the two
shipped caches share no frames at all. nuScenes carries both - a complete ring
and a 5 s ego future - so one keyframe can feed both heads.

Each record holds everything ``train_planner.py`` already reads, plus two BEV
probability fields the grounded losses sample:

  occ_risk   [200, 200]      P(an object occupies this cell), collapsed over height
  map_probs  [6, 200, 400]   per-class map probabilities; drivability is chosen at
                             training time, because a car legitimately drives over
                             road_line and crosswalk as well as driveable_surface

Both come from the frozen perception head, so they are constants per frame and
the 4.5 B VLM never has to be loaded during planner training.

    PYTHONPATH=src:. python training/cache_grounded_features.py \
        --dataroot /path/to/nuscenes --frames-per-scene 1 --max-scenes 800
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive import QwenDriveForPlanning
from qwen_drive.trajectory import normalize_history, normalize_trajectory
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.configuration_perception import NUSCENES_OCC_PC_RANGE
from qwen_drive_perception.dataset import PerceptionProcessor
from transformers import AutoTokenizer

from local.nuscenes_session import CAM_ORDER, SessionFrame, build_scene, ego_track

# Only the first 7 of the 10 occupancy classes are objects; 7/8/9 are driveable,
# background and empty, which a trajectory is supposed to drive over.
OBJECT_CLASSES = 7
GROUND_BAND_TOP = 2.5          # metres; ignore overpasses above the ego's own height


def perception_fields(head, pproc, frame, device):
    """Run the frozen head once and return (occ_risk, map_probs) as probabilities."""
    captured = {}
    handle = head.bev_modeling.register_forward_hook(
        lambda module, args, output: captured.__setitem__("outs", output))
    try:
        pin, pmeta = pproc(frame, device=device)
        with torch.no_grad():
            head.infer(pin, pmeta)
    finally:
        handle.remove()
    outs = captured["outs"]

    occ = outs["occ_pred"].float().softmax(-1)           # [1, X, Y, Z, 10]
    obj = occ[..., :OBJECT_CLASSES].sum(-1)              # [1, X, Y, Z]
    z0, z1 = NUSCENES_OCC_PC_RANGE[2], NUSCENES_OCC_PC_RANGE[5]
    nz = obj.shape[-1]
    keep = max(1, min(nz, int(math.ceil((GROUND_BAND_TOP - z0) / ((z1 - z0) / nz)))))
    occ_risk = obj[..., :keep].amax(-1)                  # [1, X, Y]

    drivable = outs["seg_preds"].float().softmax(1)      # [1, 6, 200, 400], all classes
    return occ_risk.half().cpu(), drivable.half().cpu()


def usable_indices(n_keyframes, per_scene):
    """Keyframes with 3 behind them (the 2 Hz window) and 5 s of future ahead."""
    lo, hi = 3, n_keyframes - 11
    if hi <= lo:
        return []
    if per_scene == 1:
        return [(lo + hi) // 2]
    return list(np.linspace(lo, hi, min(per_scene, hi - lo + 1)).astype(int))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/zhengzhiliu/Documents/nuscenes_tv")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--out", default="data/train_cache_grounded")
    ap.add_argument("--frames-per-scene", type=int, default=1)
    ap.add_argument("--max-scenes", type=int, default=800)
    ap.add_argument("--scenes-file", default=None,
                    help="restrict to the scene names in this file, one per line; "
                         "used to densify the held-out split without touching train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"nuScenes {args.version}: {len(nusc.scene)} scenes, "
          f"{len(nusc.sample)} keyframes  ({time.time()-t0:.0f}s)")

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(args.device).eval()
    head = QwenDrivePerception.from_pretrained(
        args.perception, dtype=torch.bfloat16).to(args.device).eval()
    pproc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, pproc)
    proc = model.processor
    scale = model.trajectory_scale(args.device)
    print(f"models loaded, GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    manifest, done, skipped, t_start = [], 0, 0, time.time()
    only = None
    if args.scenes_file:
        only = {s.strip() for s in Path(args.scenes_file).read_text().split() if s.strip()}
        print(f"restricted to {len(only)} named scenes")
    pool = nusc.scene if only else nusc.scene[: args.max_scenes]
    for si, scene in enumerate(pool):
        if only is not None and scene["name"] not in only:
            continue
        samples, tok = [], scene["first_sample_token"]
        while tok:
            s = nusc.get("sample", tok); samples.append(s); tok = s["next"]
        picks = usable_indices(len(samples), args.frames_per_scene)
        if not picks:
            continue
        track = ego_track(nusc, samples[0])

        for idx in picks:
            token = f"{scene['name']}_{idx:03d}"
            rec = out / f"{token}.pt"
            if args.resume and rec.exists():
                skipped += 1; manifest.append(rec.name); continue
            try:
                frame = SessionFrame(nusc, samples[idx], Path(args.dataroot))
                occ_risk, map_probs = perception_fields(head, pproc, frame, args.device)

                scene_obj, fut = build_scene(nusc, samples, idx, track)
                inputs = proc(scene_obj, with_reasoning=False, device=args.device)
                scene_cache, anchor = model._prefill(inputs)

                fut_t = torch.as_tensor(fut, dtype=torch.float32,
                                        device=args.device).unsqueeze(0)
                torch.save({
                    "scene_cache": [(k.cpu(), v.cpu()) for k, v in scene_cache],
                    "anchor": anchor.cpu(),
                    "history": normalize_history(inputs["history"].float(), scale).cpu(),
                    "history_velocity": inputs["history_velocity"].cpu(),
                    "history_acceleration": inputs["history_acceleration"].cpu(),
                    "nav_command": inputs["nav_command"].cpu(),
                    "ego_status": inputs["ego_status"].cpu(),
                    "target_normalized": normalize_trajectory(fut_t, scale).cpu(),
                    "future_valid": torch.ones(1, fut_t.shape[1], dtype=torch.bool),
                    "occ_risk": occ_risk,
                    "map_probs": map_probs,
                    "pc_range": torch.tensor(NUSCENES_OCC_PC_RANGE),
                    "token": token,
                }, rec)
                manifest.append(rec.name); done += 1
            except Exception as exc:                 # a bad frame must not kill an 800-scene run
                print(f"  SKIP {token}: {type(exc).__name__}: {exc}", flush=True)
                continue
            finally:
                torch.cuda.empty_cache()

            if done % 10 == 0 or done == 1:
                rate = done / max(time.time() - t_start, 1e-6)
                gb = sum((out / m).stat().st_size for m in manifest) / 2**30
                print(f"  [{done:4d}] scene {si+1}/{min(len(nusc.scene), args.max_scenes)} "
                      f"{token:24s}  occ_risk max {float(occ_risk.max()):.2f}  "
                      f"{rate*3600:.0f}/h  {gb:.1f} GiB", flush=True)

    (out / "manifest.json").write_text(json.dumps(sorted(manifest), indent=2))
    gb = sum((out / m).stat().st_size for m in manifest) / 2**30
    print(f"\nwrote {done} records ({skipped} resumed), {gb:.1f} GiB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
