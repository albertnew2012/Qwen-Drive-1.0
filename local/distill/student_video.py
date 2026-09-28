"""Demo video of the STUDENT over one held-out nuScenes scene, in the session-demo layout.

Same renderer as local/nuscenes_session.py (camera ring with 3-D boxes and the predicted /
recorded ego paths, BEV with lidar + boxes, occupancy, map), fed by the student instead of
the 4B teacher. Keyframes at 2 Hz; the previous keyframe's BEV is fused exactly as at
training time. Boxes are converted to the renderer's convention (lidar frame, bottom-centre
z, [x, y, z, w, l, h, yaw, vx, vy]); labels to the 7 teacher classes for the palette.

    python local/distill/student_video.py --scene scene-0276 --gpu 3
    -> outputs/student_video/scene-0276/student_scene-0276.mp4
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT / "src", _ROOT, _ROOT / "local"):
    sys.path.insert(0, str(p))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image

# fast local copy of data/distill (see PLAN.md); override with $DISTILL_CACHE
_CACHE = os.environ.get("DISTILL_CACHE", f"/local/{os.environ.get('USER', '')}/distill")

GROUP10 = np.array([0, 0, 0, 0, 0, 4, 2, 2, 5, 6])      # nuScenes 10 -> teacher 7 (palette)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/distill/exp/r2_long/snap_80000.pt")
    ap.add_argument("--scene", default="scene-0276")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--frames", default=_CACHE + "/frames_real")
    ap.add_argument("--teacher", default=_CACHE + "/teacher")
    ap.add_argument("--out", default="outputs/student_video")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--fps", type=float, default=2.0, help="playback rate of the 2 Hz keyframes (2 = real time)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.chdir(_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from nuscenes.nuscenes import NuScenes
    from nuscenes_session import SessionFrame, render_session_frame
    from qwen_drive_perception import geometry
    from local.distill.student import StudentConfig, StudentDetector
    from local.distill.train_student import DistillSet
    from local.distill.temporal import prev_bev_from_batch, warp_from_batch

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == args.scene)
    samples, cur = [], scene["first_sample_token"]
    while cur:
        samples.append(nusc.get("sample", cur)); cur = samples[-1]["next"]
    if args.limit:
        samples = samples[:args.limit]
    toks = [s["token"] for s in samples]

    ck = torch.load(args.ckpt, map_location="cpu"); saved = ck.get("cfg") or {}
    cfg = StudentConfig(**{k: v for k, v in saved.items() if k in StudentConfig.__init__.__code__.co_varnames})
    model = StudentDetector(cfg).eval()
    sd = ck.get("ema") or ck["model"]
    model.load_state_dict({(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() if k != "n_averaged"}, strict=False)
    model = model.cuda()
    ds = DistillSet(Path(args.frames), Path(args.teacher), cfg, toks, temporal=bool(cfg.temporal),
                    history=int(getattr(cfg, "history", 1)), occ_gt="teacher")
    assert len(ds) == len(toks), "every keyframe of the scene must have a frame dir"
    out = Path(args.out) / args.scene; (out / "frames").mkdir(parents=True, exist_ok=True)
    label = f"student {model.num_params()/1e6:.1f}M  {cfg.image_size[0]}x{cfg.image_size[1]}  step {ck.get('step')}"
    t0 = time.time(); meta = []
    with torch.no_grad():
        for i, (tok, smp) in enumerate(zip(toks, samples)):
            b = ds[i]; ego = b["ego"][None].cuda()
            if cfg.temporal:
                prev = prev_bev_from_batch(model, b, batched=False)
                pc, pb, pocc, pseg, pt, _ = model(b["image"][None].cuda(), b["bev_index"].cuda(), b["valid"].cuda(), ego, prev, warp_from_batch(b, batched=False))
            else:
                pc, pb, pocc, pseg, pt = model(b["image"][None].cuda(), b["bev_index"].cuda(), b["valid"].cuda(), ego)
            sc = torch.sigmoid(pc[0].float()); scores, lab = sc.max(-1)
            keep = scores >= args.thr
            bx = pb[0].float()[keep]                                   # ego frame, log sizes, sin/cos
            # renderer/head convention: index 3 is the extent ALONG the heading (nuScenes "length"),
            # index 4 the lateral one; the student's boxes carry nuScenes order [width, length]
            boxes_ego = torch.stack([bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 4].exp(), bx[:, 3].exp(), bx[:, 5].exp(),
                                     torch.atan2(bx[:, 6], bx[:, 7]), bx[:, 8], bx[:, 9]], -1)
            frame = SessionFrame(nusc, smp, Path(args.dataroot))
            frame.kind = label
            l2e = torch.as_tensor(frame.lidar2ego, dtype=boxes_ego.dtype, device=boxes_ego.device)
            boxes_lidar = geometry.ego_to_lidar_boxes(boxes_ego, l2e).clone()
            boxes_lidar[:, 2] -= boxes_lidar[:, 5] / 2                 # gravity centre -> bottom centre, as the head does
            result = {"boxes": boxes_lidar.cpu().numpy(), "scores": scores[keep].cpu().numpy(),
                      "labels": GROUP10[lab[keep].cpu().numpy()],
                      "occ": pocc[0].float().argmax(-1).to(torch.uint8).cpu().numpy(),
                      "map": pseg[0].float().argmax(0).to(torch.uint8).cpu().numpy()}
            traj = pt[0].float().cpu().numpy()[None]
            ef = Path("data/distill/ego") / f"{tok}.npz"
            gt_traj = None
            if ef.exists():
                e = np.load(ef)
                if ("has_future" not in e.files) or int(e["has_future"]) == 1:   # scene-end frames have a clamped future: no GT path, no ADE
                    gt_traj = e["future"]
            n_gt = int(len(frame.gt["labels"]))
            stats = [("frame", f"{i+1}/{len(toks)}"), ("boxes >%.2f" % args.thr, f"{int(keep.sum())} / {n_gt} GT")]
            if gt_traj is not None:
                err = np.linalg.norm(traj[0][:50, :2] - gt_traj[:50, :2], axis=-1); stats += [("ADE", f"{err.mean():.2f} m"), ("FDE", f"{err[-1]:.2f} m")]
            img = render_session_frame(frame, result, traj, gt_traj, stats=stats, score_threshold=0.0)
            Image.fromarray(img).save(out / "frames" / f"{i:04d}.png")
            meta.append({"i": i, "token": tok, "boxes": int(keep.sum()), "gt": n_gt})
            print(f"[{i+1}/{len(toks)}] {tok[:16]}  {int(keep.sum()):3d} boxes / {n_gt:3d} GT  ({time.time()-t0:.0f}s)", flush=True)
    (out / "frames.json").write_text(json.dumps(meta, indent=1))
    mp4 = out / f"student_{args.scene}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps}", "-i", str(out / "frames" / "%04d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)]
    subprocess.run(cmd, check=True)
    print(f"wrote {mp4} ({mp4.stat().st_size/1e6:.1f} MB, {len(toks)} keyframes at {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
