#!/usr/bin/env python
"""The `assets/nuscenes_session.gif` render, with the student in place of the teacher.

Same scene, same instants, same layout as `tools/nuscenes_session.py` (the script that made the
teacher's session video): every ~10 Hz camera sweep of the scene that has 1.5 s of history behind it
and 5 s of recorded future ahead, drawn with `render_session_frame`, the same six stats rows, played
at 0.75x real speed. Only the model differs:

  * perception and trajectory come from the student: either its PyTorch checkpoint (`--ckpt`) or its
    exported ONNX graph (`--onnx`, onnxruntime only). Both see byte-identical inputs, so their renders
    can be compared frame by frame (`--compare <other run's --out>`);
  * the student has no language model, so the chain-of-thought panel stays empty;
  * the student was trained with its BEV history 0.5 s apart (2 Hz keyframes). At the 10 Hz sweep
    rate its three history states are taken from the frames 0.5, 1.0 and 1.5 s back, warped by ego
    motion, so the graph sees the spacing it was trained on. The 1.5 s before the first rendered
    instant are run (not drawn) to fill that history, exactly as a deployment would warm up.

    source export_onnx/env_gpu.sh; export PYTHONPATH=src:.:tools
    # 1. PyTorch checkpoint: render, then export the same checkpoint to ONNX
    python distill/student_session.py --ckpt outputs/student_release_v6/student_r4_small_avg.pt \
        --out outputs/student_session_pytorch --gif outputs/student_session_pytorch/nuscenes_session_student_pytorch.gif \
        --export-onnx outputs/student_export/student_r4_small_avg.onnx
    # 2. the exported ONNX: same render, compared frame by frame with step 1
    python distill/student_session.py --onnx outputs/student_export/student_r4_small_avg.onnx \
        --out outputs/student_session --gif assets/nuscenes_session_student.gif --compare outputs/student_session_pytorch
    # 3. rebuild only a GIF from frames already rendered
    python distill/student_session.py --gif-only --out outputs/student_session --gif assets/nuscenes_session_student.gif
"""
from __future__ import annotations

import argparse, json, math, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src", _ROOT / "tools"):
    sys.path.insert(0, str(p))

from distill.demo_onnx import CAM_ORDER, PC_RANGE, DEPTH_RANGE, GROUP10, pose_matrix, boxes_for_renderer

HIST_SPACING_S = 0.5          # the keyframe spacing the student's BEV history was trained with


def ego_state_at(track, ref_pose, n_hist=16, hist_s=1.5, n_fut=50, hz=10.0):
    """The 116-d planner input, referenced to an arbitrary ego pose record (here: the lidar sweep's).

    Identical construction to demo_onnx.ego_state / nusc_ego.py: 16 history poses at 0.09375 s,
    their velocity and acceleration gradients, the nav one-hot from the 5 s lateral offset, speed.
    """
    from pyquaternion import Quaternion
    from nuscenes_session import resample
    t0 = ref_pose["timestamp"] * 1e-6
    ref = pose_matrix(ref_pose); ref_inv = np.linalg.inv(ref)
    ref_yaw = Quaternion(ref_pose["rotation"]).yaw_pitch_roll[0]
    dt = hist_s / n_hist
    hist = resample(track, t0, -np.arange(n_hist)[::-1] * dt, ref_inv, ref_yaw)
    vel = np.gradient(hist[:, :2], dt, axis=0).astype(np.float32)
    acc = np.gradient(vel, dt, axis=0).astype(np.float32)
    fut = resample(track, t0, np.arange(1, n_fut + 1) / hz, ref_inv, ref_yaw)
    lat = float(fut[-1, 1])
    nav = 0 if abs(lat) < 4.0 else (1 if lat > 0 else 2)
    vec = np.concatenate([hist.reshape(-1), vel.reshape(-1), acc.reshape(-1), np.eye(3, dtype=np.float32)[nav],
                          np.asarray([np.linalg.norm(vel[-1])], np.float32)]).astype(np.float32)
    return vec, fut, float(np.linalg.norm(vel[-1]))


def make_gif(frames_dir: Path, gif: Path, src_fps: float, gif_fps: float = 4.0, seconds: float = 12.0,
             size=(1238, 1506)):
    """The teacher GIF's recipe: the first `seconds` of the 0.75x video sampled at 4 fps.

    The reference picks source frames ceil(k * src_fps / gif_fps) for k = 0..47 (1, 3, 5, 7, 9, 11,
    13, 15, 16, 18 ... 90 of 148), 250 ms each, looping, padded to 1238x1506 on white.
    """
    pngs = sorted(p for p in frames_dir.glob("[0-9][0-9][0-9][0-9].png"))
    n_out = int(round(seconds * gif_fps))
    # the measured rate is 7.500007 fps, not 7.5: round the ratio or one ceiling tips over (17 instead of 16)
    ratio = round(src_fps / gif_fps, 3)
    pick = [min(math.ceil(k * ratio - 1e-9), len(pngs) - 1) for k in range(n_out)]
    seq = frames_dir.parent / "gif_seq"; seq.mkdir(exist_ok=True)
    for old in seq.glob("*.png"):
        old.unlink()
    for j, i in enumerate(pick):
        (seq / f"{j:04d}.png").symlink_to(pngs[i].resolve())
    w, h = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{gif_fps:g}", "-i", str(seq / "%04d.png"),
           "-vf", f"pad={w}:{h}:0:0:white,split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
           "-loop", "0", str(gif)]
    subprocess.run(cmd, check=True)
    print(f"  GIF: {len(pick)} frames (source {[i + 1 for i in pick[:10]]} ... {pick[-1] + 1}) at {gif_fps:g} fps -> {gif}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="outputs/student_release_v6/student_r4_small_avg.onnx",
                    help="run this ONNX graph with onnxruntime (ignored when --ckpt is given)")
    ap.add_argument("--ckpt", default="", help="run this PyTorch checkpoint instead of an ONNX graph")
    ap.add_argument("--export-onnx", default="", help="with --ckpt: after rendering, export the checkpoint to this .onnx")
    ap.add_argument("--compare", default="", help="another run's --out dir: compare its per-frame outputs with this run's")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--future", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=0.75)
    ap.add_argument("--thr", type=float, default=0.3, help="score threshold: the teacher render's 'boxes >0.3'")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="outputs/student_session")
    ap.add_argument("--gif", default="", help="also write the 48-frame GIF here")
    ap.add_argument("--gif-only", action="store_true", help="skip rendering; build the GIF from existing frames")
    args = ap.parse_args()
    os.chdir(_ROOT)
    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)

    if args.gif_only:
        enc = json.loads((out / "encode.json").read_text())
        make_gif(out / "frames", Path(args.gif), enc["out_fps"])
        return 0

    from nuscenes.nuscenes import NuScenes
    from nuscenes_session import (SessionFrame, render_session_frame, ego_track, sensor_chain, _nearest,
                                  interp_global_anns)
    from distill.geometry import bev_indices_all
    from distill.temporal import warp_grid_from_T

    OUT_NAMES = ["cls", "box", "occ", "seg", "trajectory", "bev_state"]
    if args.ckpt:                                   # ---- PyTorch checkpoint
        import torch
        from distill.student import StudentConfig, StudentDetector
        ck = torch.load(args.ckpt, map_location="cpu")
        mcfg = StudentConfig(**{k: v for k, v in ck["cfg"].items() if k in StudentConfig.__init__.__code__.co_varnames})
        tdev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
        model = StudentDetector(mcfg).eval().to(tdev)
        model.load_state_dict(ck.get("ema") or ck["model"], strict=True)
        W, H = mcfg.image_size; n_cams = mcfg.n_cams; depth_bins = mcfg.depth_bins
        K = int(getattr(mcfg, "history", 1)); C, bev = mcfg.bev_channels, mcfg.bev_size
        depth_range, pc_range = tuple(mcfg.depth_range), tuple(mcfg.pc_range)
        assert mcfg.temporal, "the session render needs a temporal student"

        def run(feeds):
            with torch.no_grad():
                t = {k: torch.from_numpy(v).to(tdev) for k, v in feeds.items()}
                o = model(t["image"], t["bev_index"], t["valid"], t["ego"], t["prev_bev"], t["warp_grid"])
            if tdev.type == "cuda":
                torch.cuda.synchronize(tdev)
            return {n: x.float().cpu().numpy() for n, x in zip(OUT_NAMES, o)}
        backend, tag = f"PyTorch checkpoint {args.ckpt} (step {ck.get('step')}) on {tdev}", "student PyTorch"
    else:                                           # ---- ONNX graph, onnxruntime only
        import onnxruntime as ort
        if not Path(args.onnx).exists():
            raise SystemExit(f"no ONNX graph at {args.onnx} -- run the PyTorch step with --export-onnx first")
        so = ort.SessionOptions(); so.log_severity_level = 3
        providers = [("CUDAExecutionProvider", {"device_id": args.gpu})] if args.gpu >= 0 else ["CPUExecutionProvider"]
        s = ort.InferenceSession(args.onnx, so, providers=providers)
        dev = "cuda" if "CUDAExecutionProvider" in s.get_providers() and args.gpu >= 0 else "cpu"
        shapes = {i.name: i.shape for i in s.get_inputs()}
        _, n_cams, _, H, W = shapes["image"]
        L = shapes["bev_index"][0]; depth_bins = L // (n_cams * (H // 16) * (W // 16))
        pb_shape = shapes["prev_bev"]; K = pb_shape[1] if len(pb_shape) == 5 else 1
        C, bev = pb_shape[-3], pb_shape[-1]
        depth_range, pc_range = DEPTH_RANGE, PC_RANGE
        out_names = [o.name for o in s.get_outputs()]

        def run(feeds):
            return dict(zip(out_names, s.run(out_names, feeds)))
        backend, tag = f"ONNX graph {args.onnx}, onnxruntime on {dev}", "student ONNX"
    cfg = SimpleNamespace(image_size=(W, H), depth_bins=depth_bins, depth_range=depth_range, bev_size=bev,
                          pc_range=pc_range, n_cams=n_cams)
    print(f"  {backend}: {W}x{H}, {n_cams} cams, BEV {bev}, history K={K}", flush=True)

    # ---- the teacher render's instants, verbatim (tools/nuscenes_session.py main) ----
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    samples, tok = [], scene["first_sample_token"]
    while tok:
        smp = nusc.get("sample", tok); samples.append(smp); tok = smp["next"]
    print(f"scene {args.scene}: {scene['name']}  {len(samples)} keyframes @ 2 Hz ({scene['description'][:60]})")
    track = ego_track(nusc, samples[0])
    sample_ts = np.asarray([x["timestamp"] * 1e-6 for x in samples])
    cam_chains = {c: sensor_chain(nusc, samples[0], c) for c in CAM_ORDER}
    cam_ts = {c: np.asarray([x["timestamp"] * 1e-6 for x in cam_chains[c]]) for c in CAM_ORDER}
    lidar_chain = sensor_chain(nusc, samples[0], "LIDAR_TOP")
    lidar_ts = np.asarray([x["timestamp"] * 1e-6 for x in lidar_chain])
    t_start = sample_ts[0] + 1.5
    t_end = sample_ts[-1] - args.future
    drive = cam_ts["CAM_FRONT"]
    times = drive[(drive >= t_start) & (drive <= t_end)]
    src_hz = 1.0 / float(np.median(np.diff(times)))
    out_fps = src_hz * args.speed
    # warm-up: the sweeps in the 1.5 s before the first rendered instant fill the BEV history
    warm = drive[(drive >= t_start - 3 * HIST_SPACING_S - 0.06) & (drive < t_start)]
    print(f"  rendering {len(times)} frames at {src_hz:.1f} Hz -> {out_fps:.2f} fps ({args.speed}x real speed); "
          f"{len(warm)} warm-up frames fill the 1.5 s history first", flush=True)

    past = []          # [(t, E (4x4 ego2global of the lidar sweep), state (1,C,bev,bev) float32)]
    meta, lat = [], []
    t_all = time.time()
    for n, t in enumerate(np.concatenate([warm, times])):
        render = n >= len(warm); i = n - len(warm)
        owner = samples[max(0, min(int(np.searchsorted(sample_ts, t) - 1), len(samples) - 1))]
        cam_sd = {c: _nearest(cam_chains[c], cam_ts[c], t) for c in CAM_ORDER}
        lidar_sd = _nearest(lidar_chain, lidar_ts, t)
        gt_global = interp_global_anns(nusc, samples, sample_ts, t) if render else None
        frame = SessionFrame(nusc, owner, Path(args.dataroot), cam_sd=cam_sd, lidar_sd=lidar_sd, gt_global=gt_global,
                             token=f"{owner['token'][:12]}{(t - owner['timestamp'] * 1e-6) * 1000:+.0f}ms")
        dt_ms = abs(t - owner["timestamp"] * 1e-6) * 1000
        frame.kind = ("nuScenes keyframe" if dt_ms < 1 else f"nuScenes sweep  ({src_hz:.0f} Hz, {dt_ms:+.0f} ms from keyframe)") \
            + f"   |   {tag}"

        metas = frame.img_metas(image_size=(W, H))
        idx, val = bev_indices_all(metas["lidar2img"], metas["lidar2ego"], cfg)
        imgs = np.stack([((np.asarray(frame.image(c).resize((W, H), Image.BILINEAR), np.float32) / 255.0 - 0.5) / 0.5)
                         .transpose(2, 0, 1) for c in CAM_ORDER], 0)[None].astype(np.float32)
        lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        E = pose_matrix(lidar_pose)
        ego_vec, fut, speed = ego_state_at(track, lidar_pose)
        feeds = {"image": imgs, "bev_index": idx.reshape(-1).astype(np.int64), "valid": val.reshape(-1).astype(bool),
                 "ego": ego_vec[None]}

        if not past:      # very first frame: its own state, identity warp (the training-time padding)
            z = np.zeros((1, K, C, bev, bev), np.float32) if K > 1 else np.zeros((1, C, bev, bev), np.float32)
            zg = np.zeros((1, K, bev, bev, 2), np.float32) if K > 1 else np.zeros((1, bev, bev, 2), np.float32)
            own = run(dict(feeds, prev_bev=z, warp_grid=zg))["bev_state"]
            past = [(t, E, own)]
        chosen = []
        for k in range(1, K + 1):   # the states nearest to 0.5, 1.0, 1.5 s back (repeat the oldest if short)
            target = t - k * HIST_SPACING_S
            chosen.append(min(past, key=lambda r: abs(r[0] - target)))
        grids = [warp_grid_from_T(np.linalg.inv(E) @ Ep, cfg)[None].astype(np.float32) for _, Ep, _ in chosen]
        if K > 1:
            feeds["prev_bev"] = np.stack([st[0] for _, _, st in chosen], 0)[None]
            feeds["warp_grid"] = np.stack(grids, 1)
        else:
            feeds["prev_bev"] = chosen[0][2]; feeds["warp_grid"] = grids[0]
        t0 = time.perf_counter()
        outs = run(feeds)
        lat.append((time.perf_counter() - t0) * 1e3)
        past = [r for r in past if r[0] >= t - 3 * HIST_SPACING_S - 0.3] + [(t, E, outs["bev_state"])]
        if not render:
            continue

        sc = 1.0 / (1.0 + np.exp(-outs["cls"][0].astype(np.float64)))
        scores, labels = sc.max(-1), sc.argmax(-1)
        keep = scores >= args.thr
        result = {"boxes": boxes_for_renderer(outs["box"][0][keep], frame.lidar2ego),
                  "scores": scores[keep].astype(np.float32), "labels": GROUP10[labels[keep]],
                  "occ": outs["occ"][0].argmax(-1).astype(np.uint8), "map": outs["seg"][0].argmax(0).astype(np.uint8)}
        traj = outs["trajectory"][0][None]
        err = np.linalg.norm(traj[0][:, :2] - fut[:, :2], axis=-1)
        path_len = float(np.linalg.norm(np.diff(traj[0][:, :2], axis=0), axis=1).sum())
        stats = [("frame", f"{i + 1}/{len(times)}"),
                 ("speed", f"{speed * 3.6:.1f} km/h"),
                 ("path", f"{path_len:.1f} m / 5 s"),
                 ("ADE", f"{err.mean():.2f} m"),
                 ("FDE", f"{err[-1]:.2f} m"),
                 ("boxes >0.3", f"{int(keep.sum())} / {len(frame.gt['labels'])} GT")]
        img = render_session_frame(frame, result, traj, fut, reasoning=None, stats=stats, score_threshold=args.thr)
        Image.fromarray(img).save(out / "frames" / f"{i:04d}.png")
        meta.append({"i": i, "t": float(t), "token": frame.token, "boxes": int(keep.sum()),
                     "gt_boxes": int(len(frame.gt["labels"])), "ade": float(err.mean()), "fde": float(err[-1]),
                     "model_ms": lat[-1], "traj": np.round(traj[0], 4).tolist(),
                     "score_sum": float(scores[keep].sum())})
        print(f"[{i + 1}/{len(times)}] {frame.token[:20]}  {int(keep.sum()):3d} boxes (GT {len(frame.gt['labels']):3d})  "
              f"ADE {err.mean():.3f}  FDE {err[-1]:.3f}  model {lat[-1]:.0f} ms  ({time.time() - t_all:.0f}s)", flush=True)

    (out / "session.json").write_text(json.dumps(meta, indent=2))
    (out / "encode.json").write_text(json.dumps({"frames": len(times), "source_hz": float(src_hz), "speed": args.speed,
                                                 "out_fps": float(out_fps), "backend": backend}, indent=2))
    ade = np.mean([m["ade"] for m in meta])
    print(f"\nwrote {len(meta)} frames -> {out}/frames   mean ADE {ade:.2f} m   model call median {np.median(lat):.1f} ms "
          f"(host-side numpy in/out, not the deployment loop)")
    mp4 = out / f"student_{scene['name']}_{args.speed:g}x.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{out_fps:.4f}", "-i", str(out / "frames" / "%04d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(mp4)], check=True)
    print(f"  video: {mp4}")
    if args.gif:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        make_gif(out / "frames", Path(args.gif), out_fps)
    if args.compare:
        compare_runs(out, Path(args.compare))
    if args.export_onnx:
        if not args.ckpt:
            raise SystemExit("--export-onnx needs --ckpt")
        dst = Path(args.export_onnx); dst.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(max(args.gpu, 0)))
        print(f"  exporting {args.ckpt} -> {dst}", flush=True)
        subprocess.run([sys.executable, "distill/export_student.py", "--ckpt", args.ckpt, "--out", str(dst),
                        "--report", str(dst.with_suffix(".json"))], check=True, env=env)
        print(f"  ONNX: {dst}")
    return 0


def compare_runs(a: Path, b: Path):
    """Per-frame agreement of two renders of the same instants (e.g. PyTorch checkpoint vs its ONNX)."""
    ma = {m["i"]: m for m in json.loads((a / "session.json").read_text())}
    mb = {m["i"]: m for m in json.loads((b / "session.json").read_text())}
    common = sorted(set(ma) & set(mb))
    if not common or "traj" not in ma[common[0]] or "traj" not in mb[common[0]]:
        print(f"  compare: {b} has no comparable per-frame record"); return
    dtraj = [float(np.abs(np.asarray(ma[i]["traj"]) - np.asarray(mb[i]["traj"]))[:, :2].max()) for i in common]
    dbox = [abs(ma[i]["boxes"] - mb[i]["boxes"]) for i in common]
    dscore = [abs(ma[i]["score_sum"] - mb[i]["score_sum"]) / max(ma[i]["score_sum"], 1e-6) for i in common]
    rep = {"frames": len(common), "traj_max_abs_diff_m": max(dtraj), "traj_median_abs_diff_m": float(np.median(dtraj)),
           "frames_with_different_box_count": int(sum(d > 0 for d in dbox)), "max_box_count_diff": int(max(dbox)),
           "score_sum_max_rel_diff": max(dscore), "a": str(a), "b": str(b)}
    (a / "compare.json").write_text(json.dumps(rep, indent=2))
    print(f"  compare {a.name} vs {b.name} over {len(common)} frames: trajectory max |diff| {rep['traj_max_abs_diff_m'] * 1000:.2f} mm "
          f"(median {rep['traj_median_abs_diff_m'] * 1000:.3f} mm); box count differs on {rep['frames_with_different_box_count']} "
          f"frames (max {rep['max_box_count_diff']}); kept-score sum max rel diff {rep['score_sum_max_rel_diff']:.2e}")


if __name__ == "__main__":
    raise SystemExit(main())
