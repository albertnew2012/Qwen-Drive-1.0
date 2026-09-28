"""ONNX-only demo of the student on a nuScenes scene: everything from the raw devkit data.

No training caches, no PyTorch model: the graph's shapes give the input size, depth bins,
BEV size and history length; calibration and images come from the nuScenes devkit; the
ego state (1.5 s of history, velocity, acceleration, nav command, speed) is built from the
dense ego-pose track exactly as the training cache was; the previous keyframe's BEV comes
back from the graph's own `bev_state` output (kept on the device), warped by ego motion.
Renders the session-demo layout per keyframe and encodes an MP4.

    python local/distill/demo_onnx.py --onnx student.onnx --scene scene-0276 --dataroot data/nuscenes --gpu 0
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT / "src", _ROOT, _ROOT / "local"):
    sys.path.insert(0, str(p))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image

CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]  # training order
GROUP10 = np.array([0, 0, 0, 0, 0, 4, 2, 2, 5, 6])      # nuScenes 10 classes -> teacher's 7 (renderer palette)
PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 5.4)         # the teacher's BEV grid, fixed for every student
DEPTH_RANGE = (1.0, 60.0)


def pose_matrix(rec):
    from pyquaternion import Quaternion
    m = np.eye(4); m[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix; m[:3, 3] = rec["translation"]; return m


def ego_state(nusc, track, sample, n_hist=16, hist_s=1.5, n_fut=50, hz=10.0):
    """The 116-d planner input the student was trained with, from the dense ego-pose track."""
    from nuscenes_session import resample
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"]); p = nusc.get("ego_pose", sd["ego_pose_token"])
    t0 = p["timestamp"] * 1e-6
    ref = pose_matrix(p); ref_inv = np.linalg.inv(ref)
    from pyquaternion import Quaternion
    ref_yaw = Quaternion(p["rotation"]).yaw_pitch_roll[0]
    dt = hist_s / n_hist                                                    # 0.09375 s, as the training cache sampled it
    hist = resample(track, t0, -np.arange(n_hist)[::-1] * dt, ref_inv, ref_yaw)                       # (16, 3), last row = now
    vel = np.gradient(hist[:, :2], dt, axis=0).astype(np.float32); acc = np.gradient(vel, dt, axis=0).astype(np.float32)
    t_end = track[-1][0]
    fut = None
    if t0 + n_fut / hz <= t_end + 0.1:                                       # a full 5 s of recorded future exists
        fut = resample(track, t0, np.arange(1, n_fut + 1) / hz, ref_inv, ref_yaw)
    lat = float(fut[-1, 1]) if fut is not None else 0.0
    nav = 0 if abs(lat) < 4.0 else (1 if lat > 0 else 2)                       # as the training cache derived it
    vec = np.concatenate([hist.reshape(-1), vel.reshape(-1), acc.reshape(-1), np.eye(3, dtype=np.float32)[nav],
                          np.asarray([np.linalg.norm(vel[-1])], np.float32)]).astype(np.float32)
    return vec, fut


def boxes_for_renderer(bx, lidar2ego):
    """Student boxes (ego frame, [x,y,z,log w,log l,log h,sin,cos,vx,vy]) -> renderer boxes
    (lidar frame, bottom-centre z, [x,y,z,along-heading,lateral,h,yaw,vx,vy])."""
    if len(bx) == 0:
        return np.zeros((0, 9), np.float32)
    e2l = np.linalg.inv(np.asarray(lidar2ego, np.float64))
    xyz = (e2l @ np.concatenate([bx[:, :3], np.ones((len(bx), 1))], 1).T).T[:, :3]
    yaw = np.arctan2(bx[:, 6], bx[:, 7]) + np.arctan2(e2l[1, 0], e2l[0, 0])
    v = bx[:, 8:10] @ e2l[:2, :2].T
    out = np.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2] - np.exp(bx[:, 5]) / 2, np.exp(bx[:, 4]), np.exp(bx[:, 3]), np.exp(bx[:, 5]),
                    yaw, v[:, 0], v[:, 1]], -1)
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--scene", default="scene-0276")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default="outputs/demo_onnx")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--calib", default="", help="json of per-class score thresholds {class_index: thr} (diagnose.py --write-calib); overrides --thr per class")
    ap.add_argument("--occ-mask-dir", default="", help="frames dir holding <token>/occ3d.npz: draw occupancy only inside the Occ3D camera-visible mask (the benchmark's evaluation region)")
    ap.add_argument("--dump-occ", action="store_true", help="save each frame's occupancy argmax (uint8 200x200x16) next to its PNG")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device for onnxruntime; -1 = CPU")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.chdir(_ROOT)

    import onnxruntime as ort
    from nuscenes.nuscenes import NuScenes
    from nuscenes_session import SessionFrame, render_session_frame, ego_track
    from local.distill.geometry import bev_indices_all
    from local.distill.temporal import warp_grid_from_T

    so = ort.SessionOptions(); so.log_severity_level = 3
    providers = [("CUDAExecutionProvider", {"device_id": args.gpu})] if args.gpu >= 0 else ["CPUExecutionProvider"]
    s = ort.InferenceSession(args.onnx, so, providers=providers)
    dev = "cuda" if args.gpu >= 0 and "CUDAExecutionProvider" in s.get_providers() else "cpu"
    shapes = {i.name: i.shape for i in s.get_inputs()}
    _, n_cams, _, H, W = shapes["image"]
    L = shapes["bev_index"][0]; hf, wf = H // 16, W // 16
    depth_bins = L // (n_cams * hf * wf)
    temporal = "prev_bev" in shapes
    K = 1 if (not temporal or len(shapes["prev_bev"]) == 4) else shapes["prev_bev"][1]
    C, bev = (shapes["prev_bev"][-3], shapes["prev_bev"][-1]) if temporal else (384, 200)
    cfg = SimpleNamespace(image_size=(W, H), depth_bins=depth_bins, depth_range=DEPTH_RANGE, bev_size=bev,
                          pc_range=PC_RANGE, n_cams=n_cams)
    print(f"  graph: {W}x{H}, {n_cams} cams, {depth_bins} depth bins, BEV {bev}, temporal={temporal} K={K}, ORT on {dev}", flush=True)

    thr_per_class = None
    if args.calib:                                              # per-class operating points, fitted on held-out val scenes
        cj = json.load(open(args.calib)); thr_per_class = np.array([float(cj.get(str(k), args.thr)) for k in range(10)])
        print(f"  per-class thresholds from {args.calib}: {thr_per_class.tolist()}", flush=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene = next(x for x in nusc.scene if x["name"] == args.scene)
    samples, cur = [], scene["first_sample_token"]
    while cur:
        samples.append(nusc.get("sample", cur)); cur = samples[-1]["next"]
    if args.limit:
        samples = samples[:args.limit]
    track = ego_track(nusc, samples[0])
    out = Path(args.out) / args.scene; (out / "frames").mkdir(parents=True, exist_ok=True)

    def to_dev(a):
        return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a), dev, args.gpu if dev == "cuda" else 0)

    def keep_state(ov):
        # On CUDA the output OrtValue owns its device buffer and can be fed straight back as next
        # frame's prev_bev. On CPU the binding object owns that memory, so once the binding is
        # released the state read as garbage (NaN scores -> zero detections on CPU). Copy it out.
        return ov if dev == "cuda" else to_dev(ov.numpy().copy())
    out_names = [o.name for o in s.get_outputs()]
    past, lat, meta = [], [], []                                   # past: [(state OrtValue, ego2global 4x4)]
    t_all = time.time()
    for i, smp in enumerate(samples):
        frame = SessionFrame(nusc, smp, Path(args.dataroot)); frame.kind = f"student ONNX  {W}x{H}  {dev}"
        metas = frame.img_metas(image_size=(W, H))
        idx, val = bev_indices_all(metas["lidar2img"], metas["lidar2ego"], cfg)
        imgs = np.stack([((np.asarray(frame.image(c).resize((W, H), Image.BILINEAR), np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
                         for c in CAM_ORDER], 0)[None]
        ego_vec, fut = ego_state(nusc, track, smp)
        sd = nusc.get("sample_data", smp["data"]["LIDAR_TOP"]); E = pose_matrix(nusc.get("ego_pose", sd["ego_pose_token"]))
        feeds = {"image": imgs.astype(np.float32), "bev_index": idx.reshape(-1).astype(np.int64),
                 "valid": val.reshape(-1).astype(bool), "ego": ego_vec[None]}
        # Every OrtValue bound to a run must stay alive until that run has finished: a bound
        # temporary that gets garbage-collected leaves the binding pointing at freed memory
        # (on the CPU provider that produced garbage / zero detections; CUDA happened to survive).
        held = []
        def bind_in(b, k, v):
            ov = v if isinstance(v, ort.OrtValue) else to_dev(v); held.append(ov); b.bind_ortvalue_input(k, ov)
        io = s.io_binding()
        for k, v in feeds.items():
            bind_in(io, k, v)
        if temporal:
            if not past:                                            # first frame of the scene: its own state, identity warp
                io0 = s.io_binding()
                for k, v in feeds.items(): bind_in(io0, k, v)
                z = np.zeros((1, C, bev, bev), np.float32) if K == 1 else np.zeros((1, K, C, bev, bev), np.float32)
                zg = np.zeros((1, bev, bev, 2), np.float32) if K == 1 else np.zeros((1, K, bev, bev, 2), np.float32)
                bind_in(io0, "prev_bev", z); bind_in(io0, "warp_grid", zg)
                io0.bind_output("bev_state", dev, args.gpu if dev == "cuda" else 0); s.run_with_iobinding(io0)
                past = [(keep_state(io0.get_outputs()[0]), E)]
            chosen = list(past[:K])
            while len(chosen) < K: chosen.append(chosen[-1])
            grids = [warp_grid_from_T(np.linalg.inv(E) @ Ep, cfg)[None].astype(np.float32) for _, Ep in chosen]
            if K == 1:
                bind_in(io, "prev_bev", chosen[0][0]); bind_in(io, "warp_grid", grids[0])
            else:
                bind_in(io, "prev_bev", np.stack([c_.numpy() for c_, _ in chosen], 1))
                bind_in(io, "warp_grid", np.stack(grids, 1))
        for nm in out_names:
            io.bind_output(nm, dev, args.gpu if dev == "cuda" else 0)
        t0 = time.perf_counter(); s.run_with_iobinding(io); lat.append((time.perf_counter() - t0) * 1e3)
        outs = dict(zip(out_names, [o.numpy() for o in io.get_outputs()]))
        if temporal:
            past = [(keep_state(io.get_outputs()[out_names.index("bev_state")]), E)] + past[:K]

        sc = 1 / (1 + np.exp(-outs["cls"][0].astype(np.float64))); scores = sc.max(-1); labels = sc.argmax(-1)
        keep = scores >= (thr_per_class[labels] if thr_per_class is not None else args.thr)
        occ_pred = outs["occ"][0].argmax(-1).astype(np.uint8)
        occ_note = None
        if args.occ_mask_dir:
            mp = Path(args.occ_mask_dir) / smp["token"] / "occ3d.npz"
            if mp.exists():
                occ_pred = occ_pred.copy(); occ_pred[np.load(mp)["mask"] == 0] = 9          # 9 = empty: outside the visible region nothing is drawn
                occ_note = "camera-visible region"
        if args.dump_occ:
            np.save(out / "frames" / f"{i:04d}_occ.npy", occ_pred)
        result = {"boxes": boxes_for_renderer(outs["box"][0][keep], frame.lidar2ego), "scores": scores[keep].astype(np.float32),
                  "labels": GROUP10[labels[keep]], "occ": occ_pred, "map": outs["seg"][0].argmax(0).astype(np.uint8)}
        traj = outs["trajectory"][0][None]
        stats = [("frame", f"{i+1}/{len(samples)}"), (("boxes (per-class thr)" if thr_per_class is not None else "boxes >%.2f" % args.thr), f"{int(keep.sum())} / {len(frame.gt['labels'])} GT"),
                 ("onnx", f"{lat[-1]:.0f} ms")] + ([("occupancy", occ_note)] if occ_note else [])
        if fut is not None:
            err = np.linalg.norm(traj[0][:50, :2] - fut[:50, :2], axis=-1); stats += [("ADE", f"{err.mean():.2f} m"), ("FDE", f"{err[-1]:.2f} m")]
        img = render_session_frame(frame, result, traj, fut, stats=stats, score_threshold=0.0)
        Image.fromarray(img).save(out / "frames" / f"{i:04d}.png")
        meta.append({"i": i, "token": smp["token"], "boxes": int(keep.sum()), "onnx_ms": lat[-1]})
        print(f"[{i+1}/{len(samples)}] {smp['token'][:16]}  {int(keep.sum()):3d} boxes  onnx {lat[-1]:5.1f} ms  ({time.time()-t_all:.0f}s)", flush=True)
    (out / "frames.json").write_text(json.dumps(meta, indent=1))
    mp4 = out / f"student_onnx_{args.scene}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps}", "-i", str(out / "frames" / "%04d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True)
    tl = lat[1:] if len(lat) > 1 else lat                      # frame 0 carries the warm-up
    print(f"wrote {mp4}  | onnx median {np.median(tl):.1f} ms/frame ({1000/np.median(tl):.1f} Hz) on {dev}, {len(samples)} keyframes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
