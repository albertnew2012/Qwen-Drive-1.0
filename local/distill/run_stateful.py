"""Deployment loop for the stateful (temporal) student, and its single-GPU timing.

At deployment the backbone runs ONCE per frame: the graph returns `bev_state` (this
frame's unfused lift-splat BEV) and the host feeds it back as `prev_bev` at the next
frame together with `warp_grid`, built on the host from the two ego poses
(`temporal.warp_grid_from_T(inv(E_curr) @ E_prev)`). The first frame of a scene has no
history; training gave such frames their OWN BEV with an identity warp, so the loop runs
the first frame twice (once to obtain its state) -- a one-off cost per scene.

    --check  PyTorch: recurrent loop vs the two-frame training path on one scene
    --onnx   onnxruntime: run the loop over one scene on one GPU and report latency

    python local/distill/run_stateful.py --ckpt outputs/distill/exp/e8/student.pt --check
    python local/distill/run_stateful.py --onnx outputs/onnx/student_det/student.onnx --gpu 0
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from local.distill.temporal import load_temporal_index, warp_grid_from_T


def scene_tokens(index, scene: str | None, n: int):
    """Keyframes of one scene in time order (devkit), limited to n."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes", verbose=False)
    if scene is None:
        scene = sorted(create_splits_scenes()["val"])[0]
    sc = next(s for s in nusc.scene if s["name"] == scene)
    toks, t = [], sc["first_sample_token"]
    while t and len(toks) < n:
        toks.append(t); t = nusc.get("sample", t)["next"]
    toks = [t for t in toks if t in index]
    return scene, toks


def warp_between(index, prev_tok, tok, cfg):
    Ec = np.array(index[tok]["ego2global"]).reshape(4, 4)
    Ep = np.array(index[prev_tok]["ego2global"]).reshape(4, 4)
    return warp_grid_from_T(np.linalg.inv(Ec) @ Ep, cfg)[None]


ORT_NP = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(bool)": np.bool_, "tensor(uint8)": np.uint8}
TORCH_NP = {np.float32: torch.float32, np.int64: torch.int64, np.bool_: torch.bool, np.uint8: torch.uint8}


def pinned_device_loop(s, items, toks, index, cfg, K, temporal, args):
    """Per-frame cost of the graph as a deployment would run it: fixed device buffers, pinned host
    staging, H2D of this frame's inputs, the run, D2H of every output except the BEV state (which
    only ever lives on the device). Timed with the GPU kept busy frame after frame, so the number
    is not inflated by the GPU clocking down while Python prepares the next frame -- the old loop
    measured 90 ms for a graph that runs in 60 ms for exactly that reason."""
    g = args.gpu; D = torch.device(f"cuda:{g}")
    in_specs = {i.name: (tuple(int(d) if isinstance(d, int) else 1 for d in i.shape), ORT_NP[i.type]) for i in s.get_inputs()}
    out_specs = {o.name: (tuple(int(d) if isinstance(d, int) else 1 for d in o.shape), ORT_NP[o.type]) for o in s.get_outputs()}
    dev_in = {n: torch.empty(shp, dtype=TORCH_NP[dt], device=D) for n, (shp, dt) in in_specs.items()}
    dev_out = {n: torch.empty(shp, dtype=TORCH_NP[dt], device=D) for n, (shp, dt) in out_specs.items()}
    host_out = {n: torch.empty(shp, dtype=TORCH_NP[dt]).pin_memory() for n, (shp, dt) in out_specs.items() if n != "bev_state"}
    # pinned staging of every frame's static inputs (what a camera pipeline would hand over)
    staged = []
    for b in items:
        staged.append({"image": b["image"][None].float().contiguous().pin_memory(),
                       "bev_index": b["bev_index"].long().contiguous().pin_memory(),
                       "valid": b["valid"].bool().contiguous().pin_memory(),
                       "ego": b["ego"][None].float().contiguous().pin_memory()})
    warp_host = torch.empty(in_specs["warp_grid"][0], dtype=torch.float32).pin_memory() if temporal else None
    C, n = cfg.bev_channels, cfg.bev_size
    ring = [torch.zeros((1, C, n, n), device=D) for _ in range(K + 1)]     # BEV states, newest first
    ring_tok = []

    def bind(io, prev_states):
        for nm, t in dev_in.items():
            if nm in ("prev_bev",):
                continue
            io.bind_input(nm, "cuda", g, in_specs[nm][1], tuple(t.shape), t.data_ptr())
        if temporal:
            if K == 1:
                io.bind_input("prev_bev", "cuda", g, np.float32, tuple(prev_states[0].shape), prev_states[0].data_ptr())
            else:
                torch.stack(prev_states, 1, out=dev_in["prev_bev"])
                io.bind_input("prev_bev", "cuda", g, np.float32, tuple(dev_in["prev_bev"].shape), dev_in["prev_bev"].data_ptr())
        for nm, t in dev_out.items():
            io.bind_output(nm, "cuda", g, out_specs[nm][1], tuple(t.shape), t.data_ptr())

    lat = []; phases = []
    for rep in range(2):                                      # pass 0 warms up (cuDNN autotune), pass 1 is timed
        ring_tok = []; phases = []
        for i, st in enumerate(staged):
            tok = toks[i]
            # ---- timed region: H2D, run, D2H (state stays on device) ----
            torch.cuda.synchronize(D); t0 = time.perf_counter()
            for nm in ("image", "bev_index", "valid", "ego"):
                dev_in[nm].copy_(st[nm], non_blocking=True)
            if temporal:
                if not ring_tok:                              # first frame of the scene: its own state, identity warp
                    io0 = s.io_binding()
                    zero = [torch.zeros((1, C, n, n), device=D)] * K
                    dev_in["warp_grid"].zero_()
                    bind(io0, zero)
                    torch.cuda.synchronize(D); s.run_with_iobinding(io0)
                    ring[0].copy_(dev_out["bev_state"]); ring_tok = [tok]
                prev = [ring[k % (K + 1)] for k in range(len(ring_tok))][:K]
                while len(prev) < K: prev.append(prev[-1])
                grids = [warp_grid_from_T(np.eye(4), cfg)[None] if tk == tok else warp_between(index, tk, tok, cfg)
                         for tk in (ring_tok + [ring_tok[-1]] * K)[:K]]
                warp_host.copy_(torch.from_numpy(np.ascontiguousarray(grids[0] if K == 1 else np.stack(grids, 1), dtype=np.float32)))
                dev_in["warp_grid"].copy_(warp_host, non_blocking=True)
            io = s.io_binding(); bind(io, prev if temporal else None)
            torch.cuda.synchronize(D); t1 = time.perf_counter()
            s.run_with_iobinding(io)
            t2 = time.perf_counter()
            for nm, h in host_out.items():
                h.copy_(dev_out[nm], non_blocking=True)
            torch.cuda.synchronize(D); t3 = time.perf_counter(); dt = (t3 - t0) * 1e3
            phases.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3))
            # ---- end timed region ----
            if temporal:                                      # newest state into the ring slot not referenced by prev
                slot = ring[len(ring_tok) % (K + 1)] if len(ring_tok) < K + 1 else None
                ring.insert(0, ring.pop()); ring[0].copy_(dev_out["bev_state"]); ring_tok = [tok] + ring_tok[:K - 1]
            if rep == 1:
                lat.append(dt)
    ms = float(np.median(lat)); ph = np.median(np.array(phases), 0)
    print(f"  phases (median ms): H2D+bind {ph[0]:.2f} | run {ph[1]:.2f} | D2H {ph[2]:.2f}")
    print(f"  DEPLOYMENT LOOP (pinned buffers, GPU kept busy) on GPU {g}: median {ms:.2f} ms/frame -> {1000/ms:.1f} Hz "
          f"(p90 {np.percentile(lat, 90):.2f} ms) over {len(lat)} frames: H2D of image/index/valid/ego/warp + run + D2H of "
          f"cls/box/occ/seg/trajectory; BEV state stays on the device")
    Path("outputs/distill").mkdir(parents=True, exist_ok=True)
    json.dump({"onnx": args.onnx, "frames": len(lat), "median_ms": ms, "p90_ms": float(np.percentile(lat, 90)), "hz": 1000 / ms,
               "mode": "pinned_device_loop"}, open("outputs/distill/student_onnx_stateful_pinned.json", "w"), indent=1)
    return ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--onnx", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--legacy-loop", action="store_true", help="time only session.run with fresh per-frame allocations (the old measurement)")
    args = ap.parse_args()
    os.chdir(_ROOT)
    from local.distill.student import StudentConfig, StudentDetector
    from local.distill.train_student import DistillSet
    index = load_temporal_index()
    src = args.ckpt or ""
    ck = torch.load(src, map_location="cpu") if src and Path(src).exists() else {}
    saved = ck.get("cfg") or {}
    cfg = StudentConfig(**{k: v for k, v in saved.items()
                           if k in StudentConfig.__init__.__code__.co_varnames})
    if args.onnx and not saved:
        # No checkpoint: read the graph's own contract instead of assuming 896x512 (which fed
        # 896x512 frames to a 1152x640 graph and failed with "invalid dimensions for input").
        import onnxruntime as _ort
        shp = {i.name: i.shape for i in _ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]).get_inputs()}
        _, n_cams, _, H, W = shp["image"]
        kw = dict(image_size=(int(W), int(H)), n_cams=int(n_cams), temporal="prev_bev" in shp,
                  det_head="center", ref_points=False, velocity=True, num_classes=10)
        if "prev_bev" in shp:
            pb = shp["prev_bev"]; kw["history"] = int(pb[1]) if len(pb) == 5 else 1
            kw["bev_channels"] = int(pb[-3]); kw["bev_size"] = int(pb[-1])
        cfg = StudentConfig(**kw)
        print(f"  config inferred from the ONNX: {cfg.image_size}, history {kw.get('history', 1)}, temporal {kw['temporal']}")
    scene, toks = scene_tokens(index, args.scene, args.frames)
    print(f"  scene {scene}: {len(toks)} keyframes; temporal={cfg.temporal}")
    ds = DistillSet(Path("data/distill/frames"), Path("data/distill/teacher"), cfg, toks)
    items = [ds[i] for i in range(len(toks))]           # host-side data prep, not timed

    K = int(getattr(cfg, "history", 1))

    def history_inputs(states, cur_tok):
        """(prev_bev, warp_grid) for the current frame from the deque of past (state, token),
        newest first, padded exactly as the training loader pads: a missing older frame
        repeats the last available one; no history at all repeats the current frame's own
        state with an identity warp (the caller passes it as states[0] with tok == cur_tok)."""
        chosen = list(states[:K])
        while len(chosen) < K:
            chosen.append(chosen[-1])
        bevs, grids = [], []
        for st, tk in chosen:
            bevs.append(st)
            g = warp_grid_from_T(np.eye(4), cfg)[None] if tk == cur_tok else warp_between(index, tk, cur_tok, cfg)
            grids.append(torch.from_numpy(g).cuda())
        if K == 1:
            return bevs[0], grids[0]
        return torch.stack(bevs, 1), torch.stack(grids, 1)

    if args.check:
        assert cfg.temporal, "--check needs a temporal checkpoint"
        model = StudentDetector(cfg).eval()
        sd = ck.get("ema") or ck["model"]
        sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in sd.items() if k != "n_averaged"}
        model.load_state_dict(sd, strict=False); model = model.cuda()
        ds2 = DistillSet(Path("data/distill/frames"), Path("data/distill/teacher"), cfg, toks, temporal=True, history=K)
        from local.distill.temporal import prev_bev_from_batch, warp_from_batch
        worst = 0.0
        with torch.no_grad():
            past = []                                        # [(state, token)], newest first
            for i, b in enumerate(items):
                img, idx, val, ego = (b["image"][None].cuda(), b["bev_index"].cuda(),
                                      b["valid"].cuda(), b["ego"][None].cuda())
                if not past:                                 # first frame: its own state
                    own = model.bev_from(img, idx, val)
                    fed, grid = history_inputs([(own, toks[i])], toks[i])
                else:
                    fed, grid = history_inputs(past, toks[i])
                out = model(img, idx, val, ego, fed, grid)
                hm_loop = model._hm.float().clone()
                past = [(out[-1], toks[i])] + past[:K]
                # reference: the training/eval path recomputes the previous BEV(s) from images
                b2 = ds2[i]
                prev = prev_bev_from_batch(model, b2, batched=False)
                ref = model(img, idx, val, ego, prev, warp_from_batch(b2, batched=False))
                hm_ref = model._hm.float()
                # Compare the dense heatmap logits and the fed-back state, not the decoded
                # top-k lists: a last-bit difference reorders near-ties in the top-k and
                # turns into +-100 m elementwise jumps that mean nothing.
                d_hm = float((hm_loop - hm_ref).abs().max())
                d_state = float((fed - prev).abs().max())     # fed state(s) vs recomputed prev BEV(s)
                worst = max(worst, d_hm, d_state)
                if i < 3 or d_hm > 1e-2:
                    print(f"    frame {i}: heatmap max|diff| {d_hm:.2e}, fed state vs bev_from(prev) {d_state:.2e}")
        print(f"  recurrent loop vs two-frame path over {len(items)} frames: worst max|diff| = {worst:.2e}"
              f"  -> {'EQUIVALENT' if worst < 1e-2 else 'MISMATCH'}")

    if args.onnx:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3
        s = ort.InferenceSession(args.onnx, so, providers=[("CUDAExecutionProvider", {"device_id": args.gpu})])
        names = [i.name for i in s.get_inputs()]
        temporal = "prev_bev" in names
        print(f"  ONNX inputs {names}; outputs {[o.name for o in s.get_outputs()]}")
        zero_state = np.zeros((1, cfg.bev_channels, cfg.bev_size, cfg.bev_size), np.float32)
        lat = []
        def np_history(past, cur_tok):
            chosen = list(past[:K])
            while len(chosen) < K:
                chosen.append(chosen[-1])
            bevs = [st for st, _ in chosen]
            grids = [warp_grid_from_T(np.eye(4), cfg)[None] if tk == cur_tok else warp_between(index, tk, cur_tok, cfg)
                     for _, tk in chosen]
            if K == 1:
                return bevs[0], grids[0].astype(np.float32)
            return np.stack(bevs, 1), np.stack(grids, 1).astype(np.float32)
        # The BEV state (1 x 384 x 200 x 200 float32 = 61 MB) stays on the device: IO binding
        # feeds prev_bev from the previous frame's bev_state OrtValue and binds the outputs to
        # device memory, so only the six images and the small tensors cross PCIe per frame.
        out_names = [o.name for o in s.get_outputs()]
        dev = "cuda"
        def to_dev(a): return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a), dev, args.gpu)
        if not args.legacy_loop:
            ms = pinned_device_loop(s, items, toks, index, cfg, K, temporal, args)
            lat = [ms]
        for rep in (range(2) if args.legacy_loop else []):     # legacy: pass 0 warms up, pass 1 is timed
            past = []                                          # [(state OrtValue on device, token)]
            for i, b in enumerate(items):
                io = s.io_binding()
                io.bind_ortvalue_input("image", to_dev(b["image"][None].numpy()))
                io.bind_ortvalue_input("bev_index", to_dev(b["bev_index"].numpy()))
                io.bind_ortvalue_input("valid", to_dev(b["valid"].numpy()))
                io.bind_ortvalue_input("ego", to_dev(b["ego"][None].numpy()))
                if temporal:
                    if not past:                                # first frame: own state (one extra pass per scene)
                        z = zero_state if K == 1 else np.repeat(zero_state[:, None], K, 1)
                        zg = np.zeros((1, cfg.bev_size, cfg.bev_size, 2), np.float32) if K == 1 else np.zeros((1, K, cfg.bev_size, cfg.bev_size, 2), np.float32)
                        io0 = s.io_binding()
                        for nm, val in (("image", b["image"][None].numpy()), ("bev_index", b["bev_index"].numpy()),
                                        ("valid", b["valid"].numpy()), ("ego", b["ego"][None].numpy()), ("prev_bev", z), ("warp_grid", zg)):
                            io0.bind_ortvalue_input(nm, to_dev(val))
                        io0.bind_output("bev_state", dev, args.gpu)
                        s.run_with_iobinding(io0)
                        own = io0.get_outputs()[0]
                        past = [(own, toks[i])]
                    chosen = list(past[:K])
                    while len(chosen) < K:
                        chosen.append(chosen[-1])
                    grids = [warp_grid_from_T(np.eye(4), cfg)[None] if tk == toks[i] else warp_between(index, tk, toks[i], cfg)
                             for _, tk in chosen]
                    if K == 1:
                        io.bind_ortvalue_input("prev_bev", chosen[0][0])
                        io.bind_ortvalue_input("warp_grid", to_dev(grids[0].astype(np.float32)))
                    else:                                       # K states -> one (1, K, C, H, W) device tensor
                        stacked = np.stack([c_.numpy() for c_, _ in chosen], 1)
                        io.bind_ortvalue_input("prev_bev", to_dev(stacked))
                        io.bind_ortvalue_input("warp_grid", to_dev(np.stack(grids, 1).astype(np.float32)))
                for nm in out_names:
                    io.bind_output(nm, dev, args.gpu)
                t0 = time.perf_counter(); s.run_with_iobinding(io); dt = (time.perf_counter() - t0) * 1e3
                outs = io.get_outputs()
                if temporal:
                    past = [(outs[-1], toks[i])] + past[:K]
                if rep == 1:
                    lat.append(dt)
        ms = float(np.median(lat))
        if args.legacy_loop: print(f"  DEPLOYMENT LOOP on GPU {args.gpu}: median {ms:.2f} ms/frame -> {1000/ms:.1f} Hz "
              f"(p90 {np.percentile(lat, 90):.2f} ms) over {len(lat)} frames, all outputs, state kept on device, host warp included")
        Path("outputs/distill").mkdir(parents=True, exist_ok=True)
        json.dump({"onnx": args.onnx, "scene": scene, "frames": len(lat), "median_ms": ms,
                   "hz": 1000 / ms, "p90_ms": float(np.percentile(lat, 90)), "gpu": args.gpu},
                  open("outputs/distill/student_onnx_stateful.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
