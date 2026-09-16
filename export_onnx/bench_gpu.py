"""Time the perception + trajectory ONNX graphs on the GPU, with real scene inputs.

Two phases, because they need different interpreters:

    .venv/bin/python        export_onnx/bench_gpu.py --phase prep
    .venv-ortgpu/bin/python export_onnx/bench_gpu.py --phase bench

``prep`` needs torch/transformers to build the real scene tensors once and dumps
them to ``outputs/onnx_bench``. ``bench`` needs onnxruntime-gpu, which cannot be
installed next to the CPU ``onnxruntime`` the validation harness is pinned to.

Only steady-state compute is timed. Sessions are built and warmed up first, then
timed; graph load is reported separately and excluded from the FPS figure.

The decoder hidden state is kept on the device between layers with IO binding.
Without it every one of the 32 layers round-trips ~28 MiB to the host and back,
which dominates the measurement. The planner's KV caches are constant across the
Euler loop, so they are bound to the device once.

VQA/chain-of-thought decoding is deliberately not benchmarked: it is not on the
perception or trajectory path.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
BENCH = _ROOT / "outputs" / "onnx_bench"


# ───────────────────────────────────────────────────────────────── phase: prep

def prep(args) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))
    import torch
    from transformers import AutoTokenizer
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor

    BENCH.mkdir(parents=True, exist_ok=True)

    print("== perception inputs ==")
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.float32, attn_implementation="sdpa")
    vlm = holder.vlm.eval()
    head = QwenDrivePerception.from_pretrained(args.model, dtype=torch.float32).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm, proc)
    frame = PerceptionFrame(Path(args.frames) / args.frame)
    inputs, metas = proc(frame, device="cpu")
    ids = inputs["input_ids"].numpy()
    pos = holder._rope_positions(inputs["input_ids"], inputs["image_grid_thw"]).numpy()
    grid = inputs["image_grid_thw"].numpy()
    img_tok = int(vlm.config.image_token_id)

    # the embedding gather is input-only, so it is hoisted out of the timed loop
    table = np.load(Path(args.layers) / "embed_tokens.npy", mmap_mode="r")
    embeds = np.asarray(table)[ids].astype(np.float32)
    del table

    np.savez(BENCH / "perception_inputs.npz",
             pixel_values=inputs["pixel_values"].numpy().astype(np.float32),
             input_ids=ids, position_ids=pos, image_grid_thw=grid,
             embeds=embeds, img_tok=np.int64(img_tok),
             n_cam=np.int64(len(metas["cam_order"])))
    print(f"   {ids.shape[1]} tokens   {len(metas['cam_order'])} cameras")
    del holder, vlm, head, proc
    import gc; gc.collect()

    print("== planning inputs ==")
    from export_onnx.scene_inputs import planning_inputs
    from qwen_drive.trajectory import normalize_history
    ctx = planning_inputs(args.vlm, args.planner, args.scenes, args.image_root,
                          args.image_archive, index=args.index)
    model, vlm_p, pin = ctx["holder"], ctx["vlm"], ctx["inputs"]
    ids_p = pin["input_ids"].numpy()
    pos_p = ctx["position_ids"].numpy()
    cfg_p = model.planning_expert.config
    n_points = model.config.num_future_points
    scale = model.trajectory_scale(torch.device("cpu"))
    anchor = model._rope_positions(pin["input_ids"], pin["image_grid_thw"])[:, :, -1]
    hist = normalize_history(pin["history"].float(), scale)
    noise = (model.config.noise_init_std *
             model._initial_noise(1, n_points, args.seed, torch.device("cpu")))
    nav = torch.eye(cfg_p.nav_command_classes)[
        pin["nav_command"].reshape(-1).long()].reshape(*pin["nav_command"].shape, -1)

    table = np.load(Path(args.layers_plan) / "embed_tokens.npy", mmap_mode="r")
    embeds_p = np.asarray(table)[ids_p].astype(np.float32)
    del table

    np.savez(BENCH / "planning_inputs.npz",
             pixel_values=pin["pixel_values"].numpy().astype(np.float32),
             input_ids=ids_p, position_ids=pos_p, embeds=embeds_p,
             img_tok=np.int64(vlm_p.config.image_token_id),
             noise=noise.numpy().astype(np.float32),
             history=hist.numpy().astype(np.float32),
             history_velocity=pin["history_velocity"].numpy().astype(np.float32),
             history_acceleration=pin["history_acceleration"].numpy().astype(np.float32),
             nav_onehot=nav.numpy().astype(np.float32),
             ego_status=pin["ego_status"].numpy().astype(np.float32),
             position_anchor=anchor.numpy(),
             scale=scale.numpy().astype(np.float32),
             min_one_minus_t=np.float32(model.config.min_one_minus_t))
    print(f"   {ids_p.shape[1]} tokens   {n_points} waypoints")
    print(f"\nwrote {BENCH}")
    return 0


# ──────────────────────────────────────────────────────────────── phase: bench

def premerge_grids(patches: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """numpy twin of QwenDrivePerception._premerge_grids (a reshape + permute)."""
    feats, off = [], 0
    for _, gh, gw in grid.tolist():
        size = gh * gw
        cur = patches[off:off + size]
        feats.append(cur.reshape(gh // 2, gw // 2, 2, 2, cur.shape[-1])
                        .transpose(0, 2, 1, 3, 4).reshape(gh, gw, -1))
        off += size
    return np.stack(feats, 0)


def wrap_heading(t: np.ndarray) -> np.ndarray:
    out = t.copy()
    out[..., 2] = np.arctan2(np.sin(out[..., 2]), np.cos(out[..., 2]))
    return out


def bench(args) -> int:
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("CUDAExecutionProvider missing - run this with .venv-ortgpu/bin/python "
              "and LD_LIBRARY_PATH pointing at the torch CUDA libs")
        return 1

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel,
                                          f"ORT_{args.opt}")
    prov = [("CUDAExecutionProvider", {"device_id": 0,
                                       "arena_extend_strategy": "kSameAsRequested"})]

    load_s = 0.0

    def sess(path):
        nonlocal load_s
        t = time.perf_counter()
        s = ort.InferenceSession(str(path), so, providers=prov)
        load_s += time.perf_counter() - t
        # ORT falls back to CPU silently, which would be reported as a GPU number
        if "CUDAExecutionProvider" not in s.get_providers():
            raise RuntimeError(f"{path.name} fell back to {s.get_providers()}")
        return s

    def dev(a):
        return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a), "cuda", 0)

    rule = lambda t: print(f"\n{'='*70}\n  {t}\n{'='*70}")
    results = {}

    def gpu_mem():
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
            used, total = out.stdout.strip().split("\n")[0].split(",")
            return f"GPU {int(used)/1024:.1f}/{int(total)/1024:.1f} GiB"
        except Exception:
            return ""

    # ---------------------------------------------------------- perception path
    if not args.skip_perception:
        d = np.load(BENCH / "perception_inputs.npz")
        px, ids = d["pixel_values"], d["input_ids"]
        pos, grid = d["position_ids"], d["image_grid_thw"]
        embeds, img_tok = d["embeds"], int(d["img_tok"])
        n_cam = int(d["n_cam"])
        gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
        tpi = gh // 2 * gw // 2
        mask = ids[0] == img_tok

        rule(f"PERCEPTION  {ids.shape[1]} tokens, {n_cam} cameras")
        man = json.loads((Path(args.layers) / "manifest.json").read_text())
        types = man["layer_types"]

        print(f"  loading {len(types)} layer graphs + vision + head ...")
        # small sessions first: the BFC arena cannot grow once the 17 GiB of
        # layer weights is resident, and the head wants a 655 MiB workspace
        perc = sess(Path(args.perception))
        vis = sess(Path(args.vision))
        norm = sess(Path(args.layers) / "final_norm.onnx")
        layers = [sess(Path(args.layers) / f"layer_{i:02d}.onnx")
                  for i in range(len(types))]
        print(f"  graph load {load_s:.1f}s (excluded from timings)   {gpu_mem()}")

        pos_ov = dev(pos)

        def perception_once(timed: dict | None = None):
            def mark(k, t0):
                if timed is not None:
                    timed[k] = timed.get(k, 0.0) + time.perf_counter() - t0

            t0 = time.perf_counter()
            _, vit_tap, merged = vis.run(None, {"pixel_values": px})
            mark("vision", t0)

            t0 = time.perf_counter()
            h = embeds.copy()
            h[0, mask] = merged[-int(mask.sum()):]
            mark("host_scatter", t0)

            t0 = time.perf_counter()
            hov = dev(h)
            for i, kind in enumerate(types):
                s = layers[i]
                io = s.io_binding()
                io.bind_ortvalue_input("hidden_in", hov)
                if kind != "linear_attention":
                    io.bind_ortvalue_input("position_ids", pos_ov)
                for o in s.get_outputs():
                    io.bind_output(o.name, "cuda", 0)
                s.run_with_iobinding(io)
                hov = io.get_outputs()[0]          # stays resident on the GPU
            io = norm.io_binding()
            io.bind_ortvalue_input("hidden_in", hov)
            io.bind_output("hidden_out", "cuda", 0)
            norm.run_with_iobinding(io)
            hov = io.get_outputs()[0]
            io = norm.io_binding()                  # applied twice, on purpose
            io.bind_ortvalue_input("hidden_in", hov)
            io.bind_output("hidden_out", "cuda", 0)
            norm.run_with_iobinding(io)
            hidden = io.get_outputs()[0].numpy()
            mark("decoder", t0)

            t0 = time.perf_counter()
            img_llm = hidden[0][mask][-n_cam * tpi:].reshape(n_cam, gh // 2, gw // 2, -1)
            img_vit = premerge_grids(vit_tap, grid)[-n_cam:]
            out = perc.run(None, {"img_vit_feats": img_vit, "img_llm_feats": img_llm})
            mark("head", t0)
            return hidden, out

        print("  warmup ...")
        hidden, out = perception_once()

        ref = Path(args.save_ref) / "onnx_outputs.npz"
        if ref.exists():
            r = np.load(ref)["hidden"]
            rd = np.abs(hidden - r).max() / max(np.abs(r).max(), 1e-9)
            print(f"  hidden vs CPU reference: {rd:.2e} rel")

        timed, wall = {}, []
        for _ in range(args.reps):
            t = time.perf_counter()
            perception_once(timed)
            wall.append(time.perf_counter() - t)
        n = args.reps
        print(f"\n  {'stage':<18}{'ms':>10}")
        for k in ("vision", "host_scatter", "decoder", "head"):
            print(f"  {k:<18}{1000*timed[k]/n:>10.1f}")
        med = float(np.median(wall))
        print(f"  {'-'*28}\n  {'TOTAL':<18}{1000*med:>10.1f}   ->  {1/med:.2f} FPS")
        results["perception"] = {"ms": 1000 * med, "fps": 1 / med,
                                 "stages_ms": {k: 1000 * v / n for k, v in timed.items()}}
        del vis, layers, norm, perc
        import gc; gc.collect()

    # ------------------------------------------------------------ planning path
    if not args.skip_planning:
        d = np.load(BENCH / "planning_inputs.npz")
        px, ids = d["pixel_values"], d["input_ids"]
        pos, embeds = d["position_ids"], d["embeds"]
        img_tok = int(d["img_tok"])
        mask = ids[0] == img_tok
        scale, moms = d["scale"], float(d["min_one_minus_t"])

        rule(f"PLANNING  {ids.shape[1]} tokens, {args.num_steps} Euler steps")
        man = json.loads((Path(args.layers_plan) / "manifest.json").read_text())
        types = man["layer_types"]

        load_s = 0.0
        print(f"  loading {len(types)} layer graphs + vision + planner ...")
        step = sess(Path(args.step))
        vis = sess(Path(args.vision_plan))
        layers = [sess(Path(args.layers_plan) / f"layer_{i:02d}.onnx")
                  for i in range(len(types))]
        print(f"  graph load {load_s:.1f}s (excluded from timings)   {gpu_mem()}")

        names = [x.name for x in step.get_inputs()]
        pos_ov = dev(pos)
        const = {k: dev(d[k]) for k in
                 ("history", "history_velocity", "history_acceleration",
                  "nav_onehot", "ego_status", "position_anchor")}
        dt = 1.0 / args.num_steps

        def planning_once(timed: dict | None = None):
            def mark(k, t0):
                if timed is not None:
                    timed[k] = timed.get(k, 0.0) + time.perf_counter() - t0

            t0 = time.perf_counter()
            _, _, merged = vis.run(None, {"pixel_values": px})
            mark("vision", t0)

            t0 = time.perf_counter()
            h = embeds.copy()
            h[0, mask] = merged[-int(mask.sum()):]
            mark("host_scatter", t0)

            t0 = time.perf_counter()
            hov, kv = dev(h), {}
            for i, kind in enumerate(types):
                s = layers[i]
                io = s.io_binding()
                io.bind_ortvalue_input("hidden_in", hov)
                if kind != "linear_attention":
                    io.bind_ortvalue_input("position_ids", pos_ov)
                for o in s.get_outputs():
                    io.bind_output(o.name, "cuda", 0)
                s.run_with_iobinding(io)
                outs = io.get_outputs()
                hov = outs[0]
                if kind != "linear_attention":
                    kv[i] = (outs[1].numpy().transpose(0, 2, 1, 3),
                             outs[2].numpy().transpose(0, 2, 1, 3))
            mark("decoder", t0)

            t0 = time.perf_counter()
            flat = [t for i in sorted(kv) for t in kv[i]]
            kv_ov = [dev(t.astype(np.float32)) for t in flat]   # constant over the loop
            way = d["noise"].astype(np.float32)
            for it in range(args.num_steps):
                io = step.io_binding()
                io.bind_cpu_input("waypoints", way)
                io.bind_cpu_input("flow_time", np.full((way.shape[0],), it * dt, np.float32))
                for k, v in const.items():
                    io.bind_ortvalue_input(k, v)
                for nm, ov in zip(names[8:], kv_ov):
                    io.bind_ortvalue_input(nm, ov)
                io.bind_output(step.get_outputs()[0].name, "cpu")
                step.run_with_iobinding(io)
                endpoint = io.copy_outputs_to_cpu()[0]
                way = way + (endpoint - way) / max(1.0 - it * dt, moms) * dt
            mark("planner", t0)
            return wrap_heading(way * scale.reshape(1, 1, -1))

        print("  warmup ...")
        traj = planning_once()

        ref = Path(args.save_ref_plan) / "onnx_plan.npz"
        if ref.exists():
            r = np.load(ref)["trajectory"]
            ade = float(np.linalg.norm(traj[0, :, :2] - r[0, :, :2], axis=-1).mean())
            print(f"  trajectory vs CPU reference: ADE {ade:.2e} m")

        timed, wall = {}, []
        for _ in range(args.reps):
            t = time.perf_counter()
            planning_once(timed)
            wall.append(time.perf_counter() - t)
        n = args.reps
        print(f"\n  {'stage':<18}{'ms':>10}")
        for k in ("vision", "host_scatter", "decoder", "planner"):
            print(f"  {k:<18}{1000*timed[k]/n:>10.1f}")
        med = float(np.median(wall))
        print(f"  {'-'*28}\n  {'TOTAL':<18}{1000*med:>10.1f}   ->  {1/med:.2f} FPS")
        results["planning"] = {"ms": 1000 * med, "fps": 1 / med,
                               "stages_ms": {k: 1000 * v / n for k, v in timed.items()}}

    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "gpu_latency.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {BENCH/'gpu_latency.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "bench"], required=True)
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-root", default="data/demo")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vision", default="outputs/onnx/vlm_vision/vision.onnx")
    ap.add_argument("--vision-plan", default="outputs/onnx/vlm_vision_plan/vision.onnx")
    ap.add_argument("--layers", default="outputs/onnx/vlm_layers")
    ap.add_argument("--layers-plan", default="outputs/onnx/vlm_layers_plan")
    ap.add_argument("--perception", default="outputs/onnx/perception/perception.onnx")
    ap.add_argument("--step", default="outputs/onnx/planner/planner_step.onnx")
    ap.add_argument("--save-ref", default="outputs/onnx_run")
    ap.add_argument("--save-ref-plan", default="outputs/onnx_run_plan")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--opt", default="ENABLE_ALL",
                    choices=["DISABLE_ALL", "ENABLE_BASIC", "ENABLE_EXTENDED", "ENABLE_ALL"])
    ap.add_argument("--skip-perception", action="store_true")
    ap.add_argument("--skip-planning", action="store_true")
    args = ap.parse_args()
    os.chdir(_ROOT)
    return prep(args) if args.phase == "prep" else bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
