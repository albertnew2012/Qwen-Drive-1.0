"""Run Qwen-Drive-1.0 from ONNX: 3D perception and a planned trajectory.

Chain-of-thought / VQA is not part of this pipeline. The ``vlm_decode`` graph
(the autoregressive decode step, ~14 GiB) is never loaded.

    vision.onnx ──► merged tokens ──► (host: embed gather + scatter)
                                             │
                          layer_00..31.onnx ─┼──► hidden ──► perception.onnx
                                             │              (boxes, occupancy, map)
                                             └──► 8 x (keys, values)
                                                        │
                                       planner_step.onnx x10 ──► trajectory [50, 3]

The 8 key/value tensors are NOT a decoding cache: the planning expert attends
over them as the scene memory, so they are required even with no CoT.

Two phases, because they need different interpreters:

    .venv/bin/python        export_onnx/run_onnx_drive.py --phase prep
    .venv-ortgpu/bin/python export_onnx/run_onnx_drive.py --phase run --precision fp16

``prep`` needs torch/transformers to build the real scene tensors once; ``run``
needs onnxruntime-gpu, which cannot sit next to the CPU onnxruntime the
validation harness is pinned to.

fp32 decoder weights are 17 GiB per task and do not fit resident on a 24 GiB
card, so fp32 falls back to loading the layers in chunks. Only graph execution
is timed; session construction is reported separately and excluded.
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
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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

    table = np.load(Path(args.layers or "outputs/onnx/vlm_layers") / "embed_tokens.npy",
                    mmap_mode="r")
    embeds = np.asarray(table)[ids].astype(np.float32)   # input-only: hoisted out
    del table

    np.savez(BENCH / "perception_inputs.npz",
             pixel_values=inputs["pixel_values"].numpy().astype(np.float32),
             input_ids=ids, position_ids=pos, image_grid_thw=grid, embeds=embeds,
             img_tok=np.int64(vlm.config.image_token_id),
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
    scale = model.trajectory_scale(torch.device("cpu"))
    anchor = model._rope_positions(pin["input_ids"], pin["image_grid_thw"])[:, :, -1]
    hist = normalize_history(pin["history"].float(), scale)
    noise = (model.config.noise_init_std *
             model._initial_noise(1, model.config.num_future_points,
                                  args.seed, torch.device("cpu")))
    nav = torch.eye(model.planning_expert.config.nav_command_classes)[
        pin["nav_command"].reshape(-1).long()].reshape(*pin["nav_command"].shape, -1)

    table = np.load(Path(args.layers_plan or "outputs/onnx/vlm_layers_plan") / "embed_tokens.npy",
                    mmap_mode="r")
    embeds_p = np.asarray(table)[ids_p].astype(np.float32)
    del table

    np.savez(BENCH / "planning_inputs.npz",
             pixel_values=pin["pixel_values"].numpy().astype(np.float32),
             input_ids=ids_p, position_ids=ctx["position_ids"].numpy(),
             embeds=embeds_p, img_tok=np.int64(vlm_p.config.image_token_id),
             noise=noise.numpy().astype(np.float32),
             history=hist.numpy().astype(np.float32),
             history_velocity=pin["history_velocity"].numpy().astype(np.float32),
             history_acceleration=pin["history_acceleration"].numpy().astype(np.float32),
             nav_onehot=nav.numpy().astype(np.float32),
             ego_status=pin["ego_status"].numpy().astype(np.float32),
             position_anchor=anchor.numpy(),
             scale=scale.numpy().astype(np.float32),
             min_one_minus_t=np.float32(model.config.min_one_minus_t))
    print(f"   {ids_p.shape[1]} tokens   {model.config.num_future_points} waypoints")
    print(f"\nwrote {BENCH}")
    return 0


# ──────────────────────────────────────────────────────────────── host helpers

def premerge_grids(patches: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """numpy twin of QwenDrivePerception._premerge_grids (reshape + permute)."""
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


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, key, dt):
        self.t[key] = self.t.get(key, 0.0) + dt

    def reset(self):
        self.t = {}


# ───────────────────────────────────────────────────────────────── phase: run

def run(args) -> int:
    import onnxruntime as ort

    root = _ROOT / ("outputs/onnx_fp16" if args.precision == "fp16" else "outputs/onnx")
    if not root.is_dir():
        print(f"missing {root}; run export_onnx/to_fp16.py first")
        return 1

    def comp(override: str, default_sub: str):
        """Directory for one component, so precision can be mixed per stage.

        The best configuration is not uniform: the decoder only fits resident
        with fp16 weights, while the perception head must stay fp32 because
        com.microsoft.GridSample has no fp16 CUDA kernel and silently falls
        back to the CPU.
        """
        return (_ROOT / override) if override else (root / default_sub)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, f"ORT_{args.opt}")
    if args.device == "cuda":
        prov = [("CUDAExecutionProvider", {"device_id": 0,
                                           "arena_extend_strategy": "kSameAsRequested"})]
        # 32 resident sessions would otherwise each build their own arena and the
        # activations no longer fit; one shared arena is what makes them coexist.
        if args.share_arena:
            so.enable_mem_pattern = False
            so.add_session_config_entry("session.use_env_allocators", "1")
            ort.create_and_register_allocator_v2(
                "CUDAExecutionProvider",
                ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
                                  0, ort.OrtMemType.DEFAULT),
                {"device_id": "0"},
                # max_mem, extend strategy, initial chunk, max dead bytes per chunk.
                # A zero initial chunk makes the arena grow in useless increments and
                # session construction never finishes.
                ort.OrtArenaCfg(0, 1, 1024 * 1024, 128 * 1024 * 1024))
    else:
        prov = ["CPUExecutionProvider"]

    load = {"s": 0.0}

    # Weights-only fp16 decoder layers are the one form that is both accurate and
    # small enough to stay resident, but only if ORT is stopped from constant
    # folding the weight Casts away: left alone it rebuilds fp32 initializers and
    # the layer costs 500 MiB instead of 237 MiB, which no longer fits.
    def layer_opts(level=None, dump=None):
        o = ort.SessionOptions()
        o.log_severity_level = 3
        o.graph_optimization_level = level or so.graph_optimization_level
        if args.keep_fp16_weights:
            o.add_session_config_entry(
                "optimization.disable_specified_optimizers", "ConstantFolding")
        if args.device == "cuda" and args.share_arena:
            o.enable_mem_pattern = False
            o.add_session_config_entry("session.use_env_allocators", "1")
        if dump is not None:
            o.optimized_model_filepath = str(dump)
        return o

    so_layer = layer_opts()

    def make(path, opts=None):
        t = time.perf_counter()
        s = ort.InferenceSession(str(path), opts or so, providers=prov)
        load["s"] += time.perf_counter() - t
        if args.device == "cuda" and "CUDAExecutionProvider" not in s.get_providers():
            raise RuntimeError(f"{Path(path).name} fell back to {s.get_providers()}")
        return s

    # The perception head asks for a single ~2 GiB buffer. Served out of the arena
    # that also holds the resident decoder weights it fails on fragmentation even
    # with far more than 2 GiB free, so the head gets its own allocator and takes
    # that block straight from the driver.
    so_head = so
    if args.device == "cuda" and args.share_arena and args.private_head_arena:
        so_head = ort.SessionOptions()
        so_head.log_severity_level = 3
        so_head.graph_optimization_level = so.graph_optimization_level

    on_gpu = args.device == "cuda"

    preopt_root = (_ROOT / args.preopt_cache) if args.preopt_cache else None

    def make_streamed(layer_dir, i):
        """Session for a layer that does not stay resident and is rebuilt every frame.

        Almost all of the ~1.2 s build is graph optimization, and it is repeated
        identically on every frame. Dumping the optimized graph the first time and
        reloading it with optimization disabled gives the same kernels for a
        fraction of the cost.
        """
        src = layer_dir / f"layer_{i:02d}.onnx"
        if preopt_root is None:
            return make(src, so_layer)
        # fp32 and fp16 exports share the same directory basename, so the key has
        # to include the parent or one precision silently loads the other's graphs.
        key = f"{layer_dir.parent.name}__{layer_dir.name}"
        dst = preopt_root / key / f"layer_{i:02d}.onnx"
        if dst.exists():
            return make(dst, layer_opts(ort.GraphOptimizationLevel.ORT_DISABLE_ALL))
        dst.parent.mkdir(parents=True, exist_ok=True)
        return make(src, layer_opts(dump=dst))

    # The perception head needs one ~2 GiB buffer for the view transform. The
    # decoder's arena has to hand that memory back first or it cannot be served.
    shrink = ort.RunOptions()
    if on_gpu and not args.no_shrink:
        shrink.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")

    def dev(a):
        a = np.ascontiguousarray(a)
        return (ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0) if on_gpu
                else ort.OrtValue.ortvalue_from_numpy(a))

    def gpu_mem():
        if not on_gpu:
            return ""
        import subprocess
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            u, t = o.stdout.strip().split("\n")[0].split(",")
            return f"   GPU {int(u)/1024:.1f}/{int(t)/1024:.1f} GiB"
        except Exception:
            return ""

    rule = lambda t: print(f"\n{'='*72}\n  {t}\n{'='*72}")

    def run_layer(sess, kind, hov, pos_ov, timer, want_kv):
        io = sess.io_binding()
        io.bind_ortvalue_input("hidden_in", hov)
        if kind != "linear_attention":
            io.bind_ortvalue_input("position_ids", pos_ov)
        dv = "cuda" if on_gpu else "cpu"
        for o in sess.get_outputs():
            io.bind_output(o.name, dv, 0) if on_gpu else io.bind_output(o.name)
        t = time.perf_counter()
        sess.run_with_iobinding(io)
        timer.add("decoder", time.perf_counter() - t)
        outs = io.get_outputs()
        if kind != "linear_attention" and want_kv:
            # layer graph emits [B, kv_heads, S, head]; the expert wants [B, S, kv_heads, head]
            return outs[0], (outs[1].numpy().transpose(0, 2, 1, 3),
                             outs[2].numpy().transpose(0, 2, 1, 3))
        return outs[0], None

    def decode(hidden, pos_ov, layer_dir, types, resident, timer, want_kv):
        """Push the hidden state through all 32 layers, chunking if not resident."""
        hov, kv = dev(hidden), {}
        if isinstance(resident, dict):
            # Partial residency. The perception head needs ~13 GiB of activations,
            # so the decoder cannot keep all 32 layers and still leave room; the
            # layers that do not fit are rebuilt per frame and dropped again.
            for i, kind in enumerate(types):
                s = resident.get(i)
                once = s is None
                if once:
                    s = make_streamed(layer_dir, i)
                hov, k = run_layer(s, kind, hov, pos_ov, timer, want_kv)
                if k is not None:
                    kv[i] = k
                if once:
                    del s
        elif resident is not None:
            for i, kind in enumerate(types):
                hov, k = run_layer(resident[i], kind, hov, pos_ov, timer, want_kv)
                if k is not None:
                    kv[i] = k
        else:
            n = len(types)
            for start in range(0, n, args.chunk):
                grp = [(i, make_streamed(layer_dir, i))
                       for i in range(start, min(start + args.chunk, n))]
                for i, s in grp:
                    hov, k = run_layer(s, types[i], hov, pos_ov, timer, want_kv)
                    if k is not None:
                        kv[i] = k
                del grp
        return hov, kv

    results = {}

    # ------------------------------------------------------------- perception
    if not args.skip_perception:
        d = np.load(BENCH / "perception_inputs.npz")
        px, ids, pos, grid = d["pixel_values"], d["input_ids"], d["position_ids"], d["image_grid_thw"]
        embeds, n_cam = d["embeds"], int(d["n_cam"])
        mask = ids[0] == int(d["img_tok"])
        gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
        tpi = gh // 2 * gw // 2
        ldir = _ROOT / args.layers if args.layers else root / "vlm_layers"
        types = json.loads((ldir / "manifest.json").read_text())["layer_types"]

        rule(f"PERCEPTION   {ids.shape[1]} tokens, {n_cam} cameras, {args.precision}")
        load["s"] = 0.0
        perc = make(comp(args.head_dir, "perception") / "perception.onnx", so_head)
        vis = make(comp(args.vision_dir, "vlm_vision") / "vision.onnx")
        norm = make(ldir / "final_norm.onnx", so_layer)
        resident = None
        if args.chunk == 0:
            keep = len(types) if args.resident < 0 else min(args.resident, len(types))
            resident = {}
            for i in range(keep):
                resident[i] = make(ldir / f"layer_{i:02d}.onnx", so_layer)
                if (i + 1) % 8 == 0:
                    print(f"    {i+1:2d}/{len(types)} layers   {load['s']:5.1f}s{gpu_mem()}",
                          flush=True)
        print(f"  graphs loaded in {load['s']:.1f}s (excluded from timings)"
              f"{gpu_mem()}" + ("" if resident else f"   [chunked x{args.chunk}]"))

        pos_ov = dev(pos)
        timer = Timer()

        def perception_once():
            t = time.perf_counter()
            _, vit_tap, merged = vis.run(None, {"pixel_values": px})
            timer.add("vision", time.perf_counter() - t)

            t = time.perf_counter()
            h = embeds.copy()
            h[0, mask] = merged[-int(mask.sum()):]
            timer.add("host", time.perf_counter() - t)

            hov, _ = decode(h, pos_ov, ldir, types, resident, timer, want_kv=False)
            t = time.perf_counter()
            for k in range(2):        # the final norm is applied twice, on purpose
                io = norm.io_binding()
                io.bind_ortvalue_input("hidden_in", hov)
                io.bind_output("hidden_out", "cuda", 0) if on_gpu else io.bind_output("hidden_out")
                norm.run_with_iobinding(io, shrink if (on_gpu and k == 1) else None)
                hov = io.get_outputs()[0]
            hidden = hov.numpy()
            timer.add("decoder", time.perf_counter() - t)

            t = time.perf_counter()
            img_llm = hidden[0][mask][-n_cam * tpi:].reshape(n_cam, gh // 2, gw // 2, -1)
            img_vit = premerge_grids(vit_tap, grid)[-n_cam:]
            cls, box, occ, seg = perc.run(None, {"img_vit_feats": img_vit,
                                                 "img_llm_feats": img_llm},
                                          shrink if on_gpu else None)
            timer.add("head", time.perf_counter() - t)
            return hidden, (cls, box, occ, seg)

        hidden, out = perception_once()
        ref = _ROOT / args.save_ref / "onnx_outputs.npz"
        if ref.exists():
            r = np.load(ref)["hidden"]
            print(f"  hidden vs fp32 CPU reference: "
                  f"{np.abs(hidden - r).max() / max(np.abs(r).max(), 1e-9):.2e} rel")
        cls, box, occ, seg = out
        print(f"  cls {cls.shape}  box {box.shape}  occ {occ.shape}  seg {seg.shape}")

        timer.reset()
        wall = []
        for _ in range(args.reps):
            t = time.perf_counter()
            perception_once()
            wall.append(time.perf_counter() - t)
        med = float(np.median(wall))
        print(f"\n  {'stage':<14}{'ms':>10}")
        for k in ("vision", "host", "decoder", "head"):
            print(f"  {k:<14}{1000*timer.t.get(k,0)/args.reps:>10.1f}")
        print(f"  {'-'*24}\n  {'TOTAL':<14}{1000*med:>10.1f}   ->  {1/med:6.2f} FPS")
        results["perception"] = {"ms": 1000 * med, "fps": 1 / med, "load_s": load["s"],
                                 "stages_ms": {k: 1000*v/args.reps for k, v in timer.t.items()}}
        np.savez(BENCH / f"perception_out_{args.precision}.npz",
                 cls=cls, box=box, occ=occ, seg=seg)
        del perc, vis, norm, resident
        import gc; gc.collect()

    # --------------------------------------------------------------- planning
    if not args.skip_planning:
        d = np.load(BENCH / "planning_inputs.npz")
        px, ids, pos, embeds = d["pixel_values"], d["input_ids"], d["position_ids"], d["embeds"]
        mask = ids[0] == int(d["img_tok"])
        scale, moms = d["scale"], float(d["min_one_minus_t"])
        ldir = _ROOT / args.layers_plan if args.layers_plan else root / "vlm_layers_plan"
        types = json.loads((ldir / "manifest.json").read_text())["layer_types"]

        rule(f"PLANNING   {ids.shape[1]} tokens, {args.num_steps} Euler steps, {args.precision}")
        load["s"] = 0.0
        step = make(comp(args.planner_dir, "planner") / "planner_step.onnx")
        vis = make(comp(args.vision_plan_dir, "vlm_vision_plan") / "vision.onnx")
        resident = None
        if args.chunk == 0:
            keep = len(types) if args.resident_plan < 0 else min(args.resident_plan,
                                                                 len(types))
            resident = {}
            for i in range(keep):
                resident[i] = make(ldir / f"layer_{i:02d}.onnx", so_layer)
                if (i + 1) % 8 == 0:
                    print(f"    {i+1:2d}/{len(types)} layers   {load['s']:5.1f}s{gpu_mem()}",
                          flush=True)
        print(f"  graphs loaded in {load['s']:.1f}s (excluded from timings)"
              f"{gpu_mem()}" + ("" if resident else f"   [chunked x{args.chunk}]"))

        names = [x.name for x in step.get_inputs()]
        pos_ov = dev(pos)
        const = {k: dev(d[k]) for k in ("history", "history_velocity",
                                        "history_acceleration", "nav_onehot",
                                        "ego_status", "position_anchor")}
        dt = 1.0 / args.num_steps
        timer = Timer()

        def planning_once():
            t = time.perf_counter()
            _, _, merged = vis.run(None, {"pixel_values": px})
            timer.add("vision", time.perf_counter() - t)

            t = time.perf_counter()
            h = embeds.copy()
            h[0, mask] = merged[-int(mask.sum()):]
            timer.add("host", time.perf_counter() - t)

            _, kv = decode(h, pos_ov, ldir, types, resident, timer, want_kv=True)

            t = time.perf_counter()
            kv_ov = [dev(x.astype(np.float32))          # constant across the loop
                     for i in sorted(kv) for x in kv[i]]
            way = d["noise"].astype(np.float32)
            for it in range(args.num_steps):
                io = step.io_binding()
                io.bind_cpu_input("waypoints", way)
                io.bind_cpu_input("flow_time", np.full((way.shape[0],), it*dt, np.float32))
                for k, v in const.items():
                    io.bind_ortvalue_input(k, v)
                for nm, ov in zip(names[8:], kv_ov):
                    io.bind_ortvalue_input(nm, ov)
                io.bind_output(step.get_outputs()[0].name, "cpu")
                step.run_with_iobinding(io)
                endpoint = io.copy_outputs_to_cpu()[0]
                way = way + (endpoint - way) / max(1.0 - it*dt, moms) * dt
            timer.add("planner", time.perf_counter() - t)
            return wrap_heading(way * scale.reshape(1, 1, -1))

        traj = planning_once()
        ref = _ROOT / args.save_ref_plan / "onnx_plan.npz"
        if ref.exists():
            r = np.load(ref)["trajectory"]
            ade = float(np.linalg.norm(traj[0, :, :2] - r[0, :, :2], axis=-1).mean())
            print(f"  trajectory vs fp32 CPU reference: ADE {ade:.4f} m")
        print(f"  endpoint {traj[0, -1].round(3)}")

        timer.reset()
        wall = []
        for _ in range(args.reps):
            t = time.perf_counter()
            planning_once()
            wall.append(time.perf_counter() - t)
        med = float(np.median(wall))
        print(f"\n  {'stage':<14}{'ms':>10}")
        for k in ("vision", "host", "decoder", "planner"):
            print(f"  {k:<14}{1000*timer.t.get(k,0)/args.reps:>10.1f}")
        print(f"  {'-'*24}\n  {'TOTAL':<14}{1000*med:>10.1f}   ->  {1/med:6.2f} FPS")
        results["planning"] = {"ms": 1000 * med, "fps": 1 / med, "load_s": load["s"],
                               "stages_ms": {k: 1000*v/args.reps for k, v in timer.t.items()}}
        np.savez(BENCH / f"trajectory_{args.precision}.npz", trajectory=traj)

    if results:
        p = BENCH / f"latency_{args.precision}_{args.device}.json"
        p.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {p}")
        if len(results) == 2:
            tot = results["perception"]["ms"] + results["planning"]["ms"]
            print(f"\n  perception + trajectory: {tot:.0f} ms  ->  {1000/tot:.2f} FPS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "run"], required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--chunk", type=int, default=0,
                    help="layers held resident at once; 0 = all (needs fp16 on 24 GiB)")
    ap.add_argument("--no-shrink", action="store_true",
                    help="do not shrink the CUDA arena between decoder and head; "
                         "only safe when chunking keeps memory low")
    ap.add_argument("--share-arena", action="store_true", default=True)
    ap.add_argument("--no-share-arena", dest="share_arena", action="store_false")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--opt", default="ENABLE_ALL",
                    choices=["DISABLE_ALL", "ENABLE_BASIC", "ENABLE_EXTENDED", "ENABLE_ALL"])
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-root", default="data/demo")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    # prep always needs a concrete path; run falls back to the precision root when empty
    ap.add_argument("--layers", default="")
    ap.add_argument("--layers-plan", default="")
    ap.add_argument("--head-dir", default="", help="override the perception head directory")
    ap.add_argument("--vision-dir", default="", help="override the vision tower directory")
    ap.add_argument("--vision-plan-dir", default="")
    ap.add_argument("--planner-dir", default="")
    ap.add_argument("--keep-fp16-weights", action="store_true",
                    help="build layer sessions without ConstantFolding, so "
                         "weights-only fp16 graphs stay fp16 in memory")
    ap.add_argument("--private-head-arena", action="store_true", default=True)
    ap.add_argument("--no-private-head-arena", dest="private_head_arena",
                    action="store_false")
    ap.add_argument("--preopt-cache", default="outputs/onnx_preopt",
                    help="cache of pre-optimized graphs for per-frame layer rebuilds")
    ap.add_argument("--resident", type=int, default=-1,
                    help="layers kept loaded between frames; -1 = all. The rest "
                         "are rebuilt each frame to leave room for the head")
    ap.add_argument("--resident-plan", type=int, default=-1,
                    help="same, for planning, which has no perception head "
                         "competing for VRAM and so can usually keep more")
    ap.add_argument("--save-ref", default="outputs/onnx_run")
    ap.add_argument("--save-ref-plan", default="outputs/onnx_run_plan")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-perception", action="store_true")
    ap.add_argument("--skip-planning", action="store_true")
    args = ap.parse_args()
    os.chdir(_ROOT)
    return prep(args) if args.phase == "prep" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
