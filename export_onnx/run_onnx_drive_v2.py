"""Run the v2 ONNX export: 3D perception + planned trajectory, no chain of thought.

What changes against ``run_onnx_drive.py``:

*Graphs.*  ``export_vlm_layers_v2`` traces the Gated-DeltaNet layers into 2.1k
nodes instead of 14.2k and, more to the point, into few large kernels instead of
thousands of tiny ones.  One linear layer: 125.96 -> 30.59 ms.

*Residency.*  The shipped run rebuilt the 32 layer sessions every frame because a
17 GiB fp32 decoder and the head could not share a 24 GiB card, and that rebuild
was 18.3 s of its 22.2 s.  There are two cards here, so perception gets one and
planning the other, each holding its own decoder resident for the whole run.

*Device residency between layers.*  The hidden state is bound straight from one
layer's output to the next layer's input, so it never touches the host.  The
shipped path round-tripped it through numpy once per layer.

*Concurrency.*  Perception and planning share no tensors -- they are two prefills
of the same weights over different prompts -- so with ``--parallel`` they run at
the same time on the two cards and the frame costs the slower of the two rather
than the sum.

*Padding.*  The graphs expect a sequence rounded up to a multiple of the chunk
size (``manifest.json``: ``sequence`` and ``padded``).  Appending zero rows is
safe because everything here is causal, and the host slices them off again.  It
buys the removal of five CPU-resident Pad nodes per linear layer.

    source export_onnx/env_gpu.sh
    python export_onnx/run_onnx_drive_v2.py --parallel --reps 5
"""
from __future__ import annotations

import argparse, json, os, threading, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
BENCH = _ROOT / "outputs" / "onnx_bench"

from export_onnx.run_onnx_drive import premerge_grids, wrap_heading, Timer


def _session_options(ort, opt: str, shared_arena: bool, keep_casts: bool = False):
    o = ort.SessionOptions()
    o.log_severity_level = 3
    o.graph_optimization_level = getattr(ort.GraphOptimizationLevel, f"ORT_{opt}")
    if shared_arena:
        # 32 resident sessions each building a private arena do not fit; one
        # shared allocator is what lets them coexist. Carried over verbatim from
        # run_onnx_drive.py, where it was worked out.
        o.enable_mem_pattern = False
        o.add_session_config_entry("session.use_env_allocators", "1")
    if keep_casts:
        o.add_session_config_entry(
            "optimization.disable_specified_optimizers", "ConstantFolding")
    return o


def _register_arena(ort, device_id: int, max_gib: float = 17.0) -> None:
    """One shared arena for the 32 resident layer sessions, with a ceiling.

    Uncapped (``max_mem = 0``) the arena keeps whatever it grows to, and the
    perception head's later request for a 625 MiB block then fails on a card with
    9 GiB nominally free -- the arena has it all. Capping the arena at a little
    over the 14.7 GiB the fp32 decoder weights occupy leaves the rest of the card
    for the head's own allocator.
    """
    ort.create_and_register_allocator_v2(
        "CUDAExecutionProvider",
        ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
                          device_id, ort.OrtMemType.DEFAULT),
        {"device_id": str(device_id)},
        # max_mem, extend strategy, initial chunk, max dead bytes per chunk. A
        # zero initial chunk makes the arena grow in useless increments and
        # session construction never finishes.
        ort.OrtArenaCfg(int(max_gib * 2**30), 1, 1024 * 1024, 128 * 1024 * 1024))


class Stack:
    """The 32 decoder layer graphs, resident on one device and chained on it."""

    def __init__(self, ort, ldir: Path, device_id: int, opt: str,
                 shared_arena: bool, log=print):
        self.ort = ort
        self.device_id = device_id
        man = json.loads((ldir / "manifest.json").read_text())
        self.types = man["layer_types"]
        self.sequence = man["sequence"]
        self.padded = man.get("padded", man["sequence"])
        self.hidden = man["hidden"]
        prov = [("CUDAExecutionProvider",
                 {"device_id": device_id, "arena_extend_strategy": "kSameAsRequested"})]
        so = _session_options(ort, opt, shared_arena)
        t0 = time.perf_counter()
        self.layers = []
        for i in range(len(self.types)):
            s = ort.InferenceSession(str(ldir / f"layer_{i:02d}.onnx"), so, providers=prov)
            if "CUDAExecutionProvider" not in s.get_providers():
                raise RuntimeError(f"layer {i} fell back to {s.get_providers()}")
            self.layers.append(s)
            if (i + 1) % 8 == 0:
                log(f"    gpu{device_id}  {i+1:2d}/{len(self.types)} layers  "
                    f"{time.perf_counter()-t0:5.1f}s")
        self.norm = ort.InferenceSession(str(ldir / "final_norm.onnx"), so, providers=prov)
        self.load_s = time.perf_counter() - t0
        # The perception head asks for one ~625 MiB block and one ~2 GiB block.
        # Served out of the arena that also holds 14.7 GiB of decoder weights it
        # fails on fragmentation with plenty of the card still free, so the arena
        # is told to hand back its activation chunks on the way out of the stack.
        self.shrink = ort.RunOptions()
        self.shrink.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage", f"gpu:{device_id}")

    def pad(self, embeds: np.ndarray) -> np.ndarray:
        n = self.padded - embeds.shape[1]
        if n <= 0:
            return embeds
        return np.concatenate(
            [embeds, np.zeros((embeds.shape[0], n, embeds.shape[2]), embeds.dtype)], 1)

    def __call__(self, embeds: np.ndarray, pos_ov, want_kv: bool = False):
        """Push a padded hidden state through the stack; returns it on device.

        Outputs are bound by whatever each graph declares: the perception export
        emits only the hidden state from its full-attention layers, while the
        planning export also emits the cache, already transposed and trimmed for
        the expert, so it can be handed on without touching the host.
        """
        ort = self.ort
        h = ort.OrtValue.ortvalue_from_numpy(self.pad(embeds), "cuda", self.device_id)
        kv = []
        for i, sess in enumerate(self.layers):
            io = sess.io_binding()
            io.bind_ortvalue_input("hidden_in", h)
            if self.types[i] == "full_attention":
                io.bind_ortvalue_input("position_ids", pos_ov)
            names = [o.name for o in sess.get_outputs()]
            for n in names:
                io.bind_output(n, "cuda", self.device_id)
            sess.run_with_iobinding(io)
            outs = io.get_outputs()
            h = outs[0]
            if want_kv and len(outs) == 3:
                kv.extend((outs[1], outs[2]))
        # Applied twice on purpose: the head was trained on the final norm of
        # hidden_states[-1], and transformers has already normed that.
        for k in range(2):
            io = self.norm.io_binding()
            io.bind_ortvalue_input("hidden_in", h)
            io.bind_output("hidden_out", "cuda", self.device_id)
            self.norm.run_with_iobinding(io, self.shrink if k == 1 else None)
            h = io.get_outputs()[0]
        return h, kv


def perception_stage(ort, args, log):
    d = np.load(_ROOT / args.perception_inputs)
    px, ids, pos, grid = d["pixel_values"], d["input_ids"], d["position_ids"], d["image_grid_thw"]
    embeds, n_cam = d["embeds"], int(d["n_cam"])
    mask = ids[0] == int(d["img_tok"])
    gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
    tpi = gh // 2 * gw // 2
    gid = args.gpu_perception
    prov = [("CUDAExecutionProvider", {"device_id": gid,
                                       "arena_extend_strategy": "kSameAsRequested"})]
    stack = Stack(ort, _ROOT / args.layers, gid, args.opt, args.share_arena, log)
    so = _session_options(ort, args.opt, args.share_arena)
    vis = ort.InferenceSession(str(_ROOT / args.vision_dir / "vision.onnx"), so, providers=prov)
    # The head asks for one ~2 GiB buffer and fails on fragmentation if it comes
    # out of the arena holding the decoder, so it gets a plain allocator.
    head_gid = gid if args.gpu_head < 0 else args.gpu_head
    head = ort.InferenceSession(
        str(_ROOT / args.head_dir / "perception.onnx"),
        _session_options(ort, args.opt, False),
        providers=[("CUDAExecutionProvider",
                    {"device_id": head_gid,
                     "arena_extend_strategy": "kSameAsRequested"})])
    vis_inputs = {i.name for i in vis.get_inputs()}
    pos_ov = ort.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(
            np.concatenate([pos, np.repeat(pos[:, :, -1:], stack.padded - pos.shape[2], 2)], 2)
            if stack.padded > pos.shape[2] else pos), "cuda", gid)
    timer = Timer()

    def once():
        t = time.perf_counter()
        # the one-camera vision export takes grid_thw as an input; the shipped
        # six-camera graph baked it in
        feed = {"pixel_values": px}
        if "grid_thw" in vis_inputs:
            feed["grid_thw"] = grid
        _, vit_tap, merged = vis.run(None, feed)
        timer.add("vision", time.perf_counter() - t)
        t = time.perf_counter()
        h = embeds.copy()
        h[0, mask] = merged[-int(mask.sum()):]
        timer.add("host", time.perf_counter() - t)
        t = time.perf_counter()
        hov, _ = stack(h, pos_ov)
        hidden = hov.numpy()[:, :stack.sequence]
        timer.add("decoder", time.perf_counter() - t)
        t = time.perf_counter()
        img_llm = hidden[0][mask][-n_cam * tpi:].reshape(n_cam, gh // 2, gw // 2, -1)
        img_vit = premerge_grids(vit_tap, grid)[-n_cam:]
        out = head.run(None, {"img_vit_feats": img_vit, "img_llm_feats": img_llm})
        timer.add("head", time.perf_counter() - t)
        return hidden, out

    return stack, once, timer


def planning_stage(ort, args, log):
    d = np.load(_ROOT / args.planning_inputs)
    gid = args.gpu_planning
    prov = [("CUDAExecutionProvider", {"device_id": gid,
                                       "arena_extend_strategy": "kSameAsRequested"})]
    stack = Stack(ort, _ROOT / args.layers_plan, gid, args.opt, args.share_arena, log)
    so = _session_options(ort, args.opt, args.share_arena)
    vis = ort.InferenceSession(str(_ROOT / args.vision_plan_dir / "vision.onnx"),
                               so, providers=prov)
    plan_gid = gid if args.gpu_planner < 0 else args.gpu_planner
    planner = ort.InferenceSession(
        str(_ROOT / args.planner_dir / "planner_step.onnx"),
        _session_options(ort, args.opt, False),
        providers=[("CUDAExecutionProvider",
                    {"device_id": plan_gid, "arena_extend_strategy": "kSameAsRequested"})])
    ids, pos, embeds = d["input_ids"], d["position_ids"], d["embeds"]
    mask = ids[0] == int(d["img_tok"])
    scale, moms = d["scale"], float(d["min_one_minus_t"])
    pos_ov = ort.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(
            np.concatenate([pos, np.repeat(pos[:, :, -1:], stack.padded - pos.shape[2], 2)], 2)
            if stack.padded > pos.shape[2] else pos), "cuda", gid)
    names = [x.name for x in planner.get_inputs()]
    const = {k: ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(d[k]), "cuda", gid)
             for k in ("history", "history_velocity", "history_acceleration",
                       "nav_onehot", "ego_status", "position_anchor")}
    dt = 1.0 / args.num_steps
    timer = Timer()

    def once():
        t = time.perf_counter()
        merged = vis.run(None, {"pixel_values": d["pixel_values"]})[-1]
        timer.add("vision", time.perf_counter() - t)
        t = time.perf_counter()
        h = embeds.copy()
        h[0, mask] = merged[-int(mask.sum()):]
        timer.add("host", time.perf_counter() - t)
        t = time.perf_counter()
        _, kv = stack(h, pos_ov, want_kv=True)
        timer.add("decoder", time.perf_counter() - t)

        t = time.perf_counter()
        # The cache is produced on the decoder's card. If the planner sits on the
        # other one it has to cross, which is 222 MiB once per frame rather than
        # once per Euler step.
        kv_in = kv if plan_gid == gid else [
            ort.OrtValue.ortvalue_from_numpy(x.numpy(), "cuda", plan_gid) for x in kv]
        way = d["noise"].astype(np.float32)
        for it in range(args.num_steps):
            io = planner.io_binding()
            io.bind_cpu_input("waypoints", way)
            io.bind_cpu_input("flow_time", np.full((way.shape[0],), it * dt, np.float32))
            for k, v in const.items():
                io.bind_ortvalue_input(k, v)
            # The cache is constant across the Euler steps and already on the card.
            for nm, ov in zip(names[8:], kv_in):
                io.bind_ortvalue_input(nm, ov)
            io.bind_output(planner.get_outputs()[0].name, "cpu")
            planner.run_with_iobinding(io)
            endpoint = io.copy_outputs_to_cpu()[0]
            way = way + (endpoint - way) / max(1.0 - it * dt, moms) * dt
        timer.add("planner", time.perf_counter() - t)
        return wrap_heading(way * scale.reshape(1, 1, -1)), None

    return stack, once, timer


def main() -> int:
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="outputs/onnx/vlm_layers_v2_perception")
    ap.add_argument("--layers-plan", default="outputs/onnx/vlm_layers_v2_planning")
    ap.add_argument("--head-dir", default="outputs/onnx/perception")
    ap.add_argument("--vision-dir", default="outputs/onnx/vlm_vision")
    ap.add_argument("--vision-plan-dir", default="outputs/onnx/vlm_vision_plan")
    ap.add_argument("--planner-dir", default="outputs/onnx/planner")
    ap.add_argument("--gpu-perception", type=int, default=0)
    ap.add_argument("--gpu-planning", type=int, default=1)
    ap.add_argument("--opt", default="ENABLE_ALL")
    ap.add_argument("--no-share-arena", dest="share_arena", action="store_false")
    ap.add_argument("--arena-max-gib", type=float, default=17.0,
                    help="ceiling on the shared decoder arena, so the head can "
                         "still get its large buffers from the driver")
    ap.add_argument("--perception-inputs",
                    default="outputs/onnx_bench/perception_inputs.npz",
                    help="bench npz for perception; a front-camera export needs the "
                         "front-camera prompt, not the six-camera one")
    ap.add_argument("--planning-inputs",
                    default="outputs/onnx_bench/planning_inputs.npz")
    ap.add_argument("--gpu-planner", type=int, default=-1,
                    help="device for the Planning Expert. The only placement that "
                         "fits two 24 GiB cards puts both decoders on one and the "
                         "perception head (15.4 GiB on its own) plus the planner "
                         "on the other.")
    ap.add_argument("--gpu-head", type=int, default=-1,
                    help="device for the perception head (default: same as the "
                         "perception decoder). Its inputs come from the host "
                         "anyway, so it costs nothing to put it on the other card.")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--parallel", action="store_true",
                    help="run perception and planning at the same time on the two cards")
    ap.add_argument("--skip-perception", action="store_true")
    ap.add_argument("--skip-planning", action="store_true")
    ap.add_argument("--out", default="outputs/onnx_bench/v2_result.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    lock = threading.Lock()
    def log(*a):
        with lock:
            print(*a, flush=True)

    for gid in {args.gpu_perception, args.gpu_planning}:
        if args.share_arena:
            _register_arena(ort, gid, args.arena_max_gib)

    stages = {}
    t0 = time.perf_counter()
    if not args.skip_perception:
        log("loading perception graphs")
        stages["perception"] = perception_stage(ort, args, log)
    if not args.skip_planning:
        log("loading planning graphs")
        stages["planning"] = planning_stage(ort, args, log)
    log(f"graphs loaded in {time.perf_counter()-t0:.1f}s (excluded from timings)")

    # one untimed pass, both to warm up and to print what came out
    for name, (stack, once, timer) in stages.items():
        out = once()
        if name == "perception":
            # the detection-only head emits two tensors, the full head four
            hidden, head_out = out
            keys = ["cls", "box", "occ", "seg"][:len(head_out)]
            log("  perception  " + "  ".join(
                f"{k} {v.shape}" for k, v in zip(keys, head_out)))
            np.savez(BENCH / "v2_perception_out.npz", hidden=hidden,
                     **dict(zip(keys, head_out)))
        else:
            traj, _ = out
            if traj is not None:
                log(f"  planning    trajectory {traj.shape}  endpoint "
                    f"{np.round(traj[0, -1], 3).tolist()}")
                np.savez(BENCH / "v2_planning_out.npz", trajectory=traj)
        timer.reset()

    def timed(once, reps):
        w = []
        for _ in range(reps):
            t = time.perf_counter(); once(); w.append(time.perf_counter() - t)
        return w

    result = {"reps": args.reps, "parallel": args.parallel, "stages": {}}
    if args.parallel and len(stages) == 2:
        walls = {}
        def worker(name):
            walls[name] = timed(stages[name][1], args.reps)
        th = [threading.Thread(target=worker, args=(n,)) for n in stages]
        t = time.perf_counter()
        for x in th: x.start()
        for x in th: x.join()
        frame = (time.perf_counter() - t) / args.reps
    else:
        walls = {n: timed(stages[n][1], args.reps) for n in stages}
        frame = sum(float(np.median(v)) for v in walls.values())

    log(f"\n  {'stage':<26}{'ms':>10}")
    for name, (stack, once, timer) in stages.items():
        med = float(np.median(walls[name]))
        for k, v in timer.t.items():
            log(f"    {name[:4]}.{k:<20}{1000*v/args.reps:>10.1f}")
        log(f"  {name:<26}{1000*med:>10.1f}")
        result["stages"][name] = {"ms": 1000 * med,
                                 "sub_ms": {k: 1000*v/args.reps for k, v in timer.t.items()}}
    log(f"  {'-'*36}")
    log(f"  {'FRAME':<26}{1000*frame:>10.1f}   ->  {1/frame:6.2f} FPS"
        f"   ({'parallel, 2 GPUs' if args.parallel and len(stages)==2 else 'sequential'})")
    result["frame_ms"] = 1000 * frame
    result["fps"] = 1 / frame
    Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
