"""Convert the exported fp32 graphs to fp16.

The fp32 export is 17 GiB of decoder weights per task, which does not fit
resident on a 24 GiB card alongside the vision tower and the heads. fp16 halves
it and lets the whole stack stay on the GPU, which is what makes a steady-state
FPS number meaningful.

Graph inputs and outputs stay fp32 (``keep_io_types``), so the host code and the
scene tensors are unchanged; only the weights and internal activations move.

    .venv-ortgpu/bin/python export_onnx/to_fp16.py --jobs 6
"""
from __future__ import annotations

import argparse, shutil, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# The converter's own block list is what ORT validates against. Forcing extra ops
# (LayerNormalization, Softmax) to stay fp32 leaves them fed by fp16 producers and
# the model fails to load with a type-binding error; the CUDA kernels for those
# already accumulate in fp32 internally.
BLOCK = None

# Resize takes its scales/sizes as fp32 regardless of the data type, so the
# converter must leave both it and the subgraph computing them alone.
# GridSample is kept fp32 as well: the CUDA kernel ORT registers for it (the
# com.microsoft 4-D one the head was retargeted onto) has no fp16 variant, and a
# fp16 GridSample silently falls back to the CPU, which costs more than the whole
# rest of the head.
BLOCK_BY_DIR = {"perception": ["Resize", "GridSample"]}

# ops that only ever carry shape arithmetic, safe to walk back through
SHAPE_OPS = {"Shape", "Constant", "ConstantOfShape", "Cast", "Concat", "Gather",
             "Slice", "Squeeze", "Unsqueeze", "Mul", "Div", "Add", "Sub",
             "Range", "Floor", "Ceil", "Reshape", "Identity"}


def resize_scale_nodes(graph) -> list[str]:
    """Nodes feeding a Resize's roi/scales/sizes inputs, which must stay fp32."""
    producer = {o: n for n in graph.node for o in n.output}
    blocked, stack = set(), []
    for n in graph.node:
        if n.op_type == "Resize":
            stack += [producer[i] for i in n.input[1:] if i in producer]
    while stack:
        node = stack.pop()
        if node.name in blocked or node.op_type not in SHAPE_OPS:
            continue
        blocked.add(node.name)
        stack += [producer[i] for i in node.input if i in producer]
    return sorted(blocked)


def weight_matmul_blocks(model) -> tuple[list[str], list[str]]:
    """Block lists that confine fp16 to matmuls against a constant weight.

    The Gated-DeltaNet recurrence carries ``exp(cumsum(log a))``; in fp16 the
    decay underflows to zero and the state is then rescaled by its reciprocal,
    so a whole-graph conversion loses the signal (measured 0.42-0.74 relative
    error per layer, and the activation magnitude collapses). The weights are
    what make the graph too large to stay resident, so convert only those: a
    MatMul/Gemm whose second input is an initializer. Recurrence matmuls have
    two activation inputs and are left alone, as is every other op.
    """
    init = {t.name for t in model.graph.initializer}
    keep_ops = {"MatMul", "Gemm"}
    present = {n.op_type for n in model.graph.node}
    op_block = sorted(present - keep_ops)
    node_block = [n.name for n in model.graph.node
                  if n.op_type in keep_ops
                  and not any(i in init for i in n.input)]
    return op_block, node_block


def convert_one(args) -> tuple[str, float, float, float]:
    src, dst = Path(args[0]), Path(args[1])
    block = args[2] if len(args) > 2 else BLOCK
    mode = args[3] if len(args) > 3 else "full"
    import onnx
    # onnxconverter-common 1.16 crashes in remove_unnecessary_cast_node; the
    # converter bundled with onnxruntime is the maintained one.
    from onnxruntime.transformers.float16 import convert_float_to_float16

    t0 = time.perf_counter()
    model = onnx.load(str(src))
    if mode == "weights":
        op_block, node_block = weight_matmul_blocks(model)
        kw = {"op_block_list": op_block, "node_block_list": node_block}
    else:
        kw = {} if block is None else {"op_block_list": block}
        if block and "Resize" in block:
            kw["node_block_list"] = resize_scale_nodes(model.graph)
    out = convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True,
        force_fp16_initializers=True, **kw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(out, str(dst))
    mb_in = src.stat().st_size / 1e6
    mb_out = dst.stat().st_size / 1e6
    return src.name, mb_in, mb_out, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/onnx")
    ap.add_argument("--dst", default="outputs/onnx_fp16")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", default="", help="substring filter on the subdirectory")
    ap.add_argument("--mode", choices=["full", "weights"], default="full",
                    help="'weights' keeps activations fp32 and casts only "
                         "matmul weights, which is what the Gated-DeltaNet "
                         "recurrence needs to stay numerically intact")
    args = ap.parse_args()

    src_root, dst_root = _ROOT / args.src, _ROOT / args.dst
    subdirs = ["vlm_vision", "vlm_layers", "vlm_layers_v2", "perception",
               "vlm_vision_plan", "vlm_layers_plan", "vlm_layers_plan_v2",
               "planner"]
    if args.only:
        subdirs = [s for s in subdirs if args.only in s]

    tasks = []
    for sub in subdirs:
        d = src_root / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.onnx")):
            out = dst_root / sub / f.name
            if out.exists() and out.stat().st_size > 0:
                continue
            tasks.append((str(f), str(out), BLOCK_BY_DIR.get(sub, BLOCK), args.mode))
        # carried over verbatim: the manifest and the embedding table are not graphs
        for extra in ("manifest.json", "embed_tokens.npy"):
            p = d / extra
            if p.exists():
                q = dst_root / sub / extra
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists():
                    (q.symlink_to(p) if extra.endswith(".npy")
                     else shutil.copy2(p, q))

    print(f"{len(tasks)} graphs to convert, {args.jobs} workers")
    if not tasks:
        return 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for name, a, b, dt in ex.map(convert_one, tasks):
            done += 1
            print(f"  [{done:2d}/{len(tasks)}] {name:<22} "
                  f"{a:8.0f} -> {b:7.0f} MB   {dt:5.1f}s", flush=True)
    print(f"\nwrote {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
