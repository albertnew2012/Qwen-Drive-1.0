"""Put the large projection weights in fp16, one GEMM at a time.

Both existing converters fail on this graph for different reasons.
``to_fp16.py`` converts everything and the Gated-DeltaNet recurrence loses its
range -- 0.41 relative on a layer output, against 4e-07 for what this does.
``mixed_precision.py`` converts only the heavy GEMMs, but torch exports a
projection as ``initializer -> Transpose -> MatMul``; the Transpose is not itself a
GEMM, so it lands in the block list, the initializer stays fp32, and all the pass
achieves is a per-frame Cast of 451 MiB. Measured 78.97 ms before, 78.10 ms after.
Putting the Transpose in the conversion set instead makes
``onnxruntime.transformers.float16`` emit a MatMul with one fp16 and one fp32
input, which will not load.

So the rewrite is done directly. For every MatMul or Gemm with a weight of at
least ``--min-elems`` elements:

  * the weight initializer becomes fp16 (a Transpose in front of it is
    type-generic and follows along),
  * the activation input gets a Cast to fp16,
  * the output gets a Cast back to fp32.

Everything outside those islands, the recurrence included, stays exactly as it
was. The casts are on activations -- 28 MiB, not 451 MiB -- so they cost about
0.3 ms per layer against 225 MiB less weight traffic and fp16 tensor cores.

Halving the weights is also what makes the stack fit: the fp32 decoder is 14.7 GiB
and the perception head's 625 MiB and 2 GiB buffers then cannot be served on a
24 GiB card, while capping the shared arena low enough to leave room starves the
vision tower instead.

    python export_onnx/fp16_weights_v3.py SRC_DIR DST_DIR
"""
from __future__ import annotations

import argparse, math, shutil, time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

GEMM_OPS = {"MatMul", "Gemm"}
PASSTHROUGH = {"Transpose", "Reshape", "Squeeze", "Unsqueeze"}


def _weight_chain(name, inits, produced, depth=4):
    """(element count, initializer, nodes) for the weight behind ``name``."""
    if name in inits:
        t = inits[name]
        return int(math.prod(t.dims) or 0), t, []
    node = produced.get(name)
    if depth and node is not None and node.op_type in PASSTHROUGH:
        n, t, rest = _weight_chain(node.input[0], inits, produced, depth - 1)
        return n, t, [node] + rest
    return 0, None, []


def convert_model(model: onnx.ModelProto, min_elems: int) -> tuple[int, int]:
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    produced = {o: n for n in graph.node for o in n.output}
    new_nodes, n_gemm, done = [], 0, set()

    for node in graph.node:
        if node.op_type not in GEMM_OPS:
            new_nodes.append(node)
            continue
        heavy = None
        for idx, inp in enumerate(list(node.input)[:2]):
            count, tensor, _ = _weight_chain(inp, inits, produced)
            if tensor is not None and count >= min_elems:
                heavy = (idx, tensor)
                break
        if heavy is None:
            new_nodes.append(node)
            continue
        idx, tensor = heavy
        if tensor.name not in done and tensor.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor).astype(np.float16)
            tensor.CopyFrom(numpy_helper.from_array(arr, tensor.name))
            done.add(tensor.name)

        # Cast the activation side (and a Gemm bias, which is small) to fp16.
        for other in range(len(node.input)):
            if other == idx or not node.input[other]:
                continue
            src = node.input[other]
            bias = inits.get(src)
            if bias is not None and bias.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(bias).astype(np.float16)
                bias.CopyFrom(numpy_helper.from_array(arr, bias.name))
                continue
            cast_out = f"{src}_f16_{node.name.replace('/', '_')}_{other}"
            new_nodes.append(helper.make_node(
                "Cast", [src], [cast_out], to=TensorProto.FLOAT16,
                name=f"{cast_out}__cast"))
            node.input[other] = cast_out

        # Run the GEMM in fp16 and cast the result back so nothing downstream
        # has to know this happened.
        casts_back = []
        for o in range(len(node.output)):
            orig = node.output[o]
            half = f"{orig}_f16"
            node.output[o] = half
            casts_back.append(helper.make_node(
                "Cast", [half], [orig], to=TensorProto.FLOAT,
                name=f"{orig}__cast_back"))
        new_nodes.append(node)
        new_nodes.extend(casts_back)
        n_gemm += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    # Stale value_info would contradict the new types; ORT re-infers.
    del graph.value_info[:]
    return n_gemm, len(done)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--only", default="")
    ap.add_argument("--min-elems", type=int, default=1 << 20)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for extra in ("manifest.json", "embed_tokens.npy"):
        s, d = src / extra, dst / extra
        if s.exists() and not d.exists():
            if extra.endswith(".npy") or s.is_symlink():
                d.symlink_to(s.resolve())
            else:
                shutil.copy2(s, d)

    files = sorted(p for p in src.glob("*.onnx") if args.only in p.name)
    before_tot = after_tot = 0.0
    t0 = time.time()
    for p in files:
        before = sum(f.stat().st_size for f in src.glob(f"{p.stem}.*")) / 2**20
        model = onnx.load(str(p))
        n_gemm, n_w = convert_model(model, args.min_elems)
        for stale in dst.glob(f"{p.stem}.*"):
            stale.unlink()
        onnx.save(model, str(dst / p.name), save_as_external_data=True,
                  all_tensors_to_one_file=True, location=f"{p.stem}.data",
                  size_threshold=1024)
        after = sum(f.stat().st_size for f in dst.glob(f"{p.stem}.*")) / 2**20
        before_tot += before; after_tot += after
        print(f"  {p.name:22s} {n_gemm:3d} gemms, {n_w:2d} weights   "
              f"{before:7.1f} -> {after:7.1f} MiB", flush=True)
    print(f"\n  {len(files)} graphs  {before_tot/1024:.2f} -> {after_tot/1024:.2f} GiB"
          f"  in {time.time()-t0:.0f}s -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
