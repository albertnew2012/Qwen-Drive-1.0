"""Run only the heavy GEMMs in fp16, leaving everything else in float32.

A whole-graph fp16 conversion of this model is useless: the Gated-DeltaNet
recurrence overflows and the trajectory comes out NaN (see ``to_fp16.py`` and
the notes in ``run_onnx_drive.py``).  The recurrence needs the range, but it is
not where the time goes - the projections are.  On this card a
``[2744, 2560] x [2560, 4096]`` matmul measures 1.82 ms in float32 and 0.90 ms
in float16, and a depthwise conv 0.45 ms against 0.45 ms.

So convert MatMul/Gemm/Conv and nothing else.  ``convert_float_to_float16``
inserts the Cast pairs and, with ``keep_io_types``, leaves the graph's inputs
and outputs float32, so the host code and every other graph are unaffected.
The weights of those nodes become fp16 initializers, which also halves what the
decoder occupies on the card.

    python export_onnx/mixed_precision.py outputs/onnx/vlm_layers_v2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx

# Only matmuls against a large constant weight are converted - the projections
# and the MLP. The recurrence also contains matmuls, but both of their operands
# are activations, and those are the ones that overflow: the chunk decay is an
# exp() of cumulative gates and the state accumulates across 43 chunks. Conv is
# excluded too: fp16 measured no faster for a depthwise conv (0.45 ms either
# way), and its bias makes the converter emit mixed types.
GEMM_OPS = {"MatMul", "Gemm", "FusedMatMul"}
MIN_WEIGHT_ELEMS = 1 << 20


def _projection_nodes(model: onnx.ModelProto) -> set[str]:
    sizes = {}
    for t in model.graph.initializer:
        n = 1
        for d in t.dims:
            n *= d
        sizes[t.name] = n
    # torch exports a weight as initializer -> Transpose -> MatMul, so the
    # matmul's own input is the transpose output, not the weight
    produced = {o: node for node in model.graph.node for o in node.output}

    def weight_elems(name: str, depth: int = 4) -> int:
        if name in sizes:
            return sizes[name]
        node = produced.get(name)
        if depth and node is not None and node.op_type in ("Transpose", "Reshape", "Cast"):
            return weight_elems(node.input[0], depth - 1)
        return 0

    keep = set()
    for node in model.graph.node:
        if node.op_type not in GEMM_OPS:
            continue
        if any(weight_elems(i) >= MIN_WEIGHT_ELEMS for i in node.input):
            keep.add(node.name)
    return keep


def convert(path: Path, keep_io: bool = True) -> tuple[int, int]:
    from onnxruntime.transformers.float16 import convert_float_to_float16

    # the layers of a stack share one directory, so note which external files
    # belong to THIS graph before overwriting it and clean up only those
    shell = onnx.load(str(path), load_external_data=False)
    mine = {kv.value for t in shell.graph.initializer
            for kv in t.external_data if kv.key == "location"}

    model = onnx.load(str(path))
    keep = _projection_nodes(model)
    block = [n.name for n in model.graph.node if n.name not in keep]

    out = convert_float_to_float16(
        model, keep_io_types=keep_io, node_block_list=block,
        disable_shape_infer=True)

    location = f"{path.stem}.data"
    tmp = path.parent / location
    if tmp.exists():
        tmp.unlink()
    onnx.save(out, str(path), save_as_external_data=True,
              all_tensors_to_one_file=True, location=location, size_threshold=1024)
    for name in mine - {location}:
        f = path.parent / name
        if f.exists():
            f.unlink()
    return len(keep), len(model.graph.node)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="graph files or directories")
    ap.add_argument("--no-keep-io", action="store_true",
                    help="also make the graph inputs/outputs fp16")
    args = ap.parse_args()

    graphs: list[Path] = []
    for t in args.targets:
        p = Path(t)
        graphs.extend(sorted(p.glob("*.onnx")) if p.is_dir() else [p])
    if not graphs:
        print("no .onnx files found", file=sys.stderr)
        return 1

    for g in graphs:
        n, total = convert(g, keep_io=not args.no_keep_io)
        size = g.stat().st_size + sum(
            f.stat().st_size for f in g.parent.glob(g.stem + ".data"))
        print(f"  {g.name:20s} {n:5d} heavy nodes -> fp16 "
              f"of {total} nodes  now {size/2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
