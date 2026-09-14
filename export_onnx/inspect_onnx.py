"""Inspect an exported ONNX graph: ops, size, inputs and outputs.

    python export_onnx/inspect_onnx.py --model outputs/onnx/perception.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper


def dim_str(t):
    return "x".join(str(d.dim_value) if d.dim_value else (d.dim_param or "?")
                    for d in t.type.tensor_type.shape.dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/onnx/perception.onnx")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    path = Path(args.model)
    total = sum(f.stat().st_size for f in path.parent.rglob("*")
                if f.is_file()) / 2**20
    model = onnx.load(str(path))
    print(f"{path}   graph {path.stat().st_size/2**20:.1f} MiB   "
          f"directory total {total:.1f} MiB")
    print(f"ir_version {model.ir_version}   opsets: " +
          ", ".join(f"{o.domain or 'ai.onnx'}={o.version}" for o in model.opset_import))

    print("\ninputs:")
    for i in model.graph.input:
        print(f"  {i.name:22s} {dim_str(i)}")
    print("outputs:")
    for o in model.graph.output:
        print(f"  {o.name:22s} {dim_str(o)}")

    ops = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"\n{len(model.graph.node)} nodes, {len(ops)} distinct op types")
    for k, v in sorted(ops.items(), key=lambda x: -x[1])[:args.top]:
        print(f"  {k:24s} {v}")

    init_bytes = sum(len(numpy_helper.to_array(t).tobytes())
                     for t in model.graph.initializer)
    print(f"\n{len(model.graph.initializer)} initializers, "
          f"{init_bytes/2**20:.1f} MiB of weights/constants")
    big = sorted(model.graph.initializer,
                 key=lambda t: -numpy_helper.to_array(t).nbytes)[:5]
    for t in big:
        a = numpy_helper.to_array(t)
        print(f"  {t.name[:46]:46s} {str(a.shape):20s} {a.nbytes/2**20:7.1f} MiB")
    onnx.checker.check_model(model, full_check=False)
    print("\nchecker: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
