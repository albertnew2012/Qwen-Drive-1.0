"""Lower the opset a graph *declares* so ORT will use its CUDA kernels.

onnxruntime does not keep a CUDA kernel for every opset revision of every
operator.  When a graph declares a newer opset than the newest CUDA
registration, the node silently falls back to the CPU provider - no warning,
just a node that runs on the host with a device copy on either side.

Two operators in this model are hit by that:

``Pad``         CUDA is registered up to opset 18.  Pad-19 only added support
                for negative pads, which this graph never uses, so the kernel
                would be correct - it simply is not registered.  Each decoder
                layer has five Pads on ``[1, 32, 2744, 128]`` tensors, and at
                opset 20 they cost 15 ms on the CPU plus 24 ms of round trip.

``GridSample``  handled separately in ``export_perception.py``, because there
                the fix is a domain change rather than an opset change.

``Resize``      CUDA is registered up to opset 18 as well, for both 4-D and
                5-D.  The perception head's six Resize nodes cost 132 ms on the
                CPU at opset 20 plus the device copies around them.

Lowering the declared opset can strand a node whose operator was only added
later, so ops newer than the target are rewritten into equivalent subgraphs
first (``Gelu`` arrived in opset 20).  ORT's own fusion passes fold the
replacement straight back into a single contrib kernel.

Rewriting the declared opset is only sound when no node in the graph relies on
behaviour introduced after the target.  That is not something to assume, so
``--verify`` runs the graph before and after on identical inputs and reverts
unless the outputs are bit-identical.

Only the small ``.onnx`` file is rewritten; external weight files are
referenced by relative path and are left untouched.

    python export_onnx/retarget_opset.py --opset 18 --verify outputs/onnx/vlm_layers
    python export_onnx/retarget_opset.py --revert outputs/onnx/vlm_layers
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

SUFFIX = ".opset_backup"

# operator -> the opset that introduced it, for ops this tool knows how to undo
INTRODUCED = {"Gelu": 20}


def _elem_type(model: onnx.ModelProto, name: str) -> int:
    for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if v.name == name and v.type.tensor_type.elem_type:
            return v.type.tensor_type.elem_type
    return onnx.TensorProto.FLOAT


def _expand_gelu(node, model, tag: str):
    """Gelu -> its defining formula, valid from opset 9."""
    approximate = "none"
    for a in node.attribute:
        if a.name == "approximate":
            approximate = a.s.decode()
    x, y, p = node.input[0], node.output[0], f"{tag}_"
    np_dt = onnx.helper.tensor_dtype_to_np_dtype(_elem_type(model, x))

    def const(name, v):
        return helper.make_node("Constant", [], [p + name], name=f"{p}{name}_n",
                                value=numpy_helper.from_array(np.array(v, np_dt), p + name + "_v"))

    def op(kind, ins, out, **kw):
        return helper.make_node(kind, ins, [p + out], name=f"{p}{out}_n", **kw)

    if approximate == "tanh":
        return [
            const("half", 0.5), const("one", 1.0),
            const("c3", 0.044715), const("k", 0.7978845608028654), const("three", 3.0),
            op("Pow", [x, p + "three"], "x3"),
            op("Mul", [p + "x3", p + "c3"], "x3c"),
            op("Add", [x, p + "x3c"], "inner"),
            op("Mul", [p + "inner", p + "k"], "scaled"),
            op("Tanh", [p + "scaled"], "t"),
            op("Add", [p + "t", p + "one"], "a"),
            op("Mul", [x, p + "half"], "h"),
            helper.make_node("Mul", [p + "h", p + "a"], [y], name=f"{p}out_n"),
        ]
    return [
        const("half", 0.5), const("one", 1.0), const("isqrt2", 0.7071067811865476),
        op("Mul", [x, p + "isqrt2"], "s"),
        op("Erf", [p + "s"], "e"),
        op("Add", [p + "e", p + "one"], "a"),
        op("Mul", [x, p + "half"], "h"),
        helper.make_node("Mul", [p + "h", p + "a"], [y], name=f"{p}out_n"),
    ]


def _downgrade_ops(model: onnx.ModelProto, target: int) -> int:
    """Replace standard-domain ops that do not exist at ``target``."""
    out, n = [], 0
    for i, node in enumerate(model.graph.node):
        if (node.domain or "") == "" and INTRODUCED.get(node.op_type, 0) > target:
            if node.op_type != "Gelu":
                raise ValueError(f"no rewrite for {node.op_type} at opset {target}")
            out.extend(_expand_gelu(node, model, f"{node.op_type.lower()}{i}"))
            n += 1
        else:
            out.append(node)
    if n:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return n


def _graphs(targets: list[str]) -> list[Path]:
    out: list[Path] = []
    for t in targets:
        p = Path(t)
        out.extend(sorted(p.glob("*.onnx")) if p.is_dir() else [p])
    return out


def _declared(model: onnx.ModelProto) -> int | None:
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            return o.version
    return None


def _outputs(path: Path):
    """Run the graph on CUDA with deterministic inputs."""
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        str(path), so, providers=[("CUDAExecutionProvider", {"device_id": 0})])
    dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
              "tensor(int64)": np.int64, "tensor(int32)": np.int32,
              "tensor(bool)": np.bool_}
    feed = {}
    for i in sess.get_inputs():
        if any(not isinstance(d, int) for d in i.shape):
            raise ValueError(f"{path.name}: dynamic shape {i.shape}, cannot verify")
        dt = dtypes.get(i.type, np.float32)
        rs = np.random.RandomState(0)
        feed[i.name] = (rs.randn(*i.shape).astype(dt) * 0.02 if dt in (np.float32, np.float16)
                        else np.zeros(i.shape, dt))
    return sess.run(None, feed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="graph files or directories of graphs")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--verify", action="store_true",
                    help="run each graph before and after; revert unless bit-identical")
    ap.add_argument("--revert", action="store_true", help="restore the backups")
    args = ap.parse_args()

    graphs = _graphs(args.targets)
    if not graphs:
        print("no .onnx files found", file=sys.stderr)
        return 1

    if args.revert:
        n = 0
        for g in graphs:
            bak = g.with_suffix(g.suffix + SUFFIX)
            if bak.exists():
                shutil.move(str(bak), str(g))
                n += 1
        print(f"reverted {n}/{len(graphs)} graphs")
        return 0

    changed = skipped = reverted = 0
    for g in graphs:
        model = onnx.load(str(g), load_external_data=False)
        current = _declared(model)
        if current is None or current <= args.opset:
            skipped += 1
            continue

        before = _outputs(g) if args.verify else None

        bak = g.with_suffix(g.suffix + SUFFIX)
        if not bak.exists():
            shutil.copy2(str(g), str(bak))
        lowered = _downgrade_ops(model, args.opset)
        for o in model.opset_import:
            if o.domain in ("", "ai.onnx"):
                o.version = args.opset
        onnx.save(model, str(g))

        if args.verify:
            import numpy as np
            try:
                after = _outputs(g)
                diff = max(float(np.abs(a - b).max()) for a, b in zip(before, after))
            except Exception as exc:                      # kernel missing at the lower opset
                diff, reason = None, str(exc).splitlines()[0][:90]
            else:
                reason = f"max abs diff {diff:.3e}"
            if diff != 0.0:
                shutil.move(str(bak), str(g))
                reverted += 1
                print(f"  {g.name:20s} REVERTED  {reason}")
                continue
            print(f"  {g.name:20s} opset {current} -> {args.opset}  bit-identical")
        elif lowered:
            print(f"  {g.name:20s} opset {current} -> {args.opset}  ({lowered} op(s) rewritten)")
        changed += 1

    print(f"\nretargeted {changed} graphs to opset {args.opset} "
          f"({skipped} already at or below, {reverted} reverted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
