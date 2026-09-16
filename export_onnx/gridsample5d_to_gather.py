"""Rewrite 5-D `GridSample` into gather-based trilinear sampling so it runs on CUDA.

ONNX Runtime's CUDA `GridSample` kernel is 4-D only — a 5-D node reports
"Only 4-D tensor is supported" and, at opset 20, is silently handed to the CPU
provider instead.  In the perception head that costs 772 ms of CPU sampling plus
~294 ms of forced host round-trips, because every tensor crossing the boundary
has to be copied out of device memory and back.

Trilinear sampling is just a weighted sum of the eight voxel corners around each
sample point, and `Gather` on a flattened spatial axis expresses that with
ordinary CUDA kernels.  Removing the 5-D nodes also frees the graph from needing
opset 20 at all, which matters because ORT registers its CUDA `Resize` only up
to opset 18 — so the same change lets the six `Resize` nodes move to the GPU as
well.

Everything is built from runtime `Shape` arithmetic rather than baked-in
constants, so the rewrite does not assume the frozen batch/channel sizes.

    python export_onnx/gridsample5d_to_gather.py --self-test
    python export_onnx/gridsample5d_to_gather.py outputs/onnx/perception/perception.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SUFFIX = ".gs5d_backup"


def _const(name: str, arr: np.ndarray):
    return helper.make_node("Constant", [], [name],
                            value=numpy_helper.from_array(arr, name + "_v"))


def expand_node(node, prefix: str):
    """Return the replacement nodes for one 5-D GridSample.

    Follows the ONNX spec: `grid[..., 0]` indexes W, `[..., 1]` indexes H and
    `[..., 2]` indexes D, i.e. the coordinate order is reversed with respect to
    the spatial axes.
    """
    attrs = {a.name: a for a in node.attribute}
    mode = attrs["mode"].s.decode() if "mode" in attrs else "linear"
    padding = attrs["padding_mode"].s.decode() if "padding_mode" in attrs else "zeros"
    align = attrs["align_corners"].i if "align_corners" in attrs else 0
    if mode not in ("linear", "bilinear"):
        raise ValueError(f"{node.name}: mode {mode!r} is not trilinear")
    if padding != "zeros":
        raise ValueError(f"{node.name}: padding_mode {padding!r} unsupported")

    X, G = node.input[0], node.input[1]
    Y = node.output[0]
    p = prefix
    n = []

    def add(op, ins, outs, **kw):
        n.append(helper.make_node(op, ins, outs, name=f"{outs[0]}_n", **kw))

    n.append(_const(f"{p}i0", np.array([0], np.int64)))
    n.append(_const(f"{p}i1", np.array([1], np.int64)))
    n.append(_const(f"{p}i2", np.array([2], np.int64)))
    n.append(_const(f"{p}i3", np.array([3], np.int64)))
    n.append(_const(f"{p}i4", np.array([4], np.int64)))
    n.append(_const(f"{p}i5", np.array([5], np.int64)))
    n.append(_const(f"{p}neg1", np.array([-1], np.int64)))
    n.append(_const(f"{p}zero_i", np.array(0, np.int64)))
    n.append(_const(f"{p}one_f", np.array(1.0, np.float32)))
    n.append(_const(f"{p}half", np.array(0.5, np.float32)))
    n.append(_const(f"{p}zero_f", np.array(0.0, np.float32)))

    # ---- shapes: X is [N, C, D, H, W], grid is [N, Do, Ho, Wo, 3]
    add("Shape", [X], [f"{p}xs"])
    add("Shape", [G], [f"{p}gs"])
    for tag, lo, hi in (("N", "i0", "i1"), ("C", "i1", "i2"),
                        ("D", "i2", "i3"), ("H", "i3", "i4"), ("W", "i4", "i5")):
        add("Slice", [f"{p}xs", f"{p}{lo}", f"{p}{hi}"], [f"{p}{tag}"])
    for tag, lo, hi in (("Do", "i1", "i2"), ("Ho", "i2", "i3"), ("Wo", "i3", "i4")):
        add("Slice", [f"{p}gs", f"{p}{lo}", f"{p}{hi}"], [f"{p}{tag}"])

    # ---- X -> [N, C, D*H*W], grid coords -> [N, 1, P]
    add("Mul", [f"{p}D", f"{p}H"], [f"{p}DH"])
    add("Mul", [f"{p}DH", f"{p}W"], [f"{p}DHW"])
    add("Mul", [f"{p}H", f"{p}W"], [f"{p}HW"])
    add("Concat", [f"{p}N", f"{p}C", f"{p}neg1"], [f"{p}flatshape"], axis=0)
    add("Reshape", [X, f"{p}flatshape"], [f"{p}xflat"])
    add("Concat", [f"{p}N", f"{p}i1", f"{p}neg1"], [f"{p}coordshape"], axis=0)

    add("Split", [G], [f"{p}gx5", f"{p}gy5", f"{p}gz5"], axis=-1, num_outputs=3)
    for axis, size in (("x", "W"), ("y", "H"), ("z", "D")):
        add("Reshape", [f"{p}g{axis}5", f"{p}coordshape"], [f"{p}g{axis}"])
        # unnormalise: align_corners picks which convention maps [-1,1] to pixels
        add("Cast", [f"{p}{size}"], [f"{p}{size}f"], to=TensorProto.FLOAT)
        add("Reshape", [f"{p}{size}f", f"{p}neg1"], [f"{p}{size}f1"])
        add("Squeeze", [f"{p}{size}f1", f"{p}i0"], [f"{p}{size}s"])
        add("Sub", [f"{p}{size}s", f"{p}one_f"], [f"{p}{size}m1"])
        if align:
            # x = (g + 1) * (size - 1) / 2
            add("Add", [f"{p}g{axis}", f"{p}one_f"], [f"{p}{axis}p1"])
            add("Mul", [f"{p}{axis}p1", f"{p}{size}m1"], [f"{p}{axis}m"])
            add("Mul", [f"{p}{axis}m", f"{p}half"], [f"{p}c{axis}"])
        else:
            # x = ((g + 1) * size - 1) / 2
            add("Add", [f"{p}g{axis}", f"{p}one_f"], [f"{p}{axis}p1"])
            add("Mul", [f"{p}{axis}p1", f"{p}{size}s"], [f"{p}{axis}m"])
            add("Sub", [f"{p}{axis}m", f"{p}one_f"], [f"{p}{axis}m1v"])
            add("Mul", [f"{p}{axis}m1v", f"{p}half"], [f"{p}c{axis}"])
        add("Floor", [f"{p}c{axis}"], [f"{p}f{axis}"])
        add("Sub", [f"{p}c{axis}", f"{p}f{axis}"], [f"{p}t{axis}"])
        for i in (0, 1):
            base = f"{p}{axis}{i}"
            if i == 0:
                add("Identity", [f"{p}f{axis}"], [f"{base}c"])
                add("Sub", [f"{p}one_f", f"{p}t{axis}"], [f"{base}wraw"])
            else:
                add("Add", [f"{p}f{axis}", f"{p}one_f"], [f"{base}c"])
                add("Identity", [f"{p}t{axis}"], [f"{base}wraw"])
            # zeros padding: a corner outside the volume contributes nothing
            add("GreaterOrEqual", [f"{base}c", f"{p}zero_f"], [f"{base}ge"])
            add("LessOrEqual", [f"{base}c", f"{p}{size}m1"], [f"{base}le"])
            add("And", [f"{base}ge", f"{base}le"], [f"{base}ok"])
            add("Cast", [f"{base}ok"], [f"{base}okf"], to=TensorProto.FLOAT)
            add("Mul", [f"{base}wraw", f"{base}okf"], [f"{base}w"])
            # clamp before gathering so the index is always in range
            add("Clip", [f"{base}c", f"{p}zero_f", f"{p}{size}m1"], [f"{base}cl"])
            add("Cast", [f"{base}cl"], [f"{base}idx"], to=TensorProto.INT64)

    # ---- offsets that are shared between corners
    add("Cast", [f"{p}HW"], [f"{p}HWi"], to=TensorProto.INT64)
    add("Cast", [f"{p}W"], [f"{p}Wi"], to=TensorProto.INT64)
    add("Reshape", [f"{p}HWi", f"{p}neg1"], [f"{p}HW1"])
    add("Squeeze", [f"{p}HW1", f"{p}i0"], [f"{p}HWstr"])
    add("Reshape", [f"{p}Wi", f"{p}neg1"], [f"{p}W1"])
    add("Squeeze", [f"{p}W1", f"{p}i0"], [f"{p}Wstr"])
    for k in (0, 1):
        add("Mul", [f"{p}z{k}idx", f"{p}HWstr"], [f"{p}zoff{k}"])
    for j in (0, 1):
        add("Mul", [f"{p}y{j}idx", f"{p}Wstr"], [f"{p}yoff{j}"])

    # ---- accumulate the eight corners
    acc = None
    for k in (0, 1):
        for j in (0, 1):
            for i in (0, 1):
                c = f"{p}c{i}{j}{k}"
                add("Add", [f"{p}zoff{k}", f"{p}yoff{j}"], [f"{c}zy"])
                add("Add", [f"{c}zy", f"{p}x{i}idx"], [f"{c}flat"])
                add("Reshape", [f"{c}flat", f"{p}neg1"], [f"{c}flat1"])
                add("Gather", [f"{p}xflat", f"{c}flat1"], [f"{c}g"], axis=2)
                add("Mul", [f"{p}x{i}w", f"{p}y{j}w"], [f"{c}wxy"])
                add("Mul", [f"{c}wxy", f"{p}z{k}w"], [f"{c}w"])
                add("Mul", [f"{c}g", f"{c}w"], [f"{c}v"])
                if acc is None:
                    acc = f"{c}v"
                else:
                    add("Add", [acc, f"{c}v"], [f"{c}acc"])
                    acc = f"{c}acc"

    add("Concat", [f"{p}N", f"{p}C", f"{p}Do", f"{p}Ho", f"{p}Wo"], [f"{p}outshape"], axis=0)
    n.append(helper.make_node("Reshape", [acc, f"{p}outshape"], [Y], name=f"{p}out"))
    return n


def rank_of(vi) -> int:
    t = vi.type.tensor_type
    return len(t.shape.dim) if t.HasField("shape") else -1


def rewrite(path: Path) -> int:
    inferred = Path(str(path) + ".inferred")
    onnx.shape_inference.infer_shapes_path(str(path), str(inferred))
    meta = onnx.load(str(inferred), load_external_data=False)
    inferred.unlink(missing_ok=True)
    ranks = {v.name: rank_of(v) for v in
             list(meta.graph.value_info) + list(meta.graph.input) + list(meta.graph.output)}

    model = onnx.load(str(path))
    targets = [n for n in model.graph.node
               if n.op_type == "GridSample" and ranks.get(n.input[0], -1) == 5]
    if not targets:
        print(f"  {path.name}: no 5-D GridSample")
        return 0

    out, done = [], 0
    for node in model.graph.node:
        if node in targets:
            try:
                out.extend(expand_node(node, f"{node.name.strip('/').replace('/', '_')}_gs5_"))
                done += 1
                print(f"  expanded {node.name}")
            except ValueError as e:
                print(f"  SKIP {e}")
                out.append(node)
        else:
            out.append(node)
    del model.graph.node[:]
    model.graph.node.extend(out)

    backup = Path(str(path) + SUFFIX)
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    onnx.save(model, str(path), save_as_external_data=True,
              all_tensors_to_one_file=True, location=f"{path.stem}.data",
              size_threshold=1024)
    print(f"  {path.name}: expanded {done} node(s), now {len(model.graph.node)} nodes")
    return done


def self_test() -> int:
    """Check the decomposition against ORT's own CPU GridSample."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    N, C, D, H, W = 1, 5, 6, 9, 11
    Do, Ho, Wo = 4, 7, 8
    X = rng.standard_normal((N, C, D, H, W), dtype=np.float32)
    # deliberately overshoot [-1,1] so the zeros-padding path is exercised
    G = (rng.random((N, Do, Ho, Wo, 3), dtype=np.float32) * 2.6 - 1.3)

    vX = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, D, H, W])
    vG = helper.make_tensor_value_info("G", TensorProto.FLOAT, [N, Do, Ho, Wo, 3])
    vY = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, C, Do, Ho, Wo])

    worst = 0.0
    for align in (0, 1):
        node = helper.make_node("GridSample", ["X", "G"], ["Y"], name="/gs",
                                mode="linear", align_corners=align, padding_mode="zeros")
        ref_m = helper.make_model(helper.make_graph([node], "g", [vX, vG], [vY]),
                                  opset_imports=[helper.make_opsetid("", 20)])
        ref_m.ir_version = 10
        so = ort.SessionOptions()
        so.log_severity_level = 3
        ref = ort.InferenceSession(ref_m.SerializeToString(), so,
                                   providers=["CPUExecutionProvider"]).run(None, {"X": X, "G": G})[0]

        new_m = helper.make_model(
            helper.make_graph(expand_node(node, "t_"), "g", [vX, vG], [vY]),
            opset_imports=[helper.make_opsetid("", 18)])
        new_m.ir_version = 10
        prov = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"])
        got = ort.InferenceSession(new_m.SerializeToString(), so,
                                   providers=prov).run(None, {"X": X, "G": G})[0]
        rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
        worst = max(worst, rel)
        print(f"  align_corners={align}  shape {got.shape}  rel vs CPU GridSample {rel:.3e}"
              f"   provider {prov[0][:4]}")
    print(f"  worst {worst:.3e}")
    return 0 if worst < 1e-5 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.paths:
        ap.error("give a model path or --self-test")

    for path in args.paths:
        if args.revert:
            backup = Path(str(path) + SUFFIX)
            if backup.exists():
                path.write_bytes(backup.read_bytes())
                backup.unlink()
                print(f"  reverted {path.name}")
            continue
        rewrite(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
