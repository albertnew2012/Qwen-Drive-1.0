"""Collapse the perception view transform's per-camera voxel volume.

The exported graph concatenates the six per-camera voxel grids into a single
[6, 640000, 256] tensor (3.93 GB) and then pushes it through Reshape -> Cast ->
Transpose -> Reshape -> Reshape before a ReduceSum over the camera axis. Every
one of those steps materialises another 3.93 GB, so the view transform alone
needs ~12 GB of activations and the perception head cannot coexist with a
resident decoder.

Reshape, Cast and Transpose all act independently per camera slice, so they
commute with the sum over cameras. Summing the six cameras first makes every
downstream tensor 655 MB instead of 3.93 GB, for identical results. The rest of
the chain derives its shapes dynamically and adapts on its own; only the one
hard-coded reshape target has to be told the camera axis is now 1.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

SRC = Path("outputs/onnx/perception/perception.onnx")
pos = [a for a in sys.argv[1:] if not a.startswith("-")]
DST = Path(pos[0] if pos else "outputs/onnx_vt/perception/perception.onnx")

CONCAT = "/bev/view_trans/Concat_26"
RESHAPE = "/bev/view_trans/Reshape_26"

m = onnx.load(str(SRC))
g = m.graph
by_out = {o: n for n in g.node for o in n.output}

concat = next(n for n in g.node if n.name == CONCAT)
reshape = next(n for n in g.node if n.name == RESHAPE)
assert concat.output[0] == reshape.input[0]
axis = next(a.i for a in concat.attribute if a.name == "axis")
assert axis == 0, f"expected camera concat on axis 0, got {axis}"
cams = list(concat.input)
print(f"  concat of {len(cams)} camera volumes on axis {axis}")

# Sum the cameras in place of the concat, keeping the concat's output name so no
# consumer has to be touched.
nodes = [n for n in g.node if n.name != CONCAT]
add_nodes = []
acc = cams[0]
for k, other in enumerate(cams[1:]):
    out = concat.output[0] if k == len(cams) - 2 else f"{CONCAT}_camsum_{k}"
    add_nodes.append(helper.make_node("Add", [acc, other], [out],
                                      name=f"{CONCAT}_camsum_add_{k}"))
    acc = out

# Splice the adds in where the concat used to be so the graph stays topological.
idx = next(i for i, n in enumerate(g.node) if n.name == CONCAT)
nodes = list(g.node)[:idx] + add_nodes + [n for n in list(g.node)[idx + 1:]]

# The camera axis is now 1, not 6.
shape_name = f"{RESHAPE}_camsum_shape"
g.initializer.append(numpy_helper.from_array(
    np.array([1, 1, 200, 200, 16, 256], dtype=np.int64), shape_name))
reshape.input[1] = shape_name

# After the Cast and Transpose the tensor is [1, 1, 256, 16, 200, 200], so the
# two reshapes and the sum over the camera axis that followed are now a plain
# squeeze. They are replaced by one reshape that keeps the ReduceSum's output
# name, which leaves every consumer downstream untouched.
transpose = next(n for n in nodes if n.name == "/bev/view_trans/Transpose_6")
reduce_out = "/bev/view_trans/ReduceSum_output_0"
tail = {"/bev/view_trans/Reshape_27", "/bev/view_trans/Reshape_28",
        by_out[reduce_out].name}
out_shape = f"{RESHAPE}_camsum_out_shape"
g.initializer.append(numpy_helper.from_array(
    np.array([1, 256, 16, 200, 200], dtype=np.int64), out_shape))
squeeze = helper.make_node("Reshape", [transpose.output[0], out_shape],
                           [reduce_out], name=f"{RESHAPE}_camsum_squeeze")
ti = next(i for i, n in enumerate(nodes) if n.name == transpose.name)
nodes = [n for n in nodes if n.name not in tail]
nodes.insert(ti + 1, squeeze)
print(f"  dropped {len(tail)} tail nodes, camera sum folded into the adds")

del g.node[:]
g.node.extend(nodes)


def fuse_deformable_weighted_sum(g):
    """Fold `ReduceSum(Mul(values, weights), -1)` into a single Einsum.

    Each BEV encoder layer multiplies the sampled values [48, 32, 10046, 32] by
    the attention weights [48, 1, 10046, 32] and immediately sums the sampling
    points away. The product is a 1.975 GB tensor that exists only to be
    reduced, and it is what the head runs out of memory on. Einsum contracts the
    points directly, so the intermediate never has to be allocated.
    """
    by_out = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    const = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            const[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)

    drop, add = set(), []
    order = {n.name: i for i, n in enumerate(g.node)}
    for mul in [n for n in g.node
                if n.op_type == "Mul" and n.name.endswith("/Mul_1")
                and "deformable_attention" in n.name]:
        cs = consumers.get(mul.output[0], [])
        if len(cs) != 1 or cs[0].op_type != "ReduceSum":
            continue
        rs = cs[0]
        if next((a.i for a in rs.attribute if a.name == "keepdims"), 1) != 0:
            continue
        axes = const.get(rs.input[1]) if len(rs.input) > 1 else None
        if axes is None or list(np.atleast_1d(axes)) not in ([-1], [3]):
            print(f"  skip {rs.name}: axes={axes}")
            continue
        add.append((order[mul.name],
                    helper.make_node("Einsum", list(mul.input), [rs.output[0]],
                                     name=mul.name + "_einsum",
                                     equation="bcqp,bdqp->bcq")))
        drop |= {mul.name, rs.name}

    if not add:
        return 0
    kept = list(g.node)
    for idx, node in add:
        kept[idx] = node
    kept = [n for n in kept if n.name not in drop or n.op_type == "Einsum"]
    del g.node[:]
    g.node.extend(kept)
    return len(add)


# Off by default: it is numerically fine and slightly faster, but ORT's CUDA
# Einsum transposes internally and ends up needing ~2 GB *more* than the Mul it
# replaces, which is the opposite of what the head needs.
if "--fuse-deformable" in sys.argv:
    print(f"  fused {fuse_deformable_weighted_sum(g)} deformable weighted sums")

# Stale shapes would now be wrong; ORT re-infers them.
del g.value_info[:]

DST.parent.mkdir(parents=True, exist_ok=True)
onnx.save(m, str(DST), save_as_external_data=True,
          location=DST.name + ".data", all_tensors_to_one_file=True,
          size_threshold=1024, convert_attribute=False)
print(f"  wrote {DST}")
