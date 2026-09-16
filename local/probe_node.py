"""Inspect one node and its neighbourhood: input/output shapes and producers."""
import sys
from collections import defaultdict

import onnx
from onnx import shape_inference

p = "outputs/onnx_vt/perception/perception.onnx"
target = sys.argv[1] if len(sys.argv) > 1 else \
    "/bev/head/transformer/encoder/layers.0/attentions.1/deformable_attention/Mul_1"

m = onnx.load(p, load_external_data=False)
m = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
g = m.graph
vi = {v.name: v for v in list(g.value_info) + list(g.output) + list(g.input)}
init = {i.name: i for i in g.initializer}
producer = {o: n for n in g.node for o in n.output}
consumers = defaultdict(list)
for n in g.node:
    for i in n.input:
        consumers[i].append(n)

def shape(name):
    if name in init:
        return f"INIT{list(init[name].dims)}"
    v = vi.get(name)
    if v is None:
        return "?"
    t = v.type.tensor_type
    return str([d.dim_value if d.HasField("dim_value") else "?" for d in t.shape.dim])

node = next(n for n in g.node if n.name == target)
print(f"{node.op_type}  {node.name}")
for i in node.input:
    pn = producer.get(i)
    print(f"   in  {shape(i):40s} <- {pn.op_type if pn else 'graph/init'}")
for o in node.output:
    print(f"   out {shape(o):40s} -> {[c.op_type for c in consumers[o]]}")

print("\n downstream:")
cur = node.output[0]
for _ in range(8):
    cs = consumers.get(cur)
    if not cs:
        break
    n = cs[0]
    print(f"   {n.op_type:12s} {shape(n.output[0]):42s} {n.name.split('/')[-1]}")
    cur = n.output[0]
