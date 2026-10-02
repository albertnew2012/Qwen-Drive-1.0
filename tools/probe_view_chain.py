"""Print the node chain from the per-camera scatter to the camera ReduceSum."""
import onnx, sys
from collections import defaultdict

p = "outputs/onnx/perception/perception.onnx"
m = onnx.load(p, load_external_data=False)
g = m.graph
producer = {o: n for n in g.node for o in n.output}
consumers = defaultdict(list)
for n in g.node:
    for i in n.input:
        consumers[i].append(n)

def attrs(n):
    out = []
    for a in n.attribute:
        if a.name in ("perm", "axes", "axis", "keepdims", "to"):
            v = list(a.ints) if a.ints else a.i
            out.append(f"{a.name}={v}")
    return " ".join(out)

init = {i.name for i in g.initializer}
cur = "/bev/view_trans/Concat_26_output_0"
print("--- producers of Concat_26 inputs ---")
cn = producer[cur]
for i in cn.input:
    pn = producer.get(i)
    print(f"  {i[-55:]}  <- {pn.op_type if pn else 'INPUT'}")

print("\n--- forward chain from Concat_26 ---")
for _ in range(14):
    ns = [n for n in consumers[cur] ]
    if not ns:
        print("  (no consumer)"); break
    n = ns[0]
    ins = [("INIT:" + i[-28:]) if i in init else i[-28:] for i in n.input]
    print(f"  {n.op_type:16s} {attrs(n):26s} in={ins}")
    print(f"      -> {n.output[0][-60:]}   (consumers={len(consumers[n.output[0]])})")
    cur = n.output[0]
    if n.op_type == "ReduceSum":
        print("  *** reached ReduceSum ***")
        for nn in consumers[cur][:3]:
            print(f"      next: {nn.op_type} -> {nn.output[0][-50:]}")
        break
