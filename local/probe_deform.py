"""Shapes of the biggest tensors inside the BEV encoder's deformable attention."""
import sys
import onnx
from onnx import shape_inference

p = sys.argv[1] if len(sys.argv) > 1 else "outputs/onnx_vt/perception/perception.onnx"
key = sys.argv[2] if len(sys.argv) > 2 else "deformable_attention"

m = onnx.load(p, load_external_data=False)
m = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
vi = {v.name: v for v in list(m.graph.value_info) + list(m.graph.output)}
esz = {1: 4, 10: 2, 7: 8, 6: 4, 9: 1}

rows = []
for n in m.graph.node:
    if key not in n.name:
        continue
    for o in n.output:
        v = vi.get(o)
        if v is None:
            continue
        t = v.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else "?" for d in t.shape.dim]
        cnt = 1
        for d in dims:
            cnt *= d if isinstance(d, int) and d else 1
        b = cnt * esz.get(t.elem_type, 4)
        if b > 100e6:
            rows.append((b, n.op_type, n.name, dims))

rows.sort(reverse=True)
print(f"{len(rows)} tensors > 100 MB under '{key}'")
for b, op, name, dims in rows[:18]:
    print(f"  {b/1e6:8.1f} MB  {op:12s} {dims}  {name.split('/')[-1]}")
