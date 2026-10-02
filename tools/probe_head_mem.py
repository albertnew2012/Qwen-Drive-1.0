"""Find the tensors that drive the perception head's peak GPU allocation."""
import onnx, sys
from onnx import shape_inference

p = sys.argv[1] if len(sys.argv) > 1 else "outputs/onnx/perception/perception.onnx"
m = onnx.load(p, load_external_data=False)
try:
    m = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
except Exception as e:
    print("shape inference failed:", e)

vi = {v.name: v for v in list(m.graph.value_info) + list(m.graph.output)}

def nbytes(v):
    t = v.type.tensor_type
    n = 1
    dims = []
    for d in t.shape.dim:
        x = d.dim_value if d.HasField("dim_value") else 0
        dims.append(x if x else "?")
        n *= x if x else 1
    esz = {1: 4, 10: 2, 7: 8, 6: 4, 9: 1}.get(t.elem_type, 4)
    return n * esz, dims

big = []
for n in m.graph.node:
    for o in n.output:
        if o in vi:
            b, dims = nbytes(vi[o])
            if b > 200e6:
                big.append((b, n.op_type, o, dims))
big.sort(reverse=True)
print(f"{len(big)} tensors > 200 MB")
for b, op, name, dims in big[:25]:
    print(f"  {b/1e6:8.1f} MB  {op:14s} {dims}  {name[-70:]}")

tot = sum(b for b, *_ in big)
print(f"  sum of >200MB tensors: {tot/1e9:.2f} GB")
