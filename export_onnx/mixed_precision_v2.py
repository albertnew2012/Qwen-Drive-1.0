"""Put the projection weights in fp16 and leave the recurrence in float32.

``mixed_precision.py`` has the right idea -- convert only the heavy GEMMs, because
a whole-graph fp16 conversion makes the Gated-DeltaNet recurrence lose its range
(measured here: 0.41 relative on a layer output, against 4e-07 for this) -- but it
does not actually shrink anything. torch exports a projection as
``initializer -> Transpose -> MatMul``, and because the Transpose is not itself a
heavy GEMM it lands in the block list, stays fp32, and the converter simply
inserts a Cast in front of the MatMul. The weight is still fp32 on disk and on the
card, and the per-frame Cast of 451 MiB gives back whatever the fp16 matmul won:
measured 78.97 ms against 78.10 ms, i.e. nothing.

Including the weight-producing chain in the conversion is what makes the
initializer itself fp16. Then the Cast that remains is on the activation, which is
28 MiB rather than 451 MiB.

This matters for more than speed. The fp32 decoder is 14.7 GiB of weights, and on
a 24 GiB card the perception head's 625 MiB and 2 GiB buffers then cannot be
served -- capping the shared arena low enough to leave room starves the vision
tower instead. At 7.4 GiB the whole stack fits with room to spare.

    python export_onnx/mixed_precision_v2.py outputs/onnx/vlm_layers_v2_perception
"""
from __future__ import annotations

import argparse, shutil, sys, time
from pathlib import Path

import onnx

_ROOT = Path(__file__).resolve().parent.parent

GEMM_OPS = {"MatMul", "Gemm", "FusedMatMul"}
# Nodes that only move a weight around, and so may be converted along with it.
PASSTHROUGH = {"Transpose", "Reshape", "Cast", "Squeeze", "Unsqueeze"}
MIN_WEIGHT_ELEMS = 1 << 20


def _convertible(model: onnx.ModelProto) -> set[str]:
    """Heavy GEMM nodes, plus the chain that carries each weight into them."""
    sizes = {t.name: int(__import__("math").prod(t.dims) or 0)
             for t in model.graph.initializer}
    produced = {o: n for n in model.graph.node for o in n.output}

    def chain(name: str, depth: int = 4) -> tuple[int, list]:
        """Element count of the weight behind ``name``, and the nodes on the way."""
        if name in sizes:
            return sizes[name], []
        node = produced.get(name)
        if depth and node is not None and node.op_type in PASSTHROUGH:
            n, rest = chain(node.input[0], depth - 1)
            return n, [node] + rest
        return 0, []

    keep = set()
    for node in model.graph.node:
        if node.op_type not in GEMM_OPS:
            continue
        found = [chain(i) for i in node.input]
        if any(n >= MIN_WEIGHT_ELEMS for n, _ in found):
            keep.add(node.name)
            for n, nodes in found:
                if n >= MIN_WEIGHT_ELEMS:
                    keep.update(x.name for x in nodes)
    return keep


def convert(path: Path) -> tuple[int, int, float]:
    from onnxruntime.transformers.float16 import convert_float_to_float16

    shell = onnx.load(str(path), load_external_data=False)
    mine = {kv.value for t in shell.graph.initializer
            for kv in t.external_data if kv.key == "location"}

    model = onnx.load(str(path))
    keep = _convertible(model)
    block = [n.name for n in model.graph.node if n.name not in keep]
    out = convert_float_to_float16(model, keep_io_types=True, node_block_list=block,
                                   disable_shape_infer=True)
    n_fp16 = sum(1 for t in out.graph.initializer if t.data_type == onnx.TensorProto.FLOAT16)

    location = f"{path.stem}.data"
    tmp = path.parent / location
    if tmp.exists():
        tmp.unlink()
    onnx.save(out, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=location, size_threshold=1024)
    for name in mine - {location}:
        f = path.parent / name
        if f.exists():
            f.unlink()
    size = sum(f.stat().st_size for f in (path, path.parent / location) if f.exists())
    return len(keep), n_fp16, size / 2**20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory of exported layer graphs")
    ap.add_argument("dst", help="directory to write the converted copy into")
    ap.add_argument("--only", default="", help="substring filter on the file name")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for extra in ("manifest.json", "embed_tokens.npy"):
        s, d = src / extra, dst / extra
        if s.exists() and not d.exists():
            if s.is_symlink():
                d.symlink_to(s.resolve())
            elif extra.endswith(".npy"):
                d.symlink_to(s.resolve())
            else:
                shutil.copy2(s, d)

    files = sorted(p for p in src.glob("*.onnx") if args.only in p.name)
    total_before = total_after = 0.0
    t0 = time.time()
    for p in files:
        before = p.stat().st_size / 2**20
        for stale in dst.glob(f"{p.stem}.*"):
            stale.unlink()
        target = dst / p.name
        shutil.copy2(p, target)
        for ext in (p.parent / f"{p.stem}.data",):
            if ext.exists():
                shutil.copy2(ext, dst / ext.name)
                before += ext.stat().st_size / 2**20
        n_keep, n_fp16, after = convert(target)
        total_before += before; total_after += after
        print(f"  {p.name:22s} {n_keep:4d} nodes, {n_fp16:3d} fp16 weights   "
              f"{before:7.1f} -> {after:7.1f} MiB", flush=True)
    print(f"\n  {len(files)} graphs  {total_before/1024:.2f} -> {total_after/1024:.2f} GiB"
          f"  in {time.time()-t0:.0f}s -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
