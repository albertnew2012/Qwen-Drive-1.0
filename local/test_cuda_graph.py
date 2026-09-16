"""Does CUDA Graph capture help the decoder layers and the perception head?

The per-node profile and the isolated micro-benchmarks disagree by an order of
magnitude: a `Mul [1,2744,32,128]` runs in 0.17 ms standalone but is billed
22.5 ms inside the layer graph, and ablating it only saved 3 ms.  That gap is
the signature of launch/dispatch overhead rather than arithmetic, so the fix is
to stop dispatching per node.  ORT can capture the whole session into a single
CUDA graph and replay it, which collapses thousands of launches into one.

Capture has hard preconditions: every node must land on the CUDA EP, all shapes
must be static, and every input and output must stay at a fixed device address
across runs.  Violating any of them either fails loudly at capture or, worse,
silently replays stale data, so this script checks the replayed output against
the ordinary run instead of trusting the timing alone.

    python local/test_cuda_graph.py outputs/onnx/vlm_layers_v2/layer_00.onnx
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

_ROOT = Path(__file__).resolve().parents[1]


def placements(sess) -> dict:
    """Count how many nodes ORT put on each provider, via a one-shot profile."""
    import collections
    import json as _json

    prof = sess.end_profiling()
    if not prof:
        return {}
    ev = [e for e in _json.load(open(prof)) if e.get("cat") == "Node"]
    return collections.Counter(e["args"].get("provider", "?") for e in ev)


def build(path: Path, cuda_graph: bool):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    opts = {"device_id": 0}
    if cuda_graph:
        # kNextPowerOfTwo churns the arena between runs, which moves activation
        # addresses and invalidates a captured graph.
        opts["enable_cuda_graph"] = "1"
        opts["arena_extend_strategy"] = "kSameAsRequested"
    return ort.InferenceSession(str(path), so,
                                providers=[("CUDAExecutionProvider", opts)])


def bind(sess, feeds: dict):
    """Bind every input and output to a fixed device address.

    CUDA graph replay reuses the addresses baked in at capture time, so the
    OrtValues must outlive the binding; they are returned to keep them alive.
    """
    io = sess.io_binding()
    keep = []
    for name, arr in feeds.items():
        ov = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), "cuda", 0)
        io.bind_ortvalue_input(name, ov)
        keep.append(ov)
    for o in sess.get_outputs():
        shape = [d if isinstance(d, int) else 1 for d in o.shape]
        ov = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float32, "cuda", 0)
        io.bind_ortvalue_output(o.name, ov)
        keep.append(ov)
    return io, keep


def timed(sess, io, reps: int) -> list:
    sess.run_with_iobinding(io)          # capture on the first call
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        sess.run_with_iobinding(io)
        ts.append((time.perf_counter() - t) * 1e3)
    return ts


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else _ROOT / "outputs/onnx/vlm_layers_v2/layer_00.onnx")
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    np.random.seed(0)
    plain = build(path, cuda_graph=False)
    feeds = {}
    for i in plain.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        if "int" in i.type:
            feeds[i.name] = np.zeros(shape, np.int64)
        else:
            feeds[i.name] = (np.random.randn(*shape).astype(np.float32) * 0.02)
    print(f"{path.name}   inputs " + ", ".join(f"{k}{list(v.shape)}" for k, v in feeds.items()))

    io, _keep = bind(plain, feeds)
    ts = timed(plain, io, reps)
    ref = io.copy_outputs_to_cpu()[0]
    print(f"  {'plain io_binding':26s} median {np.median(ts):7.1f} ms   min {min(ts):7.1f} ms")
    del plain

    try:
        cg = build(path, cuda_graph=True)
    except Exception as e:                       # capture is refused up front
        print(f"  cuda graph session failed: {type(e).__name__}: {str(e)[:200]}")
        return 1
    try:
        io2, _keep2 = bind(cg, feeds)
        ts2 = timed(cg, io2, reps)
    except Exception as e:
        print(f"  cuda graph replay failed: {type(e).__name__}: {str(e)[:300]}")
        return 1

    got = io2.copy_outputs_to_cpu()[0]
    rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
    print(f"  {'cuda graph replay':26s} median {np.median(ts2):7.1f} ms   min {min(ts2):7.1f} ms"
          f"   speedup {np.median(ts)/np.median(ts2):4.2f}x")
    print(f"  output vs plain: {rel:.2e}   finite={bool(np.isfinite(got).all())}")
    if rel > 1e-5:
        print("  MISMATCH - replay is not reproducing the plain result, do not ship this")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
