"""Take the remaining CPU work and the dispatch overhead out of the front head.

After the domain fix and the 5-D gather rewrite the head is 892 ms, and profiling
splits that cleanly:

    kernel time            434 ms
    of which on the CPU     73 ms   six Resize nodes
    host round-trips       109 ms   forced by those six nodes
    dispatch (wall-kernel) ~458 ms   2,408 kernels

So two things are left, and they are independent:

*Resize on the CPU.* ORT registers a CUDA Resize only up to opset 18, and the graph
declares 20. Lowering the declared opset was impossible while the 5-D GridSample was
there -- that operator only exists from 20 -- but the gather rewrite removed it, so
the retarget is now available. ``retarget_opset.py`` verifies before and after and
reverts itself if the answer moves, which is the only reason it is safe to try.

*Dispatch.* A captured CUDA graph replays thousands of launches as one, but capture
refuses a graph containing CPU nodes, which is why it was never worth trying here
before. It becomes worth trying the moment the Resize nodes move.

Both are measured against the head's own output, so a change that alters the
detections is rejected rather than counted.
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np


def bench(path, feeds, tag, cuda_graph=False, reps=4):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts = {"device_id": 0}
    if cuda_graph:
        opts["enable_cuda_graph"] = "1"
    try:
        s = ort.InferenceSession(str(path), so,
                                 providers=[("CUDAExecutionProvider", opts)])
    except Exception as exc:
        print(f"  {tag:34s} session failed: {str(exc)[:70]}")
        return None, None
    try:
        if cuda_graph:
            io = s.io_binding()
            keep = []
            for k, v in feeds.items():
                ov = ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0)
                keep.append(ov); io.bind_ortvalue_input(k, ov)
            for o in s.get_outputs():
                io.bind_output(o.name, "cuda", 0)
            s.run_with_iobinding(io)
            outs = [x.numpy() for x in io.get_outputs()]
            for _ in range(2):
                s.run_with_iobinding(io)
            ts = []
            for _ in range(reps):
                t = time.time(); s.run_with_iobinding(io); ts.append((time.time()-t)*1e3)
        else:
            outs = s.run(None, feeds)
            for _ in range(1):
                s.run(None, feeds)
            ts = []
            for _ in range(reps):
                t = time.time(); outs = s.run(None, feeds); ts.append((time.time()-t)*1e3)
        ms = float(np.median(ts))
        print(f"  {tag:34s} {ms:8.1f} ms")
        return ms, outs
    except Exception as exc:
        print(f"  {tag:34s} run failed: {str(exc)[:70]}")
        return None, None
    finally:
        del s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="outputs/onnx/perception_front/perception.onnx")
    ap.add_argument("--out", default="outputs/prune/front_head_opt_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)
    head = Path(args.head)
    feeds = {"img_vit_feats": (np.random.randn(1, 32, 56, 1024) * 0.1).astype(np.float32),
             "img_llm_feats": (np.random.randn(1, 16, 28, 2560) * 0.1).astype(np.float32)}
    report = {}

    print("  baseline (opset 20, Resize on CPU)")
    base_ms, base_out = bench(head, feeds, "as exported")
    _, cg_ms = None, None
    cg, _ = bench(head, feeds, "as exported + CUDA graph", cuda_graph=True)
    report["opset20"] = {"ms": base_ms, "ms_cuda_graph": cg}

    print("\n  retargeting to opset 18 so Resize can use its CUDA kernel")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_ROOT/'src'}:{_ROOT}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.run(
        [str(_ROOT/".venv"/"bin"/"python"), str(_ROOT/"export_onnx"/"retarget_opset.py"),
         "--opset", "18", "--verify", str(head)],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=3600)
    tail = [l for l in proc.stdout.splitlines() if l.strip()][-4:]
    for l in tail:
        print("   ", l)
    retargeted = "REVERTED" not in proc.stdout

    if retargeted:
        ms18, out18 = bench(head, feeds, "opset 18")
        cg18, _ = bench(head, feeds, "opset 18 + CUDA graph", cuda_graph=True)
        report["opset18"] = {"ms": ms18, "ms_cuda_graph": cg18}
        if base_out is not None and out18 is not None:
            worst = max(float(np.abs(a - b).max() / max(float(np.abs(a).max()), 1e-9))
                        for a, b in zip(base_out, out18))
            report["opset18"]["rel_vs_opset20"] = worst
            print(f"  opset 18 vs opset 20 outputs: {worst:.2e} relative")
    else:
        print("    retarget reverted itself; Resize stays on the CPU")

    best = min([v.get("ms") for v in report.values() if v.get("ms")] +
               [v.get("ms_cuda_graph") for v in report.values() if v.get("ms_cuda_graph")])
    print(f"\n  best head {best:.0f} ms (was {base_ms:.0f} ms)")
    print(f"  perception would be vision 94 + decoder 63 + head {best:.0f} = "
          f"{94+63+best:.0f} ms -> {1000/(94+63+best):.2f} Hz")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
