"""Compare the front head variants: as exported, and with fp16 projection weights."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
import numpy as np
import onnxruntime as ort

f = {"img_vit_feats": (np.random.randn(1, 32, 56, 1024) * 0.1).astype(np.float32),
     "img_llm_feats": (np.random.randn(1, 16, 28, 2560) * 0.1).astype(np.float32)}


def go(path, tag):
    if not Path(path).exists():
        print(f"  {tag:22s} missing"); return None, None
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        s = ort.InferenceSession(str(path), so,
                                 providers=[("CUDAExecutionProvider", {"device_id": 0})])
    except Exception as exc:
        print(f"  {tag:22s} load failed: {str(exc)[:80]}"); return None, None
    out = s.run(None, f); ts = []
    for _ in range(4):
        t = time.time(); out = s.run(None, f); ts.append((time.time() - t) * 1e3)
    ms = float(np.median(ts))
    print(f"  {tag:22s} {ms:8.1f} ms", flush=True)
    del s
    return ms, out


print(f"  ORT {ort.__version__}")
a, oa = go("outputs/onnx/perception_front/perception.onnx", "front head fp32")
b, ob = go("outputs/onnx/perception_front_f16w/perception.onnx", "front head fp16 weights")
rep = {"fp32_ms": a, "fp16w_ms": b}
if a and b:
    worst = max(float(np.abs(x - y).max() / max(float(np.abs(x).max()), 1e-9))
                for x, y in zip(oa, ob))
    rep["rel"] = worst
    print(f"  speedup {a/b:.2f}x   outputs differ by {worst:.2e} relative")
Path("outputs/prune").mkdir(parents=True, exist_ok=True)
Path("outputs/prune/head_variants.json").write_text(json.dumps(rep, indent=1))
