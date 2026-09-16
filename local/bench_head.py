"""Standalone perception head: peak GPU memory, latency and accuracy vs a reference."""
import json, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

head = sys.argv[1]
ref_path = sys.argv[2] if len(sys.argv) > 2 else ""

so = ort.SessionOptions()
so.log_severity_level = 3
prov = [("CUDAExecutionProvider", {"device_id": 0,
                                   "arena_extend_strategy": "kSameAsRequested"})]
t = time.perf_counter()
s = ort.InferenceSession(head, so, providers=prov)
build = time.perf_counter() - t
assert "CUDAExecutionProvider" in s.get_providers(), s.get_providers()

cache = Path("outputs/onnx_bench/head_inputs.npz")
if cache.exists():
    d = np.load(cache)
    feeds = {k: d[k] for k in ("img_vit_feats", "img_llm_feats")}
else:
    # No dump of real head inputs, so synthesise from the declared shapes with a
    # fixed seed. Old and new heads then see identical data, which is what the
    # equivalence check needs.
    rng = np.random.default_rng(0)
    feeds = {}
    for i in s.get_inputs():
        shape = [x if isinstance(x, int) else 6 for x in i.shape]
        feeds[i.name] = rng.standard_normal(shape, dtype=np.float32)
        print(f"  synth {i.name} {shape}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **feeds)

out = s.run(None, feeds)
best = 1e9
for _ in range(3):
    t = time.perf_counter()
    out = s.run(None, feeds)
    best = min(best, (time.perf_counter() - t) * 1e3)

import subprocess
o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
used = int(o.stdout.strip().split("\n")[0])

names = ["cls", "box", "occ", "seg"]
print(f"  {head}")
print(f"  build {build:6.2f}s   infer {best:8.1f} ms   GPU {used} MiB")
for n, a in zip(names, out):
    print(f"    {n:4s} {str(a.shape):26s} absmax {np.abs(a).max():.4f} "
          f"nan={int(np.isnan(a).sum())}")

if ref_path and Path(ref_path).exists():
    ref = np.load(ref_path)
    print("  vs reference:")
    for n, a in zip(names, out):
        r = ref[n]
        den = max(np.abs(r).max(), 1e-9)
        print(f"    {n:4s} rel {np.abs(a - r).max() / den:.3e}")
else:
    np.savez(ref_path or "/tmp/head_ref.npz", **dict(zip(names, out)))
    print(f"  saved reference -> {ref_path or '/tmp/head_ref.npz'}")
