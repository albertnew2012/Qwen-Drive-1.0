"""Find the smallest fp16 block list that keeps the Gated-DeltaNet layer accurate.

Whole-graph fp16 halves the decoder so it can stay resident, but it destroys the
linear-attention layers (0.43-0.74 relative error) because the chunked delta rule
carries ``exp(cumsum(log a))``: the decay underflows to zero in fp16 and the state
is then rescaled by its reciprocal. Weights-only fp16 is accurate but ORT rebuilds
fp32 copies of the initializers to erase the Cast nodes, so it saves no memory.

This sweeps candidate op block lists on one layer and reports accuracy, speed and
the actual resident GPU cost, so the choice is made on measurements.

    .venv-ortgpu/bin/python local/tune_fp16_blocklist.py
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers.float16 import convert_float_to_float16

SRC = "outputs/onnx/vlm_layers_v2/layer_00.onnx"
TMP = Path("/tmp/fp16tune")

# the decay path: everything that can underflow, overflow, or amplify a rounding
# error through the recurrence. Progressively widened.
DECAY = ["Exp", "CumSum", "Softplus", "Reciprocal", "Div", "Pow", "ReduceSum",
         "Neg", "Sqrt", "ReduceMean", "Sigmoid"]

CANDIDATES = {
    "none (full fp16)": [],
    "decay": DECAY,
    "decay+Sub": DECAY + ["Sub"],
    "decay+Sub+Mul": DECAY + ["Sub", "Mul"],
    "decay+Sub+Mul+Add": DECAY + ["Sub", "Mul", "Add"],
}


def gpu_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.split()[0]
    return int(out)


def measure(path: str, x, pos) -> tuple[np.ndarray, float, int]:
    base = gpu_mib()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        path, so, providers=[("CUDAExecutionProvider",
                              {"device_id": 0,
                               "arena_extend_strategy": "kSameAsRequested"})])
    feed = {"hidden_in": x}
    if "position_ids" in [i.name for i in sess.get_inputs()]:
        feed["position_ids"] = pos
    out = sess.run(None, feed)[0]
    mem = gpu_mib() - base
    sess.run(None, feed)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t) * 1e3)
    del sess
    return out, float(np.median(ts)), mem


def main() -> int:
    TMP.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 2744, 2560)) * 0.02).astype(np.float32)
    pos = np.zeros((3, 1, 2744), np.int64)

    ref, t_ref, m_ref = measure(SRC, x, pos)
    scale = max(float(np.abs(ref).max()), 1e-9)
    print(f"{'block list':22s} {'ms':>7s} {'MiB':>6s} {'rel err':>10s}")
    print(f"{'fp32 baseline':22s} {t_ref:7.1f} {m_ref:6d} {0.0:10.3e}")

    for name, block in CANDIDATES.items():
        dst = TMP / (name.replace(" ", "_").replace("(", "").replace(")", "") + ".onnx")
        if not dst.exists():
            model = onnx.load(SRC)
            kw = {"op_block_list": block} if block else {}
            out = convert_float_to_float16(
                model, keep_io_types=True, disable_shape_infer=True,
                force_fp16_initializers=True, **kw)
            onnx.save(out, str(dst))
            del model, out
        got, ms, mem = measure(str(dst), x, pos)
        rel = float(np.abs(ref - got).max()) / scale
        mb = dst.stat().st_size / 1e6
        print(f"{name:22s} {ms:7.1f} {mem:6d} {rel:10.3e}   disk {mb:6.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
