"""Export the Planning Expert's denoiser to ONNX, and verify it.

The expert is a plain diffusion transformer - no custom kernels - so it exports
far more cleanly than the perception head. What is exported is ONE call to
``predict_endpoint``: the network that, given a noisy trajectory and a flow time,
predicts the clean one.

Sampling is a 10-step Euler loop AROUND this network. Exporting the single step
rather than the loop is deliberate: the host keeps the loop, and can change the
step count without re-exporting. That also matches how the repo is written -
``sample()`` calls ``predict_endpoint`` in a Python loop.

    python export_onnx/export_planner.py --out outputs/onnx/planner_step.onnx
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive import QwenDriveForPlanning
from qwen_drive import planning_expert as _pe


_HOLDER = {}


def _attend_onnx(self, query, key, value):
    """``_attend`` without ``enable_gqa``.

    The torchscript exporter refuses SDPA when ``enable_gqa=True``
    (symbolic_opset14: "conversion of scaled_dot_product_attention not
    implemented if enable_gqa is True"). Repeating the key/value heads to match
    the query heads is exactly what enable_gqa does internally, so this is the
    same function written in ops the exporter can lower.
    """
    import torch.nn.functional as F
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    if self.num_heads != self.num_kv_heads:
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def make_one_hot_onnx(num_classes_hint: int = 16):
    """``_one_hot`` built from a constant identity matrix and one ``index_select``.

    The shipped version is three ops the ONNX exporter cannot lower on torch
    2.8: the range guard ``(index >= 0) & (index < n)`` becomes ``prims.ge``,
    ``clamp`` on an integer tensor becomes ``prims.ne(x, x)`` (a NaN probe), and
    ``F.one_hot`` itself becomes ``prims.iota``. None has a registered ONNX
    decomposition for integer input.

    Indexing rows of a precomputed identity matrix is the same function with
    none of those: a single Gather. The out-of-range guard is dropped, which is
    safe because navigation commands from the dataset are always in range.
    """
    def _one_hot(index, num_classes, dtype):
        held = _HOLDER.get("nav")
        if held is not None:
            return held.to(dtype)
        eye = torch.eye(num_classes, dtype=dtype, device=index.device)
        return eye.index_select(0, index.reshape(-1).long()).reshape(
            *index.shape, num_classes)

    return _one_hot


class PlannerStepONNX(nn.Module):
    """One denoising step. The scene cache arrives as a flat tuple of tensors."""

    def __init__(self, expert, n_groups: int):
        super().__init__()
        self.expert = expert
        self.n_groups = n_groups

    def forward(self, waypoints, flow_time, history, history_velocity,
                history_acceleration, nav_onehot, ego_status, position_anchor,
                *scene_kv):
        # nav_onehot arrives ALREADY one-hot and float. An int64 graph input is
        # what kept breaking the exporter: every op touching it (ge, ne, iota,
        # view_of) reported "no decompositions registered". Both call sites of
        # _one_hot use the same class count, so one float input serves both.
        _HOLDER["nav"] = nav_onehot
        cache = [(scene_kv[2 * i], scene_kv[2 * i + 1]) for i in range(self.n_groups)]
        hq = self.expert.encode_history(history, nav_onehot,
                                        history_velocity, history_acceleration)
        return self.expert.predict_endpoint(
            waypoints, flow_time, hq, cache, position_anchor, nav_onehot, ego_status)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--cache", default="data/train_cache_plan")
    ap.add_argument("--record", default="",
                    help="which cached scene to freeze shapes against. The graph "
                         "is shape-frozen to that scene's KV sequence length, so "
                         "it must match the layer graphs it will be fed from.")
    ap.add_argument("--out", default="outputs/onnx/planner_step.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dynamo", action="store_true")
    ap.add_argument("--attn", default="eager",
                    help="eager avoids SDPA, whose grouped-query path the "
                         "torchscript exporter rejects (enable_gqa=True)")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.float32,
        attn_implementation=args.attn)
    expert = holder.planning_expert.to(args.device).eval()
    del holder.vlm

    recs = sorted(Path(args.cache).glob("*.pt"))
    if not recs:
        raise FileNotFoundError("run training/cache_planner_features.py first")
    if args.record:
        hits = [r for r in recs if r.name.startswith(args.record)]
        if not hits:
            raise KeyError(f"no cached planning scene starting with {args.record!r}; "
                           f"have {[r.name for r in recs]}")
        chosen = hits[0]
    else:
        chosen = recs[0]
    rec = torch.load(chosen, weights_only=False)
    print(f"  shapes frozen against {chosen.name}")
    print(f"tracing from {chosen.name}")
    dev = args.device
    cache = [(k.to(dev, torch.float32), v.to(dev, torch.float32))
             for k, v in rec["scene_cache"]]
    x1 = rec["target_normalized"].to(dev).float()
    n_nav = holder.planning_expert.config.nav_command_classes
    nav_idx = rec["nav_command"].to(dev).reshape(-1).long()
    nav_onehot = torch.eye(n_nav, dtype=torch.float32, device=dev).index_select(
        0, nav_idx).reshape(*rec["nav_command"].shape, n_nav)
    flat_kv = tuple(t for kv in cache for t in kv)
    inputs = (
        torch.randn_like(x1),                               # noisy waypoints
        torch.zeros(x1.shape[0], device=dev),               # flow time
        rec["history"].to(dev).float(),
        rec["history_velocity"].to(dev).float(),
        rec["history_acceleration"].to(dev).float(),
        nav_onehot,
        rec["ego_status"].to(dev).float(),
        rec["anchor"].to(dev),
    ) + flat_kv

    _pe._one_hot = make_one_hot_onnx()   # see the note on the factory above
    _pe.PlanningExpertLayer._attend = _attend_onnx   # see the note above
    wrapper = PlannerStepONNX(expert, len(cache)).eval()
    with torch.no_grad():
        ref = wrapper(*inputs)
    print(f"{sum(p.numel() for p in expert.parameters())/1e9:.4f} B expert, "
          f"{len(cache)} KV groups   output {tuple(ref.shape)}")

    names = ["waypoints", "flow_time", "history", "history_velocity",
             "history_acceleration", "nav_onehot", "ego_status", "position_anchor"]
    names += [f"scene_{'kv'[i % 2]}_{i//2}" for i in range(len(flat_kv))]
    print(f"\nexporting ({'dynamo' if args.dynamo else 'torchscript'})...")
    t0 = time.time()
    with torch.no_grad():
        if args.dynamo:
            prog = torch.onnx.export(wrapper, inputs, dynamo=True, input_names=names,
                                     output_names=["endpoint"], opset_version=args.opset)
            prog.save(str(out))
        else:
            torch.onnx.export(wrapper, inputs, str(out), input_names=names,
                              output_names=["endpoint"], opset_version=args.opset,
                              do_constant_folding=True, dynamo=False)
    print(f"  {time.time()-t0:.0f}s -> {out} ({out.stat().st_size/2**20:.1f} MiB)")

    import onnx
    # check by PATH: a model with external data cannot be re-serialised
    # into a single proto, which is what check_model(model_obj) attempts.
    onnx.checker.check_model(str(out), full_check=False)
    m = onnx.load(str(out))
    print(f"  checker OK, {len(m.graph.node)} nodes")

    if args.skip_verify:
        return 0
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feed = {n: t.cpu().numpy() for n, t in zip(names, inputs)}
    got = sess.run(None, feed)[0]
    d = float(np.abs(got - ref.cpu().numpy()).max())
    rel = d / max(float(np.abs(ref.cpu().numpy()).max()), 1e-9)
    print(f"\nmax abs diff {d:.3e}   relative {rel:.3e}")
    ok = rel < 1e-3
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
