"""Export the VLM text model as PER-LAYER ONNX graphs.

WHY. The monolithic prefill graph exports (341,977 nodes, 13.4 GiB) but
onnxruntime cannot build a session from it - over an hour without finishing. The
node count comes from the Gated-DeltaNet forward substitution unrolling once per
layer, and it is ~244k nodes of fixed cost regardless of sequence length.

Splitting the model at layer boundaries makes each graph ~1/32 the size. One
linear-attention layer loads in about 5 seconds, so the whole stack is ~2
minutes, and the host simply runs the 32 graphs in order. This is a normal
deployment pattern for very deep models, not a workaround.

WHAT IS EMITTED

  layer_00.onnx .. layer_31.onnx   one graph per decoder layer
      linear_attention -> (hidden,)
      full_attention   -> (hidden, keys, values)   the planner reads these
  final_norm.onnx                  the language model's last norm
  embed_tokens.npy                 the embedding table, applied on the host

    python export_onnx/export_vlm_layers.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer

from qwen_drive import QwenDriveForPlanning
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor


class LinearLayerONNX(nn.Module):
    """A linear-attention block. ``position_embeddings`` is unused by this path."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states):
        return self.layer(hidden_states, position_embeddings=None)


class FullLayerONNX(nn.Module):
    """A full-attention block, also returning the post-rotary keys and values.

    The Planning Expert cross-attends to exactly these, so they have to leave the
    graph rather than staying inside a cache object.
    """

    def __init__(self, layer, rotary, n_layers):
        super().__init__()
        self.layer = layer
        self.rotary = rotary
        self.n_layers = n_layers

    def forward(self, hidden_states, position_ids):
        from transformers.cache_utils import DynamicCache
        cos, sin = self.rotary(hidden_states, position_ids)
        # THE CAUSAL MASK IS NOT OPTIONAL. The full model builds it with
        # create_causal_mask(); a per-layer wrapper that passes attention_mask=None
        # gets BIDIRECTIONAL attention and a completely different answer - measured
        # at 6.5e-01 relative against the real model. It still "verifies" per layer,
        # because both sides of that comparison share the same mistake. Only
        # chaining the layers and comparing against the whole model exposes it.
        seq = hidden_states.shape[1]
        dtype = hidden_states.dtype
        causal = torch.full((seq, seq), torch.finfo(dtype).min,
                            dtype=dtype).triu(1)[None, None]
        cache = DynamicCache(config=self.layer.self_attn.config)
        out = self.layer(hidden_states, position_embeddings=(cos, sin),
                         position_ids=position_ids, attention_mask=causal,
                         past_key_values=cache)
        entry = cache.layers[self.layer.self_attn.layer_idx]
        return out, entry.keys, entry.values


class FinalNormONNX(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, hidden_states):
        return self.norm(hidden_states)


def patch_chunk_rule_blocked(model, chunk_size: int = 64) -> int:
    """Replace the 63-step triangular inverse with a 6-step blocked one.

    ``torch_chunk_gated_delta_rule`` inverts a unit lower-triangular 64x64
    matrix by forward substitution.  The loop is sequential and indexed, so the
    tracer unrolls it into roughly 13,900 nodes per layer - about 98% of the
    exported decoder.

    Let ``A`` be the strictly lower triangular part, so the target is
    ``(I - A)^-1``.  Partitioned into 2x2 blocks,

        [[M11, 0], [M21, M22]]^-1 = [[X11, 0], [X22 A21 X11, X22]]

    which is exactly ``X + X L X`` when ``X`` holds the already-inverted
    diagonal blocks and ``L`` keeps only the odd-block/even-block coupling.
    Starting from 1x1 blocks and doubling, six steps cover all 64 rows, and
    every block can be updated at once, so it is 12 batched matmuls.

    An earlier attempt (``export_vlm.py::patch_chunk_rule_for_export``) used the
    doubling identity ``(I-A)^-1 = (I+A)(I+A^2)...`` and produced NaN, because it
    forms high powers of A.  This does not: every intermediate is a block of the
    answer, so it stays the magnitude of the answer.  Measured on real captured
    matrices it is slightly MORE accurate than the shipped loop (1.1e-07 vs
    2.2e-07 against a float64 reference) and agrees with it to 4.3e-16 in float64.

    ``chunk_size`` becomes worth raising once this is in place.  Chunking is a
    tiling of the same recurrence, so it is algebraically exact at any size, and
    the sequential chunk loop is what is left of the graph: 64 -> 256 takes it
    from 43 steps to 11.  The old cost model forbade that, because the inverse
    grew as ``chunk - 1``; the blocked one grows as ``log2(chunk)``.  Measured
    over the whole 32-layer decoder, 256 moves the final hidden state by
    7.5e-06 relative.
    """
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    src = inspect.getsource(M.torch_chunk_gated_delta_rule)
    old = """    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)"""
    new = """    _eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    _idx = torch.arange(chunk_size, device=attn.device)
    _x = _eye.expand_as(attn).contiguous()
    _b = 1
    while _b < chunk_size:
        _blk = _idx // _b
        _grp = _idx // (2 * _b)
        _m = ((_grp[:, None] == _grp[None, :]) & (_blk[:, None] % 2 == 1)
              & (_blk[None, :] % 2 == 0)).to(attn.dtype)
        _x = _x + _x @ (attn * _m) @ _x
        _b *= 2
    attn = _x"""
    if old not in src:
        raise RuntimeError("the chunk-rule loop does not match the installed "
                           "transformers source; refusing to patch blindly")
    namespace = dict(M.__dict__)
    exec(compile(src.replace(old, new), "<blocked_chunk_rule>", "exec"), namespace)
    patched = namespace["torch_chunk_gated_delta_rule"]
    if chunk_size != 64:
        import functools
        patched = functools.partial(patched, chunk_size=chunk_size)

    n = 0
    for mod in model.modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            mod.chunk_gated_delta_rule = patched
            n += 1
    return n


def export_one(module, example, names, onames, out, opset, verify=True):
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        ref = module(*example)
    refs = (ref,) if torch.is_tensor(ref) else tuple(ref)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(module, example, str(out), input_names=names,
                          output_names=onames, opset_version=opset,
                          do_constant_folding=False, dynamo=False)
    te = time.time() - t0
    import onnx
    n_nodes = len(onnx.load(str(out), load_external_data=False).graph.node)
    if not verify:
        return {"nodes": n_nodes, "export_s": te, "rel": None}
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    t0 = time.time()
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    tl = time.time() - t0
    got = sess.run(None, {n: t.cpu().numpy() for n, t in zip(names, example)})
    worst = 0.0
    for a, b in zip(got, refs):
        b = b.cpu().numpy()
        worst = max(worst, float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-9))
    return {"nodes": n_nodes, "export_s": te, "load_s": tl, "rel": worst}


def _fork_export(indices, jobs, export_layer):
    """Export `indices` across `jobs` forked children, newest-first bucketed.

    fork() rather than a Pool: the per-layer closure is not picklable, and more
    importantly the children inherit the 4.5 B of weights copy-on-write, so N
    workers cost ~31 GB in total instead of ~31 GB each. Children write their
    .onnx files as a side effect and hand back only a JSON row.
    """
    import tempfile, traceback

    jobs = min(jobs, len(indices))
    # Round-robin, so the 8 cheap full_attention layers spread across workers
    # instead of landing on one.
    buckets = [indices[k::jobs] for k in range(jobs)]
    tmp = Path(tempfile.mkdtemp(prefix="vlm_layers_"))
    parent_threads = torch.get_num_threads()
    per_child = max(1, parent_threads // jobs)
    # Quiesce the intra-op pool before forking. fork() copies only the calling
    # thread, so a child that inherits a futex held by an OpenMP worker which
    # does not exist on its side blocks forever - measured: four children at
    # 0% CPU and 00:00:00 cpu time, in futex_wait_queue_me, indefinitely.
    torch.set_num_threads(1)

    pids = {}
    for k, bucket in enumerate(buckets):
        if not bucket:
            continue
        pid = os.fork()
        if pid == 0:                                   # child
            status = 0
            try:
                torch.set_num_threads(per_child)
                rows = []
                for i in bucket:
                    rows.append(export_layer(i))
                    print(f"    worker {k}: layer {i:2d} done", flush=True)
                (tmp / f"{k}.json").write_text(json.dumps(rows))
            except Exception:
                traceback.print_exc()
                status = 1
            os._exit(status)                           # skip the parent's atexit
        pids[pid] = k

    failed = []
    for _ in range(len(pids)):
        pid, status = os.wait()
        if status != 0:
            failed.append(pids[pid])
    torch.set_num_threads(parent_threads)
    if failed:
        raise RuntimeError(f"export workers {sorted(failed)} failed; see the traceback above")

    rows = []
    for k in range(jobs):
        f = tmp / f"{k}.json"
        if f.exists():
            rows.extend(json.loads(f.read_text()))
    if len(rows) != len(indices):
        raise RuntimeError(f"expected {len(indices)} layer results, collected {len(rows)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--task", choices=["perception", "planning"],
                    default="perception",
                    help="which input shape to freeze the graphs at; see "
                         "scene_inputs.py for why each task needs its own set")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-root", default="data/demo")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--layers", default="", help="comma list, e.g. 0,3 (default: all)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="fork N workers to export layers concurrently; they share "
                         "the weights copy-on-write, so RAM stays ~1 model")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-fast-triinv", action="store_true",
                    help="keep the stock 63-step triangular inverse")
    ap.add_argument("--chunk-size", type=int, default=64,
                    help="Gated-DeltaNet chunk; larger means fewer sequential steps")
    args = ap.parse_args()

    default_out = ("outputs/onnx/vlm_layers" if args.task == "perception"
                   else "outputs/onnx/vlm_layers_plan")
    out_dir = Path(args.out or default_out); out_dir.mkdir(parents=True, exist_ok=True)
    from export_onnx.scene_inputs import perception_inputs, planning_inputs
    if args.task == "perception":
        ctx = perception_inputs(args.vlm, args.model, args.frames, args.frame)
    else:
        ctx = planning_inputs(args.vlm, args.planner, args.scenes,
                              args.image_root, args.image_archive)
    holder, vlm, inputs = ctx["holder"], ctx["vlm"], ctx["inputs"]
    pos = ctx["position_ids"]
    print(f"task={args.task}  token {str(ctx['token'])[:24]}")

    lm = vlm.model.language_model
    cfg = vlm.config.text_config if hasattr(vlm.config, "text_config") else vlm.config
    with torch.no_grad():
        embeds = vlm.model.get_input_embeddings()(inputs["input_ids"])
    print(f"sequence {embeds.shape[1]} tokens, hidden {embeds.shape[2]}, "
          f"{len(cfg.layer_types)} layers")

    if not args.no_fast_triinv:
        # Compare against the stock implementation on this exact input before
        # trusting the patch: the export's own per-layer check cannot catch a
        # bad patch, because both sides of it would share the mistake.
        first_linear = list(cfg.layer_types).index("linear_attention")
        with torch.no_grad():
            before = lm.layers[first_linear](embeds, position_embeddings=None)
        before = before if torch.is_tensor(before) else before[0]
        n_patched = patch_chunk_rule_blocked(vlm, args.chunk_size)
        with torch.no_grad():
            after = lm.layers[first_linear](embeds, position_embeddings=None)
        after = after if torch.is_tensor(after) else after[0]
        rel = float((after - before).abs().max()) / max(float(before.abs().max()), 1e-9)
        print(f"blocked triangular inverse: patched {n_patched} layers, "
              f"chunk {args.chunk_size}, layer {first_linear} output moves "
              f"{rel:.2e} relative")
        if rel > 1e-4:
            raise RuntimeError(f"blocked inverse changed the answer by {rel:.2e}")

    # the embedding table is applied on the host: it is a gather, not compute
    emb_path = out_dir / "embed_tokens.npy"
    if not emb_path.exists():
        w = vlm.model.get_input_embeddings().weight.detach().cpu().numpy()
        np.save(emb_path, w)
        print(f"  embed_tokens.npy {w.shape}  {w.nbytes/2**30:.2f} GiB")

    wanted = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(len(cfg.layer_types))))
    manifest = {"layer_types": list(cfg.layer_types), "sequence": int(embeds.shape[1]),
                "hidden": int(embeds.shape[2]), "layers": {}}

    # Build every layer's wrapper and its real tracing input in ONE forward pass.
    # The activation arriving at layer i is what makes that layer's verification
    # mean anything, and materialising all 32 up front is what lets the exports
    # then run in any order, in any process.
    plans = []
    x = embeds
    for i, kind in enumerate(cfg.layer_types):
        layer = lm.layers[i]
        if kind == "linear_attention":
            mod = LinearLayerONNX(layer).eval()
            ex, names, onames = (x,), ["hidden_in"], ["hidden_out"]
        else:
            mod = FullLayerONNX(layer, lm.rotary_emb, len(cfg.layer_types)).eval()
            ex, names, onames = (x, pos), ["hidden_in", "position_ids"], \
                                ["hidden_out", "keys", "values"]
        plans.append((i, kind, mod, ex, names, onames))
        with torch.no_grad():
            o = mod(*ex)
            x = o if torch.is_tensor(o) else o[0]

    def export_layer(i):
        _, kind, mod, ex, names, onames = plans[i]
        r = export_one(mod, ex, names, onames, out_dir / f"layer_{i:02d}.onnx",
                       args.opset, verify=not args.no_verify)
        return {"layer": i, "type": kind, "inputs": names, "outputs": onames, **r}

    print(f"\n{'layer':>6} {'type':18s} {'nodes':>8} {'export':>8} {'load':>7} {'rel':>10}")
    t_layers = time.time()
    if args.jobs > 1 and len(wanted) > 1:
        rows = _fork_export(wanted, args.jobs, export_layer)
    else:
        rows = [export_layer(i) for i in wanted]

    total_nodes = 0
    for r in sorted(rows, key=lambda d: d["layer"]):
        total_nodes += r["nodes"]
        rel = "skipped" if r["rel"] is None else f"{r['rel']:.2e}"
        print(f"{r['layer']:6d} {r['type']:18s} {r['nodes']:8d} {r['export_s']:7.1f}s "
              f"{r.get('load_s', 0):6.1f}s {rel:>10}")
        manifest["layers"][str(r["layer"])] = {
            "type": r["type"], "nodes": r["nodes"], "rel": r["rel"],
            "inputs": r["inputs"], "outputs": r["outputs"]}
    print(f"  {len(wanted)} layers in {time.time() - t_layers:.0f}s "
          f"with {args.jobs} worker(s)")

    r = export_one(FinalNormONNX(lm.norm).eval(), (x,), ["hidden_in"], ["hidden_out"],
                   out_dir / "final_norm.onnx", args.opset, verify=not args.no_verify)
    total_nodes += r["nodes"]
    print(f"{'norm':>6} {'rmsnorm':18s} {r['nodes']:8d} {r['export_s']:7.1f}s "
          f"{r.get('load_s',0):6.1f}s {r['rel']:10.2e}")
    manifest["final_norm"] = {"nodes": r["nodes"], "rel": r["rel"]}
    manifest["total_nodes"] = total_nodes
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{total_nodes} nodes across {len(wanted)+1} graphs "
          f"(monolithic was 341,977 in one)")
    print(f"total on disk: "
          f"{sum(f.stat().st_size for f in out_dir.rglob('*') if f.is_file())/2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
