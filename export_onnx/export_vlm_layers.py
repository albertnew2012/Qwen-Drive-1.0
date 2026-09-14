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
    ap.add_argument("--no-verify", action="store_true")
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
    x = embeds
    total_nodes = 0
    print(f"\n{'layer':>6} {'type':18s} {'nodes':>8} {'export':>8} {'load':>7} {'rel':>10}")
    for i, kind in enumerate(cfg.layer_types):
        layer = lm.layers[i]
        if kind == "linear_attention":
            mod = LinearLayerONNX(layer).eval()
            ex, names, onames = (x,), ["hidden_in"], ["hidden_out"]
        else:
            mod = FullLayerONNX(layer, lm.rotary_emb, len(cfg.layer_types)).eval()
            ex, names, onames = (x, pos), ["hidden_in", "position_ids"], \
                                ["hidden_out", "keys", "values"]
        if i in wanted:
            r = export_one(mod, ex, names, onames, out_dir / f"layer_{i:02d}.onnx",
                           args.opset, verify=not args.no_verify)
            total_nodes += r["nodes"]
            rel = "skipped" if r["rel"] is None else f"{r['rel']:.2e}"
            print(f"{i:6d} {kind:18s} {r['nodes']:8d} {r['export_s']:7.1f}s "
                  f"{r.get('load_s', 0):6.1f}s {rel:>10}")
            manifest["layers"][str(i)] = {"type": kind, "nodes": r["nodes"],
                                          "rel": r["rel"], "inputs": names,
                                          "outputs": onames}
        with torch.no_grad():
            o = mod(*ex)
            x = o if torch.is_tensor(o) else o[0]

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
