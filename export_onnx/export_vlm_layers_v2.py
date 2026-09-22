"""Export the VLM text model as per-layer ONNX graphs, second generation.

``export_vlm_layers.py`` splits the decoder at layer boundaries so onnxruntime can
build sessions at all; that decision stands and this reuses its wrappers. What
changes is how the Gated-DeltaNet layers are traced. Measured on one layer at
2752 tokens, fp32, RTX 3090, ORT 1.22 CUDA EP with IOBinding:

    shipped export                       125.96 ms   14,161 nodes
    + blocked triangular inverse          78.97 ms    3,952 nodes
    + gdn_onnx_v2 (stack/unbind/hoist)    30.59 ms    2,121 nodes

and within that last one the recurrence itself went 67.3 -> 19.0 ms while the MLP
sat unchanged at 12.1 ms. Projected over the stack, 24 linear + 8 full attention:
3.42 s -> 0.95 s.

Three changes, none of which alter the arithmetic:

  * ``gdn_onnx_v2`` rewrites the chunk rule so it traces into few large nodes
    instead of thousands of tiny ones (see that module for the measurements).
  * ``gdn_fast_v2.CausalDepthwiseShift`` replaces the depthwise ``Conv1d``, which
    ORT hands to a cuDNN grouped convolution costing 15.5 ms for 0.18 GFLOP.
  * The sequence handed to the graph is rounded up to a multiple of the chunk
    size, so the chunk rule's five ``Pad`` nodes -- which ORT has no CUDA kernel
    for and runs on the CPU, moving 45 MiB to the host and back each -- fall away.

The padding is the only thing the host has to know about: ``manifest.json``
records ``sequence`` (the real length) and ``padded`` (what the graphs expect), and
the runner appends zero rows and slices the output back. Both the delta rule and
the full-attention layers are causal, so the real rows are bit-unaffected; this is
checked here against the unpadded stock model before anything is written.

    python export_onnx/export_vlm_layers_v2.py --task perception
    python export_onnx/export_vlm_layers_v2.py --task planning
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from export_onnx.export_vlm_layers import (
    LinearLayerONNX, FullLayerONNX, FinalNormONNX, export_one)
from export_onnx import gdn_fast_v2 as fast
from export_onnx import gdn_onnx_v2 as gdn


class FullLayerPlanONNX(torch.nn.Module):
    """A full-attention layer whose cache leaves the graph ready for the expert.

    Two reshapes that used to happen on the host now happen here:

    *Transpose.* The layer's cache is ``[B, kv_heads, S, head]`` and the Planning
    Expert cross-attends over ``[B, S, kv_heads, head]``. ``run_onnx_drive.py``
    fixed that up in numpy, which meant pulling all eight layers' keys and values
    to the host -- 222 MiB down and back up again every frame.

    *Trim.* The stack runs on a sequence rounded up to a chunk multiple, but the
    expert's graph is built for the true length. Dropping the padding rows here
    keeps the alignment invisible outside the decoder.

    With both inside the graph the cache stays on the card from the layer that
    produces it to the planner that reads it.
    """

    def __init__(self, layer, rotary, n_layers, sequence: int):
        super().__init__()
        self.inner = FullLayerONNX(layer, rotary, n_layers)
        self.sequence = sequence

    def forward(self, hidden_states, position_ids):
        out, keys, values = self.inner(hidden_states, position_ids)
        keys = keys[:, :, : self.sequence].permute(0, 2, 1, 3)
        values = values[:, :, : self.sequence].permute(0, 2, 1, 3)
        return out, keys, values


class FullLayerHiddenONNX(torch.nn.Module):
    """A full-attention layer that emits only the hidden state.

    Perception never reads the cache, so materialising it costs 8 x 2 x 13.8 MiB
    of device memory per frame for nothing.
    """

    def __init__(self, layer, rotary, n_layers):
        super().__init__()
        self.inner = FullLayerONNX(layer, rotary, n_layers)

    def forward(self, hidden_states, position_ids):
        return self.inner(hidden_states, position_ids)[0]


def _install_side_shrink(side: float) -> None:
    """Make every scene loaded from here on carry smaller side views.

    ``planning_inputs`` builds its own scene internally, so the resize is installed
    by wrapping the loader it uses rather than by passing an argument through.
    """
    from qwen_drive import benchmarks
    from local.prune.optimise_planning_v1 import shrink_side_views
    original = benchmarks.read_scene_file

    def patched(*a, **k):
        for sample in original(*a, **k):
            try:
                sample.scene = shrink_side_views(sample.scene, side)
            except Exception:
                pass
            yield sample

    benchmarks.read_scene_file = patched
    import export_onnx.scene_inputs as si
    if hasattr(si, "read_scene_file"):
        si.read_scene_file = patched
    print(f"side views resized to {side}x", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-root", default="data/demo")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--task", choices=["perception", "planning"], default="perception")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--layers", default="", help="comma list (default: all)")
    ap.add_argument("--skip-layers", default="",
                    help="comma list of layer indices to drop. The kept layers are "
                         "renumbered 0..N-1 and the manifest records only their types, "
                         "so the host runs a shorter stack with no special casing. "
                         "Perception may drop any layer; planning may only drop "
                         "linear-attention ones, because the Planning Expert "
                         "cross-attends to every full-attention layer's cache.")
    ap.add_argument("--side", type=float, default=1.0,
                    help="planning only: resize applied to the two side views. Tokens "
                         "go as the square, and the side views are ~2/3 of the prompt. "
                         "They are shrunk rather than dropped because DrivingScene "
                         "refuses a scene missing any of CAMERA_VIEWS.")
    ap.add_argument("--cams", default="",
                    help="comma list of camera names, or 'front' for cam_order[0]")
    ap.add_argument("--embed-from", default="",
                    help="reuse embed_tokens.npy from this directory instead of "
                         "writing another 2.4 GiB copy")
    args = ap.parse_args()
    os.chdir(_ROOT)

    out_dir = Path(args.out or f"outputs/onnx/vlm_layers_v2_{args.task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from export_onnx.scene_inputs import perception_inputs, planning_inputs
    if args.task == "perception":
        if args.cams:
            from export_onnx.scene_inputs_front import perception_inputs_cams
            ctx = perception_inputs_cams(args.vlm, args.model, args.frames,
                                         args.frame, args.cams)
        else:
            ctx = perception_inputs(args.vlm, args.model, args.frames, args.frame)
    else:
        if args.side != 1.0:
            _install_side_shrink(args.side)
        ctx = planning_inputs(args.vlm, args.planner, args.scenes,
                              args.image_root, args.image_archive)
    vlm, inputs, pos = ctx["vlm"], ctx["inputs"], ctx["position_ids"]
    lm = vlm.model.language_model
    cfg = vlm.config.text_config if hasattr(vlm.config, "text_config") else vlm.config
    types = list(cfg.layer_types)

    with torch.no_grad():
        embeds = vlm.model.get_input_embeddings()(inputs["input_ids"])
    seq = embeds.shape[1]
    padded = fast.pad_to(seq, args.chunk_size)
    print(f"task={args.task}  sequence {seq} -> {padded} (chunk {args.chunk_size}), "
          f"hidden {embeds.shape[2]}, {len(types)} layers", flush=True)

    # Check the rewrite on this exact input before trusting it. The per-layer
    # check inside export_one cannot: both sides of it share whatever the patch
    # got wrong.
    first_linear = types.index("linear_attention")

    def run_linear(x):
        out = lm.layers[first_linear](x, position_embeddings=None)
        return out if torch.is_tensor(out) else out[0]

    with torch.no_grad():
        stock = run_linear(embeds).float()
    scale = max(float(stock.abs().max()), 1e-9)
    n_rule = gdn.patch(vlm, args.chunk_size)
    n_conv = fast.patch_conv(vlm)
    with torch.no_grad():
        rel_rule = float((run_linear(embeds).float() - stock).abs().max()) / scale

    pad_rows = torch.zeros(1, padded - seq, embeds.shape[2],
                           dtype=embeds.dtype, device=embeds.device)
    embeds_p = torch.cat([embeds, pad_rows], dim=1)
    with torch.no_grad():
        rel_pad = float((run_linear(embeds_p).float()[:, :seq] - stock).abs().max()) / scale
    print(f"rewrite: {n_rule} chunk rules + {n_conv} convs   "
          f"layer {first_linear} moves {rel_rule:.2e}, "
          f"padded-to-{padded} moves {rel_pad:.2e}", flush=True)
    for name, rel in (("rewrite", rel_rule), ("padding", rel_pad)):
        if rel > 1e-5:
            raise RuntimeError(f"{name} changed the answer by {rel:.2e}")

    pos_p = torch.cat([pos, pos[:, :, -1:].expand(-1, -1, padded - seq)], dim=-1)

    emb_path = out_dir / "embed_tokens.npy"
    if not emb_path.exists():
        src = Path(args.embed_from) / "embed_tokens.npy" if args.embed_from else None
        if src and src.exists():
            os.symlink(src.resolve(), emb_path)
            print(f"  embed_tokens.npy -> {src}", flush=True)
        else:
            w = vlm.model.get_input_embeddings().weight.detach().cpu().numpy()
            np.save(emb_path, w)
            print(f"  embed_tokens.npy {w.shape}  {w.nbytes/2**30:.2f} GiB", flush=True)

    dropped = {int(x) for x in args.skip_layers.split(",") if x != ""}
    if dropped:
        bad = [i for i in dropped if types[i] == "full_attention"]
        if args.task == "planning" and bad:
            raise RuntimeError(
                f"planning cannot drop full-attention layers {bad}: the Planning "
                "Expert cross-attends to every one of their caches")
        print(f"dropping {len(dropped)} layers -> {len(types)-len(dropped)} remain "
              f"({sorted(dropped)})", flush=True)
    wanted = ([int(x) for x in args.layers.split(",") if x != ""]
              if args.layers else [i for i in range(len(types)) if i not in dropped])
    renumber = {old: new for new, old in enumerate(wanted)}
    record, total = {}, 0
    print(f"\n {'layer':>5s} {'type':18s} {'nodes':>7s} {'export':>8s}", flush=True)
    t_all = time.time()
    for i in wanted:
        kind = types[i]
        if kind == "linear_attention":
            module, example = LinearLayerONNX(lm.layers[i]), (embeds_p,)
            names, onames = ["hidden_in"], ["hidden_out"]
        elif args.task == "planning":
            module = FullLayerPlanONNX(lm.layers[i], lm.rotary_emb, len(types), seq)
            example = (embeds_p, pos_p)
            names, onames = ["hidden_in", "position_ids"], ["hidden_out", "keys", "values"]
        else:
            module = FullLayerHiddenONNX(lm.layers[i], lm.rotary_emb, len(types))
            example = (embeds_p, pos_p)
            names, onames = ["hidden_in", "position_ids"], ["hidden_out"]
        t0 = time.time()
        info = export_one(module, example, names, onames,
                          out_dir / f"layer_{renumber[i]:02d}.onnx", args.opset,
                          verify=False)
        total += info["nodes"]
        record[str(renumber[i])] = {"type": kind, "nodes": info["nodes"],
                                    "source_layer": i,
                                    "inputs": names, "outputs": onames}
        print(f" {i:5d} {kind:18s} {info['nodes']:7d} {time.time()-t0:7.1f}s", flush=True)

    norm = export_one(FinalNormONNX(lm.norm), (embeds_p,), ["hidden_in"],
                      ["hidden_out"], out_dir / "final_norm.onnx", args.opset,
                      verify=False)
    total += norm["nodes"]
    manifest = {"layer_types": [types[i] for i in wanted],
                "all_layer_types": types, "kept_layers": wanted,
                "sequence": seq, "padded": padded,
                "chunk_size": args.chunk_size, "hidden": embeds.shape[2],
                "rel_rewrite": rel_rule, "rel_padding": rel_pad,
                "layers": record, "final_norm": {"nodes": norm["nodes"]},
                "total_nodes": total}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{total} nodes across {len(record)+1} graphs in "
          f"{time.time()-t_all:.0f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
