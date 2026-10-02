"""Does a larger Gated-DeltaNet chunk stay accurate?

Chunking is a tiling of the same recurrence, so any chunk size is algebraically
exact.  What can go wrong is numerical: ``decay_mask`` holds exponentials of
cumulative gates, and a wider window spans a larger range of those.

With the 63-step forward substitution a larger chunk was unaffordable - cost
grew as ``chunk - 1``.  The blocked inverse costs ``log2(chunk)``, so 256 is
almost free and cuts the sequential chunk loop from 43 steps to 11.

Scores each candidate against the stock implementation on a real prefill.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_onnx.export_vlm_layers import patch_chunk_rule_blocked


def build():
    from export_onnx.scene_inputs import perception_inputs

    ctx = perception_inputs("weights/Qwen-Drive-1.0-4B",
                           "weights/Qwen-Drive-1.0-4B/perception",
                           "data/demo/perception",
                           "90162f90eceb4ada9e595bc1adb71b5f")
    vlm = ctx["vlm"]
    with torch.no_grad():
        embeds = vlm.model.get_input_embeddings()(ctx["inputs"]["input_ids"])
    return vlm, embeds, ctx["position_ids"]


def chain(vlm, embeds, pos, n_layers=None):
    """Push the hidden state through the decoder and return the final state."""
    lm = vlm.model.language_model
    cfg = vlm.config.text_config if hasattr(vlm.config, "text_config") else vlm.config
    types = list(cfg.layer_types)
    if n_layers:
        types = types[:n_layers]
    x = embeds
    with torch.no_grad():
        for i, kind in enumerate(types):
            layer = lm.layers[i]
            if kind == "linear_attention":
                o = layer(x, position_embeddings=None)
            else:
                from transformers.cache_utils import DynamicCache
                cos, sin = lm.rotary_emb(x, pos)
                seq = x.shape[1]
                causal = torch.full((seq, seq), torch.finfo(x.dtype).min,
                                    dtype=x.dtype).triu(1)[None, None]
                o = layer(x, position_embeddings=(cos, sin), position_ids=pos,
                          attention_mask=causal,
                          past_key_values=DynamicCache(config=layer.self_attn.config))
            x = o if torch.is_tensor(o) else o[0]
    return x


def set_chunk(vlm, size):
    n = 0
    for mod in vlm.modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            mod.chunk_gated_delta_rule = functools.partial(
                mod.chunk_gated_delta_rule, chunk_size=size)
            n += 1
    return n


def main() -> None:
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    vlm, embeds, pos = build()
    print(f"reference: stock loop, chunk 64, first {n_layers} layers", flush=True)
    ref = chain(vlm, embeds, pos, n_layers)
    print(f"  |h|max {ref.abs().max():.4f}\n", flush=True)

    patch_chunk_rule_blocked(vlm)
    print(f"{'chunk':>7} {'chunks':>7} {'rel vs stock':>14}")
    for size in (64, 128, 256, 512):
        vlm_reset = vlm
        # re-bind cleanly each time: partial would otherwise stack
        patch_chunk_rule_blocked(vlm_reset)
        set_chunk(vlm_reset, size)
        got = chain(vlm_reset, embeds, pos, n_layers)
        rel = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-9)
        pad = (size - embeds.shape[1] % size) % size
        print(f"{size:7d} {(embeds.shape[1]+pad)//size:7d} {rel:14.3e}", flush=True)


if __name__ == "__main__":
    main()
