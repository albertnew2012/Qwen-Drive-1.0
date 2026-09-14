#!/usr/bin/env python
"""Dump the real parameter tree of the Qwen-Drive release, from the checkpoints.

Reads tensor shapes straight out of the safetensors headers, so it needs no GPU
and does not materialize the weights.

    python local/anatomy.py --root weights/Qwen-Drive-1.0-4B
"""
from __future__ import annotations

import argparse, json, re
from collections import OrderedDict
from pathlib import Path

from safetensors import safe_open


def tensors(path: Path):
    with safe_open(str(path), "pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            shape = sl.get_shape()
            n = 1
            for d in shape:
                n *= d
            yield k, tuple(shape), n, sl.get_dtype()


def collapse(key: str) -> str:
    """layers.7.mlp.gate -> layers.N.mlp.gate so repeated blocks group."""
    return re.sub(r"\.\d+\.", ".N.", key)


def summarize(path: Path, title: str, depth: int = 3):
    rows = list(tensors(path))
    total = sum(n for _, _, n, _ in rows)
    dtypes = {d for _, _, _, d in rows}
    print(f"\n{'='*100}\n{title}\n  file   {path}")
    print(f"  size   {path.stat().st_size/1e9:.3f} GB")
    print(f"  {len(rows)} tensors, {total/1e9:.4f} B params ({total:,}), dtype {sorted(dtypes)}")
    groups = OrderedDict()
    for k, shape, n, _ in rows:
        top = ".".join(collapse(k).split(".")[:depth])
        g = groups.setdefault(top, [0, 0])
        g[0] += n; g[1] += 1
    print(f"{'-'*100}")
    for k, (n, c) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:<62} {n/1e6:10.3f} M  {100*n/total:6.2f}%  {c:4d} tensors")
    return total, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--per-layer", action="store_true", help="also print one representative block")
    args = ap.parse_args()
    root = Path(args.root)

    cfg = json.loads((root / "config.json").read_text())
    text = cfg["vlm_config"]["text_config"]
    vis = cfg["vlm_config"]["vision_config"]
    exp = cfg["expert_config"]
    print("=" * 100)
    print("CONFIG")
    print("=" * 100)
    print(f"  VLM        {cfg['vlm_config']['architectures']}  model_type={cfg['vlm_config']['model_type']}")
    print(f"  text       hidden {text['hidden_size']}  layers {text['num_hidden_layers']}  "
          f"heads {text['num_attention_heads']} (kv {text['num_key_value_heads']})  "
          f"head_dim {text['head_dim']}  mlp {text['intermediate_size']}  vocab {text['vocab_size']}")
    lt = text["layer_types"]
    full = [i for i, t in enumerate(lt) if t == "full_attention"]
    print(f"  layer_types  {len(lt)} layers, full_attention at {full}  "
          f"(interval {text.get('full_attention_interval')})")
    print(f"  linear attn  key_heads {text['linear_num_key_heads']} x {text['linear_key_head_dim']}  "
          f"value_heads {text['linear_num_value_heads']} x {text['linear_value_head_dim']}  "
          f"conv_kernel {text['linear_conv_kernel_dim']}")
    print(f"  tie_word_embeddings {text['tie_word_embeddings']}   rope {text['rope_parameters']}")
    print(f"  vision     depth {vis['depth']}  hidden {vis['hidden_size']}  heads {vis['num_heads']}  "
          f"patch {vis['patch_size']}  merge {vis['spatial_merge_size']}  out {vis['out_hidden_size']}")
    print(f"  expert     hidden {exp['hidden_size']}  layers {exp['num_hidden_layers']}  "
          f"heads {exp['num_attention_heads']} (kv {exp['num_key_value_heads']})  "
          f"head_dim {exp['head_dim']}  mlp {exp['intermediate_size']}  "
          f"layers_per_kv {exp['layers_per_kv']}")
    print(f"  trajectory {cfg['num_future_points']} pts @ {cfg['trajectory_hz']} Hz "
          f"= {cfg['num_future_points']/cfg['trajectory_hz']:.1f} s   scale {cfg['trajectory_scale']}")

    grand = 0
    vlm_total, vlm_rows = summarize(root / "model.safetensors", "VLM  (shared by every task)")

    # Split the language model by attention kind - the fact that shapes the expert.
    import re as _re
    lin = fullat = emb = vis = misc = 0
    for k, _shape, n, _dt in vlm_rows:
        if k.startswith("vlm.model.visual"):
            vis += n; continue
        m = _re.search(r"language_model\.layers\.(\d+)\.", k)
        if m:
            if lt[int(m.group(1))] == "linear_attention":
                lin += n
            else:
                fullat += n
        elif "embed_tokens" in k:
            emb += n
        else:
            misc += n
    n_lin, n_full = lt.count("linear_attention"), lt.count("full_attention")
    print(f"{'-'*100}")
    print("  by attention kind:")
    print(f"    vision tower                        {vis/1e6:10.3f} M  {100*vis/vlm_total:6.2f}%")
    print(f"    embed_tokens (tied to lm_head)      {emb/1e6:10.3f} M  {100*emb/vlm_total:6.2f}%")
    print(f"    {n_lin:2d} x linear_attention  @ {lin/max(n_lin,1)/1e6:6.2f} M  "
          f"{lin/1e6:10.3f} M  {100*lin/vlm_total:6.2f}%   <- NO kv cache")
    print(f"    {n_full:2d} x full_attention    @ {fullat/max(n_full,1)/1e6:6.2f} M  "
          f"{fullat/1e6:10.3f} M  {100*fullat/vlm_total:6.2f}%   <- the expert reads these")
    print(f"    final norm                          {misc/1e6:10.3f} M")
    print(f"    lm_head tensor present: {any('lm_head' in k for k, *_ in vlm_rows)}")
    grand += vlm_total
    for sub, title in (("planner-rl", "PLANNING EXPERT (RL)"),
                       ("planner-sft", "PLANNING EXPERT (SFT)"),
                       ("perception", "BEV PERCEPTION HEAD")):
        p = root / sub / "model.safetensors"
        if p.exists():
            t, _ = summarize(p, title)
            if sub != "planner-sft":
                grand += t

    print(f"\n{'='*100}")
    print("TOTALS")
    print(f"{'='*100}")
    def count(sub):
        p = root / sub / "model.safetensors"
        return sum(n for _, _, n, _ in tensors(p)) if p.exists() else 0
    exp_n, perc_n = count("planner-rl"), count("perception")
    print(f"  VLM only            (VQA)                 {vlm_total/1e9:8.4f} B")
    print(f"  VLM + planner       (planning)            {(vlm_total+exp_n)/1e9:8.4f} B")
    print(f"  VLM + perception    (3D perception)       {(vlm_total+perc_n)/1e9:8.4f} B")
    print(f"  VLM + planner + perception (everything)   {(vlm_total+exp_n+perc_n)/1e9:8.4f} B")
    print(f"  bf16 resident, everything:                {(vlm_total+exp_n)*2/1e9 + perc_n*4/1e9:8.2f} GB"
          f"   (perception ships fp32)")

    if args.per_layer:
        print("\nOne representative VLM block of each kind:")
        for kind, idx in (("linear_attention", 0), ("full_attention", full[0])):
            print(f"\n--- layer {idx}  ({kind}) ---")
            for k, shape, n, _ in vlm_rows:
                if f".layers.{idx}." in k:
                    print(f"    {k:<74} {str(shape):<22} {n/1e6:9.4f} M")
        print("\nOne planning-expert block:")
        with safe_open(str(root / "planner-rl" / "model.safetensors"), "pt") as f:
            for k in sorted(f.keys()):
                if ".layers.0." in k or "layers" not in k:
                    sl = f.get_slice(k); shape = tuple(sl.get_shape())
                    n = 1
                    for d in shape: n *= d
                    print(f"    {k:<74} {str(shape):<22} {n/1e6:9.4f} M")


if __name__ == "__main__":
    main()
