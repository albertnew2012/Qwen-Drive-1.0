"""How much does each decoder layer change the answer?

Depth pruning is the cheapest structured lever available: 24 of the 32 layers are
Gated-DeltaNet blocks costing ~22 ms each in PyTorch bf16, so every layer removed
is worth about 3% of the perception decoder. The question is which ones are doing
work.

Two measurements per layer, both on real frames rather than noise:

``block_delta``  ||out - in|| / ||in|| for the block in the full forward. A block
                 whose output barely differs from its input is a candidate, but
                 this is only a local view -- a small change can still be the one
                 the next layer depends on.

``ablate_rel``   the relative change in the *final* hidden state when that single
                 layer is skipped and the other 31 run normally. This is the
                 number that matters, because it is measured where the head reads.

Skipping a layer here means passing its input straight through, which is what
removing it from the stack would do. The residual stream keeps its shape, so this
is a valid preview of a pruned model rather than an approximation of one.

    python local/prune/measure_layer_influence_v1.py --frames 4
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache


def build(vlm, frame_dir, proc, device="cuda"):
    """Token embeddings with the vision features spliced in, as the decoder sees them."""
    from qwen_drive_perception.dataset import PerceptionFrame
    with torch.no_grad():
        inputs, metas = proc(PerceptionFrame(Path(frame_dir)), device=device)
        embeds = vlm.model.get_input_embeddings()(inputs["input_ids"])
        captured = {}
        h = vlm.model.visual.merger.register_forward_hook(
            lambda m, a, o=None: captured.__setitem__("merged", o))
        try:
            vlm.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        finally:
            h.remove()
        merged = captured["merged"]
        merged = merged[0] if isinstance(merged, tuple) else merged
        mask = inputs["input_ids"][0] == vlm.config.image_token_id
        x = embeds.clone()
        x[0, mask] = merged[-int(mask.sum()):].to(embeds.dtype)
    return x, inputs, metas, mask


def run_stack(lm, types, x, pos, skip=()):
    """The 32 blocks in order, optionally passing some straight through."""
    cos, sin = lm.rotary_emb(x, pos)
    causal = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min,
                        dtype=x.dtype, device=x.device).triu(1)[None, None]
    deltas = []
    for i, kind in enumerate(types):
        if i in skip:
            deltas.append(0.0)
            continue
        prev = x
        if kind == "linear_attention":
            out = lm.layers[i](x, position_embeddings=None)
        else:
            out = lm.layers[i](
                x, position_embeddings=(cos, sin), position_ids=pos,
                attention_mask=causal,
                past_key_values=DynamicCache(config=lm.layers[i].self_attn.config))
        x = out if torch.is_tensor(out) else out[0]
        deltas.append(float((x - prev).norm() / prev.norm().clamp_min(1e-9)))
    return lm.norm(lm.norm(x)), deltas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--frames-dir", default="data/demo/perception")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out", default="outputs/prune/layer_influence_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception.dataset import PerceptionProcessor

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    vlm = model.vlm
    lm = vlm.model.language_model
    types = list(vlm.config.text_config.layer_types)
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))

    frames = sorted(p for p in Path(args.frames_dir).iterdir() if p.is_dir())[:args.frames]
    print(f"{len(types)} layers, {len(frames)} frames", flush=True)

    per_frame = []
    for fi, fd in enumerate(frames):
        x, inputs, metas, mask = build(vlm, fd, proc)
        pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
        with torch.no_grad():
            full, deltas = run_stack(lm, types, x, pos)
            scale = full.float().norm().clamp_min(1e-9)
            ablate = []
            t0 = time.time()
            for i in range(len(types)):
                got, _ = run_stack(lm, types, x, pos, skip={i})
                ablate.append(float((got.float() - full.float()).norm() / scale))
            print(f"  frame {fi+1}/{len(frames)}  {x.shape[1]} tokens  "
                  f"{time.time()-t0:.0f}s", flush=True)
        per_frame.append({"block_delta": deltas, "ablate_rel": ablate,
                          "frame": fd.name, "tokens": int(x.shape[1])})
        del x
        torch.cuda.empty_cache()

    bd = np.mean([f["block_delta"] for f in per_frame], 0)
    ab = np.mean([f["ablate_rel"] for f in per_frame], 0)
    order = np.argsort(ab)
    print(f"\n  {'layer':>5s} {'type':18s} {'block_delta':>12s} {'ablate_rel':>11s}  rank")
    rank = {int(l): r for r, l in enumerate(order)}
    for i, t in enumerate(types):
        flag = "  <- least influential" if rank[i] < 8 else ""
        print(f"  {i:5d} {t:18s} {bd[i]:12.4f} {ab[i]:11.4f} {rank[i]:5d}{flag}")
    print(f"\n  8 least influential layers: {sorted(int(x) for x in order[:8])}")
    print(f"  12 least influential layers: {sorted(int(x) for x in order[:12])}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"layer_types": types, "frames": per_frame,
                               "block_delta": bd.tolist(), "ablate_rel": ab.tolist(),
                               "rank_least_first": [int(x) for x in order]}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
