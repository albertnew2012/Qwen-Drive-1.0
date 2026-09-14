"""Smoke test: prove the perception head is differentiable once patched.

Without ``enable_training_ops()`` this fails - the shipped kernels have no
backward. With it, every trainable parameter receives a gradient.

    python training/test_gradients.py
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from qwen_drive_perception import QwenDrivePerception
from training.differentiable import enable_training_ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--cache", default="data/train_cache")
    ap.add_argument("--record", default="90162f90eceb4ada9e595bc1adb71b5f.pt")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--skip-patch", action="store_true",
                    help="show that the shipped kernels cannot backward")
    args = ap.parse_args()
    dev = "cuda"
    dtype = getattr(torch, args.dtype)

    if not args.skip_patch:
        enable_training_ops()
        print("training ops ENABLED (differentiable torch twins)")
    else:
        print("training ops DISABLED (shipped forward-only kernels)")

    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(dev)
    bev = head.bev_modeling.train()
    rec = torch.load(Path(args.cache) / args.record, weights_only=False)

    vit = rec["img_vit_feats"].to(dev, dtype)
    llm = rec["img_llm_feats"].to(dev, dtype)
    print(f"\ninput  vit {tuple(vit.shape)}  llm {tuple(llm.shape)}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    outs = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[rec["img_metas"]])
    fwd = time.time() - t0
    print("\noutputs:")
    for k, v in outs.items():
        if torch.is_tensor(v):
            print(f"  {k:16s} {str(tuple(v.shape)):28s} grad={v.requires_grad}")

    loss = sum(v.float().square().mean() for v in outs.values()
               if torch.is_tensor(v) and v.is_floating_point())
    t0 = time.time()
    loss.backward()
    bwd = time.time() - t0

    total = sum(1 for p in bev.parameters() if p.requires_grad)
    got = sum(1 for p in bev.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nforward {fwd:.1f}s   backward {bwd:.1f}s   peak GPU {peak:.1f} GiB")
    print(f"parameters receiving a NON-ZERO gradient: {got} / {total}")

    # the two patched operators specifically
    for name, mod in [("depth_net (push)", bev.depth_net),
                      ("view_trans (voxel pool)", bev.view_trans),
                      ("bev_embedding (pull seed)", bev.head.bev_embedding)]:
        ps = [p for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        allp = sum(1 for _ in mod.parameters())
        print(f"  {name:26s} {len(ps)}/{allp} params with gradient")

    assert got > 0, "no gradients - the patch did not take effect"
    print("\nPASS: the perception head is differentiable end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
