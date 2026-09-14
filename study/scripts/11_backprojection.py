"""
11_backprojection.py - the push stream is a weighted BACKPROJECTION.

Panel 2 of two_lifts.png is full of radial streaks. They are not a rendering
artifact and not noise: they are DepthNet's uncertainty, smeared along each
camera ray - the same thing unfiltered backprojection produces in PET or CT.

This proves it by changing exactly one thing, the SHARPNESS of the depth
distribution, via a temperature on the DepthNet logits:

    T = 0   -> uniform      every bin equally likely = pure backprojection
    T = 1   -> unchanged    what the model actually does
    T >> 1  -> one-hot      a single point per ray, no spreading at all

If the streaks are depth uncertainty, sharpening must collapse them. It does:
25,692 effective cells lit at T=0, 395 at T=40 - a 65x concentration.

The second result is the interesting one. Energy landing within 3 m of a real
object PEAKS at the model's own sharpness (14.5 %) and FALLS when sharpened
further (7.3 %). Hedging beats committing here, because the argmax of the
distribution is the 59.5 m reject bin for ~60 % of cells - so a hard argmax
throws the features at the far clip. See study/06 section 2.1.

    PYTHONPATH=src python study/scripts/11_backprojection.py
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from transformers import AutoTokenizer

from qwen_drive import QwenDriveForPlanning
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor

SWEEP = [(0, "depth = UNIFORM\npure backprojection"),
         (1, "depth = AS PREDICTED\nwhat the model does"),
         (8, "depth SHARPENED x8"),
         (40, "depth = NEAR ONE-HOT\none point per ray")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--figure", default="outputs/backprojection.png")
    args = ap.parse_args()

    print(f"\n{'=' * 96}\n  LOAD\n{'=' * 96}")
    dtype = getattr(torch, args.dtype)
    holder = QwenDriveForPlanning.from_pretrained(args.vlm, dtype=dtype, attn_implementation="sdpa")
    vlm = holder.vlm
    del holder.planning_expert
    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm.to(args.device).eval(), proc)
    frame = PerceptionFrame(Path(args.frames) / args.frame)
    inputs, metas = proc(frame, device=args.device)
    hc, bev = head.config, head.bev_modeling
    gt = frame.gt["boxes"]
    print(f"    {frame.token}   {len(frame.cam_order)} cameras   {len(gt)} ground-truth objects")

    def push_bev(T):
        """The pushed BEV with DepthNet's logits scaled by temperature ``T``."""
        grab = {}

        def sharpen(m, a, o):
            return torch.zeros_like(o) if T == 0 else o.float().mul(T).to(o.dtype)

        h1 = bev.depth_net.register_forward_hook(sharpen)
        h2 = bev.uvtr_query_proj.register_forward_hook(
            lambda m, a, o: grab.__setitem__("u", o.detach()))
        try:
            with torch.no_grad():
                head.infer(inputs, metas)
        finally:
            h1.remove(); h2.remove()
        u = grab["u"].float()
        u = u.norm(dim=1)[0] if u.dim() == 4 else u.norm(dim=-1)
        return u.reshape(hc.bev_h, hc.bev_w).cpu().numpy()

    lo, hi = hc.det_pc_range[0], hc.det_pc_range[3]

    def score(a):
        e = a / a.sum()
        pr = 1.0 / (e ** 2).sum()                      # effective number of cells lit
        H, W = a.shape
        ys, xs = np.mgrid[0:H, 0:W]
        X = lo + (xs + 0.5) / W * (hi - lo)            # array is [y, x]
        Y = lo + (ys + 0.5) / H * (hi - lo)
        d = np.min(np.sqrt((X[..., None] - gt[:, 0]) ** 2
                           + (Y[..., None] - gt[:, 1]) ** 2), axis=-1)
        return pr, e[d < 3.0].sum() * 100

    print(f"\n{'=' * 96}\n  SWEEP the sharpness of the depth distribution\n{'=' * 96}")
    print(f"    {'depth distribution':28s} {'effective cells lit':>21} {'% energy on objects':>21}")
    maps, rows = [], []
    for T, name in SWEEP:
        a = push_bev(T)
        pr, nr = score(a)
        maps.append(a); rows.append((name, pr, nr))
        print(f"    {name.replace(chr(10), ' / '):28s} {pr:21,.0f} {nr:20.1f}%")
    print("\n    Sharpening collapses the streaks, so the streaks ARE depth uncertainty:")
    print(f"    {rows[0][1] / rows[-1][1]:.0f}x concentration from uniform to one-hot.")
    print("    But energy ON OBJECTS peaks at the model's OWN sharpness and falls either")
    print("    side of it. A hard argmax is worse than hedging, because for ~60 % of cells")
    print("    the argmax is the 59.5 m reject bin - watch it pile into the corners.")

    if args.figure:
        _figure(args.figure, maps, rows, gt, lo, hi)
        print(f"\n    wrote {args.figure}")
    return 0


def _figure(path, maps, rows, gt, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [hi, lo, lo, hi]                              # forward UP, +y to the left
    fig, axes = plt.subplots(1, len(maps), figsize=(4.75 * len(maps), 5.2), dpi=110,
                             facecolor="white")
    for ax, a, (name, pr, nr) in zip(axes, maps, rows):
        v = np.log1p(a.astype(np.float64))
        vlo, vhi = np.percentile(v, 1), np.percentile(v, 99.5)
        ax.imshow(np.clip(v, vlo, vhi).T, cmap="magma", origin="lower", extent=ext,
                  aspect="equal")
        ax.invert_xaxis()
        ax.scatter(gt[:, 1], gt[:, 0], s=34, facecolors="none", edgecolors="#39d353",
                   linewidths=1.2)
        ax.plot(0, 0, marker="^", ms=10, color="#d6336c")
        ax.set_title(f"{name}\n{pr:,.0f} cells lit   |   {nr:.1f}% energy on objects",
                     fontsize=10.5)
        ax.set_xlabel("y left [m]  (forward is UP)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("x forward [m]", fontsize=9)
    fig.suptitle("The push stream IS a weighted backprojection: the streaks are depth "
                 "uncertainty, smeared along each ray\ngreen = real objects.  Sharpening "
                 "collapses them - but sharpening too far puts the energy in the WRONG place.",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
