"""Coarsen the lift-splat depth grid, which is 58% of the BEV head.

``view_trans`` is 246 ms of the head's 427 ms. Its cost is one scatter per frustum
cell per depth bin: 6 cameras x 56x32 cells x 118 bins (0.5 m steps from 1 m to
60 m) into 16x200x200 voxels. Nothing else in the head is worth pruning -- the BEV
encoder and the DETR decoder together are 39 ms -- so the depth axis is the lever.

Merging adjacent depth bins is not a resample: the depth head emits a probability
distribution, and the pooling computes ``sum_d p(d) * feature`` scattered to
``voxel(d)``. Summing the probabilities of a group of bins is the exact marginal of
that group, so wherever the merged bins fall in the same voxel the result is
*identical*. Only where a group straddles a voxel boundary is anything lost, and
then all the mass lands at the group's centre instead of spread across it.

That is why coarsening should be non-uniform, fine near and coarse far. The
detection voxel is 0.512 m, so radially the 0.5 m bins are matched everywhere; but
tangentially a 16-pixel frustum cell spans about 1.8 m at 60 m against 0.512 m of
voxel, so far-range radial precision is already beyond what the grid can hold.
Spending 118 bins uniformly puts the resolution where it cannot be used.

``--scheme`` selects how bins are grouped. ``uniformK`` merges every K. ``nearK_farM``
keeps 0.5 m out to ``--near-m`` and merges by K then M beyond.

    python local/prune/depth_coarsen_v1.py
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


def depth_edges(view_trans):
    lo, hi, step = view_trans.frustum_range[2], view_trans.frustum_range[5], \
        view_trans.frustum_size[2]
    return np.arange(lo, hi, step)


def make_groups(depths: np.ndarray, scheme: str, near_m: float) -> list[int]:
    """Group sizes over the original bins, summing to len(depths)."""
    n = len(depths)
    if scheme == "none":
        return [1] * n
    if scheme.startswith("uniform"):
        k = int(scheme[len("uniform"):])
        return [k] * (n // k) + ([n % k] if n % k else [])
    if scheme.startswith("near"):
        # near<K>_far<M>: 1 bin each below near_m, then K, then M past 2 x near_m
        k, m = (int(x) for x in scheme[len("near"):].split("_far"))
        groups, i = [], 0
        while i < n:
            d = depths[i]
            size = 1 if d < near_m else (k if d < 2 * near_m else m)
            size = min(size, n - i)
            groups.append(size); i += size
        return groups
    raise ValueError(scheme)


def apply_coarsening(view_trans, groups: list[int]):
    """Install a coarsened depth axis; returns a restore callable."""
    depths = depth_edges(view_trans)
    idx, centres = [], []
    at = 0
    for g in groups:
        idx.append((at, at + g))
        centres.append(float(depths[at:at + g].mean()))
        at += g
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fr = view_trans.frustum_range
    fs = view_trans.frustum_size
    xs = torch.arange(fr[0], fr[3], fs[0], device=device)
    ys = torch.arange(fr[1], fr[4], fs[1], device=device)
    ds = torch.tensor(centres, device=device, dtype=xs.dtype)
    view_trans._frustum = torch.stack(
        torch.meshgrid([xs, ys, ds], indexing="ij"), dim=-1)

    original_forward = view_trans.forward
    bounds = torch.tensor([a for a, _ in idx] + [idx[-1][1]], device=device)

    def forward(mlvl_feats, img_depth, img_metas):
        # sum each group's probability: the exact marginal over that depth interval
        merged = []
        for d in img_depth:
            parts = [d[:, a:b].sum(1, keepdim=True) for a, b in idx]
            merged.append(torch.cat(parts, dim=1))
        return original_forward(mlvl_feats, merged, img_metas)

    view_trans.forward = forward

    def restore():
        view_trans.forward = original_forward
        view_trans._frustum = None
    return restore, len(groups)


def dets(cls, box, thr=0.3):
    p = 1 / (1 + np.exp(-cls.max(-1)))
    return p >= thr, cls.argmax(-1), box[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--schemes", default="none,uniform2,uniform3,uniform4,"
                                         "near2_far4,near3_far6,near2_far6")
    ap.add_argument("--near-m", type=float, default=20.0)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/prune/depth_coarsen_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    head = QwenDrivePerception.from_pretrained(
        args.perception, dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, proc)
    with torch.no_grad():
        _, metas = proc(PerceptionFrame(Path(args.frame)), device="cuda")
    ref = np.load("outputs/onnx_bench/torch_bfloat16_perception.npz")
    dt = next(head.bev_modeling.parameters()).dtype
    vit = torch.from_numpy(ref["img_vit_feats"]).to("cuda", dt)
    llm = torch.from_numpy(ref["img_llm_feats"]).to("cuda", dt)
    bev = head.bev_modeling
    vt = bev.view_trans
    depths = depth_edges(vt)
    print(f"  original depth bins {len(depths)}  ({depths[0]:.1f} to {depths[-1]:.1f} m, "
          f"{vt.frustum_size[2]} m steps)", flush=True)

    def run():
        with torch.no_grad():
            return bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])

    def timed(n=3):
        run(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): run()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

    o = run()
    base = dets(o["all_cls_scores"][-1, 0].float().cpu().numpy(),
                o["all_bbox_preds"][-1, 0].float().cpu().numpy(), args.thr)
    base_ms = timed()
    print(f"  baseline head {base_ms:.1f} ms, {int(base[0].sum())} detections\n")
    print(f"  {'scheme':14s} {'bins':>5s} {'head ms':>8s} {'saved':>7s} "
          f"{'kept':>9s} {'spur':>5s} {'centre_med':>11s} {'argmax':>8s}")
    rows = {}
    for scheme in args.schemes.split(","):
        groups = make_groups(depths, scheme, args.near_m)
        restore, nb = apply_coarsening(vt, groups)
        try:
            oo = run()
            c = dets(oo["all_cls_scores"][-1, 0].float().cpu().numpy(),
                     oo["all_bbox_preds"][-1, 0].float().cpu().numpy(), args.thr)
            both = base[0] & c[0]
            ms = timed()
            cd = (float(np.median(np.linalg.norm(base[2][both] - c[2][both], axis=-1)))
                  if both.any() else float("nan"))
            r = {"bins": nb, "head_ms": ms, "saved_ms": base_ms - ms,
                 "kept": int(both.sum()), "of": int(base[0].sum()),
                 "spurious": int((c[0] & ~base[0]).sum()),
                 "centre_median_m": cd,
                 "argmax_agreement": float((base[1] == c[1]).mean())}
            rows[scheme] = r
            print(f"  {scheme:14s} {nb:5d} {ms:8.1f} {base_ms-ms:7.1f} "
                  f"{r['kept']:4d}/{r['of']:<4d} {r['spurious']:5d} {cd:11.3f} "
                  f"{r['argmax_agreement']:7.1%}", flush=True)
        finally:
            restore()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"baseline_head_ms": base_ms,
                               "original_bins": len(depths), "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
