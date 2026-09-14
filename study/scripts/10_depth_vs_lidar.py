"""
10_depth_vs_lidar.py — is DepthNet's predicted depth actually correct?

The honest answer needs ground truth, and the demo frames ship ``lidar.npy``.
This projects those points into CAM_FRONT, bins them onto DepthNet's own 32x56
grid, and scores every plausible way of reducing the 118-bin distribution
against them.

The headline: the 118-bin distribution is BIMODAL - a correct peak at the true
distance plus a large spike at the 59.5 m far clip. Reduce it with a plain
expectation and you get 56 m for road 5 m ahead (corr -0.55 against lidar).
Weight each bin by whether it survives the BEV range mask and you get 4.9 m
(corr +0.79, median error 1.4 m).

Second trap: the BEV volume has a z ceiling, so an upward-looking ray leaves the
grid at some depth and the network is not permitted to place anything beyond it.
85 % of lidar cells past 40 m sit outside the volume. Scored against only what is
representable, correlation is +0.879 and the median error 1.2 m - the apparent
"compression at range" is mostly the grid's limit, not the network's error.

    PYTHONPATH=src python study/scripts/10_depth_vs_lidar.py --figure outputs/depth_vs_lidar.png
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

GH, GW, IW, IH = 32, 56, 896, 512
REJECT_BIN = 110        # bins >= 110 (56 m+) are the learned far-clip reject spike
RAW_W, RAW_H = 1600, 900


def rule(t):
    print(f"\n{'=' * 96}\n  {t}\n{'=' * 96}")


def lidar_depth_grid(frame_dir: Path, cam_index: int = 0):
    """Median lidar depth per DepthNet cell, and the point count."""
    pts = np.load(frame_dir / "lidar.npy").astype(np.float64)
    c = np.load(frame_dir / "calib.npz")
    K = c["cam_intrinsic"][cam_index]
    R, t = c["sensor2lidar_rotation"][cam_index], c["sensor2lidar_translation"][cam_index]
    # sensor2lidar maps camera -> lidar, so invert it to bring points into the camera
    cam = (pts - t) @ R
    z = cam[:, 2]
    cam = cam[z > 0.5]
    uvw = (K @ cam.T).T
    u = uvw[:, 0] / uvw[:, 2] * (IW / RAW_W)
    v = uvw[:, 1] / uvw[:, 2] * (IH / RAW_H)
    d = uvw[:, 2]
    keep = (u >= 0) & (u < IW) & (v >= 0) & (v < IH)
    u, v, d = u[keep], v[keep], d[keep]
    gi = (v / IH * GH).astype(int).clip(0, GH - 1)
    gj = (u / IW * GW).astype(int).clip(0, GW - 1)
    flat = gi * GW + gj
    gt = np.full(GH * GW, np.nan)
    cnt = np.bincount(flat, minlength=GH * GW)
    for cell in np.flatnonzero(cnt >= 3):
        gt[cell] = np.median(d[flat == cell])
    return gt.reshape(GH, GW), cnt.reshape(GH, GW), len(d), (u, v, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--figure", default="outputs/depth_vs_lidar.png")
    args = ap.parse_args()

    rule("LOAD")
    dtype = getattr(torch, args.dtype)
    holder = QwenDriveForPlanning.from_pretrained(args.vlm, dtype=dtype, attn_implementation="sdpa")
    vlm = holder.vlm
    del holder.planning_expert
    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm.to(args.device).eval(), proc)
    frame_dir = Path(args.frames) / args.frame
    frame = PerceptionFrame(frame_dir)
    inputs, metas = proc(frame, device=args.device)
    print(f"    {frame.token}   {len(frame.cam_order)} cameras")

    # depth distribution + the BEV range mask, from one real forward
    grab = {}
    vt = head.bev_modeling.view_trans
    orig_cp = type(vt).coord_preparing

    def spy(self, img_metas):
        out = orig_cp(self, img_metas)
        grab.setdefault("mask", out[1][0, 0].detach())
        return out

    type(vt).coord_preparing = spy
    hk = head.bev_modeling.depth_net.register_forward_hook(
        lambda m, a, o: grab.__setitem__("logits", o.detach()))
    try:
        with torch.no_grad():
            head.infer(inputs, metas)
    finally:
        type(vt).coord_preparing = orig_cp
        hk.remove()

    p = grab["logits"].float().softmax(1)[0].cpu().numpy()   # [118, 32, 56] CAM_FRONT
    msk = grab["mask"].float()[0].cpu().numpy()              # [118, 32, 56]
    nb = p.shape[0]
    bins = np.arange(nb) * 0.5 + 1.0

    rule("A   the distribution is BIMODAL")
    for name, (r, c) in {"road right in front": (30, 28), "white van, near left": (20, 8),
                         "sky": (1, 28)}.items():
        d = p[:, r, c]
        near = d[bins < 15].sum() * 100
        mid = d[(bins >= 15) & (bins < 40)].sum() * 100
        far = d[bins >= 40].sum() * 100
        pk = bins[:110][d[:110].argmax()]
        print(f"    {name:22s}  <15 m {near:5.1f}%   15-40 m {mid:5.1f}%   >40 m {far:5.1f}%"
              f"   near peak {pk:5.1f} m")
    print("\n    The 15-40 m band is empty. A correct peak sits at the true distance and a")
    print("    large spike sits at the 59.5 m far clip, so a plain expectation averages two")
    print(f"    modes and describes neither. Only {(p * msk).sum(0).mean() * 100:.1f}% of mass "
          f"survives the BEV range mask.")

    rule("B   scored against LIDAR ground truth")
    gt, cnt, npts, uvd = lidar_depth_grid(frame_dir)
    val = ~np.isnan(gt)
    print(f"    {npts} lidar points land in CAM_FRONT; {val.sum()} of {GH * GW} cells have >=3")
    print(f"    sky (rows 0-6): lidar returns in {int((cnt[:6] >= 3).sum())} of {6 * GW} cells"
          "  <- the sky has no surface to hit")

    w = p * msk
    kept = w.sum(0)
    # The BEV volume has a z ceiling, so an upward-looking ray leaves the grid at some
    # depth. Beyond that no estimator can place a surface - the model is not allowed to.
    # Comparing against lidar points past this point measures the grid, not the network.
    d_exit = np.where(msk.any(0), bins[nb - 1 - np.argmax(msk[::-1], 0)], np.nan)
    repr_ = val & (gt <= d_exit)
    # Mass past the exit that is NOT in the learned reject spike: the network wanted to
    # name a distance the grid cannot express, so the truncated mean is an artifact of
    # the z ceiling rather than a depth. Cells over 15% of this have a median error of
    # 15.5 m against lidar; the rest have 1.2 m.
    rej = p[REJECT_BIN:].sum(0)
    trunc = np.clip(1.0 - kept - rej, 0.0, 1.0)
    # Two independent reasons to refuse to draw a cell, and only two:
    #   trunc > 0.15  the network wanted a distance past where the ray leaves the
    #                 volume, so the truncated mean is a ceiling artifact
    #   kept  < 0.005 essentially nothing survives at all (row 0 is 97.7% reject),
    #                 so the value is division noise
    # A LOW 'kept' on its own is NOT a defect - near road deposits ~11% because 87%
    # of its mass is deliberate reject, and its depth is exact. Gating at kept<0.06
    # hides the best cells and pushes drawn error 1.2 -> 1.5 m. Measured: drawn
    # cells median 1.2 m vs lidar, hidden cells 14.2 m.
    hide = (trunc > 0.15) | (kept < 0.005)
    cands = {
        "naive expectation (all 118)": (p * bins[:, None, None]).sum(0),
        "argmax (all bins)": bins[p.argmax(0)],
        "near-expectation <56 m": ((p[:110] / p[:110].sum(0, keepdims=True))
                                   * bins[:110, None, None]).sum(0),
        "near-mode <56 m": bins[p[:110].argmax(0)],
        "mask-weighted  <- USED": (w * bins[:, None, None]).sum(0) / np.maximum(kept, 1e-6),
    }
    print(f"\n    {'reducer':30s} {'corr':>7} {'MAE':>9} {'median err':>12}")
    for n, f in cands.items():
        a, b = f[val], gt[val]
        print(f"    {n:30s} {np.corrcoef(a, b)[0, 1]:+7.3f} {np.abs(a - b).mean():8.1f}m "
              f"{np.median(np.abs(a - b)):11.1f}m")

    pred = cands["mask-weighted  <- USED"]
    a, b = pred[val], gt[val]
    rule("C   where it is accurate, and where it is not")
    print(f"    {'lidar range':14s} {'n':>5} {'median pred':>12} {'median err':>11} {'bias':>9}")
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60)]:
        s = (b >= lo) & (b < hi)
        so = repr_ & (gt >= lo) & (gt < hi)
        if s.sum():
            r = (f"{np.median(pred[so] - gt[so]):+.1f} m ({int(so.sum())})"
                 if so.sum() else "none representable")
            print(f"    {f'{lo}-{hi} m':14s} {int(s.sum()):5d} {np.median(a[s]):10.1f} m "
                  f"{np.median(np.abs(a[s] - b[s])):9.1f} m {np.median(a[s] - b[s]):+8.1f} m"
                  f"   repr: {r}")
    print(f"\n    {'REPRESENTABLE?':22s} {'n':>5} {'corr':>8} {'median err':>12} {'bias':>9}")
    for nm, sel in [("all lidar cells", val), ("truth inside the volume", repr_)]:
        aa, bb = pred[sel], gt[sel]
        print(f"    {nm:22s} {int(sel.sum()):5d} {np.corrcoef(aa, bb)[0, 1]:+8.3f} "
              f"{np.median(np.abs(aa - bb)):10.1f} m {np.median(aa - bb):+8.1f} m")
    print(f"    {int((val & ~repr_).sum())} of {int(val.sum())} checkable cells have their TRUE")
    print("    surface beyond where the ray exits the grid - those are the grid's limit,")
    print("    not the network's error.")

    print(f"\n    {'image rows':22s} {'n':>5} {'pred':>9} {'lidar':>9} {'bias':>9} {'kept':>7}")
    for lo, hi, nm in [(0, 8, "0-7   sky/tops"), (8, 13, "8-12  upper buildings"),
                       (13, 19, "13-18 horizon"), (19, 25, "19-24 mid"),
                       (25, 32, "25-31 near road")]:
        sl = val[lo:hi]
        if sl.sum():
            pr, g = pred[lo:hi][sl], gt[lo:hi][sl]
            ro = repr_[lo:hi]
            extra = (f"   representable {int(ro.sum()):3d}/{int(sl.sum()):3d}, "
                     f"bias {np.median(pred[lo:hi][ro] - gt[lo:hi][ro]):+.1f} m"
                     if ro.sum() else "   none representable")
            print(f"    {nm:22s} {int(sl.sum()):5d} {np.median(pr):7.1f} m {np.median(g):7.1f} m "
                  f"{np.median(pr - g):+7.1f} m {kept[lo:hi].mean() * 100:6.1f}%{extra}")
        else:
            print(f"    {nm:22s} {0:5d} {np.median(pred[lo:hi]):7.1f} m {'-':>9} {'-':>9} "
                  f"{kept[lo:hi].mean() * 100:6.1f}%")
    print("\n    Read the 'repr' column, not the raw bias. Most of the apparent compression")
    print("    at range is lidar hitting surfaces the +-50 m grid cannot hold: 85% of cells")
    print("    beyond 40 m, and 76% of rows 8-12, exit the volume before reaching the wall")
    print("    they hit. Against what IS representable the bias is about -5 m, not -22 m.")

    shown = val & ~hide
    print(f"\n    WHAT THE FIGURE ACTUALLY DRAWS  ({100 * (~hide).mean():.0f}% of the grid)")
    if shown.sum():
        aa, bb = pred[shown], gt[shown]
        print(f"      n={int(shown.sum())}   corr {np.corrcoef(aa, bb)[0, 1]:+.3f}   "
              f"median err {np.median(np.abs(aa - bb)):.1f} m   "
              f"bias {np.median(aa - bb):+.1f} m")
        hd = val & hide
        if hd.sum():
            print(f"      hidden cells, for contrast: n={int(hd.sum())}   median err "
                  f"{np.median(np.abs(pred[hd] - gt[hd])):.1f} m  <- rightly not drawn")

    print(f"\n    within 2 m of lidar: {100 * (np.abs(a - b) < 2).mean():.0f}% of cells")
    print(f"    within 5 m of lidar: {100 * (np.abs(a - b) < 5).mean():.0f}% of cells")
    print("\n    Accurate up close, progressively COMPRESSED with range: the far half of each")
    print("    distribution is masked away, so what survives is biased toward the near peak.")
    print("    Fine for a BEV seed, which is what it is for - do not read it as a depth sensor.")

    if args.figure:
        _figure(args.figure, frame, pred, kept, gt, val, lidar_uvd=uvd, repr_=repr_,
                hide=hide)
        print(f"\n    wrote {args.figure}")
    return 0


def _figure(path, frame, pred, kept, gt, val, lidar_uvd=None, repr_=None, hide=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.asarray(frame.image(frame.cam_order[0]).resize((IW, IH)))
    ext, VM = [0, IW, IH, 0], dict(vmin=0, vmax=50, cmap="turbo")
    fig = plt.figure(figsize=(16.5, 9.6), dpi=110, facecolor="white")
    gs = fig.add_gridspec(3, 2, hspace=0.24, wspace=0.12, height_ratios=[1, 1, 1.15])

    # --- row 1: the two overlays, side by side on the same photo and colour scale ----
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, extent=ext, aspect="equal")
    hid = hide if hide is not None else (kept < 0.06)
    im = ax.imshow(np.ma.masked_where(hid, pred), **VM, extent=ext,
                   aspect="equal", alpha=0.72, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("PREDICTED depth, overlaid\nbare photo = the ray leaves the volume there",
                 fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.030, label="m")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img, extent=ext, aspect="equal")
    if lidar_uvd is not None:
        u, v, d = lidar_uvd
        im = ax.scatter(u, v, c=d, s=5, **VM)
        plt.colorbar(im, ax=ax, fraction=0.030, label="m")
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("GROUND TRUTH, same photo and colour scale\nlidar returns nothing from the sky",
                 fontsize=11)

    # --- row 2: the same two as bare grids, for cell-by-cell comparison --------------
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(np.ma.masked_where(hid, pred), **VM, extent=ext, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("predicted (grid only)", fontsize=10)
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(np.ma.masked_invalid(gt), **VM, extent=ext, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"lidar (grid only) - {int(val.sum())} of {GH * GW} cells", fontsize=10)

    # --- row 3: the quantitative checks ---------------------------------------------
    a, b = pred[val], gt[val]
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(b, a, s=7, alpha=0.35, c="#1f77b4"); ax.plot([0, 55], [0, 55], "k--", lw=1)
    ax.set_xlabel("lidar depth [m]"); ax.set_ylabel("predicted [m]"); ax.set_aspect("equal")
    ax.set_xlim(0, 55); ax.set_ylim(0, 55)
    if repr_ is not None:
        ax.scatter(gt[repr_], pred[repr_], s=7, alpha=0.55, c="#d6336c",
                   label="truth inside the volume")
        ax.legend(fontsize=7, loc="lower right")
        ar, br = pred[repr_], gt[repr_]
        ax.set_title(f"all cells: corr {np.corrcoef(a, b)[0, 1]:+.3f}   |   representable: "
                     f"corr {np.corrcoef(ar, br)[0, 1]:+.3f}\nblue points below the line are "
                     f"mostly surfaces the grid cannot hold", fontsize=10)
    else:
        ax.set_title(f"corr {np.corrcoef(a, b)[0, 1]:+.3f}", fontsize=10)

    ax = fig.add_subplot(gs[2, 1])
    # Only score cells the grid can actually represent; the rest measure the grid.
    sel = repr_ if repr_ is not None else val
    e = np.full_like(gt, np.nan); e[sel] = pred[sel] - gt[sel]
    im = ax.imshow(np.ma.masked_invalid(e), cmap="coolwarm", vmin=-15, vmax=15,
                   extent=ext, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("error on cells the volume CAN represent\n(the rest exit the grid first)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.030, label="m")

    fig.suptitle("DepthNet overlaid on the image and checked against lidar  |  corr +0.879 "
                 "and 1.2 m median error against what the BEV volume can actually hold",
                 fontsize=13)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
