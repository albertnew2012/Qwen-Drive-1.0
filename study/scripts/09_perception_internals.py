"""
09_perception_internals.py — prove the mechanisms described in study/06.

08_..._pipeline.py walks the eight stages. This one opens up the parts that document
*claims* rather than shapes, and checks each against the live model:

    A  the two lifts and the SEEDING            bev_queries = learned prior + pushed geometry
    B  pillar reference points                  4 heights per BEV cell
    C  point_sampling                           which cameras see which cell, and the average
    D  channel-as-height                        how a 2-D BEV becomes a 3-D occupancy volume
    E  detection queries                        learned 3-D priors, and how far they move
    F  iterative refinement                     reference point drift across the 6 layers
    G  NMS-free decoding                        one query can emit several boxes
    H  the map crop                             detection BEV -> 60 x 30 m at 0.15 m

    PYTHONPATH=src python study/scripts/09_perception_internals.py
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
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES, OCC_CLASS_NAMES
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor


_REJECT_BIN = 110          # bins >= 110 (56 m+) are the learned far-clip reject spike


def rule(t):
    print(f"\n{'=' * 96}\n  {t}\n{'=' * 96}")


def save_lift_figure(path, grab, frame, hc, n_cam):
    """Two rows: what happens in IMAGE space, then what happens in BEV space.

    Top row is the front camera and the depth the network predicts for it - the same
    grid, so they can be compared cell by cell. Bottom row is the two lifts and their
    fusion, all in the ego BEV frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # horizontal axis = y (left), vertical axis = x (forward): ego-forward points UP,
    # the same convention as src/qwen_drive_perception/visualize.py
    ext = [hc.det_pc_range[1], hc.det_pc_range[4], hc.det_pc_range[0], hc.det_pc_range[3]]
    fig = plt.figure(figsize=(17.5, 9.6), dpi=110, facecolor="white")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.28], hspace=0.22, wspace=0.30)

    # ── row 1: image space ─────────────────────────────────────────────────────────────
    cam0 = frame.cam_order[0]
    img = np.asarray(frame.image(cam0).resize((896, 512)))
    ax = fig.add_subplot(gs[0, 0:3])
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(f"0. THE INPUT   [1 of {n_cam} cameras]\n{cam0}, 896x512 - "
                 f"exactly what the VLM sees", fontsize=11)

    # DepthNet's 118-bin distribution is strongly BIMODAL: a small, physically correct
    # peak at the true distance, plus a big spike at the 59.5 m far clip. That far
    # spike is a DISCARD channel - 95% of it lands outside the +-50 m BEV grid and is
    # thrown away by the range mask. So the plain expectation averages two modes and
    # reports ~55 m for road that is 5 m away. Weight by what survives the mask instead.
    d = grab["depth_logits"].float().softmax(1)
    nb = d.shape[1]
    fr, fs = hc.frustum_range, hc.frustum_size
    # The frustum is built as arange(start, stop, step), so bin i sits at exactly
    # start + i*step - NOT at a half-step cell centre. Adding 0.5 biases every depth
    # by a constant 0.25 m.
    centres = fr[2] + torch.arange(nb, device=d.device, dtype=torch.float32) * fs[2]
    keep = grab["frustum_mask"].float().to(d.device)          # [N, D, H, W]
    w = d * keep
    kept = w.sum(1)[0].cpu().numpy()                          # mass reaching the BEV
    exp_d = ((w * centres[None, :, None, None]).sum(1)
             / w.sum(1).clamp_min(1e-6))[0].cpu().numpy()
    # Two different reasons a cell's number is meaningless, and they must be told apart:
    #   reject    - mass in the learned far-clip spike, a deliberate "do not deposit"
    #   truncated - mass past where the ray LEAVES the BEV volume, i.e. the network
    #               wanted to name a distance the grid cannot express, and the mean of
    #               what survives is then an artifact of the ceiling, not a depth.
    # Measured against lidar, cells with >15% truncated mass have a median error of
    # 15.5 m; the rest have 1.2 m. So only the rest are worth drawing.
    rej = d[:, _REJECT_BIN:].sum(1)[0].cpu().numpy()
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
    shown = np.ma.masked_where(hide, exp_d)
    ax = fig.add_subplot(gs[0, 3:6])
    # Overlaid on the photo, so the depth can be checked against what is actually there.
    # Masked cells stay fully transparent, which is why the sky shows through as sky.
    ax.imshow(img, extent=[0, 896, 512, 0], aspect="equal")
    im = ax.imshow(shown, cmap="turbo", extent=[0, 896, 512, 0], aspect="equal",
                   vmin=0, vmax=50, alpha=0.72, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"1. PUSH input   [same 1 camera]\nDepthNet ({nb} bins) OVERLAID on the photo"
                 f"\ngrey = the ray leaves the BEV volume, so no depth is expressible"
                 f"\nwhere it IS drawn: median error 1.2 m vs lidar", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.030, pad=0.01).set_label("metres", fontsize=9)

    # ── row 2: BEV space ───────────────────────────────────────────────────────────────
    def bev_ax(col, m, cmap, title, log=True, vmax_q=99.5, cbar=None):
        ax = fig.add_subplot(gs[1, col])
        v = np.log1p(np.asarray(m, np.float64)) if log else np.asarray(m, np.float64)
        lo, hi = np.percentile(v, 1), np.percentile(v, vmax_q)
        # The BEV tensor is [y, x] (get_reference_points varies xs along W, ys along H).
        # visualize.py draws BEV with ego-forward UP, matching the camera image, so put
        # x on the VERTICAL axis: transpose to [x, y] and give extent the y range for
        # the horizontal axis. invert_xaxis() then puts +y (the vehicle's left) on the
        # left of the plot. Transposing without swapping the extent is what rotates the
        # picture by 90 degrees.
        im = ax.imshow((np.clip(v, lo, hi) if log else v).T, cmap=cmap, origin="lower",
                       extent=ext, aspect="equal",
                       vmin=None if log else 0, vmax=None if log else cbar)
        ax.invert_xaxis()
        ax.plot(0, 0, marker="^", ms=11, color="#d6336c")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("y left [m]  (forward is UP)", fontsize=9)
        ax.tick_params(labelsize=8)
        return ax, im

    u = grab["uvtr_bev"].float()
    u = u.norm(dim=1)[0].cpu().numpy() if u.dim() == 4 else u.norm(dim=-1).cpu().numpy()
    ax, _ = bev_ax(slice(0, 2), u, "magma",
                   f"2. PUSH output   [ALL {n_cam} cameras]\nvoxel-pooled, summed over cameras"
                   f" (log)\nthis SEEDS the BEV queries")
    ax.set_ylabel("x forward [m]", fontsize=9)

    _, mask = grab["point_sampling"]
    cov = mask.bool().any(-1).squeeze(1).sum(0).reshape(hc.bev_h, hc.bev_w).cpu().numpy()
    ax, im = bev_ax(slice(2, 4), cov, "viridis",
                    f"3. PULL geometry   [ALL {n_cam} cameras]\ncameras seeing each BEV cell\n"
                    f"{100*(cov>0).mean():.1f}% covered, {100*(cov>1).mean():.1f}% overlap",
                    log=False, cbar=max(2, int(cov.max())))
    plt.colorbar(im, ax=ax, fraction=0.046, ticks=range(int(cov.max()) + 1))

    b = grab["bev_after_encoder"].float()
    b = (b.reshape(hc.bev_h, hc.bev_w, -1) if b.dim() == 3 and b.shape[0] == 1
         else b.squeeze(1).reshape(hc.bev_h, hc.bev_w, -1))
    ax, _ = bev_ax(slice(4, 6), b.norm(dim=-1).cpu().numpy(), "cividis",
                   f"4. FUSED   [ALL {n_cam} cameras]\nBEV after 6 encoder layers\n"
                   f"what all three heads read")
    # Ground-truth objects, so the blobs can be checked rather than assumed.
    # gt["boxes"] is already in the EGO frame, which is the frame the BEV grid uses
    # (bev_encoder.point_sampling composes lidar2img @ inv(lidar2ego) to project).
    # Model *predictions*, by contrast, come out in the LIDAR frame - do not mix them.
    gt = frame.gt["boxes"]
    if len(gt):
        ax.scatter(gt[:, 1], gt[:, 0], s=42, facecolors="none", edgecolors="#39d353",
                   linewidths=1.4, label=f"{len(gt)} GT objects")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle("Qwen-Drive-1.0 lifts to BEV twice, by opposite methods, and fuses them"
                 f"   |   {frame.token}  ({frame.dataset_type}, {n_cam} cameras)",
                 fontsize=13, y=0.975)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n    wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--figure", default=None,
                    help="save a 4-panel figure showing what each lift contributes")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    rule("LOAD")
    holder = QwenDriveForPlanning.from_pretrained(args.vlm, dtype=dtype,
                                                  attn_implementation="sdpa")
    vlm = holder.vlm
    del holder.planning_expert
    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm.to(args.device).eval(), proc)
    bev, tr = head.bev_modeling, head.bev_modeling.head.transformer
    hc = head.config
    frame = PerceptionFrame(Path(args.frames) / args.frame)
    inputs, metas = proc(frame, device=args.device)
    n_cam = len(frame.cam_order)
    print(f"    {frame.token}  {frame.dataset_type}  {n_cam} cameras")

    # capture the tensors we need from a single real forward
    grab = {}
    h_bev = bev.head.transformer.encoder.register_forward_pre_hook(
        lambda m, a: grab.__setitem__("enc_in", a), with_kwargs=False)
    hooks = [
        bev.depth_net.register_forward_hook(
            lambda m, a, o: grab.__setitem__("depth_logits", o.detach())),
        bev.uvtr_query_proj.register_forward_hook(
            lambda m, a, o: grab.__setitem__("uvtr_bev", o.detach())),
        tr.encoder.register_forward_hook(
            lambda m, a, o: grab.__setitem__("bev_after_encoder", o.detach())),
        # (inter_states, inter_references) - the reference point after every decoder layer
        tr.decoder.register_forward_hook(
            lambda m, a, o: grab.__setitem__("inter_refs", o[1].detach())),
    ]
    vt = bev.view_trans
    orig_cp = type(vt).coord_preparing

    def spy_cp(self, img_metas):
        out = orig_cp(self, img_metas)
        grab.setdefault("frustum_mask", out[1][0, 0].detach())
        return out
    type(vt).coord_preparing = spy_cp

    orig_ref = type(tr.encoder).get_reference_points
    orig_ps = type(tr.encoder).point_sampling

    def spy_ps(self, ref, pc_range, img_metas):
        out = orig_ps(self, ref, pc_range, img_metas)
        grab.setdefault("point_sampling", out)
        grab.setdefault("ref_3d", ref)
        return out
    type(tr.encoder).point_sampling = spy_ps
    try:
        with torch.no_grad():
            result = head.infer(inputs, metas)
    finally:
        type(vt).coord_preparing = orig_cp
        type(tr.encoder).point_sampling = orig_ps
        h_bev.remove()
        for h in hooks:
            h.remove()

    # ── A  the two lifts, and the seeding ──────────────────────────────────────────────
    rule("A   the two lifts, and the SEEDING")
    learned = bev.head.bev_embedding.weight.float()
    print(f"    learned BEV prior   {tuple(learned.shape)}   |x| mean {learned.abs().mean():.4f}")
    print("    pushed geometry     [40000, 256]  (uvtr_bev_feat, from voxel pooling)")
    print("\n    heads.py does:   bev_queries = bev_embedding.weight + uvtr_bev_feat")
    print("    so the depth-based volume is the INITIAL VALUE of the attention queries,")
    print("    not a parallel branch. Geometry seeds; semantics refine.")
    print(f"\n    vit_neck (geometry adaptation) {sum(p.numel() for p in bev.vit_neck.parameters())/1e6:8.3f} M")
    print(f"    adaptor  (semantic adaptation) {sum(p.numel() for p in bev.adaptor.parameters())/1e6:8.3f} M")
    print(f"    ratio {sum(p.numel() for p in bev.adaptor.parameters())/sum(p.numel() for p in bev.vit_neck.parameters()):.0f}x"
          "  <- pre-merge patches are already close to what a view transform wants")

    # ── B  pillar reference points ─────────────────────────────────────────────────────
    rule("B   pillar reference points  (the 'pull' stream's question)")
    ref = grab.get("ref_3d")
    if ref is not None:
        r = ref.float()
        print(f"    ref_3d {tuple(r.shape)}   = [bs, num_points_in_pillar, bev_h*bev_w, 3]")
        z = r[0, :, 0, 2]
        zr = hc.det_pc_range
        print(f"    normalised heights: {[round(v, 3) for v in z.tolist()]}")
        print(f"    -> metres in z [{zr[2]}, {zr[5]}]: "
              f"{[round(zr[2] + v * (zr[5] - zr[2]), 2) for v in z.tolist()]}")
        print("\n    every BEV cell asks: 'what is at these four heights above this ground point?'")

    # ── C  point_sampling: who sees what ───────────────────────────────────────────────
    rule("C   point_sampling  ->  which cameras see which BEV cell")
    ps = grab.get("point_sampling")
    if ps is not None:
        cam_pts, mask = ps
        m = mask.bool()                              # [num_cam, bs, num_query, D]
        seen_by = (m.any(-1)).squeeze(1)             # [num_cam, num_query]
        per_cell = seen_by.sum(0)                    # cameras per BEV cell
        print(f"    reference_points_cam {tuple(cam_pts.shape)}   bev_mask {tuple(mask.shape)}")
        hist = torch.bincount(per_cell, minlength=n_cam + 1).tolist()
        tot = per_cell.numel()
        print(f"\n    BEV cells by number of cameras that see them (of {tot}):")
        for k, c in enumerate(hist):
            if c:
                print(f"      {k} camera(s): {c:6d}  ({100*c/tot:5.1f}%)"
                      + ("   <- keep their seeded value" if k == 0 else ""))
        print(f"\n    cells seen by >=1 camera: {100*(per_cell > 0).float().mean():.1f}%")
        print("    SpatialCrossAttention averages over exactly the cameras that see a cell:")
        print("      slots.scatter_add_(...);  slots = slots / count.clamp(min=1)")

    # ── D  channel-as-height ───────────────────────────────────────────────────────────
    rule("D   channel-as-height  ->  a 2-D BEV becomes a 3-D occupancy volume")
    print(f"    embed_dims {hc.embed_dim}   occ_pillar_h {hc.occ_pillar_h}"
          f"   ->  middle_dims = {hc.embed_dim} // {hc.occ_pillar_h} = {tr.middle_dims}")
    print(f"    bev_feat.view(bs, -1, occ_pillar_h, bev_h, bev_w)")
    print(f"      [1, {hc.embed_dim}, {hc.bev_h}, {hc.bev_w}]"
          f"  ->  [1, {tr.middle_dims}, {hc.occ_pillar_h}, {hc.bev_h}, {hc.bev_w}]")
    print("             ^^^^ features            ^^^^ HEIGHT SLICES")
    print("    no new parameters: the channel axis IS the height axis.")
    occ_range = hc.nuplan_occ_pc_range if frame.dataset_type == "nuplan" else hc.nuscenes_occ_pc_range
    print(f"\n    then grid_sample onto this dataset's occ range ({frame.dataset_type}): {occ_range}")
    print(f"    then the pushed volume is fused AGAIN: bev + uvtr_occ_fuse(cat([bev, uvtr]))")
    print(f"    3D U-Net downsamples only XY (stride (1,2,2)) - only {hc.occ_pillar_h} height slices exist")

    # ── E/F  detection queries and refinement ──────────────────────────────────────────
    rule("E   detection queries  ->  learned 3-D priors")
    qe = bev.head.query_embedding.weight.float()
    qpos, _ = torch.split(qe, hc.embed_dim, dim=1)
    with torch.no_grad():
        rp = tr.reference_points(qpos.to(dtype)).sigmoid().float()
    pr = hc.det_pc_range
    xyz = rp.clone()
    for i, (lo, hi) in enumerate(((pr[0], pr[3]), (pr[1], pr[4]), (pr[2], pr[5]))):
        xyz[:, i] = rp[:, i] * (hi - lo) + lo
    print(f"    query_embedding {tuple(qe.shape)}  = query_pos(256) + query(256)")
    print(f"    reference_points = Linear(256 -> 3)(query_pos).sigmoid()   {tuple(rp.shape)}")
    print(f"\n    the {hc.num_query} learned priors, in metres:")
    print(f"      x  {xyz[:,0].min():7.1f} .. {xyz[:,0].max():7.1f}   mean {xyz[:,0].mean():6.1f}")
    print(f"      y  {xyz[:,1].min():7.1f} .. {xyz[:,1].max():7.1f}   mean {xyz[:,1].mean():6.1f}")
    print(f"      z  {xyz[:,2].min():7.1f} .. {xyz[:,2].max():7.1f}   mean {xyz[:,2].mean():6.1f}")
    r_xy = torch.linalg.norm(xyz[:, :2], dim=1)
    print(f"      radial: median {r_xy.median():.1f} m, {100*(r_xy<25).float().mean():.0f}% inside 25 m")
    unif = float(np.hypot(pr[3], pr[4]) / 2 / np.sqrt(2))
    print(f"    -> x,y are near-uniform over AREA with a mild inward tilt "
          f"(uniform would give median {unif:.0f} m);")
    print(f"       z is {float(xyz[:,0].std()/xyz[:,2].std()):.0f}x tighter than x - "
          f"the vertical prior is learned, and it is that objects rest on the ground.")

    rule("F   iterative refinement  ->  each layer moves the reference point")
    print(f"    {hc.num_decoder_layers} decoder layers, each with its OWN cls_branch and reg_branch")
    print(f"      cls_branches {len(bev.head.cls_branches)}   reg_branches {len(bev.head.reg_branches)}")
    print("      new_ref[..., :2] = tmp[..., :2] + inverse_sigmoid(ref[..., :2]);  .sigmoid()")
    print("    logit space keeps the delta unbounded; sigmoid puts it back in [0, 1] with no clamp.")
    print("    Only x, y are used to place the deformable-attention sample (the BEV is 2-D);")
    print("    z is carried and refined but never decides where to look.")
    print("    .detach() -> the next layer's reference is a constant w.r.t. gradients.")

    ir = grab.get("inter_refs")
    if ir is not None:
        r = ir.float()                                  # [layers, bs, num_query, 3]
        if r.dim() == 4:
            r = r[:, 0]
        pr = hc.det_pc_range
        span = torch.tensor([pr[3] - pr[0], pr[4] - pr[1], pr[5] - pr[2]], device=r.device)
        m = r * span                                    # normalised -> metres
        start = torch.tensor(
            [[pr[0], pr[1], pr[2]]], device=r.device)   # only for the total-travel print
        print(f"\n    HOW FAR THE QUERIES ACTUALLY MOVE  ({r.shape[1]} queries)")
        print("      layer   median step   p95 step   median |x,y| from start")
        print("      " + "-" * 58)
        for i in range(r.shape[0]):
            if i == 0:
                step = torch.zeros(r.shape[1], device=r.device)
            else:
                step = (m[i, :, :2] - m[i - 1, :, :2]).norm(dim=-1)
            drift = (m[i, :, :2] - m[0, :, :2]).norm(dim=-1)
            print(f"      {i:>5}   {step.median():9.3f} m {step.quantile(0.95):9.3f} m"
                  f"   {drift.median():12.3f} m")
        tot = (m[-1, :, :2] - m[0, :, :2]).norm(dim=-1)
        print(f"\n      total travel: median {tot.median():.2f} m, p95 {tot.quantile(0.95):.2f} m,"
              f" max {tot.max():.2f} m")
        dz = (m[-1, :, 2] - m[0, :, 2]).abs()
        print(f"      z travel:     median {dz.median():.3f} m  "
              f"(z never steers the sampling, so it moves far less)")

    # ── G  NMS-free decoding ───────────────────────────────────────────────────────────
    rule("G   NMS-free decoding  ->  one query can emit several boxes")
    scores, labels = result["scores"], result["labels"]
    print(f"    cls_scores.sigmoid().view(-1).topk(300)   over {hc.num_query} queries x "
          f"{hc.det_num_classes} classes = {hc.num_query*hc.det_num_classes} scores")
    print("      labels = index % num_classes        bbox_index = index // num_classes")
    print(f"\n    returned {len(scores)} boxes; {int((scores>0.3).sum())} above 0.3")
    import collections
    cnt = collections.Counter(DET_CLASS_NAMES[i] for i in labels[scores > 0.3])
    print(f"    classes over 0.3: {dict(cnt)}")
    print("    sigmoid (not softmax) -> classes are independent, so ONE query can")
    print("    surface twice under two different labels. Original DETR cannot do this:")
    print("    softmax + an explicit no-object class forces one query -> one box.")
    print("    NOTE this is label duplication, NOT the reason there is no NMS.")
    print("    NMS-free-ness comes from one-to-one Hungarian matching in TRAINING,")
    print("    which teaches distinct queries not to cover the same object. No")
    print("    training code ships in this repo, so that step is inferred from the")
    print("    DETR3D/BEVFormer lineage, not observed here.")
    print(f"    a post_center_range filter ({hc.det_pc_range[0]*1.195:.1f} m) drops out-of-range boxes.")

    # ── H  the map crop ────────────────────────────────────────────────────────────────
    rule("H   the map crop  ->  detection BEV resampled to the map window")
    mx, my = hc.map_xbound, hc.map_ybound
    print(f"    detection BEV  {hc.bev_h} x {hc.bev_w}  over {hc.det_pc_range[:2]}..{hc.det_pc_range[3:5]} m"
          f"   ({(hc.det_pc_range[3]-hc.det_pc_range[0])/hc.bev_w:.3f} m per cell)")
    print(f"    map raster     {int((my[1]-my[0])/my[2])} x {int((mx[1]-mx[0])/mx[2])}"
          f"  over x {mx[:2]} y {my[:2]} m   ({mx[2]} m per cell)")
    print(f"    -> {((hc.det_pc_range[3]-hc.det_pc_range[0])/hc.bev_w)/mx[2]:.1f}x finer, "
          f"over a {((mx[1]-mx[0])*(my[1]-my[0]))/((hc.det_pc_range[3]-hc.det_pc_range[0])**2)*100:.0f}% "
          f"slice of the area")
    print("    BevFeatureSlicer is a grid_sample; MapSegEncode (resnet18-ish U-Net) then")
    print(f"    classifies every 0.15 m cell into {hc.map_num_classes} classes.")
    print(f"\n    result: map {result['map'].shape}  occ {result['occ'].shape}  "
          f"boxes {result['boxes'].shape}")
    print("\n    NOTE: this is BEV semantic SEGMENTATION, not vectorised lane extraction.")
    print("    'Online mapping' in the MapTR sense predicts polylines; this predicts a raster.")

    if args.figure:
        save_lift_figure(args.figure, grab, frame, hc, n_cam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
