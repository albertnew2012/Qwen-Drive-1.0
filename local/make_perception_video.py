#!/usr/bin/env python
"""Render the perception pipeline as a walkthrough video, stage by stage.

`render_frame` draws only the FINAL result. This draws what happens on the way there,
using the intermediate tensors dumped by

    python study/scripts/08_qwen_drive_perception_pipeline.py --dump outputs/perception_stages.npz

Eight shots, in pipeline order:

    1  the camera ring, raw input
    2  the TWO taps        ViT pre-merge (geometry) vs LLM last layer (semantics)
    3  the FPN adaptors    0.853 M for geometry, 33.663 M for semantics
    4  DepthNet            expected depth per pixel, in metres - the interpretable one
    5  the frustum         which of the 1.27 M sample points land in range
    6  the voxel volume    collapsed to BEV density
    7  BEV queries         after the 1x1 conv over 16 z-slices
    8  the three outputs   detections / occupancy / map, each against ground truth

    PYTHONPATH=src python local/make_perception_video.py
"""
from __future__ import annotations

import argparse, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from qwen_drive_perception.configuration_perception import (
    DET_BOX_COLORS, DET_CLASS_NAMES, MAP_PALETTE, OCC_CLASS_NAMES, OCC_PALETTE,
)

W, H = 1920, 1080
BG, FG, DIM, ACCENT, RULE = (244, 245, 247), (32, 36, 42), (104, 112, 124), (198, 104, 8), (206, 211, 218)
GOOD = (22, 122, 72)


def _font(sz, bold=False):
    for n in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
              "/usr/share/fonts/truetype/dejavu/" + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def brighten(img, target=112.0):
    a = np.asarray(img, np.float32)
    if a.mean() >= target:
        return np.asarray(img, np.uint8)
    g = float(np.clip(np.log(target / 255) / np.log(max(a.mean(), 1) / 255), .3, 1))
    return np.clip(255 * (a / 255) ** g, 0, 255).astype(np.uint8)


def shot(title, subtitle, panel_rgb, caption_lines, note=None):
    """One frame: header, a rendered panel, and an explanation."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((58, 16), title, font=_font(31, True), fill=ACCENT)
    d.text((58, 54), subtitle, font=_font(20), fill=DIM)
    p = Image.fromarray(panel_rgb)
    box_w, box_h = W - 116, H - 120 - 30 * len(caption_lines) - (34 if note else 0)
    sc = min(box_w / p.width, box_h / p.height)
    p = p.resize((max(1, int(p.width * sc)), max(1, int(p.height * sc))), Image.LANCZOS)
    img.paste(p, ((W - p.width) // 2, 92))
    y = 92 + p.height + 16
    for ln in caption_lines:
        d.text((58, y), ln, font=_font(23), fill=FG); y += 30
    if note:
        d.line([(58, y + 4), (W - 58, y + 4)], fill=(200, 120, 40), width=2)
        d.text((58, y + 12), note, font=_font(21, True), fill=(176, 96, 16))
    return np.array(img)


def fig_to_rgb(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def grid_of_maps(maps, titles, cmap="viridis", cbar_label=None, vmin=None, vmax=None,
                 cols=3, size=3.0):
    """A row-major grid of 2-D heatmaps, one per camera."""
    n = len(maps)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size * 1.6, rows * size),
                             dpi=110, facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    im = None
    rgb = maps[0].ndim == 3 and maps[0].shape[-1] == 3
    if not rgb and vmin is None:
        allv = np.concatenate([m.ravel() for m in maps])
        vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98)
    for i, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        if i < n:
            im = (ax.imshow(maps[i], aspect="auto") if rgb
                  else ax.imshow(maps[i], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"))
            ax.set_title(titles[i], fontsize=10, color="#2f3437", pad=3)
            for sp in ax.spines.values():
                sp.set_color("#b0b7bf")
        else:
            ax.axis("off")
    if cbar_label and im is not None and not rgb:
        fig.subplots_adjust(right=0.90)
        cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
        cb = fig.colorbar(im, cax=cax); cb.set_label(cbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=9)
    fig.tight_layout(rect=(0, 0, 0.90 if cbar_label else 1, 1), pad=0.5)
    return fig_to_rgb(fig)


def _boxes_to_ego(boxes, lidar2ego):
    """Predictions come out in the LIDAR frame; the BEV grid is in the EGO frame.

    For nuScenes those differ by a ~90 degree yaw, so skipping this rotates every box
    a quarter turn against the heatmap it is drawn on. Ground truth in gt.npz is
    already in ego coordinates and must NOT be passed through this.
    """
    if boxes is None or not len(boxes) or lidar2ego is None:
        return boxes
    L = np.asarray(lidar2ego, np.float64)
    if L.ndim == 3:
        L = L[0]
    out = np.array(boxes, np.float64, copy=True)
    xyz = np.c_[out[:, :3], np.ones(len(out))]
    out[:, :3] = (L @ xyz.T).T[:, :3]
    yaw = np.arctan2(L[1, 0], L[0, 0])
    if out.shape[1] > 6:
        out[:, 6] += yaw
    if out.shape[1] > 8:                              # velocity is a direction too
        v = out[:, 7:9] @ np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]).T
        out[:, 7:9] = v
    return out


def _box_corners(b):
    """The four BEV corners of an ego-frame box, returned as (y, x) for the screen.

    Built explicitly rather than via Rectangle(angle=...) so there is no ambiguity
    about which way a rotation goes once the x axis is inverted.
    ``b`` is [x, y, z, w, l, h, yaw, ...]: length ``l`` runs along the heading.
    """
    x, y, w, l, yaw = b[0], b[1], b[3], b[4], b[6]
    ct, st = np.cos(yaw), np.sin(yaw)
    along = np.array([ct, st]) * (l / 2.0)      # heading, in (x, y)
    across = np.array([-st, ct]) * (w / 2.0)    # perpendicular
    c = np.array([x, y])
    pts = [c + along + across, c + along - across, c - along - across, c - along + across]
    pts = np.array(pts)
    return np.c_[pts[:, 1], pts[:, 0]]          # -> (y, x) = (horizontal, vertical)


def bev_heat(m, title, cmap="magma", extent=None, boxes=None, labels=None, scores=None,
             gt_boxes=None, thr=0.3, lidar2ego=None):
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=110, facecolor="white")
    v = np.log1p(np.asarray(m, np.float64))          # a few cells near ego dominate linearly
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    # The BEV tensor is [y, x]. visualize.py draws BEV with ego-forward UP, so put x on
    # the vertical axis: transpose, and give the extent the y range horizontally.
    # invert_xaxis() then places +y (the vehicle's left) on the left, matching
    # "right is -Y, up is +X" in src/qwen_drive_perception/visualize.py.
    boxes = _boxes_to_ego(boxes, lidar2ego)
    ax.imshow(np.clip(v, lo, hi).T, cmap=cmap, origin="lower",
              extent=extent or [-51.2, 51.2, -51.2, 51.2])
    ax.invert_xaxis()
    if boxes is not None and len(boxes):
        k = scores > thr
        for b, l in zip(boxes[k], labels[k]):
            c = np.array(DET_BOX_COLORS[DET_CLASS_NAMES[l]]) / 255
            ax.add_patch(plt.Polygon(_box_corners(b), closed=True, fill=False,
                                     edgecolor=c, lw=1.6))
    if gt_boxes is not None and len(gt_boxes):
        for b in gt_boxes:
            ax.add_patch(plt.Polygon(_box_corners(b), closed=True, fill=False,
                                     edgecolor=(0.23, 0.69, 0.29), lw=1.2, ls="--"))
    ax.plot(0, 0, marker="^", ms=11, color="#d6336c")
    ax.set_title(title, fontsize=11, color="#2f3437")
    ax.set_xlabel("y left [m]  (forward is UP)", fontsize=9)
    ax.set_ylabel("x forward [m]", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    return fig_to_rgb(fig)


def paint(grid, palette):
    return np.asarray(palette, np.uint8)[np.asarray(grid, np.int64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="outputs/perception_stages.npz")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--output", default="outputs/perception_walkthrough.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--hold", type=float, default=4.5)
    args = ap.parse_args()

    z = np.load(args.stages, allow_pickle=True)
    cams = [str(c) for c in z["cam_order"]]
    token, ds = str(z["token"]), str(z["dataset_type"])
    sub = f"frame {token}   ({ds}, {len(cams)} cameras, 896x512 each)"
    n = len(cams)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(args.output)],
        stdin=subprocess.PIPE)
    nf = 0

    def emit(arr, seconds=None):
        nonlocal nf
        b = np.ascontiguousarray(arr, np.uint8).tobytes()
        for _ in range(int((seconds or args.hold) * args.fps)):
            proc.stdin.write(b); nf += 1

    # ── title ──────────────────────────────────────────────────────────────────────────
    t = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(t)
    for txt, fnt, dy, col in (("BEV Perception, stage by stage", _font(70, True), 380, FG),
                              ("Qwen-Drive-1.0  ·  one VLM forward, two taps", _font(34), 480, DIM),
                              (sub, _font(24), 545, ACCENT)):
        w = d.textbbox((0, 0), txt, font=fnt)[2]
        d.text(((W - w) // 2, dy), txt, font=fnt, fill=col)
    d.line([(W // 2 - 240, 530), (W // 2 + 240, 530)], fill=RULE, width=3)
    emit(np.array(t), 2.6)

    # ── 1  the camera ring ─────────────────────────────────────────────────────────────
    ims = [brighten(np.asarray(Image.open(Path(args.frames) / token / "images" / f"{c}.jpg")
                               .convert("RGB").resize((448, 256)))) for c in cams]
    cols = 3 if n <= 6 else 4
    rows = int(np.ceil(n / cols))
    ring = Image.new("RGB", (cols * 452, rows * 282), BG)
    dr = ImageDraw.Draw(ring)
    for i, (c, im) in enumerate(zip(cams, ims)):
        x, y = (i % cols) * 452, (i // cols) * 282
        ring.paste(Image.fromarray(im), (x + 2, y + 24))
        dr.text((x + 4, y + 2), c, font=_font(19, True), fill=DIM)
    emit(shot("STEP 1   the camera ring", sub, np.array(ring), [
        f"{n} cameras, resized to 896x512, patchified at 16 px and merged 2x2.",
        f"{n} x 448 = {n*448} image tokens reach the language model.",
    ]))

    # ── 2  the two taps ────────────────────────────────────────────────────────────────
    vit, llm = z["tap_vit"], z["tap_llm"]
    emit(shot("STEP 2   TAP 1 of 2  ·  GEOMETRY", sub,
              grid_of_maps(list(vit), cams, cols=cols),
              [f"ViT PRE-merge patches, {vit.shape[1]}x{vit.shape[2]} per camera, dim 1024, shown as the top-3 principal components as RGB.",
               "Captured by a forward hook on visual.merger, then merger.norm applied by hand.",
               "These never entered the language model - this is what gets unprojected."]))
    emit(shot("STEP 2   TAP 2 of 2  ·  SEMANTICS", sub,
              grid_of_maps(list(llm), cams, cols=cols),
              [f"LLM LAST layer, {llm.shape[1]}x{llm.shape[2]} per camera, dim 2560, "
               f"2x coarser than the geometry tap - same PCA-to-RGB projection.",
               "Image-token rows only, after language_model.norm.",
               "The last layer needs no sweep to choose; a single-tap head has to search for one."]))

    # ── 3  the FPN adaptors ────────────────────────────────────────────────────────────
    lv = [z[f"fpn_llm_{i}"][0] for i in range(4) if f"fpn_llm_{i}" in z]
    emit(shot("STEP 3   the FPN adaptors", sub,
              grid_of_maps(lv + [z["fpn_vit_0"][0]],
                           [f"adaptor L{i}  scale {s}" for i, s in enumerate((4.0, 2.0, 1.0, 0.5))]
                           + ["vit_neck  scale 1.0"], cols=3),
              ["Semantics get FOUR scales (33.663 M params) - the deformable attention "
               "samples all of them.",
               "Geometry gets ONE (0.853 M) - it only has to be unprojected.",
               "39x less adaptation, because pre-merge patches are already close to what a "
               "view transform wants."]))

    # ── 4  DepthNet ────────────────────────────────────────────────────────────────────
    dep = z["depth_expected_m"]
    emit(shot("STEP 4   DepthNet  ·  expected depth per pixel", sub,
              grid_of_maps(list(dep), cams, "turbo", "metres",
                           vmin=float(dep.min()), vmax=float(dep.max()), cols=cols),
              [f"118 depth bins from 1 m to 60 m, softmax per pixel; shown here as the "
               f"expectation.",
               f"range {dep.min():.1f} - {dep.max():.1f} m, mean {dep.mean():.1f} m.",
               "This is the only place the model commits to metric depth from a single view."]))

    # ── 5  the frustum ─────────────────────────────────────────────────────────────────
    fr = z["frustum_valid"]
    emit(shot("STEP 5   the frustum  ·  what lands in range", sub,
              grid_of_maps(list(fr), cams, "magma", "fraction in range",
                           vmin=0, vmax=1, cols=cols),
              [f"Every (pixel, depth bin) pair is unprojected with inv(lidar2img), brought to "
               f"the ego frame, and quantised.",
               "Vertical axis is the depth bin, horizontal is image column.",
               "52.6 % of 1.27 M sample points fall inside the 102.4 x 102.4 x 10.4 m volume."]))

    # ── 6-7  voxel volume and BEV queries ──────────────────────────────────────────────
    emit(shot("STEP 6   voxel pooling  ·  BEV density", sub,
              bev_heat(z["voxel_bev_density"], "voxel volume, log density over channels and height"),
              ["out[b, cam, x, y, z, c] += feats[img, c, h, w] * depth[img, d, h, w]",
               "One 200x200x16 volume PER camera (1.88 GiB in bf16) - then summed over cameras.",
               "Multi-view fusion is a SUM in voxel space, not attention. That is why the "
               "geometry stream is cheap."]))
    emit(shot("STEP 7   BEV queries", sub,
              bev_heat(z["bev_tokens_norm"], "BEV tokens after the 1x1 conv"),
              ["16 z-slices x 256 channels collapsed by a 1x1 conv -> 40 000 BEV cells x 256.",
               "These initialise the BEVFormer encoder's queries, which then cross-attend the "
               "SEMANTIC stream.",
               "Geometry sets where to look; semantics say what is there."]))

    # ── 8  the three outputs ───────────────────────────────────────────────────────────
    boxes, scores, labels = z["boxes"], z["scores"], z["labels"]
    keep = scores > 0.3
    emit(shot("STEP 8   output 1 of 3  ·  3D detection", sub,
              bev_heat(z["voxel_bev_density"], "predicted boxes (solid) vs ground truth (dashed)",
                       boxes=boxes, labels=labels, scores=scores, gt_boxes=z["gt_boxes"],
                       lidar2ego=z["lidar2ego"]),
              [f"900 queries -> 300 boxes, NMS-free; {int(keep.sum())} above 0.3 against "
               f"{len(z['gt_labels'])} ground-truth objects.",
               "[x, y, z, w, l, h, yaw, vx, vy] in the LIDAR frame, z at the box bottom -",
               "rotated into the ego frame here, because that is the frame the BEV uses.",
               "Boxes are coloured by class; ego is the pink triangle at the origin."]))

    occ, gto = z["occ"], z["gt_occ"]
    E = len(OCC_CLASS_NAMES) - 1
    both = (occ != E) & (gto != E)
    agree = float((occ[both] == gto[both]).mean()) if both.sum() else float("nan")
    top = lambda g: paint(np.where((g != E).any(2), g.argmax(2) * 0 +
                                   np.take_along_axis(g, (g != E).argmax(2)[..., None], 2)[..., 0],
                                   E), OCC_PALETTE)
    pair = np.concatenate([np.pad(top(occ), ((0, 0), (0, 6), (0, 0)), constant_values=255),
                           top(gto)], axis=1)
    emit(shot("STEP 8   output 2 of 3  ·  semantic occupancy", sub,
              np.repeat(np.repeat(pair, 3, 0), 3, 1),
              [f"200 x 200 x 16 voxels, {len(OCC_CLASS_NAMES)} classes; topmost non-empty "
               f"class shown. Prediction left, ground truth right.",
               f"{100*(occ != E).mean():.2f} % of voxels non-empty; "
               f"{agree:.3f} class agreement where both are occupied.",
               "Occupancy catches what boxes cannot: kerbs, vegetation, arbitrary geometry."]))

    mp, gtm = z["map"], z["gt_map"]
    acc = float((mp == gtm).mean())
    mpair = np.concatenate([np.pad(paint(mp, MAP_PALETTE), ((0, 6), (0, 0), (0, 0)),
                                   constant_values=255), paint(gtm, MAP_PALETTE)], axis=0)
    emit(shot("STEP 8   output 3 of 3  ·  BEV map segmentation", sub,
              np.repeat(np.repeat(mpair, 2, 0), 2, 1),
              [f"200 x 400 raster, 60 m x 30 m at 0.15 m, 6 classes. Prediction top, "
               f"ground truth bottom.",
               f"pixel accuracy {acc:.3f}.",
               "Drivable surface, road line, road edge, crosswalk, walkway - the static scene."],
              note=None))

    # ── closing ────────────────────────────────────────────────────────────────────────
    t = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(t)
    d.text((110, 90), "One forward pass, three outputs", font=_font(52, True), fill=FG)
    d.line([(110, 168), (W - 110, 168)], fill=RULE, width=3)
    rows_ = [("Shared VLM (frozen)", "4.5393 B"), ("BEV perception head", "0.1251 B"),
             ("  geometry adaptation (vit_neck)", "0.853 M"),
             ("  semantic adaptation (adaptor)", "33.663 M"),
             ("3D detection", f"{int(keep.sum())} boxes / {len(z['gt_labels'])} GT"),
             ("semantic occupancy", f"{agree:.3f} class agreement"),
             ("BEV map segmentation", f"{acc:.3f} pixel accuracy"),
             ("inference, RTX 3090 bf16", "~2 s per frame")]
    y = 220
    for k, v in rows_:
        d.text((140, y), k, font=_font(27, k.startswith(("3D", "sem", "BEV", "Shared", "inf"))),
               fill=FG if not k.startswith("  ") else DIM)
        d.text((1180, y), v, font=_font(27, True), fill=GOOD if "/" in v or "." in v else FG)
        y += 46
    d.text((110, H - 76), "reproduce:  python study/scripts/08_qwen_drive_perception_pipeline.py "
                          "--dump outputs/perception_stages.npz", font=_font(20), fill=DIM)
    emit(np.array(t), 6.0)

    proc.stdin.close(); proc.wait()
    print(f"wrote {args.output}  ({nf} frames, {nf/args.fps:.1f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
