"""
08_qwen_drive_perception_pipeline.py — the BEV perception head, stage by stage.

The companion to 07_qwen_drive_pipeline.py. Same idea: every intermediate tensor between the
camera ring and the three outputs is printed with its shape, so you can put a breakpoint
anywhere and see what the stage actually receives.

    STEP 1  frame -> prompt                 6 or 8 cameras at 896x512
    STEP 2  ONE VLM forward, TWO taps       geometry from the ViT, semantics from the LLM
    STEP 3  per-camera grids                undo the merger's block ordering
    STEP 4  FPN adaptors                    0.853 M for geometry, 33.663 M for semantics
    STEP 5  DepthNet                        a 118-bin depth distribution per pixel
    STEP 6  frustum -> ego voxel indices    unproject through inv(lidar2img), then inv(lidar2ego)
    STEP 7  voxel pooling                   scatter features x depth into 200x200x16
    STEP 8  BEVFormer encoder + 3 heads     300 boxes / occupancy / map raster

THE IDEA WORTH STEALING is STEP 2. The head does not read one hidden layer. It reads:

  * the ViT's PRE-MERGE patch grid (32 x 56 per camera, dim 1024) — full spatial resolution,
    never passed through the language model. This is the GEOMETRY tap, and it is what gets
    unprojected into a frustum.
  * the LLM's FINAL hidden states (16 x 28 per camera, dim 2560) — 2x-downsampled and
    semantically abstract. This is the SEMANTICS tap, and it is what BEV queries attend into.

Compare with a single-tap head (e.g. alpamayo1.5/perception), which has to find one layer by
sweep and gets geometry that has already been through 24 layers of language modelling. The
parameter split tells you how much adaptation each stream needs: 0.853 M vs 33.663 M.

MEMORY NOTE: this is a walkthrough, so it deliberately keeps intermediates alive for
inspection - which a normal inference pass would free immediately. Two consequences were
learned the hard way on a 24 GB card that also drives a desktop:

  * `show()` reduces in 4 M-element chunks. Printing |x| mean of the voxel volume with a
    plain `t.float().abs().mean()` allocates a 3.66 GiB fp32 copy of a 1.83 GiB tensor.
  * STEP 8 continues from STEP 7's tensors rather than calling `head.infer()`, which would
    re-run the VLM and the whole view transform a second time.

CPU NOTE: two CUDA kernels back this head. `ms_deform_attn_bf16` has a torch fallback in the
upstream repo; `voxel_pool_depth` did not, and one was added in
src/qwen_drive_perception/ops/__init__.py (verified exact — local/test_voxel_pool_cpu.py).
So this runs on CPU, in fp32, which is also the dtype the head ships in.

Run:

    PYTHONPATH=src python study/scripts/08_qwen_drive_perception_pipeline.py
    PYTHONPATH=src python study/scripts/08_qwen_drive_perception_pipeline.py --frame 4d0d1ccbb1035a90
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from qwen_drive import QwenDriveForPlanning  # noqa: E402
from qwen_drive_perception import QwenDrivePerception  # noqa: E402
from qwen_drive_perception.configuration_perception import (  # noqa: E402
    DET_CLASS_NAMES, MAP_CLASS_NAMES, OCC_CLASS_NAMES,
)
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 96}\n  {title}\n{'=' * 96}")


_CHUNK = 1 << 22          # 4 M elements -> 16 MB as fp32


def abs_mean(t: torch.Tensor) -> float:
    """Mean |x|, without ever materialising an fp32 copy of the whole tensor.

    A diagnostic must never be the thing that runs out of memory. The voxel volume here
    is [1, 6, 256, 16, 200, 200] = 983 M elements: `t.float()` on that allocates 3.66 GiB,
    which is exactly what used to OOM this script on a 24 GB card.
    """
    flat = t.detach().reshape(-1)
    n = flat.numel()
    if n <= _CHUNK:
        return float(flat.float().abs().mean())
    total = 0.0
    for i in range(0, n, _CHUNK):
        total += float(flat[i : i + _CHUNK].float().abs().sum())
    return total / n


def show(name: str, t) -> None:
    if torch.is_tensor(t):
        mib = t.numel() * t.element_size() / 2**20
        print(f"    {name:<40} {str(tuple(t.shape)):<28} "
              f"{str(t.dtype).replace('torch.',''):<9} {mib:8.1f} MiB  "
              f"|x| mean {abs_mean(t):.4f}")
    elif isinstance(t, np.ndarray):
        print(f"    {name:<40} {str(t.shape):<28} {str(t.dtype):<9} "
              f"{t.nbytes/2**20:8.1f} MiB  |x| mean {np.abs(t).mean():.4f}")
    else:
        print(f"    {name:<40} {t}")


def free_vram() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram(tag: str = "") -> None:
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.max_memory_allocated() / 2**30
        print(f"    [vram] {a:5.2f} GiB allocated, {r:5.2f} GiB peak   {tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--render", default=None, help="write the summary PNG here")
    ap.add_argument("--dump", default=None,
                    help="save each stage's tensors (reduced to 2-D maps) as an .npz, for "
                         "local/make_perception_video.py")
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    dtype = getattr(torch, args.dtype)

    stages: dict = {}                      # 2-D reductions for the walkthrough video

    def keep_pca(name, t):
        """Project a [N, C, H, W] feature map onto its top-3 principal components -> RGB.

        A channel-norm heatmap of these is visually pure noise: a handful of outlier tokens
        own the whole dynamic range. PCA over the channel axis is the standard way to see
        whether a feature map carries spatial structure.
        """
        if args.dump is None:
            return
        with torch.no_grad():
            x = t.float()
            N, C, Hh, Ww = x.shape
            flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            flat = flat - flat.mean(0, keepdim=True)
            # randomised low-rank SVD is plenty for 3 components
            q = torch.linalg.qr(flat.T @ torch.randn(flat.shape[0], 8, device=flat.device,
                                                     dtype=flat.dtype))[0]
            _, _, v = torch.linalg.svd((flat @ q).T @ flat, full_matrices=False)
            comp = flat @ v[:3].T                      # [N*H*W, 3]
            lo = torch.quantile(comp, 0.02, dim=0, keepdim=True)
            hi = torch.quantile(comp, 0.98, dim=0, keepdim=True)
            comp = ((comp - lo) / (hi - lo + 1e-6)).clamp(0, 1)
            stages[name] = comp.reshape(N, Hh, Ww, 3).cpu().numpy().astype("float32")

    def keep(name, t, how="norm"):
        """Reduce a feature tensor to a small 2-D map so it can be visualised."""
        if args.dump is None:
            return
        with torch.no_grad():
            if how == "norm":                                   # [..., C, H, W] -> [H, W]
                v = t.float().flatten(0, t.dim() - 3) if t.dim() > 3 else t.float()
                v = v.norm(dim=0) if v.dim() == 3 else v
            else:
                v = t.float()
            stages[name] = v.detach().cpu().numpy().astype("float32")

    rule(f"LOAD   device={args.device}  dtype={args.dtype}")
    holder = QwenDriveForPlanning.from_pretrained(args.vlm, dtype=dtype,
                                                  attn_implementation="sdpa")
    vlm = holder.vlm
    del holder.planning_expert                      # the planner plays no part here
    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm.to(args.device).eval(), processor)
    bev = head.bev_modeling
    hc = head.config
    print(f"    VLM         {sum(p.numel() for p in vlm.parameters())/1e9:.4f} B  (shared, frozen)")
    print(f"    perception  {sum(p.numel() for p in head.parameters())/1e6:.3f} M")
    print(f"    adaptor (LLM tap)  {sum(p.numel() for p in bev.adaptor.parameters())/1e6:8.3f} M")
    print(f"    vit_neck (ViT tap) {sum(p.numel() for p in bev.vit_neck.parameters())/1e6:8.3f} M"
          f"   <- 39x less adaptation for the geometry stream")
    print(f"    BEV grid {hc.bev_h}x{hc.bev_w}   queries {hc.num_query}   "
          f"encoder x{hc.num_encoder_layers}  decoder x{hc.num_decoder_layers}")

    # ── STEP 1 ─────────────────────────────────────────────────────────────────────────
    rule("STEP 1  frame -> prompt")
    frame = PerceptionFrame(Path(args.frames) / args.frame)
    inputs, img_metas = processor(frame, device=args.device)
    print(f"    token {frame.token}   dataset {frame.dataset_type}")
    print(f"    cameras ({len(frame.cam_order)}): {frame.cam_order}")
    show("input_ids", inputs["input_ids"])
    show("pixel_values", inputs["pixel_values"])
    print(f"    image_grid_thw {inputs['image_grid_thw'][0].tolist()} per camera "
          f"-> {inputs['image_grid_thw'][0][1]*inputs['image_grid_thw'][0][2]//4} tokens each")
    show("lidar2img", img_metas["lidar2img"])
    show("lidar2ego", img_metas["lidar2ego"][0])
    print("    lidar2img folds the 896x512 resize into the projection, so it maps lidar")
    print("    points straight onto the model-resolution image plane.")

    # ── STEP 2  one forward, two taps ──────────────────────────────────────────────────
    rule("STEP 2  ONE VLM forward, TWO taps")
    captured = {}

    def hook(module, a, output=None):
        captured["patches"] = a[0]        # the INPUT to the merger = pre-merge patches

    visual = vlm.model.visual
    handle = visual.merger.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = vlm(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                      image_grid_thw=inputs["image_grid_thw"],
                      mm_token_type_ids=(inputs["input_ids"] ==
                                         vlm.config.image_token_id).long(),
                      use_cache=False, output_hidden_states=True)
    finally:
        handle.remove()
    show("raw pre-merge patches", captured["patches"])
    with torch.no_grad():
        patches = visual.merger.norm(captured["patches"])
    show("after merger.norm  (GEOMETRY)", patches)
    with torch.no_grad():
        hidden = vlm.model.language_model.norm(out.hidden_states[-1])
    show("LLM last layer + final norm", hidden)
    mask = inputs["input_ids"][0] == vlm.config.image_token_id
    llm_tokens = hidden[0][mask]
    show("image-token rows (SEMANTICS)", llm_tokens)
    print("    the geometry tap never entered the language model; the semantics tap is the")
    print("    LAST layer, so no layer sweep is needed to choose it.")

    # ── STEP 3  per-camera grids ───────────────────────────────────────────────────────
    rule("STEP 3  per-camera grids  (undo the merger's block ordering)")
    n_cam = len(frame.cam_order)
    gh, gw = (inputs["image_grid_thw"][-1, 1].item(), inputs["image_grid_thw"][-1, 2].item())
    tokens_per_img = gh // 2 * gw // 2
    vit_feats = torch.stack(head._premerge_grids(patches, inputs["image_grid_thw"])[-n_cam:], 0)
    llm_feats = llm_tokens[-n_cam * tokens_per_img:].view(n_cam, gh // 2, gw // 2, -1)
    show("img_vit_feats [N,H,W,C]", vit_feats)
    show("img_llm_feats [N,H,W,C]", llm_feats)
    if args.dump:
        keep_pca("tap_vit", vit_feats.permute(0, 3, 1, 2))     # [N,H,W,3] RGB
        keep_pca("tap_llm", llm_feats.permute(0, 3, 1, 2))
    print(f"    patch grid {gh}x{gw} -> merged {gh//2}x{gw//2}   ({tokens_per_img} tokens/cam)")
    print("    _premerge_grids un-permutes view(h/2,w/2,2,2,C).permute(0,2,1,3,4) — the merger")
    print("    orders patches in 2x2 BLOCKS, not row-major. Getting this wrong gives a")
    print("    plausible-looking but spatially scrambled grid.")

    # ── STEP 4-7  the view transform ───────────────────────────────────────────────────
    rule("STEP 4  FPN adaptors")
    vit_feats, llm_feats = vit_feats.to(dtype), llm_feats.to(dtype)
    with torch.no_grad():
        feat_main = llm_feats.permute(0, 3, 1, 2)
        mlvl = bev.adaptor(feat_main)
        feat_vit = vit_feats.permute(0, 3, 1, 2)
        vit_mlvl = bev.vit_neck(feat_vit)
    for i, f in enumerate(mlvl):
        show(f"adaptor level {i} (scale {(4.0,2.0,1.0,0.5)[i]})", f)
        keep_pca(f"fpn_llm_{i}", f)
    show("vit_neck level 0 (scale 1.0)", vit_mlvl[0])
    keep_pca("fpn_vit_0", vit_mlvl[0])
    print("    four scales for semantics (the deformable attention samples all of them),")
    print("    one for geometry (it only has to be unprojected).")

    rule("STEP 5  DepthNet -> a depth distribution per pixel")
    with torch.no_grad():
        depth_logits = bev.depth_net(vit_mlvl[0])
        depth = depth_logits.softmax(dim=1)
    show("depth logits", depth_logits)
    show("depth distribution (softmax)", depth)
    fr, fs = hc.frustum_range, hc.frustum_size
    print(f"    frustum {fr}  cell {fs}  -> {depth.shape[1]} depth bins "
          f"from {fr[2]} m to {fr[5]} m")
    print(f"    per-pixel depth sums to 1: {depth[0,:,0,0].sum().item():.4f}")
    if args.dump:
        with torch.no_grad():
            nbins = depth.shape[1]
            centres = torch.arange(nbins, device=depth.device, dtype=torch.float32)
            centres = fr[2] + (centres + 0.5) * fs[2]           # metres
            expected = (depth.float() * centres[None, :, None, None]).sum(1)
        keep("depth_expected_m", expected, "raw")               # [N, H, W] in metres
        print(f"    expected depth {expected.min():.1f} - {expected.max():.1f} m "
              f"(mean {expected.mean():.1f} m)  -> dumped for the video")

    rule("STEP 6  frustum -> ego voxel indices")
    vt = bev.view_trans
    # view_trans.forward reshapes lidar2img BEFORE coord_preparing sees it:
    #   (N_cam,4,4) -> [None] -> (1,N_cam,4,4) -> _format_lidar2img -> (1,N_cam,1,4,4)
    # calling coord_preparing without that step fails with
    #   "linalg.inv: A must be batches of square matrices, but they are 24 by 4"
    meta = dict(img_metas)
    l2i = torch.as_tensor(np.asarray(meta["lidar2img"]), dtype=torch.float32)
    meta["lidar2img"] = vt._format_lidar2img(l2i[None, ...])
    show("lidar2img, reformatted", meta["lidar2img"])
    with torch.no_grad():
        coords, vmask = vt.coord_preparing([meta])
    show("voxel_coords [B,S,N,D,H,W,4]", coords)
    show("mask", vmask)
    keep("frustum_valid", vmask[0, 0].float().mean(dim=1), "raw")   # [N, D, W] in-range frac
    print(f"    {int(vmask.sum())} of {vmask.numel()} frustum points land inside the volume "
          f"({100*vmask.float().mean():.1f}%)")
    print("    each point is unprojected with inv(lidar2img), then brought to the ego frame")
    print("    with lidar2ego, then quantised to a voxel index.")

    rule("STEP 7  voxel pooling -> a 3D volume")
    # the remaining two lines of view_trans.forward, using the coords from STEP 6
    with torch.no_grad():
        vit_feats_r = [f.view(1, n_cam, *f.shape[-3:]) for f in vit_mlvl]
        voxel_space = vt.feat_sampling(vit_feats_r, [depth], coords, vmask)
        show("after feat_sampling", voxel_space)
        print("    ^ one volume PER CAMERA, and the single largest tensor in the pipeline.")
        voxel = vt.feat_encoding(voxel_space)      # sums the camera axis away
        del voxel_space, coords, vmask             # 1.8 GiB + 0.1 GiB, no longer needed
        free_vram()
    show("voxel volume [B,C,D,H,W]", voxel)
    print("    out[b, cam, x, y, z, c] += feats[img, c, h, w] * depth[img, d, h, w]")
    print("    (a weighted scatter-add; on CPU this is _voxel_pool_depth_torch, index_add_)")
    with torch.no_grad():
        bev_tokens = bev._uvtr_voxel_to_bev_tokens(voxel)
    show("BEV init tokens", bev_tokens)
    if args.dump:
        with torch.no_grad():
            keep("voxel_bev_density", voxel[0].float().norm(dim=0).norm(dim=0), "raw")
            keep("bev_tokens_norm",
                 bev_tokens.float().norm(dim=1).view(hc.bev_h, hc.bev_w), "raw")
    print(f"    {hc.occ_pillar_h} z-slices x {hc.embed_dim} ch collapsed by a 1x1 conv "
          f"-> {hc.bev_h*hc.bev_w} BEV cells")

    # ── STEP 8  the heads ──────────────────────────────────────────────────────────────
    # CONTINUE from STEP 7's tensors. Calling head.infer() here instead would re-run the
    # VLM, both taps, the FPNs, DepthNet and the voxel pooling a second time - which is
    # what used to push this script to 20 GiB.
    rule("STEP 8  BEVFormer encoder + three heads")
    del out, hidden, patches, captured, llm_tokens, depth, depth_logits, vit_feats_r
    free_vram()
    vram("entering STEP 8")
    with torch.no_grad():
        mlvl_r = [f.view(1, n_cam, *f.shape[-3:]) for f in mlvl]
        outs = bev.head(mlvl_r, [img_metas], None, vit_bev_feat=None,
                        uvtr_bev_feat=bev_tokens, uvtr_occ_feat=voxel)
        det = bev.head.get_bboxes(outs, [img_metas])[0]
        result = {
            "boxes": det["boxes"], "scores": det["scores"], "labels": det["labels"],
            "occ": bev.head.get_occ(outs, [img_metas])[0].byte().cpu().numpy(),
            "map": bev.head.get_map_seg(outs, [img_metas])[0].byte().cpu().numpy(),
        }
    vram("after the heads")
    keep = result["scores"] > args.score_threshold
    print(f"    3D DETECTION   {len(result['boxes'])} raw -> {int(keep.sum())} over "
          f"{args.score_threshold}   ({hc.det_num_classes} classes, NMS-free)")
    print(f"       box layout [x, y, z, w, l, h, yaw, vx, vy] in the LIDAR frame, z at the "
          f"box bottom")
    import collections
    counts = collections.Counter(DET_CLASS_NAMES[i] for i in result["labels"][keep])
    gt = collections.Counter(DET_CLASS_NAMES[i] for i in frame.gt["labels"])
    print(f"       predicted {dict(counts)}")
    print(f"       ground truth {dict(gt)}")
    occ = result["occ"]
    empty = len(OCC_CLASS_NAMES) - 1
    print(f"    OCCUPANCY      {occ.shape}  {len(OCC_CLASS_NAMES)} classes, "
          f"{100*(occ != empty).mean():.2f}% non-empty  (ego frame, X fwd / Y left / Z up)")
    print(f"    MAP SEG        {result['map'].shape}  {len(MAP_CLASS_NAMES)} classes  "
          f"{hc.map_xbound[:2]} x {hc.map_ybound[:2]} m @ {hc.map_xbound[2]} m")
    agree = float((result["map"] == frame.gt["map"]).mean())
    print(f"       map pixel accuracy vs ground truth: {agree:.3f}")

    vram("final")
    if args.dump:
        stages["boxes"] = np.asarray(result["boxes"], dtype="float32")
        stages["scores"] = np.asarray(result["scores"], dtype="float32")
        stages["labels"] = np.asarray(result["labels"], dtype="int64")
        stages["occ"] = result["occ"]
        stages["map"] = result["map"]
        stages["gt_occ"] = frame.gt["occ"]
        stages["gt_map"] = frame.gt["map"]
        stages["gt_boxes"] = frame.gt["boxes"]
        stages["gt_labels"] = frame.gt["labels"]
        stages["cam_order"] = np.array(frame.cam_order)
        stages["token"] = np.array(frame.token)
        stages["dataset_type"] = np.array(frame.dataset_type)
        stages["lidar2img"] = img_metas["lidar2img"]
        stages["lidar2ego"] = img_metas["lidar2ego"]
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump, **stages)
        print(f"\n    dumped {len(stages)} stage arrays -> {args.dump}")

    if args.render:
        from qwen_drive_perception.visualize import render_frame
        from PIL import Image
        Path(args.render).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(render_frame(frame, result)).save(args.render)
        print(f"\n    wrote {args.render}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
