"""Run the lift-splat 3-D convs only where a camera can actually see.

``feat_encoding`` is three ``Conv3d(256->256, k=3x3x3)`` over the whole 16x200x200
voxel volume: 6.79 TFLOP, 228 ms, 29.8 TFLOPS. It is near peak, so it cannot be made
faster -- only smaller. And with a reduced camera rig most of that volume is never
written at all:

    all 6 cameras   48.5% of voxels occupied, spanning the full grid
    front camera     8.7% of voxels, confined to y in 44..164 and x in 105..199

The occupied region is not data-dependent. ``coord_preparing`` derives it from
``lidar2img`` and ``lidar2ego`` alone, so the writable box is fixed by the camera
calibration and can be computed once from the mask before any feature exists.

**This is exact, not an approximation.** Outside the box the conv input is zero, and
a stack of convolutions over a uniformly zero region produces a per-channel constant
(bias, then BatchNorm, then ReLU) which is computable once. The influence radius of
three 3x3x3 convs is 3 voxels, so cropping the *input* to the box plus a 6-voxel halo
leaves every output inside the box depending only on inputs that were kept, and every
output beyond the box+3 depending only on zeros. Both regions are therefore
reproduced exactly; ``--verify`` checks that against the uncropped result.

    from local.prune.voxel_crop_v1 import install
    restore = install(head.bev_modeling.view_trans)
"""
from __future__ import annotations

import torch

# three convs with a 3x3x3 kernel each reach 3 voxels; keep twice that on the input
# so the cropped border itself is also correct rather than merely unused
HALO = 6
EXACT_MARGIN = 3


def _empty_constant(conv_layer, channels: int, dtype, device) -> torch.Tensor:
    """What the stack outputs deep inside an all-zero region, per channel."""
    z = torch.zeros(1, channels, 9, 9, 9, dtype=dtype, device=device)
    with torch.no_grad():
        for layer in conv_layer:
            z = layer(z)
    return z[:, :, 4, 4, 4].clone()          # (1, C)


def _empty_volume(conv_layer, shape, dtype, device) -> torch.Tensor:
    """The stack's output for an all-zero volume of the real shape.

    A single per-channel constant is right in the interior but wrong within three
    voxels of the volume's own boundary, where ``Conv3d`` zero-pads: measured 3.5e-03
    relative, and identical in float32, so it is not rounding. Evaluating the stack
    once on a zero volume of the full shape gets the border right too, and it is a
    constant across frames -- one 410 MiB tensor and one conv pass at install time,
    rather than per frame.
    """
    z = torch.zeros(*shape, dtype=dtype, device=device)
    with torch.no_grad():
        for layer in conv_layer:
            z = layer(z)
    return z


def install(view_trans, verify: bool = False):
    """Make ``feat_encoding`` skip the unobservable part of the volume."""
    original_coord = view_trans.coord_preparing
    original_encode = view_trans.feat_encoding
    box = {}
    empty = {}

    def coord_preparing(img_metas):
        voxel_coords, mask = original_coord(img_metas)
        if mask.any():
            sel = voxel_coords[mask]
            # columns are (batch, x, y, z); the encoder's tensor is [B, C, z, y, x]
            box["w"] = (int(sel[:, 1].min()), int(sel[:, 1].max()) + 1)
            box["h"] = (int(sel[:, 2].min()), int(sel[:, 2].max()) + 1)
        else:
            box.clear()
        return voxel_coords, mask

    def feat_encoding(voxel_space):
        if not box:
            return original_encode(voxel_space)
        B, num_sweep = voxel_space.shape[:2]
        dense = voxel_space.flatten(0, 1).view(
            B, num_sweep, *voxel_space.shape[2:]).sum(1)          # [B, C, D, H, W]
        _, C, D, H, W = dense.shape

        h0, h1 = box["h"]
        w0, w1 = box["w"]
        ih0, ih1 = max(0, h0 - HALO), min(H, h1 + HALO)
        iw0, iw1 = max(0, w0 - HALO), min(W, w1 + HALO)
        frac = ((ih1 - ih0) * (iw1 - iw0)) / (H * W)
        if frac > 0.85:                     # not worth the copies
            return original_encode(voxel_space)

        crop = dense[..., ih0:ih1, iw0:iw1]
        with torch.no_grad():
            for layer in view_trans.conv_layer:
                crop = layer(crop)

        key = (B, C, D, H, W, dense.dtype)
        if empty.get("key") != key:
            empty["key"] = key
            empty["vol"] = _empty_volume(view_trans.conv_layer, key[:5],
                                         dense.dtype, dense.device)
        out = empty["vol"].clone()
        # paste back everything the kept inputs determine exactly
        eh0, eh1 = max(0, h0 - EXACT_MARGIN), min(H, h1 + EXACT_MARGIN)
        ew0, ew1 = max(0, w0 - EXACT_MARGIN), min(W, w1 + EXACT_MARGIN)
        out[..., eh0:eh1, ew0:ew1] = crop[...,
                                          eh0 - ih0:eh1 - ih0,
                                          ew0 - iw0:ew1 - iw0]
        if verify:
            ref = original_encode(voxel_space)
            d = (ref.float() - out.float()).abs()
            print(f"    [voxel_crop] kept {100*frac:.1f}% of the volume, "
                  f"max abs diff vs uncropped {d.max().item():.3e}", flush=True)
        return out

    view_trans.coord_preparing = coord_preparing
    view_trans.feat_encoding = feat_encoding

    def restore():
        view_trans.coord_preparing = original_coord
        view_trans.feat_encoding = original_encode
    return restore
