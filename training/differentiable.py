"""Make the released perception stack differentiable.

THE PROBLEM
-----------
Qwen-Drive-1.0 ships as an inference release, and two of its operators are
forward-only:

  * ``ops._VoxelPoolDepthCuda`` is a ``torch.autograd.Function`` that defines
    ``forward`` and **no** ``backward``.
  * ``ms_deform_attn_bf16_forward`` is a bare function, not wrapped in an
    ``autograd.Function`` at all, and additionally refuses anything but bf16.

Calling ``.backward()`` through either is therefore impossible. That is the real
reason the repo has no training code.

THE FIX
-------
Both operators already have pure-PyTorch twins that exist for CPU fallback, and
both of those ARE differentiable:

  * ``ops._voxel_pool_depth_torch``     - advanced indexing + ``index_add_``
  * ``layers.multi_scale_deformable_attn_pytorch`` - ``grid_sample``

``enable_training_ops()`` redirects the two call sites to the differentiable
twins. It monkey-patches rather than editing ``src/`` so the released package
stays byte-identical and inference keeps using the fast kernels.

Cost: the torch twins are slower and use more memory than the kernels. That is
the price of a backward pass, and it is only paid while training.
"""
from __future__ import annotations

import contextlib

import torch

from qwen_drive_perception import attention as _attention
from qwen_drive_perception import ops as _ops
from qwen_drive_perception.layers import multi_scale_deformable_attn_pytorch
from qwen_drive_perception.ops import _voxel_pool_depth_torch

_ORIGINALS = {}

# The scatter buffer is [B*N_cam*X*Y*Z, C]: for 6 cameras on a 200x200x16 grid
# with 256 channels that is 3.93 GiB in fp32 and the single largest allocation in
# the whole training step. The shipped kernel accumulates in fp32; for training we
# allow bf16, which halves it. Set to None to keep fp32.
ACCUM_DTYPE = None


def set_accum_dtype(dtype) -> None:
    """Accumulation dtype for the voxel scatter (None = fp32, as the kernel does)."""
    global ACCUM_DTYPE
    ACCUM_DTYPE = dtype


def voxel_pool_depth_differentiable(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z):
    """``ops.voxel_pool_depth`` with the CUDA branch removed.

    The index bookkeeping is copied verbatim from the original wrapper - it is
    integer arithmetic and carries no gradient. Only the final dispatch changes:
    the torch kernel is used on every device, not just CPU.
    """
    assert img_feats.dim() == 5 and img_depth.dim() == 4 and voxel_coords.dim() == 7
    assert mask.shape == voxel_coords.shape[:-1]

    img_feats = img_feats.contiguous()
    img_depth = img_depth.contiguous()
    voxel_coords = voxel_coords.contiguous()
    mask = mask.contiguous()

    _, n_images, feat_channels, H, W = img_feats.shape
    _, N_sweep, N_cam, D, H_coords, W_coords = mask.shape
    assert H == H_coords and W == W_coords and n_images == N_sweep * N_cam
    assert img_depth.shape[0] == B * N_sweep * N_cam
    assert img_depth.shape[1:] == (D, H, W)

    with torch.no_grad():                      # pure index arithmetic
        flat_mask = mask.reshape(-1)
        flat_coords = voxel_coords.reshape(-1, 4)
        point_indices_all = torch.arange(
            B * N_sweep * N_cam * D * H * W, device=mask.device, dtype=torch.int32)
        point_indices = point_indices_all[flat_mask]
        coords = flat_coords[flat_mask].int()
        camera_indices = (point_indices // (D * H * W)) % N_cam
        ranks = (coords[:, 0] * N_cam * X * Y * Z
                 + camera_indices * X * Y * Z
                 + coords[:, 1] * Y * Z
                 + coords[:, 2] * Z
                 + coords[:, 3])

    if ACCUM_DTYPE is None:
        out = _voxel_pool_depth_torch(
            img_feats, img_depth, coords, point_indices, ranks,
            B, N_sweep, N_cam, X, Y, Z, D, H, W)
    else:
        out = _voxel_pool_reduced(
            img_feats, img_depth, point_indices, ranks,
            B, N_cam, X, Y, Z, D, H, W, ACCUM_DTYPE)
    return out.to(torch.promote_types(img_feats.dtype, img_depth.dtype))


def _voxel_pool_reduced(img_feats, img_depth, point_indices, ranks,
                        B, N_cam, X, Y, Z, D, H, W, accum_dtype):
    """``_voxel_pool_depth_torch`` with a configurable accumulation dtype.

    Same arithmetic, same chunking; only the buffer's precision differs. Kept
    separate so the shipped fp32 path stays exactly as written.
    """
    channels = img_feats.shape[2]
    out = torch.zeros(B * N_cam * X * Y * Z, channels,
                      dtype=accum_dtype, device=img_feats.device)
    if ranks.numel() == 0:
        return out.view(B, N_cam, X, Y, Z, channels)
    feats = img_feats.reshape(-1, channels, H, W)
    depth = img_depth.reshape(-1, D, H, W)
    with torch.no_grad():
        flat = point_indices.long()
        w = flat % W; flat = flat // W
        h = flat % H; flat = flat // H
        d = flat % D
        image_index = flat // D
        rank_index = ranks.long()
    chunk = max(1, int(2**22 // max(channels, 1)))
    for start in range(0, rank_index.numel(), chunk):
        stop = start + chunk
        ic, hc, wc, dc = (image_index[start:stop], h[start:stop],
                          w[start:stop], d[start:stop])
        contribution = (feats[ic, :, hc, wc].to(accum_dtype)
                        * depth[ic, dc, hc, wc].to(accum_dtype).unsqueeze(-1))
        out = out.index_add(0, rank_index[start:stop], contribution)
    return out.view(B, N_cam, X, Y, Z, channels)


def _deform_attn_differentiable(value, value_spatial_shapes, value_level_start_index,
                                sampling_locations, attention_weights, im2col_step):
    """Always the torch implementation: differentiable, and dtype-safe.

    ``grid_sample`` requires input and grid to share a dtype. The reference
    points are built in fp32 (they come from the fp32 calibration matrices)
    while the features may be bf16, so align them before the call.
    """
    dtype = value.dtype
    return multi_scale_deformable_attn_pytorch(
        value,
        value_spatial_shapes,
        sampling_locations.to(dtype),
        attention_weights.to(dtype),
    )


def enable_training_ops() -> None:
    """Redirect both forward-only kernels to their differentiable twins."""
    if _ORIGINALS:
        return
    # view_transform does ``from .ops import voxel_pool_depth`` INSIDE the method,
    # so the name is resolved from the ops module at call time - patch it there.
    _ORIGINALS["voxel"] = _ops.voxel_pool_depth
    _ORIGINALS["deform"] = _attention.multi_scale_deformable_attn_cuda
    _ops.voxel_pool_depth = voxel_pool_depth_differentiable
    _attention.multi_scale_deformable_attn_cuda = _deform_attn_differentiable


def disable_training_ops() -> None:
    """Restore the fast inference kernels."""
    if not _ORIGINALS:
        return
    _ops.voxel_pool_depth = _ORIGINALS.pop("voxel")
    _attention.multi_scale_deformable_attn_cuda = _ORIGINALS.pop("deform")
    _ORIGINALS.clear()


@contextlib.contextmanager
def training_ops():
    """Scope the patch: ``with training_ops(): loss.backward()``."""
    enable_training_ops()
    try:
        yield
    finally:
        disable_training_ops()


def patch_frustum_device() -> None:
    """Make the view transform's frustum follow its MODULE's device.

    The released property is:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._frustum is None or self._frustum.device.type != device:
            ... torch.arange(..., device=device)

    Two problems, both invisible on a single card:

      1. ``device="cuda"`` has no index, so the tensor is allocated on the
         CURRENT device - cuda:0 - even when the module lives on cuda:1.
      2. The cache test compares ``.device.type``, which is "cuda" for every
         card, so a frustum built on cuda:0 is happily reused on cuda:1.

    Downstream the frustum becomes the voxel indices, which then index the image
    features, and torch refuses a cross-device index. Binding the frustum to
    ``next(self.parameters()).device`` fixes both.
    """
    from qwen_drive_perception.view_transform import Uni3DVoxelPoolDepth

    if getattr(Uni3DVoxelPoolDepth, "_frustum_device_patched", False):
        return

    def frustum(self):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._frustum is None or self._frustum.device != device:
            self._frustum = torch.stack(
                torch.meshgrid(
                    [
                        torch.arange(self.frustum_range[i],
                                     self.frustum_range[i + 3],
                                     self.frustum_size[i], device=device)
                        for i in range(3)
                    ],
                    indexing="ij",
                ),
                dim=-1,
            )
        return self._frustum

    Uni3DVoxelPoolDepth.frustum = property(frustum)
    Uni3DVoxelPoolDepth._frustum_device_patched = True
