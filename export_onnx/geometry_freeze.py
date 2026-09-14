"""Freeze the calibration-dependent geometry so the graph is traceable.

WHY THIS IS NEEDED
------------------
Three things in the BEV stack are not exportable as written:

1. ``torch.inverse`` on the projection matrices. ONNX has no standard ``Inverse``
   operator.
2. The geometry helpers take ``img_metas`` - a list of dicts holding numpy
   arrays - which a tracer cannot follow.
3. ``voxel_pool_depth`` selects points with a boolean mask, producing
   data-dependent shapes.

WHY FREEZING IS CORRECT, NOT A SHORTCUT
---------------------------------------
All three depend ONLY on the camera calibration, never on image content. For a
fixed sensor rig they are constants. Deployed BEVFormer-family models do exactly
this: the view-transform geometry is precomputed once per vehicle and shipped as
a lookup. Re-exporting is only needed if the rig changes.

``freeze_geometry`` runs the real functions once on a reference frame, caches
their outputs, and patches the modules to return the cache.
"""
from __future__ import annotations

import contextlib

import torch

from qwen_drive_perception import bev_encoder as _bev_encoder
from qwen_drive_perception import view_transform as _view_transform


def _shape_key(t) -> tuple:
    """Shape as plain ints.

    Under ``torch.onnx.export`` tracing, ``t.shape`` yields 0-d tensors rather
    than ints, so a raw ``tuple(t.shape)`` never matches the key recorded during
    the eager capture. Coercing to int makes the two agree.
    """
    return tuple(int(d) for d in t.shape)


class FrozenGeometry:
    """Captured outputs of the calibration-only functions."""

    def __init__(self):
        self.voxel_coords = None
        self.voxel_mask = None
        self.point_sampling = {}      # keyed by reference-point shape

    def summary(self) -> str:
        vc = tuple(self.voxel_coords.shape) if self.voxel_coords is not None else None
        keys = {k: tuple(v[0].shape) for k, v in self.point_sampling.items()}
        return f"voxel_coords {vc}   point_sampling captures: {keys}"


@contextlib.contextmanager
def capture_geometry(store: FrozenGeometry):
    """Record what the geometry functions return on one reference forward."""
    cp = _view_transform.Uni3DVoxelPoolDepth.coord_preparing
    ps = _bev_encoder.BEVFormerEncoder.point_sampling

    def cp_spy(self, img_metas):
        out = cp(self, img_metas)
        if store.voxel_coords is None:
            store.voxel_coords = out[0].detach().clone()
            store.voxel_mask = out[1].detach().clone()
        return out

    def ps_spy(self, reference_points, pc_range, img_metas):
        out = ps(self, reference_points, pc_range, img_metas)
        key = _shape_key(reference_points)
        if key not in store.point_sampling:
            store.point_sampling[key] = (out[0].detach().clone(), out[1].detach().clone())
        return out

    _view_transform.Uni3DVoxelPoolDepth.coord_preparing = cp_spy
    _bev_encoder.BEVFormerEncoder.point_sampling = ps_spy
    try:
        yield store
    finally:
        _view_transform.Uni3DVoxelPoolDepth.coord_preparing = cp
        _bev_encoder.BEVFormerEncoder.point_sampling = ps


@contextlib.contextmanager
def frozen_geometry(store: FrozenGeometry, device=None):
    """Replace the geometry functions with constant lookups."""
    assert store.voxel_coords is not None, "call capture_geometry() first"
    cp = _view_transform.Uni3DVoxelPoolDepth.coord_preparing
    ps = _bev_encoder.BEVFormerEncoder.point_sampling
    vc = store.voxel_coords if device is None else store.voxel_coords.to(device)
    vm = store.voxel_mask if device is None else store.voxel_mask.to(device)
    table = {k: (a if device is None else a.to(device), b if device is None else b.to(device))
             for k, (a, b) in store.point_sampling.items()}

    def cp_const(self, img_metas):
        return vc, vm

    def ps_const(self, reference_points, pc_range, img_metas):
        key = _shape_key(reference_points)
        if key not in table:
            raise KeyError(f"no frozen point_sampling for reference shape {key}; "
                           "re-run capture_geometry on a matching frame")
        cam, mask = table[key]
        return cam.to(reference_points.dtype), mask

    _view_transform.Uni3DVoxelPoolDepth.coord_preparing = cp_const
    _bev_encoder.BEVFormerEncoder.point_sampling = ps_const
    try:
        yield
    finally:
        _view_transform.Uni3DVoxelPoolDepth.coord_preparing = cp
        _bev_encoder.BEVFormerEncoder.point_sampling = ps


# ───────────────────────────────────────────── voxel pooling with frozen indices

def _decompose(point_indices, D, H, W):
    flat = point_indices.long()
    w = flat % W; flat = flat // W
    h = flat % H; flat = flat // H
    d = flat % D
    return flat // D, d, h, w          # image_index, d, h, w


class FrozenVoxelIndices:
    """The scatter indices, which depend only on calibration."""

    def __init__(self):
        self.ready = False

    def capture(self, image_index, d, h, w, ranks, n_rows, channels):
        self.image_index, self.d, self.h, self.w = image_index, d, h, w
        self.ranks, self.n_rows, self.channels = ranks, n_rows, channels
        self.ready = True

    def to(self, device):
        for k in ("image_index", "d", "h", "w", "ranks"):
            setattr(self, k, getattr(self, k).to(device))
        for sel in getattr(self, "per_camera", []):
            for k in list(sel):
                sel[k] = sel[k].to(device)
        return self


def make_frozen_voxel_pool(idx: FrozenVoxelIndices):
    """A voxel pool with NO data-dependent control flow.

    The shipped torch fallback branches on ``ranks.numel() == 0`` and selects
    points with a boolean mask. Both are data-dependent, and ``torch.export``
    refuses to guard on them (``GuardOnDataDependentSymNode: Eq(u0, 0)``).

    Once the geometry is frozen those indices are constants, so the branch is
    dead and the selection has already happened. This version scatters directly
    with the captured indices - a single ``index_add``, which lowers to ONNX
    ``ScatterND``.
    """
    def voxel_pool_depth_frozen(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z):
        """Scatter PER CAMERA so no single constant exceeds protobuf's 2 GiB.

        A combined buffer is [B*N_cam*X*Y*Z, C] = [3.84 M, 256], which is 3.93 GiB
        in fp32. External data lets a MODEL exceed 2 GiB, but no single TENSOR may,
        so the traced zeros alone aborted the export. One buffer per camera is
        [640 k, 256] = 655 MiB - comfortably under - and the arithmetic is
        unchanged, since ``ranks`` already encodes the camera in the row index.
        """
        channels = img_feats.shape[2]
        D, H, W = img_depth.shape[1], img_depth.shape[2], img_depth.shape[3]
        n_cam = mask.shape[2]
        feats = img_feats.reshape(-1, channels, H, W)
        depth = img_depth.reshape(-1, D, H, W)
        accum = img_feats.dtype if img_feats.dtype != torch.float64 else torch.float32
        per_cam = X * Y * Z
        slabs = []
        for c in range(n_cam):
            sel = idx.per_camera[c]
            zeros = torch.zeros(per_cam, channels, dtype=accum, device=img_feats.device)
            if sel["ranks"].numel() == 0:
                slabs.append(zeros)
                continue
            contribution = (feats[sel["image_index"], :, sel["h"], sel["w"]]
                            * depth[sel["image_index"], sel["d"], sel["h"],
                                    sel["w"]].unsqueeze(-1)).to(accum)
            # scatter_add, NOT index_add. PyTorch refuses to export index_add
            # when the index contains duplicates:
            #   "ONNX export does not support exporting 'index_add_()' function
            #    with duplicated values in 'index' parameter yet"
            # and voxel pooling is nothing BUT duplicates - many frustum points
            # land in the same voxel, and summing them is the whole operation.
            # scatter_add lowers to ScatterElements(reduction='add'), which is
            # bit-exact through onnxruntime.
            index = sel["ranks"].unsqueeze(1).expand(-1, channels)
            slabs.append(zeros.scatter_add(0, index, contribution))
        out = torch.stack(slabs, 0)                    # [N_cam, X*Y*Z, C]
        return out.view(B, n_cam, X, Y, Z, channels).to(
            torch.promote_types(img_feats.dtype, img_depth.dtype))

    return voxel_pool_depth_frozen


@contextlib.contextmanager
def capture_voxel_indices(idx: FrozenVoxelIndices):
    """Record the scatter indices during one eager forward."""
    from qwen_drive_perception import ops as _ops
    original = _ops.voxel_pool_depth

    def spy(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z):
        if not idx.ready:
            _, N_sweep, N_cam, D, H, W = mask.shape
            flat_mask = mask.reshape(-1)
            flat_coords = voxel_coords.reshape(-1, 4)
            all_pts = torch.arange(B * N_sweep * N_cam * D * H * W,
                                   device=mask.device, dtype=torch.int32)
            pts = all_pts[flat_mask]
            coords = flat_coords[flat_mask].int()
            cam = (pts // (D * H * W)) % N_cam
            ranks = (coords[:, 0] * N_cam * X * Y * Z + cam * X * Y * Z
                     + coords[:, 1] * Y * Z + coords[:, 2] * Z + coords[:, 3])
            image_index, d, h, w = _decompose(pts, D, H, W)
            idx.capture(image_index, d, h, w, ranks.long(),
                        B * N_cam * X * Y * Z, img_feats.shape[2])
            # split by camera; ranks become offsets WITHIN that camera's slab
            per_cam = X * Y * Z
            idx.per_camera = []
            r = ranks.long()
            for c in range(N_cam):
                m = (r // per_cam) % N_cam == c
                idx.per_camera.append({
                    "image_index": image_index[m], "d": d[m],
                    "h": h[m], "w": w[m], "ranks": r[m] % per_cam})
        return original(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z)

    _ops.voxel_pool_depth = spy
    try:
        yield idx
    finally:
        _ops.voxel_pool_depth = original


@contextlib.contextmanager
def frozen_voxel_pool(idx: FrozenVoxelIndices, device=None):
    from qwen_drive_perception import ops as _ops
    original = _ops.voxel_pool_depth
    _ops.voxel_pool_depth = make_frozen_voxel_pool(idx.to(device) if device else idx)
    try:
        yield
    finally:
        _ops.voxel_pool_depth = original
