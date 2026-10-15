"""Turn a frame's calibration into the flat BEV scatter indices the student lifts into.

The student's view transform is a gather plus an accumulate. Which BEV cell each
(depth bin, feature cell) lands in depends only on ``lidar2img`` and ``lidar2ego``, so it
is computed here on the host and handed to the model as an input. That keeps the graph
static and free of the projection arithmetic that, in the teacher's export, became a
GridSample and a ScatterND chain.

``lidar2img`` already has the image resize folded in by ``PerceptionFrame.img_metas``, so
the feature grid here must match the same 896x512 the teacher was given.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bev_indices", "bev_indices_all"]


def bev_indices_all(lidar2img: np.ndarray, lidar2ego: np.ndarray, cfg):
    """(index, valid) for every camera, stacked as (n_cams, depth_bins * H * W).

    All views scatter into the SAME BEV grid, which is the whole point: a front-only
    student can never reproduce the teacher, because only ~24% of the teacher's
    detections fall inside the front camera's field of view. Six views cost 2.3x rather
    than 6x, because the BEV encoder, decoder and heads downstream are paid once.
    """
    idx, val = [], []
    for cam in range(cfg.n_cams):
        i, v = bev_indices(lidar2img, lidar2ego, cfg, cam=cam)
        idx.append(i); val.append(v)
    return np.stack(idx, 0), np.stack(val, 0)


def bev_indices(lidar2img: np.ndarray, lidar2ego: np.ndarray, cfg, stride: int = 16,
                cam: int = 0):
    """(index, valid) for one camera, flattened as (depth_bins * H * W,)."""
    width, height = cfg.image_size
    h, w = height // stride, width // stride
    d0, d1 = cfg.depth_range
    depths = np.linspace(d0, d1, cfg.depth_bins, dtype=np.float64)

    us = (np.arange(w, dtype=np.float64) + 0.5) * stride
    vs = (np.arange(h, dtype=np.float64) + 0.5) * stride
    # (D, H, W) grids in the order the model flattens: depth, then rows, then columns
    dd, vv, uu = np.meshgrid(depths, vs, us, indexing="ij")
    ones = np.ones_like(dd)
    # homogeneous image point scaled by depth, as the projection expects
    pts = np.stack([uu * dd, vv * dd, dd, ones], axis=-1).reshape(-1, 4)

    l2i = np.asarray(lidar2img, dtype=np.float64)
    if l2i.ndim == 3:
        l2i = l2i[cam]
    lidar = pts @ np.linalg.inv(l2i).T
    lidar = lidar[:, :3] / np.clip(lidar[:, 3:4], 1e-6, None)

    l2e = np.asarray(lidar2ego, dtype=np.float64)
    if l2e.ndim == 3:
        l2e = l2e[cam]
    ego = lidar @ l2e[:3, :3].T + l2e[:3, 3]

    x0, y0, _, x1, y1, _ = cfg.pc_range
    n = cfg.bev_size
    ix = np.floor((ego[:, 0] - x0) / (x1 - x0) * n).astype(np.int64)
    iy = np.floor((ego[:, 1] - y0) / (y1 - y0) * n).astype(np.int64)
    valid = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    index = np.where(valid, iy * n + ix, 0).astype(np.int64)
    return index, valid


def occ_to_native(occ: np.ndarray, grid: str = "bev", pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 5.4)) -> np.ndarray:
    """Occupancy argmax (200, 200, 16) in the TEACHER's format: Occ3D's grid, 0.4 m cells over +-40 m, (x, y, z).

    That is what `qwen_drive_perception.visualize._panel_occupancy` draws. A native-grid student (occ_grid="native",
    round 8 on) already predicts it. Earlier students predicted on the 0.512 m detection grid over pc_range,
    indexed (y, x): drawn as they were, their renders came out transposed and 28% too large. Here each native
    cell centre looks up the detection-grid cell that contains it.
    """
    if grid == "native":
        return occ
    n = occ.shape[0]
    o = -40.0 + (np.arange(200) + 0.5) * 0.4
    si = np.clip(np.floor((o - pc_range[0]) / ((pc_range[3] - pc_range[0]) / n)).astype(int), 0, n - 1)
    return occ[si[None, :], si[:, None], :]
