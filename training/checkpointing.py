"""Gradient checkpointing for the BEV stack.

Without it the head peaks at ~19 GiB for a single sample, which leaves no room
for the VLM in stage 2. The two hot spots are the view transform (it briefly
holds one 200x200x16x256 volume PER CAMERA - 1875 MiB) and the six encoder
layers. Recomputing both in the backward pass trades time for memory.
"""
from __future__ import annotations

import copy

import torch
from torch.utils.checkpoint import checkpoint


def _wrap(module):
    inner = module.forward

    def forward(*args, **kwargs):
        if not torch.is_grad_enabled():
            return inner(*args, **kwargs)
        if kwargs:                       # checkpoint() cannot take kwargs directly
            keys = list(kwargs)
            def run(*flat):
                pos = flat[:len(flat) - len(keys)]
                kw = dict(zip(keys, flat[len(flat) - len(keys):]))
                return inner(*pos, **kw)
            return checkpoint(run, *args, *[kwargs[k] for k in keys],
                              use_reentrant=False)
        return checkpoint(inner, *args, use_reentrant=False)

    module.forward = forward
    module._checkpointed = True
    return module


def enable_bev_checkpointing(bev_modeling, view_transform: bool = True) -> int:
    """Checkpoint the encoder/decoder layers and the view transform.

    The view transform matters most: ``_voxel_pool_depth_torch`` accumulates into
    a [B*N_cam*X*Y*Z, C] fp32 buffer - 3.93 GB for 6 cameras - through 41 chunked
    ``index_add_`` calls, and every one of those is kept for the backward pass.
    Recomputing the whole module keeps only its input and output.
    """
    n = 0
    if view_transform and hasattr(bev_modeling, "view_trans"):
        _wrap_view_trans(bev_modeling.view_trans); n += 1
    enc = bev_modeling.head.transformer.encoder
    for layer in getattr(enc, "layers", []):
        _wrap(layer); n += 1
    dec = getattr(bev_modeling.head.transformer, "decoder", None)
    for layer in getattr(dec, "layers", []) if dec is not None else []:
        _wrap(layer); n += 1
    return n


def _wrap_view_trans(module):
    """Checkpoint the view transform, defending against its side effect.

    ``Uni3DVoxelPoolDepth.forward`` REWRITES ``img_meta["lidar2img"]`` in place,
    reshaping it to (1, 1, N, 1, 4, 4). Checkpointing runs the forward twice, and
    the second pass would then be handed the already-reshaped value and raise
    ``Unexpected lidar2img shape``. Recompute has to be idempotent, so every
    invocation gets its own deep copy of the metadata.
    """
    inner = module.forward

    def forward(mlvl_feats, img_depth, img_metas):
        pristine = copy.deepcopy(img_metas)

        def run(*tensors):
            n = len(mlvl_feats)
            return inner(list(tensors[:n]), list(tensors[n:]),
                         copy.deepcopy(pristine))

        if not torch.is_grad_enabled():
            return inner(mlvl_feats, img_depth, copy.deepcopy(pristine))
        return checkpoint(run, *mlvl_feats, *img_depth, use_reentrant=False)

    module.forward = forward
    module._checkpointed = True
    return module
