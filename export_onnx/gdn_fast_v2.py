"""Two more Gated-DeltaNet export fixes, found by profiling the layer graph.

`export_vlm_layers.patch_chunk_rule_blocked` already removed the 63-step
triangular-inverse unroll: a linear-attention layer drops from 14,161 nodes to
1,658.  It did not get faster.  Profiling the built session says why -- the graph
was never launch-bound, and the node count was never the cost:

    op                ms/run     %   n     provider
    Mul                18.56  15.3   60    CUDA
    MatMul             17.27  14.3   58    CUDA     <- the only real compute
    Pad                16.42  13.6    5    CPU  (!)
    Conv               15.46  12.8    1    CUDA
    MemcpyFromHost     14.85  12.3    5
    MemcpyToHost       14.46  11.9    5
    Split              14.07  11.6    7    CUDA

Two defects account for half of the 121 ms:

1. ONNX Runtime has no CUDA kernel for the ``Pad`` configuration the chunk rule
   emits, so all five pads run on the CPU -- and each one drags a 45 MiB
   activation to host and back.  17.4 ms of pad plus 29.3 ms of memcpy.  The pads
   only exist to round the sequence up to a multiple of ``chunk_size``, so the fix
   is to hand the graph a sequence that is already a multiple and let the padding
   branch fall away.  ``pad_to`` computes the target length; the patch here makes
   the pads conditional so they vanish when there is nothing to pad.

2. ``nn.Conv1d`` with ``groups == in_channels == 8192`` is a depthwise conv of
   0.18 GFLOP that ORT hands to cuDNN and that takes 15.5 ms.  Written out as its
   four shifted multiply-accumulates it is ordinary elementwise work.

Both are exact: (1) removes operations that were padding with zeros that were
then sliced off, and (2) is the same sum in the same order.
"""
from __future__ import annotations

import functools
import inspect

import torch
from torch import nn

# The blocked triangular inverse, kept byte-identical to the validated version in
# export_vlm_layers.patch_chunk_rule_blocked so the numerics do not move.
_TRIINV_OLD = """    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)"""
_TRIINV_NEW = """    _eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    _idx = torch.arange(chunk_size, device=attn.device)
    _x = _eye.expand_as(attn).contiguous()
    _b = 1
    while _b < chunk_size:
        _blk = _idx // _b
        _grp = _idx // (2 * _b)
        _m = ((_grp[:, None] == _grp[None, :]) & (_blk[:, None] % 2 == 1)
              & (_blk[None, :] % 2 == 0)).to(attn.dtype)
        _x = _x + _x @ (attn * _m) @ _x
        _b *= 2
    attn = _x"""

_PAD_OLD = """    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))"""
_PAD_NEW = """    if pad_size:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))"""


def pad_to(sequence_length: int, chunk_size: int) -> int:
    """The sequence length to hand the graph so the chunk rule needs no padding.

    Trailing positions are only ever appended, and both the delta rule and the
    full-attention layers are causal, so tokens the model already had keep exactly
    the hidden states they had.  The host slices the extra rows off the output.
    """
    return (sequence_length + chunk_size - 1) // chunk_size * chunk_size


def patch_chunk_rule(model, chunk_size: int = 64) -> int:
    """Blocked triangular inverse plus pads that disappear on aligned input."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    src = inspect.getsource(M.torch_chunk_gated_delta_rule)
    for old in (_TRIINV_OLD, _PAD_OLD):
        if old not in src:
            raise RuntimeError(
                "the installed transformers chunk rule does not match the text "
                "this patch replaces; refusing to patch blindly")
    src = src.replace(_TRIINV_OLD, _TRIINV_NEW).replace(_PAD_OLD, _PAD_NEW)
    namespace = dict(M.__dict__)
    exec(compile(src, "<gdn_fast_v2>", "exec"), namespace)
    patched = namespace["torch_chunk_gated_delta_rule"]
    if chunk_size != 64:
        patched = functools.partial(patched, chunk_size=chunk_size)

    n = 0
    for mod in model.modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            mod.chunk_gated_delta_rule = patched
            n += 1
    return n


class CausalDepthwiseShift(nn.Module):
    """A depthwise causal conv1d as ``kernel_size`` shifted multiply-accumulates.

    Stands in for ``nn.Conv1d(C, C, k, groups=C, padding=k-1)``.  The caller slices
    the result back to ``T`` columns, so this returns ``T`` directly and that slice
    becomes a no-op:  ``out[..., t] = b + sum_j w[:, j] * x[..., t + j - (k-1)]``,
    which is what the padded convolution evaluates over the causal range.
    """

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        weight = conv.weight.detach()                      # (C, 1, k)
        self.register_buffer("w", weight.squeeze(1).clone())
        bias = conv.bias.detach().clone() if conv.bias is not None \
            else torch.zeros(weight.shape[0], dtype=weight.dtype)
        self.register_buffer("b", bias)
        self.kernel_size = weight.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # (B, C, T)
        # Work in (B, T, C) even though the caller hands over (B, C, T). Two
        # reasons, both measured: a Mul that broadcasts over the *last* axis hits
        # ORT's vectorised path, while broadcasting over the middle axis fell back
        # to a generic strided kernel that cost 8.7 ms for a 90 MiB tensor; and
        # slicing the time axis of a (1, T+k-1, C) tensor is a contiguous block,
        # so each shift is a plain copy instead of a strided gather. The two
        # transposes added back are ~0.3 ms and pay for themselves many times.
        k = self.kernel_size
        xt = x.transpose(1, 2)                                  # (B, T, C)
        length = xt.shape[1]
        lead = torch.zeros(xt.shape[0], k - 1, xt.shape[2],
                           dtype=xt.dtype, device=xt.device)
        padded = torch.cat([lead, xt], dim=1)
        out = self.b.reshape(1, 1, -1)
        for j in range(k):
            out = out + self.w[:, j].reshape(1, 1, -1) * padded[:, j:j + length]
        return out.transpose(1, 2)


def patch_conv(model) -> int:
    """Swap every Gated-DeltaNet depthwise conv for the shifted form."""
    n = 0
    for mod in model.modules():
        conv = getattr(mod, "conv1d", None)
        if isinstance(conv, nn.Conv1d) and conv.groups == conv.in_channels:
            mod.conv1d = CausalDepthwiseShift(conv).to(
                device=conv.weight.device, dtype=conv.weight.dtype)
            n += 1
    return n
