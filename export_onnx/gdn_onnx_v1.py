"""An ONNX-exportable chunked gated delta rule.

`transformers.models.qwen3_5.torch_chunk_gated_delta_rule` is written for eager
PyTorch and traces catastrophically: its ``for i in range(1, chunk_size)`` loop
does 63 sliced in-place assignments, and each one becomes a Slice/Mul/ReduceSum/
ScatterND group plus the shape arithmetic to describe it. One decoder layer comes
out as 14,161 ONNX nodes -- 1,972 after constant folding -- and runs at 139.8 ms
against 35.3 ms for a full-attention layer of the same FLOP count. 24 of the 32
layers are this one, so the whole export is bound by shape plumbing.

Two rewrites, both exact:

*Triangular inverse.* That loop is forward substitution; what it returns is
``(I - A)^-1`` for the strictly lower triangular ``A``. Since ``A`` is nilpotent
of index ``chunk_size``,

    (I - A)^-1 = I + A + A^2 + ... + A^(C-1) = prod_k (I + A^(2^k))

because the binary expansions of 0..C-1 hit every power exactly once, and powers
of ``A`` commute so the product order is free. That is 10 batched 64x64 matmuls
for C=64 instead of 63 serial scatters.

*Chunk accumulation.* The inter-chunk recurrence is genuinely sequential, but
writing into ``core_attn_out[:, :, i]`` forces a ScatterND per step. Collecting
the per-chunk outputs in a list and stacking once at the end is the same
arithmetic with none of the indexing.

The signature matches the transformers function so it can be swapped in for the
duration of an export.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Matches the FLA kernel the reference implementation aligns itself with."""
    return x / torch.sqrt(x.pow(2).sum(dim, keepdim=True).clamp_min(eps * eps))


def _inv_unit_lower(a: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """``(I - a)^-1`` for strictly lower triangular ``a``, by binary doubling."""
    eye = torch.eye(chunk_size, dtype=a.dtype, device=a.device)
    out = eye + a
    power = a
    step = 1
    while (1 << step) < chunk_size:
        power = power @ power
        out = out @ (eye + power)
        step += 1
    return out


def chunk_gated_delta_rule_onnx(
    query,
    key,
    value,
    g,
    beta,
    chunk_size: int = 64,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size
    query = query * (k_head_dim ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # Sequence length is static at export time, so spell the chunk split out as a
    # literal shape: a -1 here would make the exporter emit shape arithmetic.
    def to_chunks(x):
        return x.reshape(batch_size, num_heads, num_chunks, chunk_size, x.shape[-1])

    query, key, value, k_beta, v_beta = [
        to_chunks(x) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    strict_lower = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(0)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(strict_lower, 0)
    attn = _inv_unit_lower(attn, chunk_size)

    value = attn @ v_beta
    g_exp = g.exp()
    k_cumdecay = attn @ (k_beta * g_exp.unsqueeze(-1))
    # The decay from each position to the end of its chunk, for the state update.
    tail_decay = (g[..., -1:] - g).exp()

    state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=value.dtype, device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )

    chunk_outs = []
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2)) * decay_mask[:, :, i]
        v_new = v_i - k_cumdecay[:, :, i] @ state
        attn_inter = (q_i * g_exp[:, :, i].unsqueeze(-1)) @ state
        chunk_outs.append(attn_inter + attn_i @ v_new)
        state = state * g_exp[:, :, i, -1].unsqueeze(-1).unsqueeze(-1) + (
            k_i * tail_decay[:, :, i].unsqueeze(-1)
        ).transpose(-1, -2) @ v_new

    core_attn_out = torch.stack(chunk_outs, dim=2)
    core_attn_out = core_attn_out.reshape(
        batch_size, num_heads, total_sequence_length, v_head_dim
    )
    if pad_size:
        core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype)
    return core_attn_out, (state if output_final_state else None)
