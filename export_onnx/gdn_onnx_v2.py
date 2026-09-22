"""A chunked gated delta rule written to trace into few, large ONNX nodes.

Measured decomposition of one linear-attention layer at 2752 tokens, fp32, on a
3090 (ORT 1.22, CUDA EP, IOBinding, warm):

    mlp                 12.1 ms    390 GFLOP    32   TFLOPS   <- optimal
    linear_attn         67.3 ms    257 GFLOP     3.9 TFLOPS   <- 3911 nodes
    (full_attention's self_attn, for comparison:  ~15 ms)

So the delta rule spends ~55 ms on 25 GFLOP of actual recurrence.  It is not
launch overhead -- a captured CUDA graph replays in the same time, so the GPU is
genuinely busy that long, in thousands of tiny serialised kernels.  Three sources,
all of them in how the reference implementation is written rather than in what it
computes:

  * ``core_attn_out[:, :, i] = ...`` inside the chunk loop emits a ScatterND per
    chunk, each one a read-modify-write of the whole output.  Collecting the
    chunks in a list and stacking once is the same arithmetic with none of that.
  * ``query[:, :, i]`` and friends emit a Gather (plus its shape arithmetic) per
    tensor per chunk -- 43 chunks x 7 tensors.  ``unbind(2)`` gives the tracer one
    Split per tensor instead.
  * ``g.exp()`` and the tail decay are recomputed inside the loop; hoisting them
    turns 43 small Exp kernels into one large one.

The triangular inverse is the blocked form from
``export_vlm_layers.patch_chunk_rule_blocked``, kept because it is the stable one:
the ``prod (I + A^2^k)`` identity also works but forms high powers of A and was
measured to produce NaN on real activations.

Padding is conditional, so handing the graph a sequence that is already a
multiple of ``chunk_size`` removes it -- worth doing, because ORT has no CUDA
kernel for the pad the reference emits and runs all five on the CPU, dragging
45 MiB activations to the host and back.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Byte-for-byte the reference normalisation.

    ``rsqrt(sum + eps)`` and ``1 / sqrt(clamp_min(sum, eps^2))`` differ by 4e-5
    relative on a layer output, which is small but needless: the point of this
    module is to change how the graph traces, not what it computes.
    """
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _inv_unit_lower(a: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """``(I - a)^-1`` for strictly lower triangular ``a``, by blocked inversion.

    Partitioned into 2x2 blocks the inverse of a unit lower triangular matrix is
    ``X + X L X`` where ``X`` holds the already-inverted diagonal blocks and ``L``
    keeps only the odd-block/even-block coupling.  Doubling the block size covers
    all ``chunk_size`` rows in ``log2(chunk_size)`` steps, and every intermediate
    is a block of the answer, so nothing grows beyond the answer's magnitude.
    The masks are constants and fold away at export.
    """
    eye = torch.eye(chunk_size, dtype=a.dtype, device=a.device)
    idx = torch.arange(chunk_size, device=a.device)
    x = eye.expand_as(a).contiguous()
    block = 1
    while block < chunk_size:
        blk = idx // block
        grp = idx // (2 * block)
        mask = ((grp[:, None] == grp[None, :]) & (blk[:, None] % 2 == 1)
                & (blk[None, :] % 2 == 0)).to(a.dtype)
        x = x + x @ (a * mask) @ x
        block *= 2
    return x


def chunk_gated_delta_rule_onnx(
    query, key, value, g, beta,
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

    batch, heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        query, key, value = [F.pad(x, (0, 0, 0, pad)) for x in (query, key, value)]
        beta, g = [F.pad(x, (0, pad)) for x in (beta, g)]
    total = seq_len + pad
    n_chunk = total // chunk_size
    query = query * (k_dim ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # Spell the split out with literal dimensions; a -1 makes the tracer emit
    # shape arithmetic to recover what is already known at export time.
    def chunks(x, last):
        return x.reshape(batch, heads, n_chunk, chunk_size, last)

    query, key = chunks(query, k_dim), chunks(key, k_dim)
    value, v_beta = chunks(value, v_dim), chunks(v_beta, v_dim)
    k_beta = chunks(k_beta, k_dim)
    g = g.reshape(batch, heads, n_chunk, chunk_size).cumsum(-1)

    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    strict_upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(0)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(strict_upper, 0)
    attn = _inv_unit_lower(attn, chunk_size)

    g_exp = g.exp()
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_exp.unsqueeze(-1))
    tail_decay = (g[..., -1:] - g).exp()

    state = (
        torch.zeros(batch, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    # One Split per tensor instead of a Gather per tensor per chunk.
    q_c, k_c, v_c = query.unbind(2), key.unbind(2), value.unbind(2)
    kd_c, dm_c = k_cumdecay.unbind(2), decay.unbind(2)
    ge_c, td_c = g_exp.unbind(2), tail_decay.unbind(2)

    outs = []
    for i in range(n_chunk):
        v_new = v_c[i] - kd_c[i] @ state
        outs.append((q_c[i] * ge_c[i].unsqueeze(-1)) @ state
                    + ((q_c[i] @ k_c[i].transpose(-1, -2)) * dm_c[i]) @ v_new)
        state = state * ge_c[i][..., -1, None, None] + (
            k_c[i] * td_c[i].unsqueeze(-1)).transpose(-1, -2) @ v_new

    core = torch.stack(outs, dim=2).reshape(batch, heads, total, v_dim)
    if pad:
        core = core[:, :, :seq_len]
    return core.transpose(1, 2).to(initial_dtype), (state if output_final_state else None)


def patch(model, chunk_size: int = 64) -> int:
    """Install this rule on every Gated-DeltaNet in ``model``."""
    fn = chunk_gated_delta_rule_onnx
    if chunk_size != 64:
        fn = functools.partial(fn, chunk_size=chunk_size)
    n = 0
    for mod in model.modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            mod.chunk_gated_delta_rule = fn
            n += 1
    return n
