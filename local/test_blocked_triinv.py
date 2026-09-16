"""Check the blocked triangular inverse against the shipped forward-substitution loop."""
import torch

CH = 64


def shipped(attn):
    attn = attn.clone()
    for i in range(1, CH):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(CH, dtype=attn.dtype, device=attn.device)


def blocked_masks(n, device, dtype):
    """One constant mask per doubling level, selecting the odd-block/even-block coupling."""
    idx = torch.arange(n, device=device)
    masks = []
    b = 1
    while b < n:
        blk = idx // b
        grp = idx // (2 * b)
        m = (grp[:, None] == grp[None, :]) & (blk[:, None] % 2 == 1) & (blk[None, :] % 2 == 0)
        masks.append(m.to(dtype))
        b *= 2
    return masks


def blocked(attn, masks):
    """(I - A)^-1 by recursive 2x2 block inversion, all blocks done in parallel."""
    x = torch.eye(CH, dtype=attn.dtype, device=attn.device).expand_as(attn).contiguous()
    for m in masks:
        x = x + x @ (attn * m) @ x
    return x


torch.manual_seed(0)
masks = blocked_masks(CH, "cpu", torch.float32)
tril = torch.tril(torch.ones(CH, CH), -1)

print(f"{'scale of A entries':>22s} {'max abs diff':>14s} {'rel':>11s} {'|answer|max':>13s}")
for scale in (1.0, 10.0, 100.0, 1000.0):
    # shape mirrors the real model: [batch, heads, chunks, 64, 64]
    a = torch.randn(1, 4, 3, CH, CH) * scale * tril
    ref = shipped(a)
    got = blocked(a, masks)
    d = (ref - got).abs().max().item()
    rel = d / max(ref.abs().max().item(), 1e-30)
    print(f"{scale:22.0f} {d:14.3e} {rel:11.3e} {ref.abs().max().item():13.3e}")

# the identity itself
a = torch.randn(1, 2, 2, CH, CH) * 100 * tril
got = blocked(a, masks)
eye = torch.eye(CH).expand_as(a)
resid = (got @ (eye - a) - eye).abs().max().item()
print(f"\nresidual  ||X(I-A) - I||_max = {resid:.3e}")
print(f"ops: {len(masks)} levels x 2 matmuls = {2*len(masks)} matmuls, "
      f"vs {CH-1} sequential steps")
