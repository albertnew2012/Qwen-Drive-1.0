"""Compare the blocked triangular inverse against the shipped loop on REAL model data.

Random strictly-lower matrices are a bad proxy: their inverses overflow for both
methods. The matrices this model actually produces are damped by ``decay_mask``,
so what matters is how the two agree on those.

Captures ``attn`` from the first linear-attention layers of a real perception
prefill, then scores both methods against a float64 reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CH = 64
CAPTURED: list[torch.Tensor] = []
WANT = 6


class _Stop(Exception):
    pass


def shipped(attn):
    attn = attn.clone()
    for i in range(1, CH):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(CH, dtype=attn.dtype, device=attn.device)


def masks_for(n, device, dtype):
    idx = torch.arange(n, device=device)
    out, b = [], 1
    while b < n:
        blk, grp = idx // b, idx // (2 * b)
        m = (grp[:, None] == grp[None, :]) & (blk[:, None] % 2 == 1) & (blk[None, :] % 2 == 0)
        out.append(m.to(dtype))
        b *= 2
    return out


def blocked(attn, masks):
    x = torch.eye(CH, dtype=attn.dtype, device=attn.device).expand_as(attn).contiguous()
    for m in masks:
        x = x + x @ (attn * m) @ x
    return x


def install_capture():
    """Grab ``attn`` right before the inversion loop, then abort the forward."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

    def spy(query, key, value, g, beta, chunk_size=64, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)]
        _, _, seq, _ = key.shape
        pad = (chunk_size - seq % chunk_size) % chunk_size
        key = F.pad(key, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
        k_beta = key * beta.unsqueeze(-1)
        key, k_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                       for x in (key, k_beta)]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(dim=-1)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
        decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(mask, 0)
        CAPTURED.append(attn.detach().clone())
        print(f"  captured layer {len(CAPTURED)}  {tuple(attn.shape)}  "
              f"|A|max {attn.abs().max():.3e}", flush=True)
        if len(CAPTURED) >= WANT:
            raise _Stop
        raise _Stop

    for name in ("torch_chunk_gated_delta_rule",):
        setattr(M, name, spy)
    return M


def main() -> None:
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
    from transformers import AutoTokenizer

    M = install_capture()
    vlm_dir = "weights/Qwen-Drive-1.0-4B"
    holder = QwenDriveForPlanning.from_pretrained(
        vlm_dir, dtype=torch.float32, attn_implementation="sdpa")
    vlm = holder.vlm
    del holder.planning_expert
    head = QwenDrivePerception.from_pretrained(vlm_dir + "/perception", dtype=torch.float32).eval()
    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(vlm_dir))
    head.attach(vlm.eval(), processor)

    # each GatedDeltaNet grabbed the function at __init__, so rebind instances too
    n = 0
    for mod in vlm.modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            mod.chunk_gated_delta_rule = M.torch_chunk_gated_delta_rule
            n += 1
    print(f"patched {n} GatedDeltaNet modules", flush=True)

    frame = PerceptionFrame(Path("data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f"))
    inputs, img_metas = processor(frame, device="cpu")
    try:
        with torch.no_grad():
            head.infer(inputs, img_metas)
    except _Stop:
        pass

    if not CAPTURED:
        print("nothing captured")
        return

    masks32 = masks_for(CH, "cpu", torch.float32)
    masks64 = masks_for(CH, "cpu", torch.float64)
    print(f"\n{'':6s} {'|A|max':>10s} {'|X|max':>10s} "
          f"{'shipped err':>13s} {'blocked err':>13s}")
    for i, a in enumerate(CAPTURED):
        ref = shipped(a.double())                       # fp64 reference
        s = shipped(a)
        b = blocked(a, masks32)
        scale = max(ref.abs().max().item(), 1e-30)
        es = (s.double() - ref).abs().max().item() / scale
        eb = (b.double() - ref).abs().max().item() / scale
        print(f"  [{i}] {a.abs().max():10.3e} {ref.abs().max():10.3e} "
              f"{es:13.3e} {eb:13.3e}")
    a = CAPTURED[0]
    print(f"\nblocked vs shipped, fp32, layer 0: "
          f"{(blocked(a, masks32) - shipped(a)).abs().max():.3e}")
    print(f"blocked in fp64 vs shipped fp64:   "
          f"{(blocked(a.double(), masks64) - shipped(a.double())).abs().max():.3e}")


if __name__ == "__main__":
    main()
