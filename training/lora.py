"""A minimal LoRA, so stage 2 fits on one card without adding a dependency.

Stage 2 makes the VLM trainable. Full fine-tuning of 4.5 B parameters needs
weights + gradients + two Adam moments - roughly 72 GB before activations, so it
is a multi-GPU job. LoRA trains rank-r updates on the attention projections
instead, which is ~0.3 % of the parameters and fits alongside the perception
head on a single 24 GB card.

This is a DEVIATION from the paper, which fine-tunes the VLM fully. It is
recorded as such in study/08; ``--full`` selects the faithful path for anyone
with the hardware.
"""
from __future__ import annotations

import math
import re

import torch
from torch import nn

DEFAULT_TARGETS = (r"\.q_proj$", r"\.k_proj$", r"\.v_proj$", r"\.o_proj$",
                   r"\.qkv$", r"\.proj$")


class LoRALinear(nn.Module):
    """``y = W x + (alpha / r) * B(A x)`` with W frozen."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        # match the frozen weight's device AND dtype, or the adapters land on CPU
        dtype, device = base.weight.dtype, base.weight.device
        self.lora_a = nn.Parameter(
            torch.zeros(rank, base.in_features, dtype=dtype, device=device))
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))   # B stays zero, so
        nn.init.zeros_(self.lora_b)                             # the model starts unchanged

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_a), self.lora_b) * self.scaling


def apply_lora(model: nn.Module, rank: int = 16, alpha: int = 32,
               targets=DEFAULT_TARGETS) -> int:
    """Wrap matching ``nn.Linear`` modules in place. Returns how many were wrapped."""
    pats = [re.compile(p) for p in targets]
    todo = [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and any(p.search(n) for p in pats)]
    for name, mod in todo:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], LoRALinear(mod, rank, alpha))
    return len(todo)


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters()
            if ("lora_a" in n or "lora_b" in n) and p.requires_grad]


def freeze_all_but_lora(model: nn.Module) -> tuple[int, int]:
    trainable = frozen = 0
    for n, p in model.named_parameters():
        if "lora_a" in n or "lora_b" in n:
            p.requires_grad_(True); trainable += p.numel()
        else:
            p.requires_grad_(False); frozen += p.numel()
    return trainable, frozen
