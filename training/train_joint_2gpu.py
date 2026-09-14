"""Stage 2 across BOTH GPUs by MODULE parallelism, not data parallelism.

    python training/train_joint_2gpu.py --steps 8

WHY MODULE PARALLEL AND NOT FSDP / DDP
--------------------------------------
Measured on this model:

  * DDP replicates. Stage 2 does not fit on one card to begin with, so
    replicating it twice does not help.
  * FSDP shards parameters, gradients and optimizer state. Full stage-2
    fine-tuning needs ~37 GiB of those; sharded over two cards that is 18.7 GiB
    each, and the perception head alone peaks at ~19 GiB of ACTIVATIONS, which
    FSDP does not shard. 18.7 + 19 > 23.6, so it still does not fit.
  * The VLM and the BEV head exchange exactly two tensors - the ViT tap and the
    LLM tap, ~34 MiB total. That is a natural cut, and putting each module on
    its own card splits both the parameters AND the activations.

So: VLM on cuda:0, perception head on cuda:1. Autograd crosses the device
boundary on its own, because ``.to(device)`` is differentiable.

WHAT THIS BUYS
--------------
The single-card version had to LoRA everything to fit. With the head's ~19 GiB
moved off cuda:0 there is room to train the **vision encoder in full** (0.33 B)
while the 4.2 B language model stays frozen behind LoRA. The report's stage 2
trains "the vision encoder, VLM and perception head" jointly, so this is a step
closer to it than LoRA-only.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer

from qwen_drive import QwenDriveForPlanning
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.configuration_perception import OCC_EMPTY_LABEL
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor

from training.checkpointing import enable_bev_checkpointing
from training.config import JointTrainConfig
from training.differentiable import (enable_training_ops, patch_frustum_device,
                                     set_accum_dtype)
from training.lora import apply_lora, lora_parameters
from training.losses import (HungarianMatcher3D, detection_loss, map_loss,
                             occupancy_loss)
from training.train_joint import taps_with_grad


def metas_to(metas, device):
    """Move any tensors inside img_metas to the head's device.

    ``PerceptionProcessor`` builds the calibration on whatever device it was
    given - here the VLM's - but the view transform derives its frustum and
    voxel indices from those tensors, then indexes the FEATURES with them. With
    the modules on different cards that is a cross-device index and torch
    refuses it.
    """
    out = {}
    for k, v in metas.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--out", default="outputs/train_joint2")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--vlm-device", default="cuda:0")
    ap.add_argument("--head-device", default="cuda:1")
    ap.add_argument("--freeze-vision", action="store_true",
                    help="keep the vision encoder frozen (the single-card setting)")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="override the config. Dataclass defaults are baked into "
                         "__init__, so setting the class attribute after import "
                         "does nothing - this flag is the working way.")
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()

    cfg = JointTrainConfig()
    if args.lora_rank:
        cfg.lora_rank = args.lora_rank
        cfg.lora_alpha = 2 * args.lora_rank
    torch.manual_seed(cfg.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    enable_training_ops()
    patch_frustum_device()   # the frustum must follow the head, not cuda:0
    dtype = getattr(torch, cfg.amp_dtype)
    set_accum_dtype(dtype)

    d_vlm, d_head = args.vlm_device, args.head_device
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=dtype, attn_implementation="sdpa")
    vlm = holder.vlm
    del holder.planning_expert
    vlm = vlm.to(d_vlm)
    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(d_head)
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm, proc)
    bev = head.bev_modeling.train()
    print(f"  VLM  -> {d_vlm}     head -> {d_head}")

    # LoRA on the language model; the vision encoder trains in full.
    n_lora = apply_lora(vlm.model.language_model, cfg.lora_rank, cfg.lora_alpha)
    for p in vlm.parameters():
        p.requires_grad_(False)
    for n, p in vlm.named_parameters():
        if "lora_a" in n or "lora_b" in n:
            p.requires_grad_(True)
    vision_params = []
    if not args.freeze_vision:
        for p in vlm.model.visual.parameters():
            p.requires_grad_(True)
        vision_params = [p for p in vlm.model.visual.parameters() if p.requires_grad]
    lora_params = lora_parameters(vlm)
    head_params = [p for p in bev.parameters() if p.requires_grad]
    vlm.train()

    nc = enable_bev_checkpointing(bev)
    for m in (vlm, getattr(vlm, "model", None)):
        if m is not None and hasattr(m, "gradient_checkpointing_enable"):
            try:
                m.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                break
            except Exception:
                pass
    tot = lambda ps: sum(p.numel() for p in ps)
    print(f"  LoRA r={cfg.lora_rank} on {n_lora} LM projections : {tot(lora_params)/1e6:8.2f} M")
    print(f"  vision encoder {'FROZEN' if args.freeze_vision else 'TRAINABLE'}"
          f"              : {tot(vision_params)/1e6:8.2f} M")
    print(f"  perception head                          : {tot(head_params)/1e6:8.2f} M")
    print(f"  gradient checkpointing on {nc} BEV modules")

    head_lr = cfg.vlm_lr * cfg.head_lr_multiplier
    groups = [{"params": lora_params, "lr": cfg.vlm_lr},
              {"params": head_params, "lr": head_lr}]
    if vision_params:
        groups.append({"params": vision_params, "lr": cfg.vlm_lr})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    matcher = HungarianMatcher3D(cfg.match_cls_cost, cfg.match_reg_cost)

    frames = sorted(p for p in Path(args.frames).iterdir() if p.is_dir())
    history, t0 = [], time.time()
    for step in range(args.steps):
        fp = frames[0] if args.overfit else frames[step % len(frames)]
        frame = PerceptionFrame(fp)
        inputs, metas = proc(frame, device=d_vlm)
        metas = metas_to(metas, d_head)
        gt = np.load(fp / "gt.npz")
        n_cam = len(metas["cam_order"])

        vit, llm = taps_with_grad(head, vlm, inputs, n_cam)
        # the only tensors that cross the device boundary
        vit = vit.to(d_head, dtype)
        llm = llm.to(d_head, dtype)
        outs = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])

        d = detection_loss(outs["all_cls_scores"].float(),
                           outs["all_bbox_preds"].float(),
                           [torch.from_numpy(gt["boxes"]).float().to(d_head)],
                           [torch.from_numpy(gt["labels"]).long().to(d_head)],
                           matcher, cls_weight=cfg.det_cls_weight,
                           reg_weight=cfg.det_reg_weight)
        d.pop("det_matched")
        o = occupancy_loss(outs["occ_pred"],
                           torch.from_numpy(gt["occ"]).long().to(d_head),
                           empty_label=OCC_EMPTY_LABEL,
                           focal_weight=cfg.occ_focal_weight,
                           max_points=cfg.occ_max_points)
        m = map_loss(outs["seg_preds"],
                     torch.from_numpy(gt["map"]).long().to(d_head),
                     focal_weight=cfg.map_focal_weight,
                     max_points=cfg.map_max_points)
        perc = sum(d.values()) + sum(o.values()) + sum(m.values())

        opt.zero_grad(set_to_none=True)
        perc.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            lora_params + head_params + vision_params, cfg.grad_clip_norm)
        opt.step()

        g_lora = sum(float(p.grad.abs().sum()) for p in lora_params if p.grad is not None)
        g_vis = sum(float(p.grad.abs().sum()) for p in vision_params if p.grad is not None)
        mem0 = torch.cuda.max_memory_allocated(0) / 2**30
        mem1 = torch.cuda.max_memory_allocated(1) / 2**30
        row = {"step": step, "L_perc": float(perc.detach()),
               "lora_grad": g_lora, "vision_grad": g_vis,
               "grad_norm": float(gn), "gib_cuda0": mem0, "gib_cuda1": mem1,
               **{k: float(v.detach()) for k, v in {**d, **o, **m}.items()}}
        history.append(row)
        print(f"  step {step:3d}  L_perc {row['L_perc']:9.4f}   "
              f"|LoRA g| {g_lora:.2e}  |vision g| {g_vis:.2e}   "
              f"cuda0 {mem0:5.1f} G  cuda1 {mem1:5.1f} G", flush=True)

    dt = time.time() - t0
    (out / "history.json").write_text(json.dumps(history, indent=2))
    first, last = history[0]["L_perc"], history[-1]["L_perc"]
    print(f"\n{args.steps} steps in {dt:.0f}s ({dt/args.steps:.1f}s/step)")
    print(f"  peak cuda:0 {max(r['gib_cuda0'] for r in history):.1f} GiB   "
          f"peak cuda:1 {max(r['gib_cuda1'] for r in history):.1f} GiB")
    print(f"L_perc {first:.4f} -> {last:.4f}  ({100*(first-last)/abs(first):+.1f}%)")
    grads_ok = any(r["lora_grad"] > 0 for r in history)
    vis_ok = args.freeze_vision or any(r["vision_grad"] > 0 for r in history)
    print(f"gradient reached the LM (LoRA): {grads_ok}")
    print(f"gradient reached the vision encoder: {vis_ok}")
    ok = last < first and grads_ok and vis_ok
    print("JOINT 2-GPU TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
