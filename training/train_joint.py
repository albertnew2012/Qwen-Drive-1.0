"""Stage 2 - joint perception + VQA training, VLM trainable.

    perception samples -> L_perc = L_det + L_occ + L_map
    language samples   -> L_ntp  (next-token prediction)

and, from the report, "the BEV perception head uses a learning rate 20x that of
the VLM". Mixing general vision-language data in is what stops the model
forgetting how to talk; that is the entire point of the stage.

MEMORY. The paper fine-tunes the VLM fully. Weights + grads + two Adam moments
for 4.5 B parameters is ~72 GB before activations, so that is a multi-GPU job.
This script defaults to LoRA on the attention projections (~0.3 % of the
parameters) plus gradient checkpointing, which fits on one 24 GB card. ``--full``
selects the faithful path for anyone with the hardware. The deviation is
recorded in study/08.

    python training/train_joint.py --steps 20 --overfit
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
from training.differentiable import enable_training_ops, set_accum_dtype
from training.lora import apply_lora, freeze_all_but_lora, lora_parameters
from training.losses import (HungarianMatcher3D, detection_loss, map_loss,
                             occupancy_loss)


def taps_with_grad(head, vlm, inputs, n_cam):
    """The two VLM taps, WITH gradient - stage 2 trains the VLM, so the cache
    used by stages 1 and 3 is not available here."""
    captured = {}
    visual = vlm.model.visual
    h = visual.merger.register_forward_hook(
        lambda m, a, o=None: captured.__setitem__("p", a[0]))
    try:
        out = vlm(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                  image_grid_thw=inputs["image_grid_thw"],
                  mm_token_type_ids=head._modality_ids(inputs["input_ids"]),
                  use_cache=False, output_hidden_states=True)
    finally:
        h.remove()
    hidden = vlm.model.language_model.norm(out.hidden_states[-1])
    patches = visual.merger.norm(captured["p"])
    vit_feats = head._premerge_grids(patches, inputs["image_grid_thw"])
    gh = inputs["image_grid_thw"][-1, 1].item()
    gw = inputs["image_grid_thw"][-1, 2].item()
    tpi = gh // 2 * gw // 2
    mask = inputs["input_ids"][0] == vlm.config.image_token_id
    llm = hidden[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1)
    return torch.stack(vit_feats[-n_cam:], 0), llm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--out", default="outputs/train_joint")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--full", action="store_true",
                    help="fine-tune the whole VLM (needs far more than 24 GB)")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--fp32-accum", action="store_true",
                    help="keep the voxel scatter in fp32, as the kernel does")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = JointTrainConfig()
    torch.manual_seed(cfg.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    enable_training_ops()
    dtype = getattr(torch, cfg.amp_dtype)
    if not args.fp32_accum:
        set_accum_dtype(dtype)     # halves the 3.93 GiB voxel scatter buffer
        print(f'voxel scatter accumulates in {cfg.amp_dtype} (fp32 needs 3.93 GiB)')

    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=dtype, attn_implementation="sdpa")
    vlm = holder.vlm
    del holder.planning_expert
    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).to(args.device)
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm.to(args.device), proc)
    bev = head.bev_modeling.train()

    if args.full:
        for p in vlm.parameters():
            p.requires_grad_(True)
        vlm_params = [p for p in vlm.parameters() if p.requires_grad]
        print(f"FULL fine-tuning: {sum(p.numel() for p in vlm_params)/1e9:.3f} B VLM params")
    else:
        n = apply_lora(vlm, cfg.lora_rank, cfg.lora_alpha)
        tr, fr = freeze_all_but_lora(vlm)
        vlm_params = lora_parameters(vlm)
        print(f"LoRA r={cfg.lora_rank} on {n} projections: "
              f"{tr/1e6:.2f} M trainable / {fr/1e9:.3f} B frozen "
              f"({100*tr/(tr+fr):.3f} %)")
    vlm.train()

    if not args.no_checkpoint:
        nc = enable_bev_checkpointing(bev)
        print(f"gradient checkpointing on {nc} BEV modules (incl. the view transform)")
        # The VLM is the other half of the budget: 4.5 B parameters of activations
        # over ~2700 tokens. transformers can recompute those too.
        for m in (vlm, getattr(vlm, "model", None)):
            if m is not None and hasattr(m, "gradient_checkpointing_enable"):
                try:
                    m.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                    print(f"  VLM gradient checkpointing enabled on {type(m).__name__}")
                    break
                except Exception as exc:      # not all stacks support it
                    print(f"  VLM checkpointing unavailable ({exc})")

    head_params = [p for p in bev.parameters() if p.requires_grad]
    head_lr = cfg.vlm_lr * cfg.head_lr_multiplier
    print(f"lr: VLM {cfg.vlm_lr:.1e}   head {head_lr:.1e}  "
          f"({cfg.head_lr_multiplier:g}x, from the paper)")
    opt = torch.optim.AdamW([
        {"params": vlm_params, "lr": cfg.vlm_lr, "name": "vlm"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ], weight_decay=cfg.weight_decay)
    matcher = HungarianMatcher3D(cfg.match_cls_cost, cfg.match_reg_cost)

    frames = sorted(p for p in Path(args.frames).iterdir() if p.is_dir())
    import numpy as np
    history, t0 = [], time.time()
    for step in range(args.steps):
        fp = frames[0] if args.overfit else frames[step % len(frames)]
        frame = PerceptionFrame(fp)
        inputs, metas = proc(frame, device=args.device)
        gt = np.load(fp / "gt.npz")
        n_cam = len(metas["cam_order"])

        vit, llm = taps_with_grad(head, vlm, inputs, n_cam)
        outs = bev(img_vit_feats=vit.to(dtype), img_llm_feats=llm.to(dtype),
                   img_metas=[metas])

        dev = args.device
        d = detection_loss(outs["all_cls_scores"].float(), outs["all_bbox_preds"].float(),
                           [torch.from_numpy(gt["boxes"]).float().to(dev)],
                           [torch.from_numpy(gt["labels"]).long().to(dev)],
                           matcher, cls_weight=cfg.det_cls_weight,
                           reg_weight=cfg.det_reg_weight)
        d.pop("det_matched")
        o = occupancy_loss(outs["occ_pred"],
                           torch.from_numpy(gt["occ"]).long().to(dev),
                           empty_label=OCC_EMPTY_LABEL,
                           focal_weight=cfg.occ_focal_weight,
                           max_points=cfg.occ_max_points)
        m = map_loss(outs["seg_preds"], torch.from_numpy(gt["map"]).long().to(dev),
                     focal_weight=cfg.map_focal_weight, max_points=cfg.map_max_points)
        perc = sum(d.values()) + sum(o.values()) + sum(m.values())

        opt.zero_grad(set_to_none=True)
        perc.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in list(vlm.parameters()) + head_params if p.requires_grad],
            cfg.grad_clip_norm)
        opt.step()

        vlm_grad = sum(float(p.grad.abs().sum()) for p in vlm_params if p.grad is not None)
        row = {"step": step, "L_perc": float(perc.detach()),
               "vlm_grad_abs_sum": vlm_grad, "grad_norm": float(gn),
               **{k: float(v.detach()) for k, v in {**d, **o, **m}.items()}}
        history.append(row)
        if step % 2 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  L_perc {row['L_perc']:9.4f}   "
                  f"det {row['det_cls']:.3f}/{row['det_reg']:.3f}  "
                  f"occ {row['occ_focal']:.3f}  map {row['map_focal']:.3f}   "
                  f"|VLM grad| {vlm_grad:.3e}   peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    dt = time.time() - t0
    (out / "history.json").write_text(json.dumps(history, indent=2))
    first, last = history[0]["L_perc"], history[-1]["L_perc"]
    print(f"\n{args.steps} steps in {dt:.0f}s ({dt/args.steps:.1f}s/step)   "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    print(f"L_perc {first:.4f} -> {last:.4f}  ({100*(first-last)/abs(first):+.1f}%)")
    grads_flowed = any(r["vlm_grad_abs_sum"] > 0 for r in history)
    print(f"gradient reached the VLM: {grads_flowed}")
    if args.overfit:
        ok = last < first * 0.8 and grads_flowed
        print("JOINT TEST:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
