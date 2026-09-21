#!/usr/bin/env python
"""LoRA fine-tune of the Qwen-Drive VLM on ego-lane traffic-light colour.

The prompt is exactly the one the baseline is scored with, so the fine-tune moves the
same distribution the evaluation reads. Loss is next-token prediction restricted to the
answer tokens: the prompt, which contains ~1k image tokens, is masked out with -100.

Class balance matters here. Train is 3186 red / 1891 green / 118 yellow, so an unweighted
run learns the prior as much as the picture; `--balance` samples classes evenly instead.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE))
ROOT = BASE / 'data/OpenLane-V2'

OVERSAMPLE_CAP = 5.0        # most a rare class may be repeated, relative to the rarest

QUESTION = ("What is the colour of the traffic light controlling the lane the ego "
            "vehicle is in? Answer with one word: red, yellow, or green.")


# The model does not answer straight after "assistant\n": it emits an empty reasoning
# block first, and generate_text strips it. Training on the bare colour would teach it to
# drop that block and mismatches inference, which is what made the initial loss ~13.5.
# Keep the native prefix as unsupervised context and supervise only the answer.
THINK_PREFIX = "<think>\n\n</think>\n\n"


def build_sample(model, row, cond):
    """(input_ids, pixel_values, image_grid_thw, labels) for one frame."""
    from qwen_drive import CameraFrame
    proc = model.processor
    p = ROOT / row['image']
    if cond == 'full':
        images = [str(p)]
    else:
        im = Image.open(p).convert('RGB')
        images = [CameraFrame(im, target_size=im.size)]
    enc = proc.encode_vqa(images, QUESTION, device='cpu')
    prompt = enc['input_ids'][0]
    pre = torch.tensor(proc._encode(THINK_PREFIX), dtype=torch.long)
    ans = torch.tensor(proc._encode(row['label']) + [proc.im_end_id], dtype=torch.long)
    ids = torch.cat([prompt, pre, ans])
    labels = torch.full_like(ids, -100)
    labels[len(prompt) + len(pre):] = ans          # supervise the answer only
    return ids, enc['pixel_values'], enc['image_grid_thw'], labels


def forward_loss(model, ids, px, grid, labels, device):
    vlm = model.vlm
    ids = ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    out = vlm(input_ids=ids, pixel_values=px.to(device),
              image_grid_thw=grid.to(device),
              mm_token_type_ids=model._modality_ids(ids),
              use_cache=False)
    logits = out.logits[:, :-1]
    tgt = labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                           tgt.reshape(-1), ignore_index=-100)


def make_order(rows, balance, epochs, seed):
    rnd = random.Random(seed)
    if not balance:
        order = []
        for _ in range(epochs):
            e = list(range(len(rows))); rnd.shuffle(e); order += e
        return order
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r['label'], []).append(i)
    # Full balancing would repeat the 103 yellow frames ~25x and teach the model that
    # scene rather than the colour. Cap the oversample so rare classes are lifted, not
    # memorised.
    per = max(len(v) for v in by.values())
    per = min(per, int(min(len(v) for v in by.values()) * OVERSAMPLE_CAP))
    order = []
    for _ in range(epochs):
        e = []
        for c, idxs in by.items():
            pool = idxs * (per // len(idxs) + 1)
            e += rnd.sample(pool, per)
        rnd.shuffle(e); order += e
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/tlb/train.jsonl')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    # 'full' (921600 px -> ~880 image tokens) rather than 'hires' (~1400): hires OOMs a
    # 24 GB card once LoRA adds a backward pass, and it scored within a point of full.
    ap.add_argument('--cond', default='full', choices=['full', 'hires'])
    ap.add_argument('--max-pixels', type=int, default=0,
                    help='override the image pixel budget (0 = model default)')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--balance', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', default='outputs/tlb/lora_vqa.pt')
    ap.add_argument('--log-every', type=int, default=20)
    # A 2.5 h run that dies at step 180 with nothing on disk is a total loss, and this
    # box has segfaulted mid-run once already. Checkpoint periodically and resume.
    ap.add_argument('--save-every', type=int, default=50)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda'
    rows = [json.loads(l) for l in open(BASE / args.train)]
    if args.limit:
        rows = rows[:args.limit]

    from qwen_drive import QwenDriveForPlanning
    from training.lora import apply_lora, freeze_all_but_lora, lora_parameters
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    if args.max_pixels:
        model.processor.config.current_image_pixels = args.max_pixels
        print(f"image budget -> {args.max_pixels} px", flush=True)
    # HF gates checkpointing on `self.gradient_checkpointing and self.training`, so
    # eval() silently disables it and the 32 decoder layers keep every activation --
    # that, not the module path, was the OOM. Dropout is 0.0 in this config, so train()
    # costs nothing.
    model.vlm.train()
    n = apply_lora(model.vlm, args.rank, args.alpha)
    tr, fr = freeze_all_but_lora(model.vlm)
    for p in model.parameters():
        if not any(p is q for q in lora_parameters(model.vlm)):
            pass
    params = lora_parameters(model.vlm)
    print(f"LoRA r={args.rank} on {n} projections: {tr/1e6:.2f} M trainable / {fr/1e9:.3f} B frozen",
          flush=True)

    try:
        # non-reentrant: the reentrant path needs an input that requires grad, which a
        # LoRA-only run does not have (the embeddings are frozen).
        model.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        # The composite call only reaches the vision tower; the 32-layer language model
        # keeps every activation and is what actually blows the 24 GB card. Enable it on
        # the text stack explicitly.
        for sub in (getattr(model.vlm.model, 'language_model', None),
                    getattr(model.vlm, 'language_model', None)):
            if sub is not None and hasattr(sub, 'gradient_checkpointing_enable'):
                sub.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False})
            elif sub is not None:
                sub.gradient_checkpointing = True
                for blk in getattr(sub, 'layers', []):
                    blk.gradient_checkpointing = True
        on = [n for n, m in model.vlm.named_modules()
              if getattr(m, 'gradient_checkpointing', False)]
        print(f"gradient checkpointing: on (non-reentrant) for {len(on)} modules"
              + (f", e.g. {on[:2]}" if on else " -- NONE ENABLED, expect OOM"), flush=True)
    except Exception as e:
        print("gradient checkpointing unavailable:", e, flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    order = make_order(rows, args.balance, args.epochs, args.seed)
    total = len(order)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(total // args.accum, 1),
        pct_start=0.05, anneal_strategy='cos')

    print(f"{total} samples ({args.epochs} epoch(s), balance={args.balance}), "
          f"accum {args.accum} -> {total//args.accum} steps", flush=True)
    sp = BASE / args.save
    sp.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(step, done):
        torch.save({'lora': {k: v for k, v in model.vlm.state_dict().items() if 'lora_' in k},
                    'args': vars(args), 'step': step, 'samples_done': done}, sp)

    start_at = 0
    if args.resume and sp.exists():
        ck = torch.load(sp, map_location='cpu')
        model.vlm.load_state_dict({k: v.to(device) for k, v in ck['lora'].items()},
                                  strict=False)
        start_at = int(ck.get('samples_done', 0))
        print(f"resumed from {sp} at sample {start_at}", flush=True)

    t0 = time.time(); run = []; step = 0
    opt.zero_grad(set_to_none=True)
    for i, idx in enumerate(order):
        if i < start_at:
            continue
        ids, px, grid, labels = build_sample(model, rows[idx], args.cond)
        loss = forward_loss(model, ids, px, grid, labels, device)
        (loss / args.accum).backward()
        run.append(float(loss.detach()))
        if i == 0:
            print(f"  first step ok: loss {float(loss.detach()):.4f}, "
                  f"tokens {ids.numel()}, peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        if (i + 1) % args.accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % args.save_every == 0:
                snapshot(step, i + 1)
            if step % args.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                el = time.time() - t0
                print(f"  step {step:5d}/{total//args.accum}  loss {sum(run[-args.accum*args.log_every:])/len(run[-args.accum*args.log_every:]):.4f}  "
                      f"|g| {float(gn):.2f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{el:6.0f}s  eta {el/max(step,1)*(total//args.accum-step)/60:5.1f}m  peak {mem:.1f}G",
                      flush=True)
    snapshot(step, len(order))
    print(f"saved {sp}  ({time.time()-t0:.0f}s, final loss {sum(run[-200:])/len(run[-200:]):.4f})")


if __name__ == '__main__':
    main()
