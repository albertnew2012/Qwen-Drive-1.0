#!/usr/bin/env python
"""Task 1 v2: fine-tune the VLM on the question it will actually be asked.

v1 (train_vqa.py) trained only on frames that HAVE an ego-lane light, with a question that
offers no way to say "none". On the whole validation split that model names a colour on
77% of the frames where lights are visible but belong to another lane -- it was never
shown an example where the right answer is "none", so it cannot produce one.

v2 trains on the four-way question including `none`, with the training mix deliberately
containing the hard negatives: 1100 frames where lights are visible but none is the ego's,
alongside 500 with no light at all.

Everything else matches v1: LoRA r=16 on attention projections, loss on the answer tokens
only, the model's own thinking prefix kept as unsupervised context.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE))
ROOT = BASE / 'data/OpenLane-V2'

QUESTION = ("Is there a traffic light controlling the lane the ego vehicle is in, and if "
            "so what colour is it? Other traffic lights may be visible that control "
            "different lanes or cross traffic; those do not count. Answer with one word: "
            "red, yellow, green, or none.")


def build_sample(model, row, cond='full'):
    proc = model.processor
    p = str(ROOT / row['image'])
    if cond == 'hires':
        from qwen_drive import CameraFrame
        from PIL import Image
        im = Image.open(p).convert('RGB')
        imgs = [CameraFrame(im, target_size=im.size)]
    else:
        imgs = [p]
    enc = proc.encode_vqa(imgs, QUESTION, device=model.device)
    ids = enc['input_ids']
    # the released model opens with an empty thinking block; keep it so the fine-tune does
    # not teach the model to drop its own format
    ans = proc._encode("<think>\n\n</think>\n\n" + row['label']) + [proc.im_end_id]
    ans_t = torch.tensor([ans], dtype=torch.long, device=model.device)
    full = torch.cat([ids, ans_t], dim=1)
    labels = full.clone()
    labels[0, :ids.shape[1]] = -100          # supervise the answer only
    return full, enc['pixel_values'], enc['image_grid_thw'], labels


def forward_loss(model, ids, px, grid, labels):
    out = model.vlm(input_ids=ids, pixel_values=px, image_grid_thw=grid,
                    mm_token_type_ids=model._modality_ids(ids), use_cache=False)
    logits = out.logits[:, :-1]
    tgt = labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                           tgt.reshape(-1), ignore_index=-100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/tlb/ft_full_train.jsonl')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--cond', default='full')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--save', default='outputs/tlb/lora_vqa_v2.pt')
    args = ap.parse_args()

    device = 'cuda'
    rows = [json.loads(l) for l in open(BASE / args.train)]
    if args.limit:
        rows = rows[:args.limit]

    from qwen_drive import QwenDriveForPlanning
    from training.lora import apply_lora, freeze_all_but_lora, lora_parameters
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    n = apply_lora(model.vlm, args.rank, args.alpha)
    tr, fr = freeze_all_but_lora(model.vlm)
    params = lora_parameters(model.vlm)
    print(f"LoRA r={args.rank} on {n} projections: {tr/1e6:.2f} M trainable / {fr/1e9:.3f} B frozen",
          flush=True)

    # train() is required: HF gates checkpointing on self.training, and eval() silently
    # disables it -- the difference between 13 GiB and an OOM on a 24 GB card
    model.vlm.train()
    try:
        model.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        for sub in (getattr(getattr(model.vlm, 'model', None), 'language_model', None),
                    getattr(model.vlm, 'language_model', None)):
            if sub is None:
                continue
            if hasattr(sub, 'gradient_checkpointing_enable'):
                sub.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False})
            sub.gradient_checkpointing = True
            for blk in getattr(sub, 'layers', []):
                blk.gradient_checkpointing = True
        on = [k for k, m in model.vlm.named_modules() if getattr(m, 'gradient_checkpointing', False)]
        print(f"gradient checkpointing on for {len(on)} modules", flush=True)
    except Exception as e:
        print("gradient checkpointing unavailable:", e, flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    order = []
    rnd = random.Random(0)
    for _ in range(args.epochs):
        e = list(range(len(rows))); rnd.shuffle(e); order += e
    steps = max(len(order) // args.accum, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.05, anneal_strategy='cos')
    sp = BASE / args.save
    sp.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(step, done):
        torch.save({'lora': {k: v for k, v in model.vlm.state_dict().items() if 'lora_' in k},
                    'args': vars(args), 'step': step, 'samples_done': done,
                    'question': QUESTION}, sp)

    start_at = 0
    if args.resume and sp.exists():
        ck = torch.load(sp, map_location='cpu')
        model.vlm.load_state_dict({k: v.to(device) for k, v in ck['lora'].items()}, strict=False)
        start_at = int(ck.get('samples_done', 0))
        print(f"resumed at sample {start_at}", flush=True)

    print(f"{len(order)} samples -> {steps} steps", flush=True)
    t0 = time.time(); step = 0; run = []
    opt.zero_grad(set_to_none=True)
    for i, idx in enumerate(order):
        if i < start_at:
            continue
        ids, px, grid, labels = build_sample(model, rows[idx], args.cond)
        loss = forward_loss(model, ids, px, grid, labels)
        (loss / args.accum).backward()
        run.append(float(loss.detach()))
        if i == 0 or (start_at and i == start_at):
            print(f"  first step ok: loss {run[-1]:.4f}, peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        if (i + 1) % args.accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % args.save_every == 0:
                snapshot(step, i + 1)
            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"  step {step:5d}/{steps}  loss {sum(run[-80:])/len(run[-80:]):.4f}  "
                      f"|g| {float(gn):.2f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{el:5.0f}s eta {el/step*(steps-step)/60:5.1f}m", flush=True)
    snapshot(step, len(order))
    print(f"saved {sp}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
