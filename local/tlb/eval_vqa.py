#!/usr/bin/env python
"""Baseline: can Qwen-Drive name the colour of the light governing the ego's lane?

Two things are measured separately, because they come apart:

  assoc    "the light controlling the lane the ego vehicle is in"
  salient  "the traffic light ahead"            -- no association asked for

On `discriminative` frames another visible light disagrees with the ego's, so a model
that just reports the most salient lamp scores at chance there while a model that truly
associates does not. That gap is the measurement.

Conditions control how much resolution reaches the model:
  full   default budget, 1600x900 -> ~1280x720 (a 18px lamp becomes ~14px)
  hires  target_size pins native 1600x900 (13.1 MP budget)
  crop   oracle crop around the governing box: an upper bound, not deployable
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from collections import Counter

import torch
from PIL import Image

sys.path.insert(0, '/home/albert/Desktop/Qwen-Drive-1.0/src')
ROOT = Path('/home/albert/Desktop/Qwen-Drive-1.0/data/OpenLane-V2')

COLOUR_RE = re.compile(r'\b(red|green|yellow|amber)\b', re.I)
QUESTIONS = {
    'assoc': "What is the colour of the traffic light controlling the lane the ego "
             "vehicle is in? Answer with one word: red, yellow, or green.",
    'salient': "What colour is the traffic light ahead? Answer with one word: "
               "red, yellow, or green.",
    # Overexposure destroys hue but not geometry. The crop colour head wins on exactly
    # those frames by reading which third of the housing is lit; this asks the VLM to do
    # the same, and is a targeted test of that diagnosis rather than a tuned prompt.
    'hint': "What is the colour of the traffic light controlling the lane the ego vehicle "
            "is in? If the lamp looks washed out or overexposed, judge it by which "
            "position in the housing is lit: the top lamp is red, the middle is yellow, "
            "the bottom is green. Answer with one word: red, yellow, or green.",
    # For the whole split, where 78% of frames have no ego-lane light at all. Without a
    # 'none' option the model can only ever be wrong on those, so the question has to
    # offer it -- and the interesting failure is naming a colour when other lights are
    # visible but none of them is the ego's.
    'full': "Is there a traffic light controlling the lane the ego vehicle is in, and if "
            "so what colour is it? Other traffic lights may be visible that control "
            "different lanes or cross traffic; those do not count. Answer with one word: "
            "red, yellow, green, or none.",
    'full_hint': "Is there a traffic light controlling the lane the ego vehicle is in, and "
                 "if so what colour is it? Other traffic lights may be visible that "
                 "control different lanes or cross traffic; those do not count. If a lamp "
                 "looks washed out, judge it by which position in the housing is lit: top "
                 "is red, middle is yellow, bottom is green. Answer with one word: red, "
                 "yellow, green, or none.",
}


NONE_RE = re.compile(r'\b(none|no traffic light|no light|not visible|there is no)\b', re.I)


def parse(text):
    t = text or ''
    m = COLOUR_RE.search(t)
    if not m:
        # 'none' is a real answer on the full split, distinct from an unparseable reply
        return 'none' if NONE_RE.search(t) else None
    c = m.group(1).lower()
    return 'yellow' if c == 'amber' else c


def build(cond, row):
    from qwen_drive import CameraFrame
    p = ROOT / row['image']
    if cond == 'full':
        return [str(p)]
    im = Image.open(p).convert('RGB')
    if cond == 'hires':
        return [CameraFrame(im, target_size=im.size)]
    # oracle crop around the governing light, with context, upscaled
    bs = row['gov_boxes']
    x1 = min(b[0] for b in bs); y1 = min(b[1] for b in bs)
    x2 = max(b[2] for b in bs); y2 = max(b[3] for b in bs)
    w, h = x2 - x1, y2 - y1
    px, py = max(w * 1.5, 100), max(h * 1.5, 100)
    c = im.crop((int(max(0, x1-px)), int(max(0, y1-py)),
                 int(min(im.width, x2+px)), int(min(im.height, y2+py))))
    if max(c.size) < 640:
        s = 640 / max(c.size)
        c = c.resize((int(c.width*s), int(c.height*s)), Image.LANCZOS)
    return [CameraFrame(c, target_size=c.size)]


def report(res):
    """Accuracy, plus macro-averaged recall.

    The discriminative subset is ~88% green, so plain accuracy there is nearly free:
    always answering green scores 87.6%. Macro recall over the classes actually present
    is what separates association from guessing the prior.
    """
    def acc(sel):
        v = [r for r in res if sel(r)]
        n = len(v)
        return (sum(r['pred'] == r['gt'] for r in v) / n if n else 0.0), n

    def macro(sel):
        # include 'none' when the split contains it: on the whole validation set it is
        # 78% of frames and leaving it out would hide the only class that matters there
        v = [r for r in res if sel(r)]
        recs = []
        for c in ('red', 'green', 'yellow', 'none'):
            cls = [r for r in v if r['gt'] == c]
            if cls:
                recs.append(sum(r['pred'] == c for r in cls) / len(cls))
        return sum(recs) / len(recs) if recs else 0.0

    out = {}
    for name, sel in (('all', lambda r: True),
                      ('disc', lambda r: r['disc']),
                      ('nondisc', lambda r: not r['disc']),
                      ('big', lambda r: r['wh'][0] >= 20),
                      ('small', lambda r: r['wh'][0] < 20)):
        out[name], out['n_' + name] = acc(sel)
        out['m_' + name] = macro(sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/tlb/val.jsonl')
    ap.add_argument('--cond', default='full', choices=['full', 'hires', 'crop'])
    ap.add_argument('--question', default='assoc', choices=list(QUESTIONS))
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-new-tokens', type=int, default=16)
    ap.add_argument('--out', default='')
    ap.add_argument('--lora', default='', help='LoRA checkpoint to load before scoring')
    args = ap.parse_args()

    base = Path('/home/albert/Desktop/Qwen-Drive-1.0')
    rows = [json.loads(l) for l in open(base / args.src)]
    if args.limit:
        rows = rows[:args.limit]
    out_path = base / (args.out or f"outputs/tlb/base_{args.question}_{args.cond}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        done = {r['key']: r for r in json.loads(out_path.read_text())}
        print(f"  resuming with {len(done)} scored", flush=True)

    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if args.lora:
        import sys as _s
        _s.path.insert(0, str(base))
        from training.lora import apply_lora
        ck = torch.load(base / args.lora, map_location='cpu')
        rank = ck['args'].get('rank', 16); alpha = ck['args'].get('alpha', 32)
        apply_lora(model.vlm, rank, alpha)
        missing, unexpected = model.vlm.load_state_dict(
            {k: v.to(model.device) for k, v in ck['lora'].items()}, strict=False)
        got = len(ck['lora'])
        assert not unexpected, f"unexpected LoRA keys: {unexpected[:3]}"
        print(f"  loaded LoRA r={rank} ({got} tensors) from {args.lora}", flush=True)
        model.eval()

    Q = QUESTIONS[args.question]
    res = list(done.values()); t0 = time.time()
    for i, r in enumerate(rows):
        key = f"{r['segment']}/{r['timestamp']}"
        if key in done:
            continue
        txt = model.generate_text(build(args.cond, r), Q,
                                  max_new_tokens=args.max_new_tokens).text
        res.append({'key': key, 'gt': r['label'], 'pred': parse(txt),
                    'raw': txt.strip()[:60],
                    # the full split has no governing box, so these are absent there
                    'disc': r.get('discriminative', False),
                    'wh': r.get('max_gov_wh', [0, 0]),
                    'why': r.get('why', 'governed'),
                    'n_other': r.get('n_other', 0)})
        if (i + 1) % 100 == 0:
            m = report(res)
            print(f"    {len(res):5d}/{len(rows)}  {time.time()-t0:5.0f}s  "
                  f"all {m['all']:.1%} (n={m['n_all']})  disc {m['disc']:.1%} (n={m['n_disc']})",
                  flush=True)
            out_path.write_text(json.dumps(res, indent=1))

    out_path.write_text(json.dumps(res, indent=1))
    m = report(res)
    print(f"\n===== {args.question} / {args.cond} =====")
    print(f"  {'subset':9s} {'acc':>7s} {'macroR':>8s}   n")
    for k in ('all', 'disc', 'nondisc', 'big', 'small'):
        print(f"  {k:9s} {m[k]:7.1%} {m['m_'+k]:8.1%}   {m['n_'+k]}")
    import itertools
    print("  confusion (gt -> pred):")
    for g in ('red', 'green', 'yellow', 'none'):
        rowc = Counter(r['pred'] for r in res if r['gt'] == g)
        if rowc:
            print(f"      {g:7s} -> {dict(rowc)}")
    print("  predicted distribution:", Counter(r['pred'] for r in res).most_common())
    print("  gt distribution:       ", Counter(r['gt'] for r in res).most_common())
    print(f"  unparseable: {sum(r['pred'] is None for r in res)}")
    print(f"  wrote {out_path}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
