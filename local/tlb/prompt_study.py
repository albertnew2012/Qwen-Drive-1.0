#!/usr/bin/env python
"""Choose the VQA question on dev, never on val.

A question written by hand and never checked is an untested instrument: a high score may
be measuring the phrasing rather than the model. This sweeps candidate phrasings on the
dev split (46 train segments, disjoint from val) so the final val number is reported with
a question that was selected before val was ever looked at.

Three groups, and the contrast between them is the point:

  ASSOCIATION  asks for the light governing the ego's own lane -- the real task
  CONTROL      asks for the salient/most visible light, no association at all.
               If a control scores the same as an association prompt, the association
               clause is doing nothing and the score is salience, not understanding.
  FORMAT       same question, different answer format or option order, to expose
               ordering bias and word-frequency bias rather than measuring driving.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
ROOT = BASE / 'data/OpenLane-V2'

PROMPTS = {
  # ---- association ----
  'ego_jargon': ("assoc",
    "What is the colour of the traffic light controlling the lane the ego vehicle is in? "
    "Answer with one word: red, yellow, or green."),
  'first_person': ("assoc",
    "You are driving this car. What colour is the traffic light for your own lane? "
    "Answer with one word: red, yellow, or green."),
  'exclude_cross': ("assoc",
    "Other traffic lights may be visible for cross traffic or for other lanes. Ignoring "
    "those, what colour is the traffic light that applies to the lane you are driving in? "
    "Answer with one word: red, yellow, or green."),
  'straight_ahead': ("assoc",
    "Look at the traffic light directly facing you that controls your lane of travel. "
    "What colour is it? Answer with one word: red, yellow, or green."),
  'action': ("assoc",
    "You are driving. Does the signal for your own lane tell you to stop or to go? "
    "Answer with its colour only: red, yellow, or green."),
  'cot': ("assoc",
    "First find the traffic light that governs the lane you are driving in, ignoring "
    "signals for cross traffic and for other lanes. Then answer. "
    "Reply with exactly: ANSWER: <red|yellow|green>"),
  'ground_then_colour': ("assoc",
    "Several traffic lights may be visible. Identify the one that controls your own lane "
    "by its position in the image (left, centre, or right), then give its colour. "
    "Reply with exactly: POSITION, COLOUR"),
  'allow_none': ("assoc",
    "What colour is the traffic light controlling the lane you are driving in? Answer "
    "with one word: red, yellow, green, or none if no traffic light controls your lane."),
  # ---- format / bias controls (same question, different surface) ----
  # Targets the one failure mode that dominates the VQA path: shooting into the sun blows
  # the lamp out to white, hue is gone, and the model says yellow. Position within the
  # housing survives overexposure -- that is exactly what the crop colour head uses.
  'position_hint': ("assoc",
    "What is the colour of the traffic light controlling the lane the ego vehicle is in? "
    "If the lamp looks washed out or overexposed, judge it by which position in the "
    "housing is lit: the top lamp is red, the middle is yellow, the bottom is green. "
    "Answer with one word: red, yellow, or green."),
  'ego_reversed': ("format",
    "What is the colour of the traffic light controlling the lane the ego vehicle is in? "
    "Answer with one word: green, yellow, or red."),
  'mcq': ("format",
    "Which describes the traffic light controlling your own lane? "
    "(A) red  (B) yellow  (C) green. Answer with one letter."),
  # ---- association-free controls ----
  'salient': ("control",
    "What colour is the traffic light ahead? Answer with one word: red, yellow, or green."),
  'any_light': ("control",
    "What colour is the most clearly visible traffic light in this image? "
    "Answer with one word: red, yellow, or green."),
}

COLOUR_RE = re.compile(r'\b(red|green|yellow|amber)\b', re.I)
MCQ_RE = re.compile(r'\b([ABC])\b')
MCQ_MAP = {'A': 'red', 'B': 'yellow', 'C': 'green'}


def parse(name, text):
    t = (text or '').strip()
    if name == 'allow_none' and re.search(r'\bnone\b', t, re.I) and not COLOUR_RE.search(t):
        return 'none'
    if name == 'mcq':
        m = MCQ_RE.search(t)
        if m:
            return MCQ_MAP[m.group(1).upper()]
        # fall through: the model may answer with the word anyway
    if name == 'cot' and 'ANSWER' in t.upper():
        t = t[t.upper().rindex('ANSWER'):]
    m = COLOUR_RE.search(t)
    if not m:
        return None
    c = m.group(1).lower()
    return 'yellow' if c == 'amber' else c


def macro(items):
    rs = []
    for c in ('red', 'green', 'yellow'):
        cl = [r for r in items if r['gt'] == c]
        if cl:
            rs.append(sum(r['pred'] == c for r in cl) / len(cl))
    return sum(rs) / len(rs) if rs else 0.0


def acc(items):
    return sum(r['pred'] == r['gt'] for r in items) / len(items) if items else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/tlb/dev_probe.jsonl')
    ap.add_argument('--prompts', nargs='+', default=list(PROMPTS))
    ap.add_argument('--cond', default='hires', choices=['full', 'hires'])
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out-dir', default='outputs/tlb/prompts')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / args.src)]
    if args.limit:
        rows = rows[:args.limit]
    out_dir = BASE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from qwen_drive import QwenDriveForPlanning, CameraFrame
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()

    for name in args.prompts:
        kind, q = PROMPTS[name]
        op = out_dir / f"{name}_{args.cond}.json"
        if op.exists():
            print(f"  skip {name} (done)", flush=True)
            continue
        res = []
        t0 = time.time()
        mnt = 96 if name == 'cot' else 16
        for r in rows:
            p = ROOT / r['image']
            if args.cond == 'full':
                img = [str(p)]
            else:
                im = Image.open(p).convert('RGB')
                img = [CameraFrame(im, target_size=im.size)]
            txt = model.generate_text(img, q, max_new_tokens=mnt).text
            res.append({'key': f"{r['segment']}/{r['timestamp']}", 'gt': r['label'],
                        'pred': parse(name, txt), 'raw': txt.strip()[:120],
                        'disc': r['discriminative'], 'w': r['max_gov_wh'][0]})
        op.write_text(json.dumps({'name': name, 'kind': kind, 'question': q,
                                  'cond': args.cond, 'res': res}, indent=1))
        d = [x for x in res if x['disc']]
        print(f"  {name:15s} [{kind:7s}] acc {acc(res):6.1%}  macroR {macro(res):6.1%}  | "
              f"disc acc {acc(d):6.1%} macroR {macro(d):6.1%}  | "
              f"pred={dict(Counter(x['pred'] for x in res))}  {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
