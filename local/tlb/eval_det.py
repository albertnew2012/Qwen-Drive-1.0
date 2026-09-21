#!/usr/bin/env python
"""Score the 2D head, then run it as an ego-lane traffic-light pipeline.

Two levels, because they answer different questions:

  detection   precision / recall / AP at IoU 0.3 and 0.5 over every labelled light.
              IoU 0.5 is harsh for an 18x27 px object on 8 px cells, so 0.3 is reported
              alongside it rather than instead of it.

  downstream  the thing that actually matters: pick the detection with the highest
              score*P(gov) and report its colour as the ego's light. Scored on exactly
              the frames the VQA baseline is scored on, so the two methods are directly
              comparable, with the same episode-clustered intervals.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'local/tlb'))
ROOT = BASE / 'data/OpenLane-V2'

from det_model import TLDetHead, decode, COLOURS            # noqa: E402
from train_det import ViTTap                                # noqa: E402
from analyze import boot_ci                                 # noqa: E402


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float); b = np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/tlb/det_head.pt')
    ap.add_argument('--det-gt', default='data/tlb/det_val.jsonl')
    ap.add_argument('--vqa-gt', default='data/tlb/val.jsonl')
    ap.add_argument('--model', default='weights/Qwen-Drive-1.0-4B')
    ap.add_argument('--thr', type=float, default=0.3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='outputs/tlb/det_eval.json')
    args = ap.parse_args()

    device = 'cuda'
    ck = torch.load(BASE / args.ckpt, map_location='cpu')
    up, patch = ck['up'], ck['patch']

    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tap = ViTTap(model)

    det_rows = {f"{r['segment']}/{r['timestamp']}": r
                for r in (json.loads(l) for l in open(BASE / args.det_gt))}
    vqa_rows = [json.loads(l) for l in open(BASE / args.vqa_gt)]
    if args.limit:
        vqa_rows = vqa_rows[:args.limit]

    head = None
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    col_ok = col_n = 0
    downstream = []
    t0 = time.time()
    with torch.no_grad():
        for i, vr in enumerate(vqa_rows):
            key = f"{vr['segment']}/{vr['timestamp']}"
            feat, r, c, W0, H0 = tap.features(ROOT / vr['image'], device)
            if head is None:
                head = TLDetHead(in_dim=feat.shape[0], hid=ck['args']['hid'], up=up).to(device).float()
                head.load_state_dict(ck['head']); head.eval()
            out = head(feat.float().unsqueeze(0))
            cell = patch / up
            dets = decode(out, cell, thr=args.thr)[0]
            sx = (c * patch) / W0; sy = (r * patch) / H0
            for d in dets:                       # back to original image pixels
                d['box'] = [d['box'][0] / sx, d['box'][1] / sy, d['box'][2] / sx, d['box'][3] / sy]

            gt = det_rows.get(key)
            if gt:
                gb = [L['box'] for L in gt['lights']]
                M = iou_mat([d['box'] for d in dets], gb)
                for t in (0.3, 0.5):
                    used = set(); matched = 0
                    for di in np.argsort([-d['score'] for d in dets]):
                        if not len(gb):
                            break
                        j = int(np.argmax(M[di])) if M.shape[1] else -1
                        if j >= 0 and M[di, j] >= t and j not in used:
                            used.add(j); matched += 1
                            if t == 0.3:
                                if gt['lights'][j]['colour'] != 'unknown':
                                    col_n += 1
                                    col_ok += dets[di]['colour'] == gt['lights'][j]['colour']
                    tp[t] += matched; fp[t] += len(dets) - matched; fn[t] += len(gb) - matched

            best = max(dets, key=lambda d: d['score'] * d['gov'], default=None)
            pred = None
            if best is not None and best['gov'] > 0.5:
                pred = best['colour'] if best['colour'] != 'unknown' else None
            downstream.append({'key': key, 'gt': vr['label'], 'pred': pred,
                               'disc': vr['discriminative'], 'w': vr['max_gov_wh'][0],
                               'n_det': len(dets),
                               'gov_score': float(best['gov']) if best else 0.0})
            if (i + 1) % 200 == 0:
                ok = sum(d['pred'] == d['gt'] for d in downstream) / len(downstream)
                print(f"    {i+1}/{len(vqa_rows)}  {time.time()-t0:.0f}s  downstream {ok:.1%}",
                      flush=True)

    print("\n===== detection =====")
    for t in (0.3, 0.5):
        p = tp[t] / max(tp[t] + fp[t], 1); r_ = tp[t] / max(tp[t] + fn[t], 1)
        print(f"  IoU {t}:  P {p:.1%}  R {r_:.1%}  F1 {2*p*r_/max(p+r_,1e-9):.1%}  "
              f"(tp {tp[t]}, fp {fp[t]}, fn {fn[t]})")
    print(f"  colour accuracy on matched, non-unknown lights: {col_ok/max(col_n,1):.1%} (n={col_n})")

    # episode ids for clustered intervals
    vqa_rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    eid = {}; n = 0
    for seg, g in groupby(vqa_rows, key=lambda r: r['segment']):
        for lab, gg in groupby(list(g), key=lambda r: r['label']):
            for r_ in gg:
                eid[f"{r_['segment']}/{r_['timestamp']}"] = n
            n += 1
    for d in downstream:
        d['ep'] = eid.get(d['key'], -1)

    def acc(items):
        return sum(x['pred'] == x['gt'] for x in items) / len(items) if items else 0.0

    def prec(items):
        a = [x for x in items if x['pred'] is not None]
        return sum(x['pred'] == x['gt'] for x in a) / len(a) if a else 0.0

    print("\n===== downstream: ego-lane light colour =====")
    for name, sel in (('all', lambda d: True),
                      ('discriminative', lambda d: d['disc']),
                      ('light <12px', lambda d: d['w'] < 12),
                      ('light >=20px', lambda d: d['w'] >= 20)):
        s = [d for d in downstream if sel(d)]
        if not s:
            continue
        lo, hi = boot_ci(s, acc)
        plo, phi = boot_ci(s, prec)
        cov = sum(d['pred'] is not None for d in s) / len(s)
        print(f"  {name:16s} n={len(s):5d} eps={len({d['ep'] for d in s}):3d}  "
              f"acc {acc(s):6.1%} [{lo:.1%},{hi:.1%}]   "
              f"precision-when-answered {prec(s):6.1%} [{plo:.1%},{phi:.1%}]  coverage {cov:.1%}")
    print("  confusion:", {g: dict(Counter(d['pred'] for d in downstream if d['gt'] == g))
                           for g in ('red', 'green', 'yellow')})
    (BASE / args.out).write_text(json.dumps(downstream, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
