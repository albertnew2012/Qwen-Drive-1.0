#!/usr/bin/env python
"""The full detector-path pipeline, end to end, on the same frames as the VQA baseline.

    lights  ->  selector picks the ego's  ->  crop  ->  colour head  ->  answer

Boxes come either from annotation (`--boxes gt`, which isolates selection + colour) or
from the trained detector (`--boxes det`, the deployable path). Both are reported because
the gap between them is exactly what detection quality costs.

Scored on data/tlb/val.jsonl, with episode-clustered intervals, so the number is directly
comparable to the VQA baseline and the fine-tune.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'local/tlb'))
ROOT = BASE / 'data/OpenLane-V2'

from selector import Selector                                   # noqa: E402
from train_colour import build as build_colour, COLOURS         # noqa: E402
from extract_crops import crop_one                              # noqa: E402


def acc(it):
    return sum(r['pred'] == r['gt'] for r in it) / len(it) if it else 0.0


def macro(it):
    rs = []
    for c in ('red', 'green', 'yellow'):
        cl = [r for r in it if r['gt'] == c]
        if cl:
            rs.append(sum(r['pred'] == c for r in cl) / len(cl))
    return sum(rs) / len(rs) if rs else 0.0


def boot(items, stat, iters=3000, seed=0):
    by = defaultdict(list)
    for r in items:
        by[r['ep']].append(r)
    eps = list(by)
    if not eps:
        return 0., 0.
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(iters):
        p = rng.choice(len(eps), len(eps), replace=True)
        v.append(stat([x for i in p for x in by[eps[i]]]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selector', default='outputs/tlb/selector.pt')
    ap.add_argument('--colour', default='outputs/tlb/colour_head_clean.pt')
    ap.add_argument('--roi', default='data/tlb/roi/val.npz')
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--det-gt', default='data/tlb/det_val.jsonl',
                    help='per-light annotation for the same split as --gt')
    ap.add_argument('--out', default='outputs/tlb/pipeline_val.json')
    ap.add_argument('--vote', default='argmax', choices=['argmax', 'soft'],
                    help="argmax: read the single top-scoring light. "
                         "soft: every light votes, weighted by P(governs)*P(colour)")
    ap.add_argument('--topk', type=int, default=6, help='lights considered when voting')
    args = ap.parse_args()

    device = 'cuda'
    # mmap_mode does nothing for an .npz: every d['roi'][j] re-reads the whole 1.6 GB
    # array out of the zip, which made this 5x slower than the model it is feeding.
    # Materialise once.
    _z = np.load(BASE / args.roi)
    d = {k: _z[k] for k in ('roi', 'ctx', 'nl', 'box', 'col', 'gov', 'keys')}
    keys = list(d['keys'])
    kidx = {k: i for i, k in enumerate(keys)}

    ck = torch.load(BASE / args.selector, map_location='cpu')
    sel = Selector(d=ck['args']['d'], layers=ck['args']['layers']).to(device)
    sel.load_state_dict(ck['model']); sel.eval()

    cck = torch.load(BASE / args.colour, map_location='cpu')
    col = build_colour().to(device)
    col.load_state_dict(cck['model']); col.eval()

    rows = [json.loads(l) for l in open(BASE / args.gt)]
    rows.sort(key=lambda r: (r['segment'], r['timestamp']))
    eid = {}; n = 0
    for s, g in groupby(rows, key=lambda r: r['segment']):
        for lb, gg in groupby(list(g), key=lambda r: r['label']):
            for r in gg:
                eid[f"{r['segment']}/{r['timestamp']}"] = n
            n += 1

    det_rows = {f"{r['segment']}/{r['timestamp']}": r
                for r in (json.loads(l) for l in open(BASE / args.det_gt))}

    out = []
    t0 = time.time()
    with torch.no_grad():
        for r in rows:
            key = f"{r['segment']}/{r['timestamp']}"
            if key not in kidx:
                continue
            j = kidx[key]
            nl = int(d['nl'][j])
            if nl == 0:
                continue
            roi = torch.from_numpy(np.asarray(d['roi'][j], np.float32))[None].to(device)
            box = torch.from_numpy(np.asarray(d['box'][j], np.float32))[None].to(device)
            ctx = torch.from_numpy(np.asarray(d['ctx'][j], np.float32))[None].to(device)
            mask = torch.zeros(1, roi.shape[1], dtype=torch.bool, device=device)
            mask[0, :nl] = True
            lg = sel(roi, box, ctx, mask).masked_fill(~mask, -1e4)
            pick = int(lg.argmax(1))

            # the selector indexes lights sorted by area -- same order as the cache
            lights = sorted(det_rows[key]['lights'],
                            key=lambda L: -((L['box'][2]-L['box'][0])*(L['box'][3]-L['box'][1])))
            chosen = lights[pick]
            im = Image.open(ROOT / r['image']).convert('RGB')

            def colour_probs(boxes):
                ts = []
                for b in boxes:
                    c = crop_one(im, b)
                    t = torch.from_numpy(np.asarray(c, np.uint8).copy())
                    ts.append(((t.permute(2, 0, 1).float() / 255.) - 0.5) / 0.5)
                x = torch.stack(ts).to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    p = torch.softmax(col(x).float(), 1).cpu().numpy()
                return p[:, 1:4]                       # red, green, yellow

            if args.vote == 'argmax':
                rgy = colour_probs([chosen['box']])[0]
            else:
                # Reading one light is brittle: a single bad pick flips the answer. Let
                # every plausible light vote, weighted by how strongly the selector
                # believes it governs the ego -- agreement among several lights then
                # outweighs one confident mistake.
                k = min(args.topk, nl)
                order = torch.argsort(lg[0, :nl], descending=True)[:k].cpu().numpy()
                w = torch.sigmoid(lg[0, :nl]).cpu().numpy()[order]
                P = colour_probs([lights[int(i)]['box'] for i in order])
                rgy = (w[:, None] * P).sum(0)
            rgy = rgy / max(rgy.sum(), 1e-9)
            pred = ['red', 'green', 'yellow'][int(rgy.argmax())]

            out.append({'key': key, 'gt': r['label'], 'pred': pred,
                        'disc': r['discriminative'], 'w': r['max_gov_wh'][0],
                        'ep': eid[key], 'picked_is_gov': int(chosen['is_gov'] == 1),
                        'picked_colour_ann': chosen['colour'],
                        'conf': float(rgy.max() / rgy.sum())})
            if len(out) % 300 == 0:
                print(f"    {len(out)}/{len(rows)}  {time.time()-t0:.0f}s  acc {acc(out):.2%}",
                      flush=True)

    print(f"\n===== PIPELINE (selector + colour head) on {len(out)} val frames, "
          f"{len({o['ep'] for o in out})} episodes =====")
    for name, s in (('all', out),
                    ('discriminative', [o for o in out if o['disc']]),
                    ('light <12px', [o for o in out if o['w'] < 12]),
                    ('light >=20px', [o for o in out if o['w'] >= 20])):
        if not s:
            continue
        lo, hi = boot(s, acc)
        print(f"  {name:16s} n={len(s):5d} eps={len({o['ep'] for o in s}):3d}  "
              f"acc {acc(s):6.2%} [{lo:.1%},{hi:.1%}]  macroR {macro(s):6.2%}")
    print(f"  selector picked an annotated governing light: "
          f"{np.mean([o['picked_is_gov'] for o in out]):.2%}")
    print("  confusion:", {g: dict(Counter(o['pred'] for o in out if o['gt'] == g))
                           for g in ('red', 'green', 'yellow')})
    (BASE / args.out).write_text(json.dumps(out, indent=1))
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
