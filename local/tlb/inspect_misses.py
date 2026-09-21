#!/usr/bin/env python
"""Render the frames a run got wrong, so the failures can be read rather than guessed.

Some 'errors' are the ground truth's fault -- a 2-hop propagation can tie the ego to a
light past the intersection. Looking at them is the only way to tell that apart from a
model error, and it decides whether 95% is even reachable on this label set.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
R = BASE / 'data/OpenLane-V2'
RGB = {'red': (255, 60, 60), 'green': (60, 255, 60), 'yellow': (255, 220, 60), None: (200, 200, 200)}
T = 420


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/tlb/val.jsonl')
    ap.add_argument('--res', required=True)
    ap.add_argument('--n', type=int, default=9)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--context', action='store_true', help='whole frame instead of a crop')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    gt = {f"{r['segment']}/{r['timestamp']}": r
          for r in (json.loads(l) for l in open(BASE / args.gt))}
    res = json.loads((BASE / args.res).read_text())
    bad = [x for x in res if x['pred'] != x['gt'] and x['key'] in gt]
    print(f"{len(bad)} misses of {len(res)}")
    # one per episode-ish: dedupe by segment so the sheet is not 9 copies of one scene
    seen = set(); pick = []
    for x in bad:
        seg = x['key'].split('/')[0]
        if seg in seen:
            continue
        seen.add(seg); pick.append(x)
    print(f"{len(pick)} distinct segments among the misses; showing {min(args.n, len(pick))}")
    pick = pick[:args.n]

    cols = args.cols; rn = (len(pick) + cols - 1) // cols
    sheet = Image.new('RGB', (T * cols, T * rn), (18, 18, 18))
    for i, x in enumerate(pick):
        r = gt[x['key']]
        im = Image.open(R / r['image']).convert('RGB')
        bs = r['gov_boxes']
        if args.context:
            crop = im; ox = oy = 0; s = T / max(im.size)
            crop = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
            sx = sy = s
        else:
            x1 = min(b[0] for b in bs); y1 = min(b[1] for b in bs)
            x2 = max(b[2] for b in bs); y2 = max(b[3] for b in bs)
            pad = max((x2 - x1), (y2 - y1)) * 2.5 + 90
            ox, oy = max(0, x1 - pad), max(0, y1 - pad)
            crop = im.crop((int(ox), int(oy), int(min(im.width, x2 + pad)), int(min(im.height, y2 + pad))))
            sx = T / crop.width; sy = T / crop.height
            crop = crop.resize((T, T), Image.LANCZOS)
        d = ImageDraw.Draw(crop)
        for b in r['other_boxes']:
            d.rectangle([(b[0]-ox)*sx, (b[1]-oy)*sy, (b[2]-ox)*sx, (b[3]-oy)*sy],
                        outline=(255, 255, 255), width=2)
        for b in bs:
            d.rectangle([(b[0]-ox)*sx, (b[1]-oy)*sy, (b[2]-ox)*sx, (b[3]-oy)*sy],
                        outline=RGB[r['label']], width=3)
        d.rectangle([0, 0, T, 34], fill=(0, 0, 0))
        d.text((4, 3), f"GT {x['gt'].upper()}  ->  SAID {str(x['pred']).upper()}", fill=RGB[x['gt']])
        d.text((4, 19), f"{x['key']}  gov {r['max_gov_wh'][0]:.0f}x{r['max_gov_wh'][1]:.0f}px  "
                        f"disc={int(r['discriminative'])} other={r['other_colours']}",
               fill=(190, 190, 190))
        sheet.paste(crop, (T * (i % cols), T * (i // cols)))
    sheet.save(args.out)
    print('wrote', args.out)
    for x in pick:
        r = gt[x['key']]
        extra = ''
        if 'picked_is_gov' in x:      # pipeline results carry which light was chosen
            extra = (f" picked_gov={x['picked_is_gov']} "
                     f"picked_ann={x.get('picked_colour_ann')} conf={x.get('conf', 0):.2f}")
        print(f"  {x['key']}  gt={x['gt']:6s} said={str(x['pred']):6s} "
              f"{r['max_gov_wh']} disc={int(r['discriminative'])} other={r['other_colours']} "
              f"n_gov={r['n_gov']} n_other={r['n_other']}{extra}")


if __name__ == '__main__':
    main()
