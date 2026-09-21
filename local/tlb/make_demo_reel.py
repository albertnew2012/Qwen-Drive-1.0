#!/usr/bin/env python
"""Stitch the per-session clips into one reel with a title card before each.

The cards say what each session is for, including the one where the pipeline loses. A reel
that only showed wins would be a sales deck, not a result.
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/albert/Desktop/Qwen-Drive-1.0')
OUT = BASE / 'outputs/tlb/demo'
F = '/usr/share/fonts/truetype/dejavu/'
W, H = 1280, 896

CARDS = [
    ('11064', 'Overexposed: the hue is gone',
     ['Filmed into the sun. The lit lamp is a blown-out white blob and the',
      'colour that says "red" is no longer in the pixels. What survives is',
      'geometry: the lit lamp sits at the TOP of the housing.',
      '',
      'VQA 5%        PIPELINE 95%',
      '',
      'The colour head reads lamp position, so exposure does not destroy it.',
      'Two fine-tunes and a prompt spelling out the rule all failed here.']),
    ('11063', 'Several lights, only one is yours',
     ['19 of these frames are discriminative: another visible light shows a',
      'different colour, so reporting the most salient lamp gives the wrong',
      'answer. Up to 11 lights are detected at once.',
      '',
      'VQA 62%       PIPELINE 71%',
      '',
      'Cyan boxes are detections; the thick one is what the selector chose.']),
    ('11144', 'The ordinary case, including a yellow',
     ['A clean daytime approach. Both methods are right almost everywhere,',
      'which is the usual situation and why overall accuracy flatters both.',
      '',
      'VQA 97%       PIPELINE 97%']),
    ('11149', 'Where the pipeline LOSES',
     ['Night, heavy sensor noise, a 16 px lamp. The detector and selector',
      'both degrade, and the VLM does better here.',
      '',
      'VQA 80%       PIPELINE 40%',
      '',
      'Included deliberately: the pipeline wins on average, not everywhere.']),
]


def card(seg, title, lines, secs=4.0, fps=4.0):
    fb = ImageFont.truetype(F + 'DejaVuSans-Bold.ttf', 40)
    fs = ImageFont.truetype(F + 'DejaVuSans.ttf', 23)
    ft = ImageFont.truetype(F + 'DejaVuSans.ttf', 18)
    im = Image.new('RGB', (W, H), (14, 14, 17))
    d = ImageDraw.Draw(im)
    d.text((70, 150), f"session {seg}", font=ft, fill=(140, 140, 150))
    d.text((70, 185), title, font=fb, fill=(245, 245, 250))
    d.line([(70, 255), (W - 70, 255)], fill=(60, 60, 70), width=2)
    y = 300
    for ln in lines:
        colour = (235, 190, 90) if ('VQA' in ln and 'PIPELINE' in ln) else (200, 200, 210)
        d.text((70, y), ln, font=fs, fill=colour)
        y += 36
    return im, int(secs * fps)


def main():
    tmp = OUT / '_reel'
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    for seg, title, lines in CARDS:
        clip = OUT / f'session_{seg}.mp4'
        if not clip.exists():
            print(f"  missing {clip.name}, skipping")
            continue
        im, nrep = card(seg, title, lines)
        cdir = tmp / f'card_{seg}'
        cdir.mkdir(exist_ok=True)
        for i in range(nrep):
            im.save(cdir / f'{i:04d}.png')
        cmp4 = tmp / f'card_{seg}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '4',
                        '-i', str(cdir / '%04d.png'), '-c:v', 'libx264', '-pix_fmt',
                        'yuv420p', '-crf', '20', str(cmp4)], check=True)
        for f_ in cdir.glob('*.png'):
            f_.unlink()
        cdir.rmdir()
        parts += [cmp4, clip]

    lst = tmp / 'list.txt'
    lst.write_text('\n'.join(f"file '{p}'" for p in parts) + '\n')
    reel = OUT / 'reel_all_sessions.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(lst), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                    '-vf', f'scale={W}:{H}:force_original_aspect_ratio=decrease,'
                           f'pad={W}:{H}:(ow-iw)/2:(oh-ih)/2', str(reel)], check=True)
    for p in tmp.glob('card_*.mp4'):
        p.unlink()
    lst.unlink()
    tmp.rmdir()
    dur = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'csv=p=0', str(reel)], capture_output=True, text=True)
    print(f"  wrote {reel.name}  ({float(dur.stdout.strip()):.0f}s, "
          f"{reel.stat().st_size/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
