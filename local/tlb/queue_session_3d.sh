#!/usr/bin/env bash
# v3 render: distance on the ego light and a magnified inset where the 3D cuboid is
# actually legible. At 30-50 m a 0.30 m-deep object has no visible perspective in the
# full frame, so the inset is the only place the box reads as a volume.
# New filenames throughout -- nothing existing is overwritten.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
$P local/tlb/nuscenes_session_tl_v3.py --scene 0 --rate sweep \
   --out outputs/tlb/session_day_turn_3d > outputs/tlb/log_session_3d.log 2>&1
N=$(ls outputs/tlb/session_day_turn_3d/frames/*.png 2>/dev/null | wc -l)
FR=$($P -c "import json;print(json.load(open('outputs/tlb/session_day_turn_3d/encode.json'))['framerate'])" 2>/dev/null || echo 7.5)
ffmpeg -y -loglevel error -framerate "$FR" -i outputs/tlb/session_day_turn_3d/frames/%04d.png \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30' -c:v libx264 -pix_fmt yuv420p -crf 20 \
  outputs/tlb/demo_v2/session_day_turn_3d_cuboid.mp4
echo "  video: $(du -h outputs/tlb/demo_v2/session_day_turn_3d_cuboid.mp4|cut -f1)  ($N frames)"
echo "GIF_SRC_READY"
