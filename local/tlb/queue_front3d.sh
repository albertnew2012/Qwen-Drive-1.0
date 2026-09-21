#!/usr/bin/env bash
# Front-view comparison with the full 3D stack, over the three scenes that have both
# traffic-light labels and full nuScenes sensor data (red / green / yellow between them).
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
$P local/tlb/make_front3d_compare_v1.py --scenes 2 5 1 --fps 2.5 \
   > outputs/tlb/log_front3d.log 2>&1
tail -3 outputs/tlb/log_front3d.log
N=$(ls outputs/tlb/_front3d/*.png | wc -l)
# GIF at native resolution -- no scale filter, per-clip palette
ffmpeg -y -loglevel error -framerate 2.5 -i outputs/tlb/_front3d/%04d.png \
  -vf "palettegen=max_colors=256:stats_mode=diff" /tmp/pal_f3d.png
ffmpeg -y -loglevel error -framerate 2.5 -i outputs/tlb/_front3d/%04d.png -i /tmp/pal_f3d.png \
  -lavfi "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  assets/front3d_perception_vs_vqa.gif
ls -la outputs/tlb/demo_v2/front3d_perception_vs_vqa.mp4 assets/front3d_perception_vs_vqa.gif \
  | awk '{printf "  %-52s %6.1f MB\n",$9,$5/1e6}'
echo "  frames: $N"
echo "FRONT3D DONE"
