cd "$(dirname "$0")/../../.."
until [ "$(ps -eo args | grep -c '[n]usc_depth.py')" = "0" ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] extraction finished: 896 $(find data/distill/frames -maxdepth 2 -name depth.npz | wc -l), 1152 $(find data/distill/frames -maxdepth 2 -name depth_1152x640.npz | wc -l), 1408 $(find data/distill/frames -maxdepth 2 -name depth_1408x768.npz | wc -l)"
for D in ${DISTILL_CACHE:-/local/$USER/distill}/frames ${DISTILL_CACHE:-/local/$USER/distill}/frames_real; do
  rsync -a --include='*/' --include='depth.npz' --include='depth_1152x640.npz' --include='depth_1408x768.npz' --exclude='*' data/distill/frames/ $D/
  echo "$D synced: 1152 $(find $D -maxdepth 2 -name depth_1152x640.npz | wc -l)"; done
echo "DEPTHFIX_DONE"
