# After the ego-cache extension finishes: sync it to the local copy, re-render scene-0276.
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
until grep -q "wrote" outputs/logs/nusc_ego_short.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] $(grep wrote outputs/logs/nusc_ego_short.log | tail -1)"
rsync -a data/distill/ego/ ${DISTILL_CACHE:-/local/$USER/distill}/ego/ && echo "local ego files: $(ls ${DISTILL_CACHE:-/local/$USER/distill}/ego | wc -l)"
source export_onnx/env_gpu.sh
.venv/bin/python local/distill/student_video.py --scene scene-0276 --gpu 3 --thr 0.4 2>&1 | grep -E "wrote|Traceback|Error" | tail -3
echo "VIDEO_RERENDER_DONE"
