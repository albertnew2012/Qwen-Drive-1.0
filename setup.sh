#!/usr/bin/env bash
# Bootstrap this repo on a fresh machine.
#
#   bash setup.sh            environment + weights + smoke test
#   bash setup.sh --no-fla   same (this is the default; see the note below)
#   bash setup.sh --with-fla install the fused Gated-DeltaNet kernels too
#
# THE ONE DECISION THAT MATTERS
# requirements.txt pins flash-attn, causal-conv1d and flash-linear-attention.
# With flash-linear-attention installed the model runs FUSED Gated-DeltaNet
# kernels, which are faster but CANNOT be traced - the ONNX export breaks. This
# repo's export works precisely because those packages are absent and the model
# falls back to torch_chunk_gated_delta_rule. Default here is therefore to leave
# them out. Add --with-fla only if you want maximum inference speed and do not
# care about ONNX.
set -euo pipefail
cd "$(dirname "$0")"

WITH_FLA=0
for a in "$@"; do
  case "$a" in
    --with-fla) WITH_FLA=1 ;;
    --no-fla)   WITH_FLA=0 ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }

say "1/5  checks"
command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2]>=(3,12)' || { echo "need Python >= 3.12"; exit 1; }
command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "  WARNING: no nvcc. The two perception CUDA kernels compile on first use."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /' || echo "  no GPU detected"

say "2/5  virtual environment"
[ -d .venv ] || uv venv --python 3.12 .venv
export VIRTUAL_ENV="$PWD/.venv"
if [ "$WITH_FLA" = "1" ]; then
  uv pip install -r requirements.txt --no-build-isolation
else
  grep -vE '^(flash-attn|causal-conv1d|flash-linear-attention)' requirements.txt > /tmp/req.noflash.txt
  uv pip install -r /tmp/req.noflash.txt
  echo "  fused kernels SKIPPED so ONNX export stays possible (--with-fla to include)"
fi
uv pip install onnx onnxruntime onnxscript scipy      # export + Hungarian matcher

say "3/5  weights (~13 GB)"
if [ -d weights/Qwen-Drive-1.0-4B ]; then
  echo "  already present"
else
  # the `hf` CLI arrives transitively via transformers and its entry-point name
  # has changed across versions; snapshot_download is the stable interface.
  .venv/bin/python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen-Drive-1.0-4B",
                  local_dir="weights/Qwen-Drive-1.0-4B",
                  max_workers=8)
PYEOF
fi

say "4/5  feature cache + smoke test"
export PYTHONPATH=src:. CUDA_HOME="${CUDA_HOME:-/usr}"
# test_gradients.py loads data/train_cache/<frame>.pt, so the cache has to exist
# FIRST. Caching runs the 4.5 B VLM once per frame - practical on a GPU, very slow
# on CPU - so skip both if there is no GPU rather than appear to hang.
if .venv/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  if ls data/train_cache/*.pt >/dev/null 2>&1; then
    echo "  feature cache already present"
  else
    .venv/bin/python training/cache_features.py 2>&1 | tail -2
  fi
  .venv/bin/python training/test_gradients.py --dtype bfloat16 2>&1 | tail -3
else
  echo "  no CUDA device - skipping. ONNX export and validation still work on CPU;"
  echo "  training does not. See SETUP.md section 4."
fi

say "5/5  done"
cat <<'TXT'
  Next:
    .venv/bin/python training/cache_features.py          # ~2 min
    .venv/bin/python training/run_all_stages.py          # ~13 min, 5/5 expected
  ONNX (graphs must be exported first, ~45 min total - see ONNX_EXPORT.md §2):
    .venv/bin/python export_onnx/run_onnx_pipeline.py --phase run
    .venv/bin/python export_onnx/run_onnx_pipeline.py --phase compare
TXT
