# ONNX Runtime GPU environment.
# The venv ships onnxruntime-gpu but not the CUDA/cuDNN shared objects on the loader path;
# torch's bundled nvidia-* wheels provide them. Without this, ORT silently falls back to
# CPU with only a warning, which is how a "GPU" benchmark can quietly measure the CPU.
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$SP/cudnn/lib:$SP/cublas/lib:$SP/cuda_runtime/lib:$SP/curand/lib:$SP/cufft/lib:$SP/cusparse/lib:$SP/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
