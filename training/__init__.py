"""Training pipeline for Qwen-Drive-1.0.

The released repository is inference-only. Two custom CUDA kernels have no
backward pass at all, so nothing in ``src/`` can be differentiated as shipped.
``differentiable.py`` is what makes training possible; read it first.
"""
