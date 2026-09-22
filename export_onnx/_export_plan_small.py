"""Re-export the whole planning path with smaller side views.

Planning is the binding constraint on the ONNX frame: 1,685 ms against perception's
665, and it is the half that cannot be pruned. The Planning Expert cross-attends to
all eight full-attention caches, so removing any of those layers breaks it outright,
and removing twelve of the twenty-four linear-attention layers doubles ADE. The only
lever is the token count, which is 98% images and goes as the square of each frame's
target size.

Four things carry the token count and all four have to move together:

    the decoder layer graphs      traced at the sequence length
    the vision tower              traced at the patch count
    the cached scene KV           what the planner's shapes are derived from
    the Planning Expert graph     scene_k_i / scene_v_i are [1, tokens, 4, 256]

So this shrinks the side views, re-caches, and re-exports all four consistently.
``--side`` is the resize; 0.5 took PyTorch ADE from 0.335 to 0.429 m and 0.25 to
0.75 m, so the point of measuring is to see what ONNX buys for that.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

PY = _ROOT / ".venv" / "bin" / "python"


def sh(cmd, tag, timeout=7200):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_ROOT/'src'}:{_ROOT}"
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    print(f"  [{tag}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=_ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    for l in r.stdout.splitlines()[-12:]:
        if l.strip() and "it/s]" not in l and "Loading weights" not in l:
            print("    " + l, flush=True)
    if r.returncode != 0:
        print(f"  [{tag}] FAILED rc={r.returncode}", flush=True)
        for l in r.stderr.splitlines()[-15:]:
            print("    " + l, flush=True)
        return False
    print(f"  [{tag}] ok {time.time()-t0:.0f}s", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=float, default=0.5)
    args = ap.parse_args()
    os.chdir(_ROOT)
    tag = f"s{str(args.side).replace('.','')}"
    cache = _ROOT / f"data/train_cache_plan_{tag}"
    layers = _ROOT / f"outputs/onnx/vlm_layers_plan_{tag}"
    planner = _ROOT / f"outputs/onnx/planner_{tag}"
    vision = _ROOT / f"outputs/onnx/vlm_vision_plan_{tag}"

    # 1. cache the scene KV at the reduced token count
    if not any(cache.glob("*.pt")):
        cache.mkdir(parents=True, exist_ok=True)
        if not sh([PY, _ROOT/"export_onnx"/"_cache_plan_small.py",
                   "--side", args.side, "--out", cache], "cache"):
            return 1
    rec = sorted(cache.glob("*.pt"))
    if not rec:
        print("  no cache records produced")
        return 1
    print(f"  cached {len(rec)} scenes -> {cache}", flush=True)

    # 2. the Planning Expert, whose scene_k/scene_v shapes come from that cache
    if not (planner / "planner_step.onnx").exists():
        planner.mkdir(parents=True, exist_ok=True)
        sh([PY, _ROOT/"export_onnx"/"export_planner.py",
            "--cache", cache, "--record", rec[0].name,
            "--out", planner/"planner_step.onnx", "--skip-verify"], "planner")

    # 3. the decoder layers at the reduced sequence length
    if not (layers / "manifest.json").exists():
        sh([PY, _ROOT/"export_onnx"/"export_vlm_layers_v2.py",
            "--task", "planning", "--side", args.side, "--out", layers,
            "--embed-from", _ROOT/"outputs/onnx/vlm_layers_plan"], "layers")

    # 4. fp16 projection weights
    f16 = Path(str(layers) + "_f16")
    if (layers / "manifest.json").exists() and not (f16 / "manifest.json").exists():
        sh([PY, _ROOT/"export_onnx"/"fp16_weights_v3.py", layers, f16], "fp16")

    print(f"\n  done: layers={layers.exists()} planner={(planner/'planner_step.onnx').exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
