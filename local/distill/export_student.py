"""Export the student to ONNX and time it.

The student was shaped for this: ordinary attention rather than deformable (which has no
ONNX representation and is what pushed the teacher onto GridSample), a single
``index_add`` for the view transform rather than a voxel-pooling kernel, and no 5-D
anything. So the export is a plain trace with no post-hoc graph surgery.

``bev_index`` and ``valid`` are graph inputs, not constants: they are pure calibration and
cost nothing to compute on the host, and keeping them out of the graph leaves it static.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from local.distill.student import StudentConfig, StudentDetector


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/distill/student/student.pt")
    ap.add_argument("--out", default="outputs/onnx/student_det/student.onnx")
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--bench", action="store_true", default=True)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="only without a checkpoint: input size of the untrained shape to time")
    ap.add_argument("--arch", default="resnet50", help="only without a checkpoint: backbone of the shape")
    ap.add_argument("--report", default="outputs/distill/student_onnx.json", help="where to write the timing json")
    ap.add_argument("--head-upsample", type=int, default=1, help="untrained shape only: detection head grid factor")
    args = ap.parse_args()
    os.chdir(_ROOT)

    _ck = torch.load(args.ckpt, map_location="cpu") if Path(args.ckpt).exists() else {}
    _saved = _ck.get("cfg") or {}
    cfg = StudentConfig(**{k: v for k, v in _saved.items()
                           if k in StudentConfig.__init__.__code__.co_varnames})
    if args.image_size and not _saved:
        cfg = StudentConfig(det_head="center", ref_points=False, center_hidden=128, center_blocks=3,
                            temporal=True, velocity=True, num_classes=10,
                            image_size=tuple(args.image_size), arch=args.arch,
                            head_upsample=args.head_upsample)
        print(f"  untrained final-recipe shape at {cfg.image_size}")
    if _saved:
        print(f"  config from checkpoint: pool={cfg.pool} ref_points={cfg.ref_points}")
    model = StudentDetector(cfg).eval()
    if Path(args.ckpt).exists():
        st = _ck
        use_ema = bool(st.get("ema")) and not args.no_ema
        model.load_state_dict(st["ema"] if use_ema else st["model"])
        print(f"  weights: {'EMA' if use_ema else 'raw'}")
        print(f"  loaded step {st.get('step')} from {args.ckpt}")
    else:
        print(f"  no checkpoint at {args.ckpt}; exporting the untrained shape to time it")

    w, h = cfg.image_size
    H, W = h // 16, w // 16
    # one index block per camera, concatenated exactly as LiftSplat consumes them
    n = cfg.n_cams * cfg.depth_bins * H * W
    img = torch.randn(1, cfg.n_cams, 3, h, w)
    idx = torch.randint(0, cfg.bev_size ** 2, (n,))
    # ~45% of real rays fall outside the grid or behind the camera; an all-valid dummy hid a
    # 30 ms scratch-row contention in the scatter (see LiftSplat.forward), so time realistically.
    valid = torch.rand(n) > 0.45
    ego_dim = cfg.hist_points * 3 + cfg.hist_points * 4 + 4
    ego = torch.randn(1, ego_dim)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with torch.no_grad():
        temporal = bool(getattr(cfg, "temporal", False))
        if temporal:
            K = int(getattr(cfg, "history", 1))
            if K == 1:
                prev_bev = torch.randn(1, cfg.bev_channels, cfg.bev_size, cfg.bev_size)
                warp_grid = torch.zeros(1, cfg.bev_size, cfg.bev_size, 2)
            else:
                prev_bev = torch.randn(1, K, cfg.bev_channels, cfg.bev_size, cfg.bev_size)
                warp_grid = torch.zeros(1, K, cfg.bev_size, cfg.bev_size, 2)
            args_t = (img, idx, valid, ego, prev_bev, warp_grid)
            in_names = ["image", "bev_index", "valid", "ego", "prev_bev", "warp_grid"]
            out_names = ["cls", "box", "occ", "seg", "trajectory", "bev_state"]
        else:
            args_t = (img, idx, valid, ego)
            in_names = ["image", "bev_index", "valid", "ego"]
            out_names = ["cls", "box", "occ", "seg", "trajectory"]
        torch.onnx.export(model, args_t, str(out),
                          input_names=in_names,
                          output_names=out_names,
                          opset_version=args.opset,
                          do_constant_folding=True, dynamo=False)
    import onnx
    g = onnx.load(str(out), load_external_data=False).graph
    print(f"  exported {len(g.node)} nodes in {time.time()-t0:.0f}s -> {out}")

    if not args.bench:
        return 0
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True
    # ORT will not create the prefix's directory; without it profiling silently writes
    # nothing and end_profiling() hands back a path that does not exist.
    Path("outputs/onnx_probe").mkdir(parents=True, exist_ok=True)
    so.profile_file_prefix = "outputs/onnx_probe/student"
    s = ort.InferenceSession(str(out), so,
                             providers=[("CUDAExecutionProvider", {"device_id": 0})])
    feeds = {"image": img.numpy(), "bev_index": idx.numpy(),
             "valid": valid.numpy(), "ego": ego.numpy()}
    if temporal:
        feeds["prev_bev"] = prev_bev.numpy(); feeds["warp_grid"] = warp_grid.numpy()
    s.run(None, feeds)
    ts = []
    for _ in range(20):
        t = time.time(); s.run(None, feeds); ts.append((time.time() - t) * 1e3)
    ms = float(np.median(ts))
    p = s.end_profiling()
    import collections
    ev = [e for e in json.load(open(p)) if e.get("cat") == "Node"
          and e["name"].endswith("_kernel_time")]
    cpu = sum(e["dur"] for e in ev if "CUDA" not in e["args"].get("provider", ""))
    ops = collections.Counter()
    for e in ev:
        ops[e["args"].get("op_name")] += e["dur"]
    os.remove(p)
    print(f"  WHOLE MODEL (detection + planning) ONNX: {ms:.2f} ms -> {1000/ms:.1f} Hz   "
          f"cpu-fallback {cpu/1e3/21:.2f} ms")
    print("  top ops: " + ", ".join(f"{k} {v/1e3/21:.1f}ms" for k, v in ops.most_common(5)))
    Path("outputs/distill").mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(
        {"ms": ms, "hz": 1000 / ms, "nodes": len(g.node),
         "params_m": model.num_params() / 1e6}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
