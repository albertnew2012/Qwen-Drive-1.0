"""The three 3-D convs in the view transform are 228 ms and 6.79 TFLOP.

Measured breakdown of ``view_trans``, which is itself 58% of the BEV head:

    feat_encoding    228.0 ms     3 x Conv3d(256 -> 256, k=3x3x3) over 16x200x200
    feat_sampling     15.0 ms     the lift-splat scatter
    coord_preparing    3.9 ms     the frustum unprojection

That is 29.8 TFLOPS, i.e. the convs are running near peak -- they are not slow, they
are large. 6.79 TFLOP is 14% of the whole frame's arithmetic spent here, which makes
this the single densest target in the model. (Coarsening the depth grid, which was
the obvious guess, saves 0.1-4.7 ms: the scatter was never the cost.)

Two reductions, measured here:

  depth      truncate the stack to fewer convs -- linear in the count
  width      narrow the channels -- quadratic, so 256 -> 128 is 4x

Truncation is testable as-is. Narrowing is not: the weights would have to be
retrained, so what is measured for width is the achievable time with randomly
initialised narrow convs, to confirm the scaling before committing to distillation.

    python local/prune/eval_voxel_convs_v1.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer


def dets(cls, box, thr=0.3):
    p = 1 / (1 + np.exp(-cls.max(-1)))
    return p >= thr, cls.argmax(-1), box[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--widths", default="256,192,128,96,64")
    ap.add_argument("--out", default="outputs/prune/voxel_convs_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    head = QwenDrivePerception.from_pretrained(
        args.perception, dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, proc)
    with torch.no_grad():
        _, metas = proc(PerceptionFrame(Path(args.frame)), device="cuda")
    ref = np.load("outputs/onnx_bench/torch_bfloat16_perception.npz")
    dt = next(head.bev_modeling.parameters()).dtype
    vit = torch.from_numpy(ref["img_vit_feats"]).to("cuda", dt)
    llm = torch.from_numpy(ref["img_llm_feats"]).to("cuda", dt)
    bev = head.bev_modeling
    vt = bev.view_trans
    stack = list(vt.conv_layer)

    def run():
        with torch.no_grad():
            return bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])

    def timed(n=3):
        run(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): run()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

    o = run()
    base = dets(o["all_cls_scores"][-1, 0].float().cpu().numpy(),
                o["all_bbox_preds"][-1, 0].float().cpu().numpy(), args.thr)
    base_ms = timed()
    print(f"  baseline head {base_ms:.1f} ms, {int(base[0].sum())} detections above {args.thr}")

    report = {"baseline_head_ms": base_ms, "truncate": {}, "width_timing": {}}
    print(f"\n  DEPTH -- keep the first k convs (testable as shipped)")
    print(f"  {'keep':>5s} {'head ms':>8s} {'saved':>7s} {'kept':>9s} {'spur':>5s} "
          f"{'centre_med':>11s} {'argmax':>8s}")
    for keep in range(len(stack), -1, -1):
        vt.conv_layer._modules = {str(i): m for i, m in enumerate(stack[:keep])}
        oo = run()
        c = dets(oo["all_cls_scores"][-1, 0].float().cpu().numpy(),
                 oo["all_bbox_preds"][-1, 0].float().cpu().numpy(), args.thr)
        both = base[0] & c[0]
        ms = timed()
        cd = (float(np.median(np.linalg.norm(base[2][both] - c[2][both], axis=-1)))
              if both.any() else float("nan"))
        report["truncate"][keep] = {
            "head_ms": ms, "saved_ms": base_ms - ms, "kept": int(both.sum()),
            "of": int(base[0].sum()), "spurious": int((c[0] & ~base[0]).sum()),
            "centre_median_m": cd, "argmax_agreement": float((base[1] == c[1]).mean())}
        r = report["truncate"][keep]
        print(f"  {keep:5d} {ms:8.1f} {base_ms-ms:7.1f} {r['kept']:4d}/{r['of']:<4d} "
              f"{r['spurious']:5d} {cd:11.3f} {r['argmax_agreement']:7.1%}", flush=True)
    vt.conv_layer._modules = {str(i): m for i, m in enumerate(stack)}

    # What would a narrower stack cost? Random weights: timing only, to confirm the
    # quadratic scaling before spending training on it.
    print(f"\n  WIDTH -- achievable time with narrow convs (random weights, timing only)")
    print(f"  {'width':>6s} {'head ms':>8s} {'saved':>7s} {'TFLOP':>7s}")
    vol = int(np.prod(vt.voxel_shape))
    for w in [int(x) for x in args.widths.split(",")]:
        if w == 256:
            report["width_timing"][w] = {"head_ms": base_ms, "saved_ms": 0.0}
            print(f"  {w:6d} {base_ms:8.1f} {0.0:7.1f} {3*2*vol*256*256*27/1e12:7.2f}")
            continue
        narrow = nn.ModuleList()
        for i in range(len(stack)):
            cin = 256 if i == 0 else w
            cout = 256 if i == len(stack) - 1 else w
            narrow.append(nn.Sequential(
                nn.Conv3d(cin, cout, 3, 1, 1, bias=True),
                nn.BatchNorm3d(cout), nn.ReLU(inplace=True)).to("cuda", dt))
        vt.conv_layer = narrow
        ms = timed()
        fl = sum(2 * vol * (256 if i == 0 else w) * (256 if i == len(stack)-1 else w) * 27
                 for i in range(len(stack))) / 1e12
        report["width_timing"][w] = {"head_ms": ms, "saved_ms": base_ms - ms, "tflop": fl}
        print(f"  {w:6d} {ms:8.1f} {base_ms-ms:7.1f} {fl:7.2f}", flush=True)
        del narrow
    vt.conv_layer = nn.ModuleList(stack)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
