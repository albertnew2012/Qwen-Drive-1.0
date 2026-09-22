"""What does removing decoder layers do to the detections, not to the norm?

``measure_layer_influence_v1.py`` says no layer is free: skipping any one of the 32
moves the final hidden state by 17-76%, where a prunable LLM block is usually 1-5%.
But the head does not read the norm of the hidden state, it reads the detections,
and a large change in an unnormalised residual stream can still leave the argmax
alone. So this measures the thing that matters.

For each candidate set of skipped layers: run the decoder without them, run the
head on the result, and compare against the unpruned model on the same frame.

  kept@0.3      detections the pruned model still finds, of those the full model
                reports above 0.3 confidence
  spurious      detections the pruned model adds that the full model does not have
  centre_med    median centre displacement over the matched ones, in metres
  argmax        fraction of the 900 queries whose predicted class is unchanged

Candidate sets are contiguous where possible, because contiguous removal is both
what the depth-pruning literature finds works and what actually shortens the
graph. The head is left untouched here on purpose -- this is the floor, before any
distillation is allowed to adapt it.

    python local/prune/eval_depth_prune_v1.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from local.prune.measure_layer_influence_v1 import build, run_stack

CANDIDATES = {
    "none": [],
    "L13": [13],
    "L12-13": [12, 13],
    "L12-17 (6)": [12, 13, 14, 15, 16, 17],
    "least8": [4, 9, 12, 13, 14, 15, 16, 17],
    "L10-17 (8)": [10, 11, 12, 13, 14, 15, 16, 17],
    "least12": [1, 3, 4, 8, 9, 12, 13, 14, 15, 16, 17, 20],
    "L8-19 (12)": list(range(8, 20)),
}


def detections(cls, box, thr=0.3):
    """Queries above threshold, with their class and centre."""
    p = 1 / (1 + np.exp(-cls.max(-1)))
    keep = p >= thr
    return keep, cls.argmax(-1), box[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frame", default="data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/prune/depth_prune_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    head = QwenDrivePerception.from_pretrained(
        args.perception, dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, proc)
    vlm = model.vlm
    lm = vlm.model.language_model
    types = list(vlm.config.text_config.layer_types)

    x, inputs, metas, mask = build(vlm, args.frame, proc)
    pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
    grid = inputs["image_grid_thw"]
    n_cam = len(metas["cam_order"])
    gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
    tpi = gh // 2 * gw // 2
    dt = next(head.bev_modeling.parameters()).dtype

    # the ViT tap, constant across every candidate
    captured = {}
    hk = vlm.model.visual.merger.register_forward_hook(
        lambda m, a, o=None: captured.__setitem__("p", a[0]))
    with torch.no_grad():
        vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
    hk.remove()
    with torch.no_grad():
        vit = torch.stack(head._premerge_grids(
            vlm.model.visual.merger.norm(captured["p"]), grid)[-n_cam:], 0).to(dt)
    del captured
    torch.cuda.empty_cache()

    def forward(skip):
        with torch.no_grad():
            hidden, _ = run_stack(lm, types, x, pos, skip=set(skip))
            llm = hidden[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1).to(dt)
            o = head.bev_modeling(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
            return (o["all_cls_scores"][-1, 0].float().cpu().numpy(),
                    o["all_bbox_preds"][-1, 0].float().cpu().numpy())

    def timed(skip, n=3):
        for _ in range(1):
            forward(skip)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            with torch.no_grad():
                run_stack(lm, types, x, pos, skip=set(skip))
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1e3

    cls0, box0 = forward([])
    keep0, lab0, ctr0 = detections(cls0, box0, args.thr)
    base_ms = timed([])
    print(f"  frame {Path(args.frame).name[:16]}  {x.shape[1]} tokens, "
          f"{int(keep0.sum())} detections above {args.thr}")
    print(f"  baseline decoder {base_ms:.1f} ms\n")
    print(f"  {'skipped':14s} {'n':>3s} {'dec ms':>8s} {'speedup':>8s} "
          f"{'kept':>9s} {'spurious':>9s} {'centre_med':>11s} {'argmax':>8s}")

    rows = {}
    for name, skip in CANDIDATES.items():
        cls, box = forward(skip)
        keep, lab, ctr = detections(cls, box, args.thr)
        both = keep0 & keep
        kept = int(both.sum())
        spurious = int((keep & ~keep0).sum())
        cd = (float(np.median(np.linalg.norm(ctr0[both] - ctr[both], axis=-1)))
              if kept else float("nan"))
        argmax = float((lab0 == lab).mean())
        ms = timed(skip) if skip else base_ms
        rows[name] = {"skip": skip, "n": len(skip), "decoder_ms": ms,
                      "speedup": base_ms / ms, "kept": kept,
                      "of": int(keep0.sum()), "spurious": spurious,
                      "centre_median_m": cd, "argmax_agreement": argmax}
        print(f"  {name:14s} {len(skip):3d} {ms:8.1f} {base_ms/ms:7.2f}x "
              f"{kept:4d}/{int(keep0.sum()):<4d} {spurious:9d} {cd:11.3f} {argmax:7.1%}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frame": args.frame, "threshold": args.thr,
                               "baseline_decoder_ms": base_ms, "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
