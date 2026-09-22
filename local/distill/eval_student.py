"""How closely does the student reproduce the teacher, on frames it did not train on?

Speed is settled -- the student exports to 287 nodes and runs in single-digit
milliseconds. The open question is agreement, so this reports it the way it matters:
not tensor norms but whether the same objects come out.

    recall      teacher detections above threshold that the student also asserts
    precision   student detections the teacher agrees with
    centre      median distance between matched centres, in metres

Matching is by query index, which is exact here: the student is trained so that its
query i corresponds to the teacher's query i.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from local.distill.student import StudentConfig, StudentDetector
from local.distill.train_student import DistillSet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/distill/student/student.pt")
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--out", default="outputs/distill/eval.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    if not Path(args.ckpt).exists():
        print("  no checkpoint yet")
        return 0
    cfg = StudentConfig()
    model = StudentDetector(cfg).cuda().eval()
    st = torch.load(args.ckpt, map_location="cuda")
    model.load_state_dict(st["model"])
    step = st.get("step", 0)

    everything = DistillSet(Path(args.frames), Path(args.teacher), cfg)
    n_val = max(1, int(len(everything) * args.val_frac))
    val = DistillSet(Path(args.frames), Path(args.teacher), cfg,
                     everything.tokens[:n_val][:args.limit])
    if not len(val):
        print("  no validation frames")
        return 0

    tp = fp = fn = 0
    dists = []
    ades = []
    with torch.no_grad():
        for i in range(len(val)):
            b = val[i]
            pc, pb, pt = model(b["image"][None].cuda(),
                               b["bev_index"].cuda(), b["valid"].cuda(),
                               b["ego"][None].cuda())
            if float(b["has_traj"]) > 0:
                ade = np.linalg.norm(
                    pt[0, :, :2].float().cpu().numpy() - b["future"][:, :2].numpy(),
                    axis=-1).mean()
                ades.append(float(ade))
            ps = torch.sigmoid(pc[0].float()).max(-1).values.cpu().numpy()
            ts = torch.sigmoid(b["cls"].float()).max(-1).values.numpy()
            sk, tk = ps >= args.thr, ts >= args.thr
            tp += int((sk & tk).sum()); fp += int((sk & ~tk).sum())
            fn += int((~sk & tk).sum())
            both = sk & tk
            if both.any():
                d = np.linalg.norm(
                    pb[0].float().cpu().numpy()[both, :3] - b["box"].numpy()[both, :3],
                    axis=-1)
                dists += d.tolist()
    rep = {"step": int(step), "frames": len(val), "threshold": args.thr,
           "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
           "tp": tp, "fp": fp, "fn": fn,
           "centre_median_m": float(np.median(dists)) if dists else None,
           "trajectory_ade_m": float(np.mean(ades)) if ades else None,
           "trajectory_frames": len(ades)}
    print(f"  step {step}: recall {rep['recall']:.1%}  precision {rep['precision']:.1%}  "
          f"centre median {rep['centre_median_m'] if rep['centre_median_m'] is None else round(rep['centre_median_m'],3)} m"
          f"   (tp {tp} fp {fp} fn {fn})")
    if ades:
        print(f"           trajectory ADE {np.mean(ades):.3f} m over {len(ades)} frames "
              f"(teacher against the same ground truth: 0.335 m)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
