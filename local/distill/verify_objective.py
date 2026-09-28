"""Sanity-check a training objective BEFORE spending GPU-hours on it.

THE RULE THIS ENFORCES
A loss must rank a known-good model below a known-bad one. If it does not, training
will faithfully move the model toward the bad one, and every downstream symptom looks
like "needs more steps" or "not enough capacity".

WHY IT EXISTS
The Hungarian objective added in E1 was inverted. Measured on two real checkpoints:

    distilled 78k   F1 34.5%   l_cls 1.293
    E2 hungarian    F1  1.1%   l_cls 0.537     <- the BAD model scored BETTER

Three 20k-step runs (E1, E1b, E2, ~7 GPU-hours) produced 0.0%, 0.0% and 1.1% F1 before
this was checked. The cause was a matching cost that compared a class term bounded by 2
against a box term in raw metres (5 x tens of metres), so assignment was purely
geometric; confident queries were never matched and their confidence was then charged as
a false positive. The only descent direction was silence.

Two failure modes are distinguished here, and they need different fixes:

    INVERTED   the good model scores worse. The loss is wrong. Fix the loss.
    FLAT       the two score within noise. The loss cannot see the difference, so there
               is no gradient signal toward good. Fix the loss.
    ORDERED    the good model scores better. The loss is learnable; if training still
               fails the problem is optimisation or architecture, not the objective.

Note that ORDERED does not promise the path is downhill from the current init -- focal
loss over 900 queries for ~34 objects is correctly ordered yet still has an uphill climb
out of silence, which is why E3 warm-starts from a model that already fires.

    python local/distill/verify_objective.py --good <ckpt> --bad <ckpt>
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", default="outputs/distill/student/student.pt",
                    help="a checkpoint known to perform BETTER on the target metric")
    ap.add_argument("--bad", required=True,
                    help="a checkpoint known to perform WORSE")
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ref-points", default="off", choices=["on", "off"])
    args = ap.parse_args()
    os.chdir(_ROOT)

    from local.distill.student import StudentConfig, StudentDetector
    from local.distill.train_student import DistillSet
    from local.distill.det_loss import hungarian_detection_loss, gt_to_teacher_encoding

    cfg = StudentConfig(ref_points=(args.ref_points == "on"))
    st = np.load("data/distill/box_stats.npz")
    box_std = torch.from_numpy(st["std"]).cuda()
    ds = DistillSet(Path(args.frames), Path(args.teacher), cfg)

    out = {}
    for name, ck in (("good", args.good), ("bad", args.bad)):
        m = StudentDetector(cfg).cuda().eval()
        m.load_state_dict(torch.load(ck, map_location="cuda")["model"], strict=False)
        m.set_box_stats(st["mean"], st["std"])
        cls_l, box_l = [], []
        for i in range(args.n):
            b = ds[i]
            g = np.load(Path(args.frames) / ds.tokens[i] / "gt_boxes.npz")
            gb = torch.from_numpy(gt_to_teacher_encoding(g["boxes"])).cuda()
            gl = torch.from_numpy(g["labels"]).cuda()
            with torch.no_grad():
                pc, pb, *_ = m(b["image"][None].cuda(), b["bev_index"].cuda(),
                               b["valid"].cuda(), b["ego"][None].cuda())
                lc, lb, _ = hungarian_detection_loss(pc.float(), pb.float(), [gb], [gl],
                                                     box_std)
            cls_l.append(lc.item()); box_l.append(lb.item())
        out[name] = (float(np.mean(cls_l)), float(np.std(cls_l)),
                     float(np.mean(box_l)))
        print(f"  {name:5s} {Path(ck).parent.name:14s} l_cls {out[name][0]:7.4f} "
              f"+/- {out[name][1]:.4f}   l_box {out[name][2]:7.4f}")

    gc, gs, _ = out["good"]; bc, bs, _ = out["bad"]
    margin = bc - gc
    noise = (gs + bs) / 2 + 1e-9
    print()
    if margin > noise:
        print(f"  ORDERED  good is lower by {margin:.4f} ({margin/noise:.1f}x the spread)")
        print("           the objective is learnable. If training still fails, look at "
              "optimisation\n           or architecture, not the loss.")
        return 0
    if margin < -noise:
        print(f"  INVERTED the BAD model scores {-margin:.4f} LOWER. Training on this "
              f"will move the\n           model toward the bad one. Do not launch.")
        return 2
    print(f"  FLAT     the two differ by {margin:.4f} against a spread of {noise:.4f}. "
          f"The loss\n           cannot distinguish them, so there is no signal toward "
          f"good. Do not launch.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
