"""Score the TEACHER against real ground truth, and optionally the student beside it.

This is the second reading of "performance degradation under 10%":

    replication   student reproduces >= 90% of the teacher's raw output. Demands the
                  student copy a 4 B model's mistakes; this is what eval_student.py does.
    task quality  student scores within 10% of what the TEACHER scores against ground
                  truth. The ordinary meaning, and what this measures.

Matching follows the nuScenes detection convention: greedy by centre distance in BEV,
2 m threshold, class must agree. No IoU, because the nuScenes metric is centre-distance
based and box extents are not what distillation is judged on here.

The teacher number alone is worth having: it sets the bar the student is held to, and
nothing in the repo had ever measured it.

    python local/distill/eval_vs_gt.py                      # teacher only, CPU
    python local/distill/eval_vs_gt.py --ckpt outputs/distill/student/student.pt
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np

RANGE = 51.2
DIST_THR = 2.0


def match(det_xy, det_lab, gt_xy, gt_lab, thr=DIST_THR):
    """Greedy nearest-first matching, class-constrained. Returns tp, fp, fn."""
    if not len(det_xy):
        return 0, 0, len(gt_xy)
    if not len(gt_xy):
        return 0, len(det_xy), 0
    d = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=-1)
    d[det_lab[:, None] != gt_lab[None, :]] = np.inf
    tp = 0
    used_d, used_g = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for i, j in order:
        if d[i, j] > thr:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(int(i)); used_g.add(int(j)); tp += 1
    return tp, len(det_xy) - tp, len(gt_xy) - tp


def in_range(xy):
    return (np.abs(xy[:, 0]) <= RANGE) & (np.abs(xy[:, 1]) <= RANGE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.0,
                    help="BEV centre-distance NMS radius in m; 0 disables. Roughly half "
                         "the student's false positives are duplicates stacked on real "
                         "objects, so this is the cheapest precision lever there is.")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="outputs/distill/eval_vs_gt.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    toks = sorted(p.stem for p in Path(args.teacher).glob("*.npz"))
    toks = [t for t in toks if (Path(args.frames) / t / "gt_boxes.npz").exists()]
    n_val = max(1, int(len(toks) * args.val_frac))
    toks = toks[:n_val][:args.limit]
    if not toks:
        print("  no frames with both teacher output and gt_boxes.npz")
        return 1

    model = None
    if args.ckpt and Path(args.ckpt).exists():
        import torch
        from local.distill.student import StudentConfig, StudentDetector
        from local.distill.train_student import DistillSet
        # Build the config to match the CHECKPOINT, not the defaults. ref_logit exists
        # only when ref_points is on, and forward applies it whenever it exists -- so
        # loading a --ref-points off checkpoint into a default config silently applies a
        # RANDOM reference-point transform to every box. Measured cost of getting this
        # wrong: boxes thrown to +-150 m and F1 read as 2.2% for a model actually worth
        # far more.
        ck = torch.load(args.ckpt, map_location="cpu")
        sd = ck["model"]
        # Rebuild the config the checkpoint was TRAINED with. Guessing one field at a
        # time does not scale -- inferring ref_points but not pool is what made this
        # silently fail on the pool=100 run. The checkpoint carries cfg; use it.
        saved = ck.get("cfg") or {}
        cfg = StudentConfig(**{k: v for k, v in saved.items()
                               if k in StudentConfig.__init__.__code__.co_varnames})
        print(f"  config from checkpoint: pool={cfg.pool} ref_points={cfg.ref_points} "
              f"n_cams={cfg.n_cams}")
        model = StudentDetector(cfg).cuda().eval()
        # strict=False so checkpoints from other configurations still load (e.g.
        # --ref-points off has no ref_logit), but say which keys were left at their
        # random initialisation -- a silently random head reads as a trained model.
        miss, unexp = model.load_state_dict(sd, strict=False)
        if miss or unexp:
            print(f"  !! randomly-initialised: {list(miss)}   unexpected: {list(unexp)}")
        ds = DistillSet(Path(args.frames), Path(args.teacher), cfg, toks)

    acc = {"teacher": [0, 0, 0], "student": [0, 0, 0]}
    for i, tok in enumerate(toks):
        g = np.load(Path(args.frames) / tok / "gt_boxes.npz")
        gxy, glab = g["boxes"][:, :2], g["labels"]
        k = in_range(gxy) if len(gxy) else np.zeros(0, bool)
        gxy, glab = gxy[k], glab[k]

        d = np.load(Path(args.teacher) / f"{tok}.npz")
        cls = d["cls"].astype(np.float32)
        s = 1.0 / (1.0 + np.exp(-cls.max(-1)))
        sel = s >= args.thr
        txy, tlab = d["box"][sel][:, :2], cls[sel].argmax(-1)
        kk = in_range(txy) if len(txy) else np.zeros(0, bool)
        tp, fp, fn = match(txy[kk], tlab[kk], gxy, glab)
        acc["teacher"][0] += tp; acc["teacher"][1] += fp; acc["teacher"][2] += fn

        if model is not None:
            import torch
            b = ds[i]
            with torch.no_grad():
                pc, pb, *_ = model(b["image"][None].cuda(), b["bev_index"].cuda(),
                                   b["valid"].cuda(), b["ego"][None].cuda())
            pcl = pc[0].float().cpu().numpy()
            ps = 1.0 / (1.0 + np.exp(-pcl.max(-1)))
            sel2 = ps >= args.thr
            sxy = pb[0].float().cpu().numpy()[sel2][:, :2]
            slab = pcl[sel2].argmax(-1)
            if args.nms > 0 and len(sxy):
                order = np.argsort(-ps[sel2])          # confident first
                keep_i, taken = [], np.zeros(len(sxy), bool)
                for j in order:
                    if taken[j]:
                        continue
                    keep_i.append(j)
                    d = np.linalg.norm(sxy - sxy[j], axis=-1)
                    taken |= (d < args.nms) & (slab == slab[j])
                sxy, slab = sxy[keep_i], slab[keep_i]
            k2 = in_range(sxy) if len(sxy) else np.zeros(0, bool)
            tp, fp, fn = match(sxy[k2], slab[k2], gxy, glab)
            acc["student"][0] += tp; acc["student"][1] += fp; acc["student"][2] += fn

    def rates(a):
        tp, fp, fn = a
        r = tp / max(tp + fn, 1); p = tp / max(tp + fp, 1)
        return {"recall": r, "precision": p,
                "f1": 2 * r * p / max(r + p, 1e-9), "tp": tp, "fp": fp, "fn": fn}

    rep = {"frames": len(toks), "score_thr": args.thr, "dist_thr_m": DIST_THR,
           "teacher_vs_gt": rates(acc["teacher"])}
    print(f"  {len(toks)} frames, centre-distance {DIST_THR} m, score >= {args.thr}")
    t = rep["teacher_vs_gt"]
    print(f"  TEACHER vs ground truth: recall {t['recall']:.1%}  "
          f"precision {t['precision']:.1%}  F1 {t['f1']:.1%}   "
          f"(tp {t['tp']} fp {t['fp']} fn {t['fn']})")
    if model is not None:
        rep["student_vs_gt"] = rates(acc["student"])
        s = rep["student_vs_gt"]
        print(f"  STUDENT vs ground truth: recall {s['recall']:.1%}  "
              f"precision {s['precision']:.1%}  F1 {s['f1']:.1%}   "
              f"(tp {s['tp']} fp {s['fp']} fn {s['fn']})")
        deg = 1.0 - (s["f1"] / t["f1"]) if t["f1"] > 0 else 1.0
        rep["f1_degradation"] = deg
        print(f"  DEGRADATION on F1: {deg:.1%}   "
              f"{'PASS' if deg < 0.10 else 'FAIL'} against the < 10% bar")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
