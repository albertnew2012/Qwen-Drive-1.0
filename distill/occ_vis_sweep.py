"""Choose the visibility cut of a `--occ-vis` occupancy head on held-out frames, then (optionally) write a copy
of the checkpoint with that cut in its config.

The head predicts classes and a camera-visibility logit per voxel; at inference a voxel whose logit is at or below
`cfg.occ_vis_thr` becomes "empty". This runs the head raw once per frame and scores every candidate cut:
visible-voxel mIoU against Occ3D (9 classes present in the GT, camera-visible voxels -- the benchmark region) and
occupied voxels per frame inside / outside the visible region, beside the teacher's on the same frames.

Calibration frames default to val frames from scene fold A that are NOT in eval250, so the 250-frame gate stays
out-of-sample. The chosen cut is the one with the best visible mIoU among cuts whose unseen-space fill is at most
--max-outside x the teacher's.

    PYTHONPATH=src:.:tools python distill/occ_vis_sweep.py --ckpt outputs/distill/exp/r8_occ_e/student.pt \
        --write outputs/distill/exp/r8_occ_e/student_vis_cal.pt
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

THRS = (-6.0, -5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)
BIASES = (-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="vis", choices=["vis", "empty-bias"],
                    help="vis: sweep the visibility cut of an --occ-vis head; empty-bias: sweep an offset on the 'empty' logit "
                         "(any native head; for an --occ-vis head at its configured cut)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tokens", default="valfoldA-not250", choices=["valfoldA-not250", "scene"])
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--max-outside", type=float, default=1.0, help="allowed unseen-space fill, as a multiple of the teacher's")
    ap.add_argument("--write", default="", help="write a copy of the checkpoint with the chosen cut here (never overwrites)")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from distill.student import StudentConfig, StudentDetector
    from distill.train_student import DistillSet
    from distill.temporal import prev_bev_from_batch, warp_from_batch

    ck = torch.load(args.ckpt, map_location="cpu")
    allowed = StudentConfig.__init__.__code__.co_varnames
    cfg = StudentConfig(**{k: v for k, v in ck["cfg"].items() if k in allowed})
    assert args.mode == "empty-bias" or getattr(cfg, "occ_vis", False), "not an --occ-vis checkpoint"
    knob = "occ_vis_thr" if args.mode == "vis" else "occ_empty_bias"
    cands = THRS if args.mode == "vis" else BIASES
    model = StudentDetector(cfg).cuda().eval()
    model.load_state_dict(ck.get("ema") or ck["model"], strict=True)
    model.occ_raw = True                                   # classes + visibility logit, no gating

    ss = json.load(open("data/distill/scene_split.json"))
    ok = lambda t: (Path(args.teacher) / f"{t}.npz").exists() and (Path(args.frames) / t / "occ3d_native.npz").exists() \
        and (Path(args.frames) / t / "gt_boxes10.npz").exists()
    if args.tokens == "scene":
        toks = [t for t in ss["eval250"] if ok(t)]
    else:
        samp = {x["token"]: x["scene_token"] for x in json.load(open("data/nuscenes/v1.0-trainval/sample.json"))}
        scenes = sorted({samp[t] for t in ss["val_tokens"] if t in samp}); fold = set(scenes[0::2])
        e250 = set(ss["eval250"])
        toks = [t for t in ss["val_tokens"] if samp.get(t) in fold and t not in e250 and ok(t)]
        toks = toks[::max(1, len(toks) // args.limit)]
    toks = toks[:args.limit]
    ds = DistillSet(Path(args.frames), Path(args.teacher), cfg, toks, temporal=True, gt_name="gt_boxes10.npz",
                    min_pts=1, history=int(getattr(cfg, "history", 1)))
    nc = cfg.occ_num_classes; E = nc - 1
    T = len(cands)
    I = np.zeros((T, nc)); U = np.zeros((T, nc)); G = np.zeros(nc)
    cnt = np.zeros((T, 2))                                  # student occupied outside / inside the visible region
    ti = np.zeros(nc); tu = np.zeros(nc); tcnt = np.zeros(2)
    vis_stats = np.zeros((T, 4))                            # tp fp fn tn of the visibility cut vs the camera mask
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            out = model(b["image"][None].cuda(), b["bev_index"].cuda(), b["valid"].cuda(), b["ego"][None].cuda(),
                        prev_bev_from_batch(model, b, batched=False), warp_from_batch(b, batched=False))
            raw = out[2][0].float()
            if args.mode == "vis":
                cls = raw[..., :nc].argmax(-1).cpu().numpy(); vl = raw[..., nc].cpu().numpy()
            else:
                lg = raw[..., :nc].clone()
                if getattr(cfg, "occ_vis", False):          # gate at the configured cut first
                    lg[..., E] = torch.where(raw[..., nc] > float(getattr(cfg, "occ_vis_thr", 0.0)), lg[..., E], torch.full_like(lg[..., E], 1e4))
                top2 = lg[..., :E].max(-1)                  # best non-empty class and its margin over 'empty'
                cls = top2.indices.cpu().numpy(); vl = (top2.values - lg[..., E]).cpu().numpy()   # occupied iff margin > -bias
            nat = np.load(Path(args.frames) / toks[i] / "occ3d_native.npz"); gt = nat["occ"].astype(np.int64); m = nat["mask"] > 0
            tn_ = b["occ"].numpy().astype(np.int64)
            for c in range(E):
                g = (gt == c) & m; G[c] += g.sum()
                a = (tn_ == c) & m; ti[c] += (g & a).sum(); tu[c] += (g | a).sum()
            tcnt += [((tn_ != E) & ~m).sum(), ((tn_ != E) & m).sum()]
            for k, t in enumerate(cands):
                pv = vl > t if args.mode == "vis" else vl > t        # empty-bias b: occupied iff margin > b
                p = np.where(pv, cls, E)
                for c in range(E):
                    g = (gt == c) & m; a = (p == c) & m
                    I[k, c] += (g & a).sum(); U[k, c] += (g | a).sum()
                cnt[k] += [((p != E) & ~m).sum(), ((p != E) & m).sum()]
                vis_stats[k] += [(pv & m).sum(), (pv & ~m).sum(), (~pv & m).sum(), (~pv & ~m).sum()]
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ds)}", flush=True)
    n = len(ds); keep = [c for c in range(E) if G[c] > 0]
    miou = lambda ii, uu: float(np.mean([ii[c] / max(uu[c], 1) for c in keep]))
    t_miou = miou(ti, tu); t_out, t_in = tcnt / n
    print(f"\n  {n} frames ({args.tokens}); teacher: visible mIoU {100 * t_miou:.1f}%  outside {t_out:,.0f}  inside {t_in:,.0f}")
    print(f"  {knob:>14}  mIoU   outside   inside   predicted-occupied-or-visible vs mask: precision recall")
    rows = []
    for k, t in enumerate(cands):
        tp, fp, fn, _ = vis_stats[k]
        r = {"thr": t, "miou": miou(I[k], U[k]), "outside": cnt[k, 0] / n, "inside": cnt[k, 1] / n,
             "vis_precision": tp / max(tp + fp, 1), "vis_recall": tp / max(tp + fn, 1)}
        rows.append(r)
        print(f"  {t:+14.2f}  {100 * r['miou']:5.1f}%  {r['outside']:8,.0f}  {r['inside']:7,.0f}"
              f"   {100 * r['vis_precision']:5.1f}%  {100 * r['vis_recall']:5.1f}%")
    ok_rows = [r for r in rows if r["outside"] <= args.max_outside * t_out]
    best = max(ok_rows or rows, key=lambda r: r["miou"])
    print(f"  chosen {knob} {best['thr']:+.2f}: visible mIoU {100 * best['miou']:.1f}% (teacher {100 * t_miou:.1f}%), "
          f"outside {best['outside']:,.0f} (teacher {t_out:,.0f}), inside {best['inside']:,.0f} (teacher {t_in:,.0f})")
    rep = {"ckpt": args.ckpt, "frames": n, "tokens": args.tokens, "teacher": {"miou": t_miou, "outside": t_out, "inside": t_in},
           "rows": rows, "chosen": best}
    Path(args.ckpt).with_suffix(f".{args.mode}_sweep.json").write_text(json.dumps(rep, indent=1))
    if args.write:
        dst = Path(args.write)
        if dst.exists():
            raise SystemExit(f"refusing to overwrite {dst}")
        ck2 = dict(ck); ck2["cfg"] = dict(ck["cfg"], **{knob: float(best["thr"])})
        ck2["note"] = f"{ck.get('note', '')} | {knob} {best['thr']:+.2f} from occ_vis_sweep.py on {n} {args.tokens} frames".strip(" |")
        torch.save(ck2, dst)
        print(f"  wrote {dst} with {knob} {best['thr']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
