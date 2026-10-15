"""Fit per-class logit offsets for a native-grid occupancy head on held-out frames (no retraining).

The head's classes come out differently calibrated (inverse-frequency class weights, the unobserved-space term),
and the Occ3D metric weighs the 9 occupied classes equally. Ten offsets added to the logits at inference
(`cfg.occ_class_bias`, applied in-graph) move each class's operating point. They are fitted by coordinate ascent
on cached raw logits of calibration frames, maximising visible-voxel mIoU (classes present in the GT,
camera-visible voxels) while the occupied voxels per frame outside the visible region stay at or below the
teacher's on the same frames.

Calibration frames default to fold-A val frames outside eval250 (as occ_vis_sweep.py), so the 250-frame check and
fold B stay out-of-sample.

    PYTHONPATH=src:.:tools python distill/occ_bias_fit.py --ckpt outputs/distill/exp/r8_occ_a/snap_8000.pt \
        --write outputs/distill/exp/r8_occ_a/snap_8000_classbias.pt
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--max-outside", type=float, default=1.0, help="allowed unobserved fill, multiple of the teacher's")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--write", default="", help="copy of the checkpoint with the fitted offsets (never overwrites)")
    ap.add_argument("--ensemble", default="", help="comma-separated extra checkpoints whose heads read the SAME BEV "
                    "(head-only runs on one base): their log-probabilities are averaged with --ckpt's before the fit "
                    "(measurement only; --write is refused)")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from distill.student import StudentConfig, StudentDetector
    from distill.train_student import DistillSet
    from distill.temporal import prev_bev_from_batch, warp_from_batch

    ck = torch.load(args.ckpt, map_location="cpu")
    allowed = StudentConfig.__init__.__code__.co_varnames
    cfg = StudentConfig(**{k: v for k, v in ck["cfg"].items() if k in allowed})
    assert getattr(cfg, "occ_grid", "bev") == "native", "native-grid occupancy heads only"
    assert not getattr(cfg, "occ_vis", False), "--occ-vis heads: use occ_vis_sweep.py"
    model = StudentDetector(cfg).cuda().eval()
    model.load_state_dict(ck.get("ema") or ck["model"], strict=True)
    model.occ_raw = True                                           # raw logits: no configured offsets
    extra = []
    for path in [x for x in args.ensemble.split(",") if x]:
        cke = torch.load(path, map_location="cpu")
        cfe = StudentConfig(**{k: v for k, v in cke["cfg"].items() if k in allowed})
        me = StudentDetector(cfe).cuda().eval(); me.load_state_dict(cke.get("ema") or cke["model"], strict=True); me.occ_raw = True
        extra.append(me); print(f"  ensemble member: {path}", flush=True)
    if extra and args.write:
        raise SystemExit("--write with --ensemble is not supported (the model has one occupancy head)")

    ss = json.load(open("data/distill/scene_split.json"))
    samp = {x["token"]: x["scene_token"] for x in json.load(open("data/nuscenes/v1.0-trainval/sample.json"))}
    scenes = sorted({samp[t] for t in ss["val_tokens"] if t in samp}); fold = set(scenes[0::2]); e250 = set(ss["eval250"])
    ok = lambda t: (Path(args.teacher) / f"{t}.npz").exists() and (Path(args.frames) / t / "occ3d_native.npz").exists() \
        and (Path(args.frames) / t / "gt_boxes10.npz").exists()
    toks = [t for t in ss["val_tokens"] if samp.get(t) in fold and t not in e250 and ok(t)]
    toks = toks[::max(1, len(toks) // args.limit)][:args.limit]
    ds = DistillSet(Path(args.frames), Path(args.teacher), cfg, toks, temporal=True, gt_name="gt_boxes10.npz",
                    min_pts=1, history=int(getattr(cfg, "history", 1)))
    nc = cfg.occ_num_classes; E = nc - 1
    L, G, M = [], [], []; t_out = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            out = model(b["image"][None].cuda(), b["bev_index"].cuda(), b["valid"].cuda(), b["ego"][None].cuda(),
                        prev_bev_from_batch(model, b, batched=False), warp_from_batch(b, batched=False))
            lg = out[2][0, ..., :nc].float()
            if extra:                                                  # mean log-probability over the heads
                lg = lg.log_softmax(-1)
                for me in extra:
                    oe = me(b["image"][None].cuda(), b["bev_index"].cuda(), b["valid"].cuda(), b["ego"][None].cuda(),
                            prev_bev_from_batch(me, b, batched=False), warp_from_batch(b, batched=False))
                    lg = lg + oe[2][0, ..., :nc].float().log_softmax(-1)
                lg = lg / (1 + len(extra))
            L.append(lg.half())
            nat = np.load(Path(args.frames) / toks[i] / "occ3d_native.npz")
            G.append(torch.from_numpy(nat["occ"].astype(np.int64)).cuda().to(torch.int8)); M.append(torch.from_numpy(nat["mask"] > 0).cuda())
            t_out += float(((b["occ"] != E) & ~torch.from_numpy(nat["mask"] > 0)).sum())
            if (i + 1) % 50 == 0:
                print(f"  cached {i + 1}/{len(ds)}", flush=True)
    n = len(ds); t_out /= n; limit = args.max_outside * t_out
    L = torch.stack(L); G = torch.stack(G).long(); M = torch.stack(M)
    present = [c for c in range(E) if bool(((G == c) & M).any())]

    def score(bias):
        inter = torch.zeros(E, device="cuda"); union = torch.zeros(E, device="cuda"); out_n = 0.0; in_n = 0.0
        bt = torch.as_tensor(bias, dtype=L.dtype, device="cuda")
        for s in range(0, n, 25):
            p = (L[s:s + 25] + bt).argmax(-1); g = G[s:s + 25]; m = M[s:s + 25]
            for c in present:
                a = (p == c) & m; gg = (g == c) & m
                inter[c] += (a & gg).sum(); union[c] += (a | gg).sum()
            occ = p != E; out_n += float((occ & ~m).sum()); in_n += float((occ & m).sum())
        iou = (inter / union.clamp_min(1)).cpu().numpy()
        return float(np.mean([iou[c] for c in present])), out_n / n, in_n / n, iou

    def objective(r):   # mIoU, minus a steep penalty for unobserved fill above the limit
        miou, out_n = r[0], r[1]
        return miou - (0.5 * (out_n - limit) / max(limit, 1.0) if out_n > limit else 0.0)

    bias = np.zeros(nc)
    base = score(bias); best = base; print(f"\n  {n} frames; teacher outside {t_out:,.0f}/frame (limit {limit:,.0f})")
    print(f"  raw: visible mIoU {100 * base[0]:.1f}%  outside {base[1]:,.0f}  inside {base[2]:,.0f}")
    # start from the best single 'empty' offset, then refine every class
    for e in np.arange(-1.0, 3.01, 0.25):
        trial = bias.copy(); trial[E] = e; r = score(trial)
        if objective(r) > objective(best): best, bias = r, trial
    print(f"  best empty offset {bias[E]:+.2f}: {100 * best[0]:.1f}%  outside {best[1]:,.0f}  inside {best[2]:,.0f}", flush=True)
    step = 0.5
    for rnd in range(args.rounds):
        for c in list(range(E)) + [E]:
            for d in (-2 * step, -step, step, 2 * step):
                trial = bias.copy(); trial[c] += d; r = score(trial)
                if objective(r) > objective(best) + 1e-5:
                    best, bias = r, trial
        print(f"  round {rnd + 1} (step {step}): {100 * best[0]:.2f}%  outside {best[1]:,.0f}  inside {best[2]:,.0f}  "
              f"bias {np.round(bias, 2).tolist()}", flush=True)
        step /= 2
    print("  per-class IoU (present classes):", {c: round(float(best[3][c]), 3) for c in present})
    rep = {"ckpt": args.ckpt, "frames": n, "teacher_outside": t_out, "raw": base[:3], "fitted": best[:3], "bias": bias.tolist()}
    if not extra:
        Path(args.ckpt).with_suffix(".classbias_fit.json").write_text(json.dumps(rep, indent=1))
    if args.write:
        dst = Path(args.write)
        if dst.exists():
            raise SystemExit(f"refusing to overwrite {dst}")
        ck2 = dict(ck); ck2.pop("opt", None)
        ck2["cfg"] = dict(ck["cfg"], occ_class_bias=[float(x) for x in bias], occ_empty_bias=0.0)
        ck2["note"] = f"{ck.get('note', '')} | occ_class_bias {np.round(bias, 2).tolist()} from occ_bias_fit.py on {n} fold-A frames".strip(" |")
        torch.save(ck2, dst); print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
