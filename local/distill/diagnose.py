"""Full diagnostic suite for a student checkpoint. One JSON record per experiment.

WHY THIS EXISTS RATHER THAN eval_student.py
Five defects in this project were numbers that looked plausible and were never checked
against a baseline (see PLAN.md section 6). So every metric here is reported WITH its
trivial baseline, and the things that caught those defects are measured every time:

    detection    duplicate vs hallucination split, and a threshold sweep. A precision
                 that cannot be raised at ANY threshold is a ranking problem, not a
                 calibration one.
    occupancy    predicted vs teacher class histogram. Over-prediction inflates the IoU
                 union and freezes mIoU while the loss falls -- exactly what a 0.75
                 weighting exponent did here.
    trajectory   ADE beside zeros, constant velocity and a least-squares ego probe, plus
                 the magnitude of the learned residual. A residual of ~0 means the head
                 is returning the anchor and has learned nothing.

    python local/distill/diagnose.py --ckpt outputs/distill/student/student.pt --tag e0
    python local/distill/diagnose.py --ckpt ... --nms 2.0        # add NMS at inference
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

DIST_TOL = 2.0          # nuScenes detection convention: centre distance, not IoU
NOTEBOOK = _ROOT / "outputs" / "distill" / "lab_notebook.json"


# --------------------------------------------------------------------------- matching

RANGE_EDGES = [10, 20, 30, 40]      # recall bins 0-10 / 10-20 / 20-30 / 30-40 / 40+ m
RANGE_NAMES = ["0-10", "10-20", "20-30", "30-40", "40+"]
CLASS_NAMES = ("vehicle", "czone_sign", "bicycle", "generic_object", "pedestrian",
               "traffic_cone", "barrier")                      # nusc_gt_boxes.py order
DET10 = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
         "motorcycle", "bicycle", "traffic_cone", "barrier")   # official order
# 10-class -> teacher group, so a 10-class student and the 7-class teacher are matched in
# the same label space for the headline F1 (motorcycle joins bicycle, as in nusc_gt_boxes)
GROUP10 = np.array([0, 0, 0, 0, 0, 4, 2, 2, 5, 6])


def greedy_match(dxy, dlb, gxy, glb, tol=DIST_TOL):
    """Greedy by centre distance, class must agree.

    Returns (tp, fp, fn, dists, hit); `hit` marks which GT boxes were matched, so recall
    can be split by range -- the student's gap to the teacher lives at 10-30 m.
    """
    used = np.zeros(len(gxy), bool)
    greedy_match.pred_hit = np.zeros(len(dxy), bool)      # which predictions matched (for per-class stats)
    if len(dxy) == 0:
        return 0, 0, len(gxy), [], used
    if len(gxy) == 0:
        return 0, len(dxy), 0, [], used
    tp, dists = 0, []
    for i in range(len(dxy)):
        best, bd = -1, tol
        for j in range(len(gxy)):
            if used[j] or glb[j] != dlb[i]:
                continue
            d = float(np.hypot(*(dxy[i] - gxy[j])))
            if d < bd:
                bd, best = d, j
        if best >= 0:
            used[best] = True; tp += 1; dists.append(bd); greedy_match.pred_hit[i] = True
    return tp, len(dxy) - tp, len(gxy) - tp, dists, used


def nms(xy, lb, sc, radius):
    """Centre-distance NMS within a class. The student has no duplicate suppression of
    any kind -- no Hungarian matching in training, no NMS at inference -- and 51% of its
    detections sit within 2 m of a real object, i.e. piled on top of each other."""
    order = np.argsort(-sc)
    keep = []
    for i in order:
        ok = True
        for j in keep:
            if lb[i] == lb[j] and np.hypot(*(xy[i] - xy[j])) < radius:
                ok = False; break
        if ok:
            keep.append(i)
    return np.array(keep, dtype=np.int64)


# --------------------------------------------------------------------------- baselines

def ego_probe_ade(ego_dir: Path, eval_tokens, n_fit=6000):
    """Least squares from the ego state to the 50x2 future, FIT ON NON-EVAL RECORDS AND
    SCORED ON THE EXACT EVAL FRAMES the student is scored on.

    The first version fit and scored the probe on its own 90/10 split of unrelated
    frames and reported 1.469 m; on the student's actual eval frames the same probe
    scores 1.566 m. That 0.1 m is the difference between "student loses to a linear
    probe" (what was reported for two days) and "every checkpoint beats it". A baseline
    is only a baseline on the same data as the thing it baselines.
    """
    def vec(d):
        return np.concatenate([d["history"].astype(np.float64).reshape(-1),
                               d["velocity"].astype(np.float64).reshape(-1),
                               d["acceleration"].astype(np.float64).reshape(-1),
                               np.eye(3)[int(d["nav"])], [float(d["speed"])], [1.0]])
    ev = set(eval_tokens)
    tr = [(vec(np.load(f)), np.load(f)["future"][:, :2].astype(np.float64))
          for f in sorted(ego_dir.glob("*.npz")) if f.stem not in ev][:n_fit]
    te = [(vec(np.load(ego_dir / f"{t}.npz")),
           np.load(ego_dir / f"{t}.npz")["future"][:, :2].astype(np.float64))
          for t in eval_tokens if (ego_dir / f"{t}.npz").exists()]
    if not tr or not te:
        return {"zeros_ade_m": None, "constant_velocity_ade_m": None,
                "ego_linear_probe_ade_m": None, "probe_eval_frames": 0}
    X = np.array([a for a, _ in tr]); Y = np.array([b.reshape(-1) for _, b in tr])
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Xe = np.array([a for a, _ in te]); Ye = np.array([b for _, b in te])
    probe = np.linalg.norm((Xe @ W).reshape(-1, 50, 2) - Ye, axis=-1).mean()
    t = np.arange(1, 51) / 10.0
    cv = np.mean([np.linalg.norm(y - np.stack([x[78] * t, x[79] * t], -1), axis=-1).mean()
                  for x, y in te])
    zer = np.mean([np.linalg.norm(y, axis=-1).mean() for _, y in te])
    return {"zeros_ade_m": float(zer), "constant_velocity_ade_m": float(cv),
            "ego_linear_probe_ade_m": float(probe), "probe_eval_frames": len(te)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/distill/student/student.pt")
    ap.add_argument("--tag", required=True, help="experiment id, e.g. e0 / e1")
    ap.add_argument("--note", default="", help="what changed and why")
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--classes", default="auto", choices=["auto", "teacher7", "nuscenes10"],
                    help="label space of the checkpoint; auto = 10 classes iff the "
                         "checkpoint's num_classes is 10")
    ap.add_argument("--write-calib", default="", help="write per-class score thresholds chosen on THESE frames to this json")
    ap.add_argument("--calib-from", default="", help="also score the per-class thresholds in this json (chosen on other frames)")
    ap.add_argument("--tokens", default="sorted", choices=["sorted", "scene", "valall", "valfoldA", "valfoldB"],
                    help="sorted: first --limit cached tokens (the frame-split val set); "
                         "scene: the 250-frame subset of the official val scenes")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--nms", type=float, default=0.0, help="NMS radius in m; 0 disables")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--no-ema", action="store_true", help="evaluate raw weights even if EMA saved")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from local.distill.student import StudentConfig, StudentDetector
    from local.distill.train_student import DistillSet

    # Config comes FROM THE CHECKPOINT, never from defaults. Building a default config
    # around a checkpoint trained with different settings silently corrupts the model:
    # a run trained with --ref-points off, evaluated with ref_points=True, has a
    # reference grid ADDED to boxes that are already absolute. That is what produced the
    # "96-97% hallucinated" readings for E3/E4 and an e5 line whose occupancy, map and
    # trajectory were byte-identical to e0 while its F1 differed by 30 points.
    ck = torch.load(args.ckpt, map_location="cuda")
    saved = ck.get("cfg") or {}
    allowed = StudentConfig.__init__.__code__.co_varnames
    cfg = StudentConfig(**{k: v for k, v in saved.items() if k in allowed})
    if saved:
        print(f"  config from checkpoint: det_head={cfg.det_head} "
              f"ref_points={cfg.ref_points} pool={cfg.pool}")
    else:
        print("  WARNING: checkpoint carries no cfg; using defaults, results may be wrong")
    model = StudentDetector(cfg).cuda().eval()
    use_ema = bool(ck.get("ema")) and not args.no_ema
    sd = ck["ema"] if use_ema else ck["model"]
    print(f"  weights: {'EMA' if use_ema else 'raw'}")
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"  state_dict: missing {list(miss)[:4]} unexpected {list(unexp)[:4]}")
    step = int(ck.get("step", 0))

    # Validation split, and only frames that carry ground truth.
    toks = sorted(p.stem for p in Path(args.teacher).glob("*.npz"))
    toks = [t for t in toks if (Path(args.frames) / t / "gt_boxes.npz").exists()]
    if args.tokens == "scene":
        # frames from the 150 official val scenes (never trained on under --split scene)
        ss = json.load(open("data/distill/scene_split.json"))
        toks = [t for t in ss["eval250"] if (Path(args.teacher) / f"{t}.npz").exists()
                and (Path(args.frames) / t / "gt_boxes.npz").exists()]
    if args.tokens == "valall":
        # every cached val-scene keyframe with a teacher record (4,452): the same protocol
        # as eval250 with ~18x the frames, for a retained % whose noise is well under a point
        ss = json.load(open("data/distill/scene_split.json"))
        toks = [t for t in ss["val_tokens"] if (Path(args.teacher) / f"{t}.npz").exists()
                and (Path(args.frames) / t / "gt_boxes.npz").exists()]
    if args.tokens in ("valfoldA", "valfoldB"):
        # the 150 val scenes split in two by sorted scene token: fit per-class thresholds on one
        # half, score them on the other, so a calibrated number is out-of-sample by SCENE
        ss = json.load(open("data/distill/scene_split.json"))
        samp = {x["token"]: x["scene_token"] for x in json.load(open("data/nuscenes/v1.0-trainval/sample.json"))}
        scenes = sorted({samp[t] for t in ss["val_tokens"] if t in samp})
        fold = set(scenes[0::2] if args.tokens == "valfoldA" else scenes[1::2])
        toks = [t for t in ss["val_tokens"] if samp.get(t) in fold and (Path(args.teacher) / f"{t}.npz").exists()
                and (Path(args.frames) / t / "gt_boxes.npz").exists()]
        print(f"  {args.tokens}: {len(fold)} val scenes, {len(toks)} frames")
    toks = toks[:args.limit]
    ten = (cfg.num_classes == 10) if args.classes == "auto" else (args.classes == "nuscenes10")
    gt_name = "gt_boxes10.npz" if ten else "gt_boxes.npz"
    names = DET10 if ten else CLASS_NAMES
    ds = DistillSet(Path(args.frames), Path(args.teacher), cfg, toks,
                    temporal=bool(getattr(cfg, 'temporal', False)), gt_name=gt_name,
                    min_pts=(1 if ten else 0), history=int(getattr(cfg, "history", 1)))
    print(f"  {len(ds)} frames with ground truth, checkpoint step {step}")

    # Extends down to 0.05: the center head's best F1 landed at 0.2, the previous floor,
    # so the true optimum was below the sweep. A dense heatmap head is precise and
    # under-recalls, the opposite of the query head, so its operating point is lower.
    sweep = {t: [0, 0, 0] for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7)}
    t_tp = t_fp = t_fn = 0
    nb = len(RANGE_NAMES)
    rng_tot, rng_t = np.zeros(nb, np.int64), np.zeros(nb, np.int64)
    rng_s = {t: np.zeros(nb, np.int64) for t in sweep}
    C = cfg.num_classes
    cls_tot, cls_t = np.zeros(C, np.int64), np.zeros(C, np.int64)
    cls_s = {t: np.zeros(C, np.int64) for t in sweep}
    # per-class tp/fp (by PREDICTED class) and fn (by GT class) at every sweep threshold, for
    # per-class operating points chosen on calibration frames
    pc_tp = {t: np.zeros(C, np.int64) for t in sweep}; pc_fp = {t: np.zeros(C, np.int64) for t in sweep}
    pc_fn = {t: np.zeros(C, np.int64) for t in sweep}
    calib = json.load(open(args.calib_from)) if args.calib_from else None
    cal_tp = cal_fp = cal_fn = 0
    near = far = 0
    ndet, centres, clus = [], [], []
    occ_p = np.zeros(cfg.occ_num_classes, np.int64); occ_t = np.zeros_like(occ_p)
    occ_i, occ_u = {}, {}
    seg_p = np.zeros(cfg.map_num_classes, np.int64); seg_t = np.zeros_like(seg_p)
    seg_i, seg_u = {}, {}
    ades, anchor_ades, resid = [], [], []
    # occupancy against Occ3D ground truth (data/distill/frames/<tok>/occ3d.npz, val scenes),
    # for the student AND the teacher, inside the camera-visible mask -- the honest yardstick
    Co = cfg.occ_num_classes
    o3_n = 0; o3_i_s = np.zeros(Co); o3_u_s = np.zeros(Co); o3_i_t = np.zeros(Co); o3_u_t = np.zeros(Co)

    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            ego = b["ego"][None].cuda()
            if getattr(cfg, "temporal", False):
                from local.distill.temporal import prev_bev_from_batch, warp_from_batch
                prev_bev = prev_bev_from_batch(model, b, batched=False)
                pc, pb, pocc, pseg, pt, _ = model(b["image"][None].cuda(), b["bev_index"].cuda(),
                                                  b["valid"].cuda(), ego, prev_bev,
                                                  warp_from_batch(b, batched=False))
            else:
                pc, pb, pocc, pseg, pt = model(b["image"][None].cuda(), b["bev_index"].cuda(),
                                               b["valid"].cuda(), ego)
            g = np.load(Path(args.frames) / toks[i] / gt_name)
            gxy, glb = g["boxes"][:, :2], g["labels"]
            if ten and "num_pts" in g.files:       # official rule: no return, no GT box
                k = g["num_pts"] >= 1; gxy, glb = gxy[k], glb[k]
            glb_m = GROUP10[glb] if ten else glb    # label space shared with the teacher
            gbin = np.digitize(np.hypot(gxy[:, 0], gxy[:, 1]), RANGE_EDGES)
            rng_tot += np.bincount(gbin, minlength=nb)
            cls_tot += np.bincount(glb, minlength=C)[:C]

            sc = torch.sigmoid(pc[0].float()).cpu().numpy()
            bx = pb[0].float().cpu().numpy()
            conf, lab = sc.max(-1), sc.argmax(-1)

            # --- teacher vs GT, the bar
            tsc = torch.sigmoid(b["cls"].float()).numpy()
            tk = tsc.max(-1) >= args.thr
            a, bb, c, _, hit = greedy_match(b["box"].numpy()[tk][:, :2], tsc[tk].argmax(-1), gxy, glb_m)
            t_tp += a; t_fp += bb; t_fn += c
            rng_t += np.bincount(gbin[hit], minlength=nb)
            cls_t += np.bincount(glb[hit], minlength=C)[:C]

            # --- student vs GT, swept
            for th in sweep:
                sel = conf >= th
                xy, lb, s2 = bx[sel][:, :2], lab[sel], conf[sel]
                if args.nms > 0 and len(xy):
                    k = nms(xy, lb, s2, args.nms); xy, lb = xy[k], lb[k]
                a, bb, c, dd, hit = greedy_match(xy, (GROUP10[lb] if ten else lb), gxy, glb_m)
                r = sweep[th]; r[0] += a; r[1] += bb; r[2] += c
                rng_s[th] += np.bincount(gbin[hit], minlength=nb)
                cls_s[th] += np.bincount(glb[hit], minlength=C)[:C]
                ph = greedy_match.pred_hit
                if len(lb):
                    pc_tp[th] += np.bincount(lb[ph], minlength=C)[:C]; pc_fp[th] += np.bincount(lb[~ph], minlength=C)[:C]
                pc_fn[th] += np.bincount(glb[~hit], minlength=C)[:C]
                if th == args.thr:
                    ndet.append(len(xy)); centres += dd
                    if len(xy) and len(gxy):
                        dist = np.linalg.norm(xy[:, None] - gxy[None], axis=-1).min(1)
                        near += int((dist <= DIST_TOL).sum()); far += int((dist > DIST_TOL).sum())
                        dd2 = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
                        np.fill_diagonal(dd2, 1e9)
                        clus.append(float((dd2 <= DIST_TOL).sum(1).mean()))

            if calib is not None:
                thr_c = np.array([calib.get(str(k), args.thr) for k in range(C)], np.float32)
                sel = conf >= thr_c[lab]
                xy, lb = bx[sel][:, :2], lab[sel]
                if args.nms > 0 and len(xy):
                    k = nms(xy, lb, conf[sel], args.nms); xy, lb = xy[k], lb[k]
                a, bb, c, _, _ = greedy_match(xy, (GROUP10[lb] if ten else lb), gxy, glb_m)
                cal_tp += a; cal_fp += bb; cal_fn += c
            # --- occupancy vs Occ3D GT where available
            f3 = Path(args.frames) / toks[i] / "occ3d.npz"
            if f3.exists():
                o3 = np.load(f3); gt3 = o3["occ"].astype(np.int64); m3 = o3["mask"] > 0
                ps = pocc[0].float().argmax(-1).cpu().numpy(); pt_ = b["occ"].numpy().astype(np.int64)
                for c_ in range(Co):
                    g = (gt3 == c_) & m3
                    a_s = (ps == c_) & m3; a_t = (pt_ == c_) & m3
                    o3_i_s[c_] += (g & a_s).sum(); o3_u_s[c_] += (g | a_s).sum()
                    o3_i_t[c_] += (g & a_t).sum(); o3_u_t[c_] += (g | a_t).sum()
                o3_n += 1
            # --- occupancy / map, predicted vs teacher distribution
            po = pocc[0].argmax(-1).cpu().numpy().reshape(-1); to = b["occ"].numpy().reshape(-1)
            occ_p += np.bincount(po, minlength=cfg.occ_num_classes)
            occ_t += np.bincount(to, minlength=cfg.occ_num_classes)
            for cc in np.unique(to):
                occ_i[cc] = occ_i.get(cc, 0) + int(((po == cc) & (to == cc)).sum())
                occ_u[cc] = occ_u.get(cc, 0) + int(((po == cc) | (to == cc)).sum())
            ps = pseg[0].argmax(0).cpu().numpy().reshape(-1); ts = b["seg"].numpy().reshape(-1)
            seg_p += np.bincount(ps, minlength=cfg.map_num_classes)
            seg_t += np.bincount(ts, minlength=cfg.map_num_classes)
            for cc in np.unique(ts):
                seg_i[cc] = seg_i.get(cc, 0) + int(((ps == cc) & (ts == cc)).sum())
                seg_u[cc] = seg_u.get(cc, 0) + int(((ps == cc) | (ts == cc)).sum())

            # --- trajectory, with the anchor isolated so a dead head is visible
            if float(b["has_traj"]) > 0:
                fut = b["future"][:, :2].numpy()
                v0 = cfg.hist_points * 3; v1 = v0 + cfg.hist_points * 2
                vel = ego[:, v1 - 2:v1]
                dt = torch.arange(1, cfg.traj_points + 1, device=ego.device,
                                  dtype=vel.dtype) * 0.1
                anc = (vel.unsqueeze(1) * dt.reshape(1, -1, 1))[0, :, :2].float().cpu().numpy()
                ades.append(float(np.linalg.norm(pt[0, :, :2].float().cpu().numpy() - fut,
                                                 axis=-1).mean()))
                anchor_ades.append(float(np.linalg.norm(anc - fut, axis=-1).mean()))
                resid.append(float(np.abs(pt[0, :, :2].float().cpu().numpy() - anc).mean()))

    def f1(tp, fp, fn):
        r = tp / max(tp + fn, 1); p = tp / max(tp + fp, 1)
        return r, p, 2 * r * p / max(r + p, 1e-9)

    def miou(inter, union, drop=None):
        v = [inter[c] / union[c] for c in union if union[c] > 0 and c != drop]
        return float(np.mean(v)) if v else None

    tr, tp_, tf = f1(t_tp, t_fp, t_fn)
    best = max(sweep, key=lambda t: f1(*sweep[t])[2])
    sr, sp, sf = f1(*sweep[args.thr])
    br, bp, bf = f1(*sweep[best])
    free = max(occ_u, key=lambda c: occ_u[c]) if occ_u else None
    base = ego_probe_ade(Path("data/distill/ego"), toks)
    rr_t = rng_t / np.maximum(rng_tot, 1)
    rr_s = rng_s[best] / np.maximum(rng_tot, 1)
    # gate G2 of PLAN.md section 19/20: the teacher-student recall gap at 10-30 m
    mid_gap = float(100.0 * max(rr_t[1] - rr_s[1], rr_t[2] - rr_s[2]))
    rc_t = cls_t / np.maximum(cls_tot, 1)
    rc_s = cls_s[best] / np.maximum(cls_tot, 1)
    if args.write_calib:
        # per class: the sweep threshold with the best per-class F1 (ties -> the higher one)
        chosen = {}
        for k in range(C):
            scores = [(f1(int(pc_tp[t][k]), int(pc_fp[t][k]), int(pc_fn[t][k]))[2], t) for t in sweep]
            chosen[str(k)] = max(scores)[1] if cls_tot[k] > 0 else args.thr
        Path(args.write_calib).parent.mkdir(parents=True, exist_ok=True)
        json.dump(chosen, open(args.write_calib, "w"), indent=1)
        print(f"  per-class thresholds written -> {args.write_calib}: " + " ".join(f"{names[k][:6]}={chosen[str(k)]}" for k in range(C)))
    cal = f1(cal_tp, cal_fp, cal_fn) if calib is not None else None

    occ3d = None
    if o3_n:
        keep_c = [c_ for c_ in range(Co) if c_ != Co - 1 and o3_u_t[c_] + o3_u_s[c_] > 0]   # drop 'empty'
        ms_ = float(np.mean([o3_i_s[c_] / max(o3_u_s[c_], 1) for c_ in keep_c]))
        mt_ = float(np.mean([o3_i_t[c_] / max(o3_u_t[c_], 1) for c_ in keep_c]))
        occ3d = {"frames": o3_n, "student_miou": ms_, "teacher_miou": mt_,
                 "retained_pct": 100.0 * ms_ / max(mt_, 1e-9),
                 "per_class_student": [round(float(o3_i_s[c_] / max(o3_u_s[c_], 1)), 4) for c_ in range(Co)],
                 "per_class_teacher": [round(float(o3_i_t[c_] / max(o3_u_t[c_], 1)), 4) for c_ in range(Co)]}
    rec = {
        "tag": args.tag, "note": args.note, "step": step, "ckpt": args.ckpt,
        "occ3d": occ3d,
        "at": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": len(ds),
        "nms_radius_m": args.nms,
        "teacher_vs_gt": {"recall": tr, "precision": tp_, "f1": tf},
        "student_vs_gt": {"recall": sr, "precision": sp, "f1": sf, "thr": args.thr},
        "student_vs_gt_best_thr": {"thr": best, "recall": br, "precision": bp, "f1": bf},
        "detection_gate_f1": 0.9 * tf,
        "detection_pass": bf >= 0.9 * tf,
        "degradation_pct": 100.0 * (1 - bf / max(tf, 1e-9)),
        # the user's bar: keep at least 80% of the teacher's F1
        "retained_pct": 100.0 * bf / max(tf, 1e-9),
        "bar_80pct_f1": 0.8 * tf,
        "bar_80pct_pass": bf >= 0.8 * tf,
        "recall_by_range": {"bins_m": RANGE_NAMES, "gt_count": rng_tot.tolist(),
                            "teacher": rr_t.round(4).tolist(),
                            "student_best_thr": rr_s.round(4).tolist()},
        "mid_range_gap_pts": mid_gap,
        "classes": "nuscenes10" if ten else "teacher7",
        "recall_by_class": {"names": list(names[:C]), "gt_count": cls_tot.tolist(),
                            "teacher": rc_t.round(4).tolist(),
                            "student_best_thr": rc_s.round(4).tolist()},
        "eval_tokens": args.tokens,
        "student_vs_gt_calibrated": ({"recall": cal[0], "precision": cal[1], "f1": cal[2], "thresholds": calib}
                                     if cal is not None else None),
        "sweep": {str(t): dict(zip(("recall", "precision", "f1"), f1(*v)))
                  for t, v in sweep.items()},
        "detections_per_frame": float(np.mean(ndet)) if ndet else 0.0,
        "frac_near_real_object": near / max(near + far, 1),
        "frac_hallucinated": far / max(near + far, 1),
        "mean_neighbours_within_2m": float(np.mean(clus)) if clus else 0.0,
        "centre_median_m": float(np.median(centres)) if centres else None,
        "occ_miou_occupied": miou(occ_i, occ_u, drop=free),
        "occ_miou_all": miou(occ_i, occ_u),
        "occ_pred_frac": (occ_p / max(occ_p.sum(), 1)).round(5).tolist(),
        "occ_teacher_frac": (occ_t / max(occ_t.sum(), 1)).round(5).tolist(),
        "map_miou": miou(seg_i, seg_u),
        "map_pred_frac": (seg_p / max(seg_p.sum(), 1)).round(5).tolist(),
        "map_teacher_frac": (seg_t / max(seg_t.sum(), 1)).round(5).tolist(),
        "trajectory_ade_m": float(np.mean(ades)) if ades else None,
        "trajectory_anchor_ade_m": float(np.mean(anchor_ades)) if anchor_ades else None,
        "trajectory_mean_residual_m": float(np.mean(resid)) if resid else None,
        "trajectory_baselines": base,
        "trajectory_beats_probe": (bool(np.mean(ades) < base["ego_linear_probe_ade_m"])
                                   if ades else False),
    }

    print(f"\n=== {args.tag} === {args.note}")
    print(f"  TEACHER vs GT   recall {tr:6.1%} precision {tp_:6.1%} F1 {tf:6.1%}")
    print(f"  STUDENT vs GT   recall {sr:6.1%} precision {sp:6.1%} F1 {sf:6.1%}   @thr {args.thr}")
    print(f"  STUDENT best    recall {br:6.1%} precision {bp:6.1%} F1 {bf:6.1%}   @thr {best}"
          + (f"  NMS {args.nms} m" if args.nms else ""))
    print(f"  detection gate  F1 >= {0.9*tf:.1%}   -> {'PASS' if rec['detection_pass'] else 'FAIL'}"
          f"   (degradation {rec['degradation_pct']:.1f}%)")
    print(f"  retained {rec['retained_pct']:.1f}% of teacher F1   (bar 80% -> F1 >= {0.8*tf:.1%})"
          f"   -> {'PASS' if rec['bar_80pct_pass'] else 'FAIL'}")
    if cal is not None:
        print(f"  STUDENT per-class thresholds (calibrated elsewhere): recall {cal[0]:6.1%} precision {cal[1]:6.1%} "
              f"F1 {cal[2]:6.1%}   -> {100*cal[2]/max(tf,1e-9):.1f}% of teacher")
    if occ3d:
        print(f"  occupancy vs Occ3D GT ({occ3d['frames']} frames, camera-visible voxels, mIoU over non-empty classes): "
              f"student {occ3d['student_miou']:.1%}  teacher {occ3d['teacher_miou']:.1%}  -> {occ3d['retained_pct']:.0f}% of teacher")
    print("  recall by range     " + "".join(f"{n:>8}" for n in RANGE_NAMES)
          + "      (GT " + "/".join(str(int(v)) for v in rng_tot) + ")")
    print("    teacher           " + "".join(f"{v:8.1%}" for v in rr_t))
    print("    student @best     " + "".join(f"{v:8.1%}" for v in rr_s)
          + f"   mid-range gap {mid_gap:.1f} pts (G2 wants < 8)")
    print("  recall by class     " + "".join(f"{n[:9]:>10}" for n in names[:C])
          + "      (GT " + "/".join(str(int(v)) for v in cls_tot) + ")")
    print("    teacher           " + "".join(f"{v:10.1%}" for v in rc_t))
    print("    student @best     " + "".join(f"{v:10.1%}" for v in rc_s))
    print(f"  {rec['detections_per_frame']:.1f} det/frame   "
          f"{rec['frac_near_real_object']:.1%} near a real object, "
          f"{rec['frac_hallucinated']:.1%} hallucinated, "
          f"{rec['mean_neighbours_within_2m']:.2f} neighbours")
    print(f"  occupancy mIoU  {rec['occ_miou_occupied']:.1%} occupied "
          f"({rec['occ_miou_all']:.1%} incl free)     map mIoU {rec['map_miou']:.1%}")
    if ades:
        print(f"  trajectory ADE  {rec['trajectory_ade_m']:.3f} m   "
              f"anchor {rec['trajectory_anchor_ade_m']:.3f}  residual "
              f"{rec['trajectory_mean_residual_m']:.4f} m")
        print(f"     baselines: zeros {base['zeros_ade_m']:.3f}  cv "
              f"{base['constant_velocity_ade_m']:.3f}  ego-probe "
              f"{base['ego_linear_probe_ade_m']:.3f}  -> beats probe: "
              f"{rec['trajectory_beats_probe']}")

    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    book = json.loads(NOTEBOOK.read_text()) if NOTEBOOK.exists() else []
    book.append(rec)
    NOTEBOOK.write_text(json.dumps(book, indent=1))
    print(f"\n  appended to {NOTEBOOK.relative_to(_ROOT)} ({len(book)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
