"""Official nuScenes detection metrics (mAP / NDS) for the student or the cached teacher.

WHY
The 2 m centre-distance F1 in diagnose.py is a home-made yardstick over 250 frames. The
deliverable is judged in the metric every camera-only paper reports, on the official val
split (150 scenes, 6,019 keyframes), computed by `nuscenes.eval.detection` itself.

WHAT IS EVALUATED
  --who student   the checkpoint, run on every val keyframe (frames the teacher was never
                  run on get their calibration from calib.npz -- see DistillSet).
  --who teacher   the cached teacher outputs; only 4,452 of the 6,019 val keyframes are
                  cached, so pair it with --subset cached (and evaluate the student on the
                  same subset for a like-for-like comparison).
A 7-class model is mapped vehicle->car, bicycle->bicycle, pedestrian, traffic_cone,
barrier (czone_sign / generic_object dropped): its trucks, buses and trailers count as
misses, which is the taxonomy's own limitation. A 10-class model maps one to one.
Velocity is what the head predicts (zero for heads without it); attributes follow the
BEVDet convention from the predicted speed (parked / standing / without_rider below
0.2 m/s), so AVE/AAE are honest but not flattering.

    python local/distill/eval_official.py --ckpt outputs/distill/exp/e5c/student.pt --tag e5c
    python local/distill/eval_official.py --who teacher --subset cached --tag teacher
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

DET10 = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
         "motorcycle", "bicycle", "traffic_cone", "barrier")
MAP7 = {0: "car", 2: "bicycle", 4: "pedestrian", 5: "traffic_cone", 6: "barrier"}
NOTEBOOK = _ROOT / "outputs" / "distill" / "lab_notebook.json"


def attribute(name: str, speed: float) -> str:
    moving = speed > 0.2
    if name in ("car", "truck", "bus", "trailer", "construction_vehicle"):
        return "vehicle.moving" if moving else "vehicle.parked"
    if name == "pedestrian":
        return "pedestrian.moving" if moving else "pedestrian.standing"
    if name in ("motorcycle", "bicycle"):
        return "cycle.with_rider" if moving else "cycle.without_rider"
    return ""


def load_model(ckpt_path: str, no_ema: bool):
    from local.distill.student import StudentConfig, StudentDetector
    ck = torch.load(ckpt_path, map_location="cpu")
    saved = ck.get("cfg") or {}
    cfg = StudentConfig(**{k: v for k, v in saved.items()
                           if k in StudentConfig.__init__.__code__.co_varnames})
    model = StudentDetector(cfg)
    sd = ck.get("ema") if (ck.get("ema") and not no_ema) else ck["model"]
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items() if k != "n_averaged"}
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval(), cfg, ck.get("step")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/distill/exp/e5c/student.pt")
    ap.add_argument("--who", default="student", choices=["student", "teacher"])
    ap.add_argument("--subset", default="all", choices=["all", "cached"],
                    help="all: every val keyframe (6,019); cached: only those with a "
                         "teacher npz (4,452)")
    ap.add_argument("--limit", type=int, default=0, help="first N val keyframes (test runs)")
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--nms", type=float, default=0.0, help="class-wise centre NMS radius, m")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--root", default="data/nuscenes")
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--out", default="outputs/distill/official")
    ap.add_argument("--tag", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.chdir(_ROOT)

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion
    t0 = time.time()
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.root, verbose=False)
    val_scenes = set(create_splits_scenes()["val"])
    toks = [s["token"] for s in nusc.sample
            if nusc.get("scene", s["scene_token"])["name"] in val_scenes]
    if args.subset == "cached" or args.who == "teacher":
        toks = [t for t in toks if (Path(args.teacher) / f"{t}.npz").exists()]
    if args.limit:
        toks = toks[:args.limit]
    print(f"  devkit ready in {time.time()-t0:.0f}s; {len(toks)} val keyframes to score", flush=True)

    def ego_pose(tok):
        s = nusc.get("sample", tok)
        e = nusc.get("ego_pose", nusc.get("sample_data", s["data"]["LIDAR_TOP"])["ego_pose_token"])
        return Quaternion(e["rotation"]).rotation_matrix, np.asarray(e["translation"])

    results, n_boxes = {}, 0

    def emit(tok, scores, labels, boxes, ten):
        """boxes: (N, 10) [x, y, z, log w, log l, log h, sin, cos, vx, vy] in the ego frame."""
        nonlocal n_boxes
        R, T = ego_pose(tok)
        yaw_ego = float(np.arctan2(R[1, 0], R[0, 0]))
        keep = scores >= args.thr
        scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
        if args.nms > 0 and len(boxes):
            from local.distill.diagnose import nms
            k = nms(boxes[:, :2], labels, scores, args.nms)
            scores, labels, boxes = scores[k], labels[k], boxes[k]
        order = np.argsort(-scores)[:500]                  # evaluator cap per sample
        out = []
        for i in order:
            name = DET10[int(labels[i])] if ten else MAP7.get(int(labels[i]))
            if name is None:
                continue
            b = boxes[i]
            xyz = R @ b[:3] + T
            yaw = float(np.arctan2(b[6], b[7])) + yaw_ego
            v = R[:2, :2] @ b[8:10]
            out.append({"sample_token": tok, "translation": xyz.tolist(),
                        "size": np.exp(b[3:6]).tolist(),
                        "rotation": Quaternion(axis=[0, 0, 1], angle=yaw).elements.tolist(),
                        "velocity": v.tolist(), "detection_name": name,
                        "detection_score": float(scores[i]),
                        "attribute_name": attribute(name, float(np.hypot(*v)))})
        results[tok] = out; n_boxes += len(out)

    step, ten = None, False
    if args.who == "teacher":
        for tok in toks:
            d = np.load(Path(args.teacher) / f"{tok}.npz")
            sc = 1 / (1 + np.exp(-d["cls"].astype(np.float64)))
            emit(tok, sc.max(-1), sc.argmax(-1), d["box"].astype(np.float64), ten=False)
    else:
        from local.distill.train_student import DistillSet
        model, cfg, step = load_model(args.ckpt, args.no_ema)
        ten = cfg.num_classes == 10
        temporal = bool(getattr(cfg, "temporal", False))
        ds = DistillSet(Path(args.frames), Path(args.teacher), cfg, toks, temporal=temporal,
                        history=int(getattr(cfg, "history", 1)))
        dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=args.workers,
                                         collate_fn=lambda x: x[0])
        print(f"  {'temporal ' if temporal else ''}{'10' if ten else '7'}-class student, "
              f"step {step}, {'raw' if args.no_ema else 'EMA if saved'} weights", flush=True)
        with torch.no_grad():
            for i, b in enumerate(dl):
                ego = b["ego"][None].cuda()
                if temporal:
                    from local.distill.temporal import prev_bev_from_batch, warp_from_batch
                    prev = prev_bev_from_batch(model, b, batched=False)
                    pc, pb, *_ = model(b["image"][None].cuda(), b["bev_index"].cuda(),
                                       b["valid"].cuda(), ego, prev, warp_from_batch(b, batched=False))
                else:
                    pc, pb, *_ = model(b["image"][None].cuda(), b["bev_index"].cuda(),
                                       b["valid"].cuda(), ego)
                sc = torch.sigmoid(pc[0].float()).cpu().numpy()
                emit(toks[i], sc.max(-1), sc.argmax(-1), pb[0].float().cpu().numpy().astype(np.float64), ten)
                if (i + 1) % 500 == 0:
                    print(f"    {i+1}/{len(toks)} frames, {time.time()-t0:.0f}s", flush=True)

    out_dir = Path(args.out) / (args.tag or args.who); out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results_nusc.json"
    json.dump({"meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                        "use_map": False, "use_external": False}, "results": results},
              open(res_path, "w"))
    print(f"  {n_boxes/ max(len(toks),1):.1f} boxes/frame written -> {res_path}", flush=True)
    if n_boxes == 0:
        print("  no predictions above --thr at all; the devkit cannot evaluate an empty set")
        return 1

    # --- the official evaluator, restricted to the scored keyframes when a subset is used
    import nuscenes.eval.detection.evaluate as E
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.config import config_factory
    keep = set(toks); orig_load_gt = E.load_gt

    def load_gt_subset(nusc_, eval_split, box_cls, verbose=False):
        gt = orig_load_gt(nusc_, eval_split, box_cls, verbose)
        sub = EvalBoxes()
        for t in gt.sample_tokens:
            if t in keep:
                sub.add_boxes(t, gt[t])
        return sub
    E.load_gt = load_gt_subset
    ev = E.DetectionEval(nusc, config=config_factory("detection_cvpr_2019"),
                         result_path=str(res_path), eval_set="val",
                         output_dir=str(out_dir), verbose=False)
    m = ev.main(plot_examples=0, render_curves=False)
    aps = {c: round(float(m["mean_dist_aps"][c]), 4) for c in m["mean_dist_aps"]}
    rec = {"tag": args.tag or args.who, "note": args.note, "who": args.who, "ckpt": args.ckpt,
           "step": step, "classes": "nuscenes10" if ten else "teacher7", "subset": args.subset,
           "frames": len(toks), "thr": args.thr, "nms_radius_m": args.nms,
           "official": True, "mAP": round(float(m["mean_ap"]), 4), "NDS": round(float(m["nd_score"]), 4),
           "class_ap": aps, "tp_errors": {k: round(float(v), 4) for k, v in m["tp_errors"].items()},
           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"\n=== OFFICIAL nuScenes detection ({rec['tag']}, {len(toks)} val keyframes, "
          f"{rec['classes']}, subset {args.subset}) ===")
    print(f"  mAP {rec['mAP']:.4f}   NDS {rec['NDS']:.4f}")
    print("  AP per class: " + "  ".join(f"{c[:9]} {v:.3f}" for c, v in aps.items()))
    print("  TP errors: " + "  ".join(f"{k} {v:.3f}" for k, v in rec["tp_errors"].items()))
    nb = json.load(open(NOTEBOOK)) if NOTEBOOK.exists() else []
    nb.append(rec); json.dump(nb, open(NOTEBOOK, "w"), indent=1)
    print(f"  appended to {NOTEBOOK} ({len(nb)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
