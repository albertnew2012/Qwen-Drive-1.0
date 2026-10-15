"""Same-frame occupancy panels: Occ3D ground truth, the teacher, and one or more student checkpoints.

Every grid is shown on Occ3D's native grid (0.4 m, +-40 m, index (x, y, z)) through the session renderer's own
`_panel_occupancy`, so a column here looks exactly like the occupancy panel of the session GIF. A student whose
head predicts on the 0.512 m detection grid (v6 and earlier, index (y, x)) is looked up at each native cell
centre first -- the same lookup diagnose.py scores. The GT column shows only camera-visible voxels (the Occ3D
evaluation region); the right-most number under each panel counts non-empty voxels OUTSIDE that region.

    PYTHONPATH=src:.:tools python distill/occ_compare.py --out outputs/debug/occ_compare.png \
        --ckpt v6=outputs/student_release_v6/student_r4_small_avg.pt --ckpt r8a=outputs/distill/exp/r8_occ_a/student.pt
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))


def student_native(model, cfg, b) -> np.ndarray:
    """One frame through a student, occupancy argmax as (200, 200, 16) on the native grid in (x, y, z)."""
    from distill.temporal import prev_bev_from_batch, warp_from_batch
    with torch.no_grad():
        ego = b["ego"][None].cuda()
        out = model(b["image"][None].cuda(), b["bev_index"].cuda(), b["valid"].cuda(), ego,
                    prev_bev_from_batch(model, b, batched=False), warp_from_batch(b, batched=False))
    from distill.geometry import occ_to_native
    return occ_to_native(out[2][0].float().argmax(-1).cpu().numpy(), getattr(cfg, "occ_grid", "bev"), cfg.pc_range)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], help="LABEL=path, repeatable; one column each")
    ap.add_argument("--tokens", default="", help="comma-separated sample tokens; default: --n frames spread over eval250")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--out", default="outputs/debug/occ_compare.png")
    args = ap.parse_args()
    os.chdir(_ROOT)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import qwen_drive_perception.visualize as V
    from distill.student import StudentConfig, StudentDetector
    from distill.train_student import DistillSet

    if args.tokens:
        toks = args.tokens.split(",")
    else:
        ev = json.load(open("data/distill/scene_split.json"))["eval250"]
        ev = [t for t in ev if (Path(args.frames) / t / "occ3d_native.npz").exists() and (Path(args.teacher) / f"{t}.npz").exists()]
        toks = [ev[int(k)] for k in np.linspace(0, len(ev) - 1, args.n + 2)[1:-1]]

    cols = {}                                  # label -> per-token native grid
    teacher = {}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        allowed = StudentConfig.__init__.__code__.co_varnames
        cfg = StudentConfig(**{k: v for k, v in (ck.get("cfg") or {}).items() if k in allowed})
        model = StudentDetector(cfg).cuda().eval()
        model.load_state_dict(ck.get("ema") or ck["model"], strict=True)
        ds = DistillSet(Path(args.frames), Path(args.teacher), cfg, toks, temporal=bool(getattr(cfg, "temporal", False)),
                        gt_name="gt_boxes10.npz", min_pts=1, history=int(getattr(cfg, "history", 1)))
        cols[label] = []
        for i in range(len(toks)):
            b = ds[i]
            cols[label].append(student_native(model, cfg, b))
            teacher.setdefault(i, b["occ"].numpy().astype(np.int64))       # bev-mode DistillSet: occ = the teacher's
        del model; torch.cuda.empty_cache()
        print(f"  {label}: {path} (occ_grid={getattr(cfg, 'occ_grid', 'bev')}, step {ck.get('step')})", flush=True)

    empty = V.OCC_EMPTY_LABEL
    names = ["Occ3D GT (visible)", "teacher"] + list(cols)
    fig = plt.figure(figsize=(4.2 * len(names), 4.0 * len(toks)))
    for r, tok in enumerate(toks):
        nat = np.load(Path(args.frames) / tok / "occ3d_native.npz"); mask = nat["mask"] > 0
        gt = np.where(mask, nat["occ"].astype(np.int64), empty)
        grids = [gt, teacher[r]] + [cols[k][r] for k in cols]
        win = V._occupancy_window(gt)
        for c, (name, g) in enumerate(zip(names, grids)):
            ax = fig.add_subplot(len(toks), len(names), r * len(names) + c + 1, projection="3d")
            V._panel_occupancy(ax, g, win)
            ax.set_axis_off()
            inside, outside = int(((g != empty) & mask).sum()), int(((g != empty) & ~mask).sum())
            ax.set_title(f"{name}\n{tok[:8]}  in {inside // 1000}k / out {outside // 1000}k", fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"  wrote {args.out}  ({len(toks)} frames x {len(names)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
