"""Distil the teacher's 900-query output into the compact student.

No ground truth and no matching: the student's query i is trained against the teacher's
query i, which is well posed because the teacher's queries are fixed learned embeddings.

Two losses, weighted:

  classification   binary cross-entropy against the teacher's per-class logits, taken as
                   soft targets. Most of the 900 queries are background on any frame, so
                   the positives (the queries the teacher actually asserts) are upweighted
                   or the model learns to predict nothing.
  box              L1 on the ten box parameters, applied only where the teacher asserts
                   something -- the box output of a background query is meaningless and
                   regressing towards it is pure noise.
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def channels_last_4d(model):
    """channels_last for the 4-D (Conv2d) weights only: Module.to(memory_format=channels_last)
    raises on the 5-D Conv3d weights of the 3D occupancy head, which stay contiguous."""
    for t in list(model.parameters()) + list(model.buffers()):
        if t.dim() == 4:
            t.data = t.data.to(memory_format=torch.channels_last)
    return model


from local.distill.student import StudentConfig, StudentDetector
from local.distill.geometry import bev_indices_all
from local.distill.det_loss import (
    gt_to_teacher_encoding, hungarian_detection_loss)
from local.distill.center_head import center_targets, center_loss
from local.distill.temporal import (load_temporal_index, relative_T, warp_grid_from_T,
                                    prev_bev_from_batch)

TEACHER_CLASSES = 7        # the teacher's taxonomy; the student may use the 10 nuScenes ones


class DistillSet(Dataset):
    def __init__(self, frames: Path, teacher: Path, cfg: StudentConfig, tokens=None,
                 ego: Path | None = None, feats: str = "", cam_flip: bool = False,
                 temporal: bool = False, gt_name: str = "gt_boxes.npz", min_pts: int = 0,
                 history: int = 1, occ_gt: str = "teacher"):
        self.feats = feats
        # Camera-flip augmentation: mirror every image AND mirror lidar2img in pixel x,
        # so the scatter indices geometry.py derives from it follow the flipped image.
        # The world (GT boxes, occupancy, map, ego future) is untouched -- only the
        # cameras' view of it is mirrored. Not compatible with feature distillation
        # (the teacher's ViT tap is un-flipped), so it requires feat_weight == 0.
        self.cam_flip = cam_flip
        # Temporal fusion: also load the PREVIOUS cached keyframe's six images and
        # calibration, and the warp grid built from prev-ego -> curr-ego. A frame with no
        # cached predecessor (26.8%) reuses itself with an identity warp and has_prev=0,
        # which keeps every tensor a fixed shape (BEVDet4D does the same for the first
        # frame of a sequence).
        self.temporal = temporal
        self.history = history
        # occ_gt="occ3d": occupancy target from Occ3D ground truth (occ3d.npz, camera-visible
        # voxels only, -100 elsewhere) where the frame has it; the teacher's grid otherwise
        self.occ_gt = occ_gt
        # gt_boxes.npz: the 7 teacher classes; gt_boxes10.npz: the 10 nuScenes detection
        # classes (+ num_pts, so GT without a lidar/radar return can be dropped as the
        # official evaluator does)
        self.gt_name, self.min_pts = gt_name, min_pts
        self.tindex = load_temporal_index() if temporal else None
        self.frames, self.teacher, self.cfg = frames, teacher, cfg
        self.ego = ego or (frames.parent / "ego")
        if tokens is None:
            tokens = sorted(p.stem for p in teacher.glob("*.npz"))
        # One stat per UNIQUE token: a CBGS-resampled list repeats each token ~3x, and
        # 74k NFS stats per rank (x4 ranks) outlasted a 600 s launch window twice (E8).
        ok = {t: (frames / t).is_dir() for t in set(tokens)}
        self.tokens = [t for t in tokens if ok[t]]

    def __len__(self):
        return len(self.tokens)

    def _teacher_or_calib(self, tok):
        """The teacher npz (targets + calibration) or, for a keyframe the teacher was never
        run on (8,550 of 34,149 have images and GT only), the same calibration rebuilt from
        calib.npz with zero targets -- so the student can be scored on every official val
        sample. The teacher's lidar2img is at cfg.image_size (verified: rows 0-1 are the
        1600x900 projection scaled by 896/1600 and 512/900)."""
        f = self.teacher / f"{tok}.npz"
        w, h = self.cfg.image_size
        if f.exists():
            d = np.load(f)
            if (w, h) == (896, 512):
                return d
            # the teacher's lidar2img is at 896x512; rescale rows 0-1 to the student's size
            S = np.diag([w / 896.0, h / 512.0, 1.0, 1.0]).astype(np.float32)
            return {**{k: d[k] for k in d.files}, "lidar2img": S[None] @ d["lidar2img"].astype(np.float32)}
        c = np.load(self.frames / tok / "calib.npz")
        S = np.diag([w / 1600.0, h / 900.0, 1.0, 1.0])
        l2i = []
        for i in range(c["cam_intrinsic"].shape[0]):
            s2l = np.eye(4); s2l[:3, :3] = c["sensor2lidar_rotation"][i]
            s2l[:3, 3] = c["sensor2lidar_translation"][i]
            Kp = np.eye(4); Kp[:3, :3] = c["cam_intrinsic"][i]
            l2i.append(S @ Kp @ np.linalg.inv(s2l))
        n, q = len(l2i), self.cfg.num_queries
        return {"lidar2img": np.asarray(l2i, np.float32),
                "lidar2ego": np.repeat(c["lidar2ego"][None], n, 0).astype(np.float32),
                "cls": np.zeros((q, TEACHER_CLASSES), np.float32),
                "box": np.zeros((q, 10), np.float32), "keep": np.zeros((0,), np.int64),
                "occ": np.full((self.cfg.bev_size, self.cfg.bev_size, self.cfg.occ_pillar_h),
                               self.cfg.occ_num_classes - 1, np.int64),
                "seg": np.zeros(tuple(self.cfg.map_size), np.int64)}

    def __getitem__(self, i):
        from PIL import Image
        tok = self.tokens[i]
        d = self._teacher_or_calib(tok)
        fr = self.frames / tok
        cams = json.loads((fr / "frame.json").read_text())["cam_order"]
        w, h = self.cfg.image_size
        # All six views. cam_order is the order lidar2img is stored in, so the images and
        # the per-camera scatter indices stay aligned.
        views = []
        flip = bool(self.cam_flip and np.random.rand() < 0.5)
        l2i = d["lidar2img"].astype(np.float32).copy()
        if flip:
            M = np.eye(4, dtype=np.float32); M[0, 0] = -1.0; M[0, 2] = float(w)  # u' = W - u
            l2i = M[None] @ l2i
        for c in cams[:self.cfg.n_cams]:
            im = Image.open(fr / "images" / f"{c}.jpg").convert("RGB").resize(
                (w, h), Image.BILINEAR)
            if flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            v = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
            views.append((v.permute(2, 0, 1) - 0.5) / 0.5)
        x = torch.stack(views, 0)                     # (N, 3, H, W)
        index, valid = bev_indices_all(l2i, d["lidar2ego"], self.cfg)
        prev = {}
        if self.temporal:
            # The K previous keyframes, walking the index chain. A missing link repeats the
            # last available frame (its warp included); no history at all repeats the
            # current frame with an identity warp. run_stateful.py pads the same way.
            chain, cur = [], tok
            for _ in range(self.history):
                nxt = self.tindex.get(cur, {}).get("prev")
                if not nxt:
                    break
                chain.append(nxt); cur = nxt
            while len(chain) < self.history:
                chain.append(chain[-1] if chain else tok)
            Ec = np.array(self.tindex[tok]["ego2global"]).reshape(4, 4) if tok in self.tindex else np.eye(4)
            loaded, per = {}, []
            for ptok in chain:
                if ptok not in loaded:
                    dp = self._teacher_or_calib(ptok)
                    pl2i = dp["lidar2img"].astype(np.float32).copy()
                    if flip:                                 # same mirror on every frame
                        pl2i = M[None] @ pl2i
                    pviews = []
                    for c in cams[:self.cfg.n_cams]:
                        im = Image.open(self.frames / ptok / "images" / f"{c}.jpg").convert("RGB").resize(
                            (w, h), Image.BILINEAR)
                        if flip:
                            im = im.transpose(Image.FLIP_LEFT_RIGHT)
                        v = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
                        pviews.append((v.permute(2, 0, 1) - 0.5) / 0.5)
                    pindex, pvalid = bev_indices_all(pl2i, dp["lidar2ego"], self.cfg)
                    T = (np.linalg.inv(Ec) @ np.array(self.tindex[ptok]["ego2global"]).reshape(4, 4)
                         if ptok != tok else np.eye(4))
                    loaded[ptok] = (torch.stack(pviews, 0), torch.from_numpy(pindex.reshape(-1)),
                                    torch.from_numpy(pvalid.reshape(-1)),
                                    torch.from_numpy(warp_grid_from_T(T, self.cfg)),
                                    torch.tensor(np.float32(ptok != tok)))
                per.append(loaded[ptok])
            if self.history == 1:
                pimg, pidx, pval, pgrid, phas = per[0]
            else:
                pimg, pidx, pval, pgrid, phas = (torch.stack([q[i] for q in per], 0) for i in range(5))
            prev = {"prev_image": pimg, "prev_bev_index": pidx, "prev_valid": pval,
                    "warp_grid": pgrid, "has_prev": phas}
        index, valid = index.reshape(-1), valid.reshape(-1)
        pos = np.zeros(self.cfg.num_queries, dtype=np.float32)
        pos[d["keep"]] = 1.0
        # Planning is supervised by the recorded ego future, not by the teacher: for
        # trajectory the ground truth *is* the objective, and the teacher's own ADE
        # against it is 0.335 m. Frames too close to the end of a scene have no future,
        # so they carry a zero weight rather than a fabricated target.
        ego_f = self.ego / f"{tok}.npz"
        if ego_f.exists():
            e = np.load(ego_f)
            state = np.concatenate([
                e["history"].reshape(-1), e["velocity"].reshape(-1),
                e["acceleration"].reshape(-1),
                np.eye(3, dtype=np.float32)[int(e["nav"])],
                np.asarray([float(e["speed"])], dtype=np.float32)]).astype(np.float32)
            future = e["future"].astype(np.float32)
            # scene-end keyframes carry a valid ego state but a clamped future (has_future=0):
            # the state feeds the planner, the trajectory loss is masked
            has_traj = np.float32(float(e["has_future"]) if "has_future" in e.files else 1.0)
        else:
            dim = self.cfg.hist_points * 3 + self.cfg.hist_points * 4 + 4
            state = np.zeros(dim, dtype=np.float32)
            future = np.zeros((self.cfg.traj_points, 3), dtype=np.float32)
            has_traj = np.float32(0.0)
        has_teacher = (self.teacher / f"{tok}.npz").exists()
        occ_t = d["occ"].astype(np.int64); has_occ = has_teacher
        if self.occ_gt == "occ3d" and (fr / "occ3d.npz").exists():
            o3 = np.load(fr / "occ3d.npz")
            occ_t = o3["occ"].astype(np.int64); occ_t[o3["mask"] == 0] = -100; has_occ = True
        gt_f = fr / self.gt_name
        if gt_f.exists():
            g = np.load(gt_f)
            raw, gl = g["boxes"], g["labels"].astype(np.int64)
            if self.min_pts > 0 and "num_pts" in g.files:
                k = g["num_pts"] >= self.min_pts
                raw, gl = raw[k], gl[k]
            gb = gt_to_teacher_encoding(raw); graw = raw.astype(np.float32)
        else:
            gb = np.zeros((0, 10), np.float32); gl = np.zeros((0,), np.int64)
            graw = np.zeros((0, 9), np.float32)
        w_, h_ = self.cfg.image_size
        df = fr / ("depth.npz" if (w_, h_) == (896, 512) else f"depth_{w_}x{h_}.npz")
        depth = (np.load(df)["depth_bin"].astype(np.int64) if df.exists()
                 else np.full((self.cfg.n_cams, self.cfg.image_size[1] // 16,
                               self.cfg.image_size[0] // 16), -1, np.int64))
        if flip:
            # The depth target lives in image space: (N, H/16, W/16), so it mirrors with
            # the image. Without this, half of every --cam-flip batch trained the depth
            # head on the wrong side of the picture (E7's depth loss started at 3.65 vs
            # 1.28 for E6 from the same weights).
            depth = np.ascontiguousarray(depth[:, :, ::-1])
        vf = Path(self.feats) / f"{tok}.npy" if self.feats else None
        if vf is not None and vf.exists():
            tvit = torch.from_numpy(np.load(vf).astype(np.float32)); has_vit = 1.0
        else:
            tvit = torch.zeros(self.cfg.n_cams, self.cfg.image_size[1] // 16,
                               self.cfg.image_size[0] // 16, self.cfg.teacher_dim)
            has_vit = 0.0
        return {**prev, "image": x,
                "tvit": tvit, "has_vit": torch.tensor(np.float32(has_vit)),
                "depth_bin": torch.from_numpy(depth),
                "gt_box": torch.from_numpy(gb), "gt_label": torch.from_numpy(gl),
                "gt_raw": torch.from_numpy(graw),
                "occ": torch.from_numpy(occ_t),
                "seg": torch.from_numpy(d["seg"].astype(np.int64)),
                "bev_index": torch.from_numpy(index),
                "valid": torch.from_numpy(valid),
                "cls": torch.from_numpy(d["cls"].astype(np.float32)),
                "box": torch.from_numpy(d["box"]),
                "pos": torch.from_numpy(pos),
                "ego": torch.from_numpy(state),
                "future": torch.from_numpy(future),
                "has_traj": torch.tensor(has_traj),
                # frames the teacher was never run on have no map target (and no occupancy
                # target unless Occ3D labels are used); their losses are masked out
                "has_seg": torch.tensor(np.float32(has_teacher)),
                "has_occ": torch.tensor(np.float32(has_occ))}


def cbgs_resample(frames: Path, tokens, num_classes: int, seed: int = 0, log=print,
                  gt_name: str = "gt_boxes.npz"):
    """mmdet3d's class-balanced grouping and sampling, at the frame level.

    Measured over 3,000 frames: vehicle in 99.5% of frames, pedestrian 80%, cone 45%,
    bicycle 41%, barrier 34%, generic 24%. Uniform sampling therefore shows the model a
    barrier one frame in three. For each class present in the set, frames containing
    it are re-drawn with ratio (1/C_eff) / (its frame fraction), so every class ends up
    with about an equal share of the (larger) epoch. Classes absent from the data
    (czone_sign here) are excluded from C_eff rather than dividing by zero.
    """
    rng = np.random.RandomState(seed)
    per_tok = {}
    # 23k small npz reads on a cold NFS cache took longer than a launch self-check window
    # (E8, 2026-09-25); data/distill/gt_labels_index.json holds every frame's label set
    # for both GT files, so the resampler needs one read.
    index_f = Path("data/distill/gt_labels_index.json")
    cached = json.load(open(index_f)).get(gt_name, {}) if index_f.exists() else {}
    for t in tokens:
        if t in cached:
            per_tok[t] = set(cached[t]); continue
        f = frames / t / gt_name
        per_tok[t] = set(np.load(f)["labels"].tolist()) if f.exists() else set()
    cls_frames = {c: [t for t in tokens if c in per_tok[t]] for c in range(num_classes)}
    present = [c for c in cls_frames if len(cls_frames[c]) > 0]
    total = sum(len(cls_frames[c]) for c in present)
    frac = 1.0 / len(present)
    out = []
    for c in present:
        ratio = frac / (len(cls_frames[c]) / total)
        n = int(round(len(cls_frames[c]) * ratio))
        out.extend(rng.choice(cls_frames[c], n, replace=True).tolist())
    rng.shuffle(out)
    before = {c: len(cls_frames[c]) / max(len(tokens), 1) for c in present}
    after = {c: sum(1 for t in out if c in per_tok[t]) / max(len(out), 1) for c in present}
    log(f"  CBGS: {len(tokens)} -> {len(out)} frames/epoch ({len(out)/max(len(tokens),1):.2f}x); "
        "frame share per class before -> after: " +
        " ".join(f"{c}:{before[c]:.0%}->{after[c]:.0%}" for c in present), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--out", default="outputs/distill/student")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pos-weight", type=float, default=5.0,
                    help="20.0 made the student assert 157 boxes/frame "
                         "against the teacher's 21.5, capping precision "
                         "at 27.7% at ANY score threshold")
    ap.add_argument("--box-weight", type=float, default=2.0)
    ap.add_argument("--traj-weight", type=float, default=5.0,
                    help="at 1.0 trajectory was ~8% of the total loss "
                         "against box's ~6.0, too little to compete")
    ap.add_argument("--occ-weight", type=float, default=1.0)
    ap.add_argument("--hm-class-weight", default="", help="per-class heatmap loss weights, e.g. '5:2,6:2,7:2,8:2' (nuScenes10 indices: pedestrian, motorcycle, bicycle, traffic cone)")
    ap.add_argument("--head-dilate", action="store_true", help="2x-grid center head with dilation-2 3x3 blocks (exact warm start from a 1x head)")
    ap.add_argument("--occ-head", default="conv1x1", choices=["conv1x1", "conv3d"], help="occupancy head: original 1x1 conv, or the 3D conv head")
    ap.add_argument("--occ-3d-channels", type=int, default=16)
    ap.add_argument("--train-only", default="", help="comma-separated parameter-name prefixes to train; everything else frozen (e.g. 'occ_head')")
    ap.add_argument("--occ-outside-empty", type=float, default=0.0,
                    help="weight of a CE term pulling voxels OUTSIDE the Occ3D camera-visible mask toward 'empty' (0 = off, the r2/r3 recipe)")
    ap.add_argument("--seg-weight", type=float, default=1.0)
    ap.add_argument("--depth-weight", type=float, default=1.0,
                    help="lidar depth supervision on the lift-splat distribution. This "
                         "was absent entirely: the depth was learned only through the "
                         "detection gradient, so features scattered into wrong BEV "
                         "cells. ~60% of feature cells carry a lidar return.")
    ap.add_argument("--ref-points", default="on", choices=["on","off"])
    ap.add_argument("--det-head", default="query", choices=["query","center"])
    ap.add_argument("--center-hidden", type=int, default=64)
    ap.add_argument("--center-blocks", type=int, default=2)
    ap.add_argument("--center-min-radius", type=int, default=1)
    ap.add_argument("--snap-every", type=int, default=10000, help="copy the checkpoint to snap_<step>.pt every N steps (0 = off)")
    ap.add_argument("--head-upsample", type=int, default=1,
                    help="detection head grid factor: 2 = 400x400 / 0.256 m heatmap via a learned 2x deconv")
    ap.add_argument("--cbgs", action="store_true", help="class-balanced frame resampling")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay, e.g. 0.999; 0 = off")
    ap.add_argument("--cam-flip", action="store_true", help="horizontal camera-flip aug")
    ap.add_argument("--temporal", action="store_true", help="fuse the previous keyframe BEV")
    ap.add_argument("--velocity", action="store_true",
                    help="regress ego-frame velocity in the center head (gt_boxes10.npz only)")
    ap.add_argument("--occ-gt", default="teacher", choices=["teacher", "occ3d"],
                    help="occupancy target: the teacher's grid, or Occ3D ground truth where a frame has occ3d.npz")
    ap.add_argument("--history", type=int, default=1,
                    help="previous keyframes fused by --temporal (1 = BEVDet4D, 3 = 1.5 s of history)")
    ap.add_argument("--arch", default="resnet50", help="torchvision backbone name (resnet50/101, ...)")
    ap.add_argument("--image-size", type=int, nargs=2, default=[896, 512], metavar=("W", "H"),
                    help="student input size; needs nusc_depth.py --image-size W H --out-name "
                         "depth_WxH.npz for the depth targets (lidar2img is rescaled in the loader)")
    ap.add_argument("--feat-weight", type=float, default=0.0,
                    help="feature-level distillation against the teacher's ViT tap. The "
                         "measured gap is representational -- the student reaches only "
                         "74% of the teacher even on large objects at 0-15 m -- and 900 "
                         "output logits are a thin channel through which to move a 4 B "
                         "encoder. 0 disables and the adapter is not even built.")
    ap.add_argument("--feats", default="data/distill/teacher_vit")
    ap.add_argument("--pool", type=int, default=40,
                    help="BEV pooling for the detection decoder. 40 gives 2.56 m per "
                         "token -- a pedestrian is 0.5 m and disappears inside one. 100 "
                         "gives 1.02 m and still measures 13.5 Hz against a 10 Hz bar. "
                         "The occupancy and map heads always read the full 200x200, "
                         "which is why they score far better than detection does.")
    ap.add_argument("--distill-box-weight", type=float, default=1.0,
                    help="hybrid only: weight on the teacher per-query box anchor")
    ap.add_argument("--det-objective", default="distill",
                    choices=["distill", "hungarian", "hybrid", "center"],
                    help="distill = per-query regression against the teacher (F1 34.5%); "
                         "hungarian = one-to-one GT matching with focal loss (F1 2.1%, "
                         "unmatched queries' boxes drift); hybrid = hungarian class + "
                         "teacher box anchor on every query")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--classes", default="teacher7", choices=["teacher7", "nuscenes10"],
                    help="teacher7: the teacher's 7 classes (gt_boxes.npz); nuscenes10: the "
                         "10 official detection classes (gt_boxes10.npz, GT with no lidar/"
                         "radar return dropped) so official mAP/NDS apply")
    ap.add_argument("--split", default="frame", choices=["frame", "scene", "scene-all"],
                    help="frame: first val-frac of sorted tokens held out (every scene is "
                         "trained on); scene: official scene split, val scenes never seen")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    os.chdir(_ROOT)

    # Distributed data parallel, opt-in via torchrun. Caching occupies GPUs 1-3 while
    # GPU 0 trains, but once every frame is cached those three cards sit idle for the
    # rest of the run -- which is most of it. Measured: 9.2 samples/s on one card, and
    # the step is compute bound (batch 4 -> 8 buys 4%), so the only way to more epochs
    # is more cards.
    #   torchrun --nproc_per_node=4 local/distill/train_student.py ...
    # Without torchrun the environment variables are absent and this is a no-op, so the
    # single-GPU path behaves exactly as before.
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    is_main = (rank == 0)

    def say(*a, **k):
        if is_main:
            print(*a, **k)

    gt_name = "gt_boxes10.npz" if args.classes == "nuscenes10" else "gt_boxes.npz"
    min_pts = 1 if args.classes == "nuscenes10" else 0
    cfg = StudentConfig(num_classes=(10 if args.classes == "nuscenes10" else TEACHER_CLASSES),
                        ref_points=(args.ref_points == "on"), pool=args.pool,
                        feat_distill=args.feat_weight > 0, det_head=args.det_head,
                        center_hidden=args.center_hidden, center_blocks=args.center_blocks,
                        center_min_radius=args.center_min_radius, head_upsample=args.head_upsample, temporal=args.temporal,
                        head_dilate=args.head_dilate, occ_head=args.occ_head, occ_3d_channels=args.occ_3d_channels,
                        velocity=args.velocity, image_size=tuple(args.image_size), arch=args.arch,
                        history=args.history)
    ds_all = DistillSet(Path(args.frames), Path(args.teacher), cfg, feats=args.feats)
    if os.environ.get("DISTILL_SUBSET"):   # tiny-subset smoke tests: force fast epochs
        ds_all.tokens = ds_all.tokens[:int(os.environ["DISTILL_SUBSET"])]
    if args.split in ("scene", "scene-all"):
        # Official nuScenes scene split. The frame split below holds out frames, not
        # scenes: 324 of the 250 eval frames' 500 neighbouring keyframes (0.5 s away) are
        # in the training set, which rewards memorising scenes and flatters the student
        # against a teacher that never saw nuScenes. The deliverable trains on the 700
        # train scenes only and is scored on the 150 val scenes it has never seen.
        ss = json.load(open("data/distill/scene_split.json"))
        have = set(ds_all.tokens)
        train_tokens = [t for t in ss["train_tokens"] if t in have]
        if args.split == "scene-all":
            # + the 6,983 train-scene keyframes the teacher was never run on: images, GT
            # boxes, Occ3D occupancy and (now) depth and ego futures exist for them; the map
            # loss is masked (no teacher target), detection is supervised by GT as always
            extra = [t for t in ss["uncached_train_tokens"] if (Path(args.frames) / t).is_dir()]
            train_tokens = train_tokens + extra
        val_tokens = [t for t in ss["eval250"] if t in have]
        say(f"  scene split: {len(train_tokens)} train frames (700 scenes), "
            f"{len(val_tokens)} val frames from the 150 held-out scenes", flush=True)
    else:
        n_val = max(1, int(len(ds_all) * args.val_frac))
        val_tokens = ds_all.tokens[:n_val]
        train_tokens = ds_all.tokens[n_val:]
    if args.cbgs:
        train_tokens = cbgs_resample(Path(args.frames), train_tokens, cfg.num_classes,
                                     seed=0, log=say, gt_name=gt_name)
    train = DistillSet(Path(args.frames), Path(args.teacher), cfg, train_tokens, feats=args.feats, cam_flip=args.cam_flip, temporal=args.temporal, gt_name=gt_name, min_pts=min_pts, history=args.history, occ_gt=args.occ_gt)
    val = DistillSet(Path(args.frames), Path(args.teacher), cfg, val_tokens, feats=args.feats,
                     gt_name=gt_name, min_pts=min_pts)
    say(f"  {len(train)} train / {len(val)} val frames", flush=True)
    if not len(train):
        say("  nothing cached yet")
        return 1

    sampler = None
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train, shuffle=True, drop_last=True)
    def collate(batch):
        # gt_box / gt_label are ragged (a frame carries 0-60 objects), so they travel as
        # lists while everything else stacks normally.
        # Anything ragged travels as a list. gt_* are ragged by nature (0-60 objects per
        # frame); the teacher feature tap is ragged too when frames were cached at
        # different widths, so shape is tested rather than assumed.
        RAGGED = ("gt_box", "gt_label", "gt_raw")
        out = {}
        for k in batch[0]:
            if k in RAGGED:
                continue
            vals = [b[k] for b in batch]
            if all(v.shape == vals[0].shape for v in vals):
                out[k] = torch.stack(vals)
            else:
                out[k] = vals
        out["gt_box"] = [b["gt_box"] for b in batch]
        out["gt_label"] = [b["gt_label"] for b in batch]
        out["gt_raw"] = [b["gt_raw"] for b in batch]
        return out

    dl = DataLoader(train, batch_size=args.batch, shuffle=(sampler is None),
                    sampler=sampler, num_workers=args.workers, drop_last=True,
                    pin_memory=True, collate_fn=collate)
    vdl = DataLoader(val, batch_size=args.batch, num_workers=2, collate_fn=collate)

    model = channels_last_4d(StudentDetector(cfg).cuda())
    stats_f = Path("data/distill/box_stats.npz")
    if stats_f.exists():
        st_ = np.load(stats_f)
        model.set_box_stats(st_["mean"], st_["std"])
        box_std = torch.from_numpy(st_["std"]).cuda()
        say(f"  box stats installed (x std {float(st_['std'][0]):.2f} m)", flush=True)
    else:
        box_std = torch.ones(cfg.box_dim).cuda()
        say("  no box stats; box loss will be dominated by position", flush=True)
    say(f"  student {model.num_params()/1e6:.2f} M params", flush=True)

    # Occupancy is 95.6% free space (measured over 300 cached frames), so an unweighted
    # cross-entropy has a trivial minimum: predict free everywhere for 95.6% accuracy
    # and 0.10 mIoU. Inverse-sqrt-frequency weights make the rare classes worth
    # learning. The map grid is far better balanced (dominant class 41.9%) and is left
    # unweighted.
    if args.hm_class_weight:
        import local.distill.center_head as _CH
        _w = torch.ones(cfg.num_classes)
        for kv in args.hm_class_weight.split(","):
            k, v = kv.split(":"); _w[int(k)] = float(v)
        _CH.CLASS_WEIGHT = _w
        say(f"  heatmap class weights: {_w.tolist()}")
    occ_w = torch.ones(cfg.occ_num_classes).cuda()
    # With Occ3D targets the supervised voxels are ~77% empty (vs 95.7% on the teacher's
    # grid), so the weights come from the label distribution instead (occ3d_labels.py stats).
    freq_f = Path("data/distill/occ3d_freq.npy" if args.occ_gt == "occ3d" else "data/distill/occ_freq.npy")
    counts = np.zeros(cfg.occ_num_classes, np.int64)
    if freq_f.exists():
        counts = np.load(freq_f)
    else:
        for t in train.tokens[:300]:
            counts += np.bincount(np.load(train.teacher / f"{t}.npz")["occ"].reshape(-1),
                                  minlength=cfg.occ_num_classes)
        if counts.sum():
            freq_f.parent.mkdir(parents=True, exist_ok=True)
            np.save(freq_f, counts)
    if counts.sum():
        frac = counts / counts.sum()
        # Exponent set by measuring what the model then PREDICTS, not by loss mass.
        # 0.75 balanced the loss nicely (free 43%) but over-corrected the model: it
        # predicted occupied classes 3-40x more often than the teacher (class 6:
        # 0.020% teacher vs 0.761% student), which inflates the IoU union and froze
        # occupancy mIoU at 3.5% across two rounds. 0.45 backs that off.
        # The 1e-4 floor stops a class with ~zero voxels becoming a 1000x outlier that
        # distorts the mean, and the 4.0 cap bounds what such a class can still claim.
        w = 1.0 / np.clip(frac, 1e-4, None) ** 0.45
        w = np.minimum(w / w.mean(), 4.0)
        occ_w = torch.from_numpy(w.astype(np.float32)).cuda()
        say(f"  occupancy class weights (free class {int(frac.argmax())} at "
              f"{frac.max():.1%}): " + " ".join(f"{v:.2f}" for v in w), flush=True)
    # The map grid is better balanced than occupancy (dominant class 46%, not 96%) and
    # was left unweighted -- but it then under-predicted its rare classes by 3-5x
    # (class 2: 4.13% teacher vs 0.74% student). A gentle 0.3 exponent lifts them
    # without the over-correction that 0.75 caused on occupancy.
    seg_w = torch.ones(cfg.map_num_classes).cuda()
    sfreq = np.zeros(cfg.map_num_classes, np.int64)
    n_seen = 0
    for t in train.tokens:                     # first 300 frames that have a teacher map target
        if not (train.teacher / f"{t}.npz").exists():
            continue
        sfreq += np.bincount(np.load(train.teacher / f"{t}.npz")["seg"].reshape(-1),
                             minlength=cfg.map_num_classes)
        n_seen += 1
        if n_seen >= 300:
            break
    if sfreq.sum():
        sf = sfreq / sfreq.sum()
        sw = 1.0 / np.clip(sf, 1e-4, None) ** 0.3
        sw = np.minimum(sw / sw.mean(), 4.0)
        seg_w = torch.from_numpy(sw.astype(np.float32)).cuda()
        print("  map class weights: " + " ".join(f"{v:.2f}" for v in sw), flush=True)

    # The trajectory target has the same pathology box_stats.py was written to fix, and
    # it was never applied here. Measured over 4,000 ego futures the mean |value| per
    # dimension is x 12.86, y 0.79, heading 0.09 -- so an unweighted L1 spends 94% of its
    # gradient on longitudinal distance and 6% on y, the dimension that encodes turning.
    # Measured consequence at 34k steps: the student's endpoint spread was x 17.06 m
    # against ground truth's 17.32 (longitudinal solved) but y 0.22 m against 4.49 --
    # it had learned to drive straight at the right speed and never turn, which is why
    # ADE sat at 1.988 m across two rounds while every other metric moved.
    # Dividing the loss by the per-dimension scale equalises them. The model still emits
    # metres, so the ONNX signature and every consumer are unchanged.
    traj_f = Path("data/distill/traj_scale.npy")
    if traj_f.exists():
        traj_scale = torch.from_numpy(np.load(traj_f)).cuda()
    else:
        fs = sorted((_ROOT / "data/distill/ego").glob("*.npz"))[:4000]
        if fs:
            # not `F` -- that is torch.nn.functional at module scope
            futs = np.stack([np.load(f)["future"] for f in fs])
            sc = np.maximum(np.abs(futs).mean((0, 1)), 1e-2).astype(np.float32)
        else:
            sc = np.ones(3, dtype=np.float32)
        np.save(traj_f, sc)
        traj_scale = torch.from_numpy(sc).cuda()
    say("  trajectory loss scale (x, y, heading): "
        + " ".join(f"{v:.3f}" for v in traj_scale.tolist()), flush=True)

    if args.train_only:
        # freeze everything outside the named heads: a fresh head can then train at a high LR
        # without moving the backbone, and every other output stays bit-identical
        prefixes = tuple(x.strip() for x in args.train_only.split(",") if x.strip())
        n_train = 0
        for name, prm in model.named_parameters():
            prm.requires_grad = name.startswith(prefixes)
            n_train += prm.numel() if prm.requires_grad else 0
        say(f"  train-only {prefixes}: {n_train/1e6:.2f} M trainable parameters, the rest frozen", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    ema = None
    if args.ema > 0:
        # Averaged copy of the raw module (never the DDP wrapper). use_buffers=True so
        # BatchNorm running statistics are averaged as well, which is what makes the
        # EMA weights usable for evaluation on their own.
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema),
                            use_buffers=True)
        print(f"  EMA weights on, decay {args.ema}", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.1)
    # Constructed before the resume below so `last_epoch` can be fast-forwarded: a fresh
    # OneCycleLR restarts its warm-up from zero every time training resumes, so each
    # escalating round (and each manual restart) re-warmed the LR mid-convergence.
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "student.pt"
    step = 0
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cuda")
        # strict=False so a checkpoint predating a newly added head still loads; the new
        # parameters keep their initialisation (ego_direct is zero-init, so the model
        # resumes behaving exactly as the checkpoint did). Anything missing is printed
        # rather than swallowed -- a silent shape drift here would be invisible for hours.
        # strict=False tolerates missing/extra keys but still RAISES on a shape
        # mismatch, so changing --pool (which resizes `pos` from pool^2 to pool'^2)
        # would kill the warm start. Drop the mismatched tensors and keep the rest:
        # the backbone and BEV stack are the expensive part and transfer unchanged.
        cur = model.state_dict()
        sd = dict(st["model"])
        resized = [k for k, v in sd.items()
                   if k in cur and tuple(cur[k].shape) != tuple(v.shape)]
        for k in resized:
            sd.pop(k)
        if resized:
            print(f"  shape changed, reinitialising: {resized}", flush=True)
        miss, unexp = model.load_state_dict(sd, strict=False)
        if miss or unexp:
            print(f"  checkpoint is older than the model: missing {list(miss)}, "
                  f"unexpected {list(unexp)} -- keeping their fresh init", flush=True)
        try:
            if resized:
                # Adam's exp_avg/exp_avg_sq for a resized parameter still carry the OLD
                # shape, and load_state_dict accepts them -- the mismatch only surfaces
                # at the first optimizer.step(), which is minutes later and looks like a
                # training bug rather than a resume bug. Start the optimiser fresh.
                raise ValueError(f"parameter shapes changed: {resized}")
            opt.load_state_dict(st["opt"])
        except (ValueError, KeyError) as e:   # KeyError: warm-start ckpt with opt dropped
            print(f"  optimiser state does not match the new parameter set ({e}); "
                  f"starting the optimiser fresh", flush=True)
        step = st.get("step", 0)
        say(f"  resumed at step {step}", flush=True)

    # Wrapped after the resume so the checkpoint keeps unprefixed keys and stays loadable
    # by eval/export, which are single-process.
    net = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=False)
        say(f"  DDP over {os.environ.get('WORLD_SIZE')} GPUs, "
            f"effective batch {args.batch * int(os.environ['WORLD_SIZE'])}", flush=True)

    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    hist = []
    epoch = 0
    bad_batches = 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)      # or every rank sees the same order every epoch
        epoch += 1
        for b in dl:
            if step >= args.steps:
                break
            # Guard: under 4-rank DDP the loader has yielded a bare tensor instead of the
            # collated dict exactly at an epoch boundary (98,003 + 1,440 = 99,443), which
            # killed a run 20 minutes in. Not reproducible single-process, so rather than
            # lose hours to it, record what arrived and skip the batch.
            if not isinstance(b, dict):
                bad_batches += 1
                if bad_batches <= 3:
                    print(f"  !! batch {bad_batches} at step {step} is "
                          f"{type(b).__name__} shape "
                          f"{tuple(b.shape) if hasattr(b, 'shape') else '?'} "
                          f"-- skipping", flush=True)
                if bad_batches > 50:
                    raise RuntimeError("loader keeps yielding non-dict batches")
                continue
            img = b["image"].cuda(non_blocking=True)
            idx = b["bev_index"].cuda()          # per-sample, see LiftSplat
            val_m = b["valid"].cuda()
            tc = b["cls"].cuda(); tb = b["box"].cuda(); pos = b["pos"].cuda()
            ego = b["ego"].cuda(); fut = b["future"].cuda(); hw = b["has_traj"].cuda()
            tocc = b["occ"].cuda(); tseg = b["seg"].cuda()
            tdep = b["depth_bin"].cuda(non_blocking=True)
            if args.temporal:
                # Previous-frame BEV under no_grad (BEVDet4D detaches history) in its OWN
                # autocast context, closed before the main one opens. autocast caches the
                # bf16 copy of every fp32 weight it casts; a copy first made under no_grad
                # has no autograd edge, so reusing it in the grad-mode forward silently
                # cut every conv weight in the backbone and lift-splat off the gradient
                # (only DDP's reducer noticed, in the 2-GPU smoke test; a single process
                # "trained" with a frozen backbone and no error).
                core = model.module if hasattr(model, "module") else model
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    prev_bev = prev_bev_from_batch(core, b, batched=True).float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if args.temporal:
                    pc, pb, pocc, pseg, pt, _ = net(img, idx, val_m, ego,
                                                    prev_bev, b["warp_grid"].cuda())
                else:
                    pc, pb, pocc, pseg, pt = net(img, idx, val_m, ego)
                if args.det_objective == "center":
                    # Dense heatmap supervision: every cell is supervised every step, so
                    # there is no assignment to be unstable and no unconstrained box.
                    core = model.module if hasattr(model, "module") else model
                    hm_t, rg_t, msk = [], [], []
                    for gr, gl_ in zip(b["gt_raw"], b["gt_label"]):
                        a, c, d = center_targets(gr.numpy(), gl_.numpy(), cfg, "cuda")
                        hm_t.append(a); rg_t.append(c); msk.append(d)
                    l_cls, l_box = center_loss(
                        core._hm.float(), core._reg.float(),
                        torch.stack(hm_t), torch.stack(rg_t), torch.stack(msk))
                if args.det_objective in ("hungarian", "hybrid"):
                    l_cls, l_box_h, n_m = hungarian_detection_loss(
                        pc.float(), pb.float(),
                        [t.cuda(non_blocking=True) for t in b["gt_box"]],
                        [t.cuda(non_blocking=True) for t in b["gt_label"]],
                        box_std)
                w = 1.0 + (args.pos_weight - 1.0) * pos.unsqueeze(-1)
                if pc.shape[-1] != tc.shape[-1]:
                    # 10-class student against the 7-class teacher: the distillation term
                    # has no meaning (and is unused under the center objective anyway)
                    l_cls_d = pc.new_zeros(())
                else:
                    l_cls_d = (F.binary_cross_entropy_with_logits(
                        pc.float(), tc.sigmoid(), reduction="none") * w).mean()
                m = pos.unsqueeze(-1)
                # standardised, or the two position dimensions (std 24.8 and 17.2 m
                # against 0.3-1.9 for the rest) account for 94% of the gradient
                l_box_d = (((pb.float() - tb) / box_std).abs() * m).sum() \
                    / (m.sum().clamp_min(1.0) * 1.0)
                if args.det_objective == "center":
                    pass
                elif args.det_objective == "hungarian":
                    l_box = l_box_h
                elif args.det_objective == "hybrid":
                    # Hungarian supervises boxes on MATCHED pairs only -- ~34 of 900
                    # queries. The other 866 get background pressure on their class but
                    # nothing at all on their box, so those boxes drift. Measured in E3:
                    # 37.6 detections/frame of which 96.3% landed nowhere near a real
                    # object, i.e. the model fired on queries whose boxes had wandered.
                    # Keeping the teacher's per-query box target on every query anchors
                    # them, which is exactly what the pure-distillation objective did
                    # right. Hungarian then only has to fix WHICH queries fire.
                    l_box = l_box_h + args.distill_box_weight * l_box_d
                else:
                    l_cls, l_box = l_cls_d, l_box_d
                wt = hw.view(-1, 1, 1)
                l_traj = (((pt.float() - fut).abs() / traj_scale) * wt).sum() / \
                    wt.expand_as(pt).sum().clamp_min(1.0)
                # Occupancy and the map are plain cross-entropy against the teacher's
                # argmax. (B,200,200,16,C) -> (B,C,...) is what cross_entropy expects.
                # per-sample weighted cross-entropy (same value as reduction="mean" when every
                # sample is real), masked to the samples that carry a real target
                def masked_wce(logits, target, weight, has, ignore=None):
                    kw = {"ignore_index": ignore} if ignore is not None else {}
                    ls = F.cross_entropy(logits, target, weight=weight, reduction="none", **kw)
                    valid = (target != ignore) if ignore is not None else torch.ones_like(target, dtype=torch.bool)
                    den = (weight[target.clamp_min(0)] * valid).flatten(1).sum(1)
                    num = ls.flatten(1).sum(1)
                    # global weighted mean over the real samples == reduction="mean" when all are real
                    return (num * has).sum() / (den * has).sum().clamp_min(1e-6)
                l_occ = masked_wce(pocc.float().permute(0, 4, 1, 2, 3), tocc, occ_w, b["has_occ"].cuda(), ignore=-100)
                if args.occ_outside_empty > 0:
                    # Occ3D supervises only camera-visible voxels (-100 elsewhere), so the head was
                    # free to fill the other 88% of the grid with "driveable"/"background" -- a
                    # blanket in every render. A light pull toward "empty" (class 9) out there
                    # leaves the benchmark region untouched (it is masked in the metric) and
                    # makes the exported occupancy show only what the cameras can see.
                    outside = (tocc == -100).float()
                    ls_out = F.cross_entropy(pocc.float().permute(0, 4, 1, 2, 3), torch.full_like(tocc, cfg.occ_num_classes - 1), reduction="none")
                    per_sample = (ls_out * outside).flatten(1).sum(1) / outside.flatten(1).sum(1).clamp_min(1.0)
                    has = b["has_occ"].cuda()
                    l_occ = l_occ + args.occ_outside_empty * (per_sample * has).sum() / has.sum().clamp_min(1.0)
                l_seg = masked_wce(pseg.float(), tseg, seg_w, b["has_seg"].cuda())
                # Depth: the lift-splat distribution against the lidar bin, on the ~60%
                # of feature cells that have a return. ignore_index=-1 covers the rest.
                # net may be DDP-wrapped, so reach the module for the stashed logits.
                # NOT `dl` -- that is the DataLoader. Shadowing it made the loop
                # iterate the depth-logits tensor at the next epoch boundary and yield
                # (64, 32, 56) slices instead of batches, which killed two runs.
                core = net.module if hasattr(net, "module") else net
                dlog = core.lss.last_depth_logits
                l_depth = F.cross_entropy(
                    dlog.float(), tdep.reshape(-1, *tdep.shape[-2:]), ignore_index=-1)
                # Under hungarian the box term is already weighted inside the loss and
                # the teacher no longer supervises detection at all -- occupancy, map and
                # trajectory still distil, because no ground truth exists for them here.
                det = (l_cls + l_box) if args.det_objective in ("hungarian", "hybrid", "center") \
                    else (l_cls + args.box_weight * l_box)
                # Feature distillation: the student's stride-16 backbone map against
                # the teacher's ViT tap on the identical (32, 56) grid. Cosine rather
                # than MSE -- the two encoders have no reason to share a scale, and only
                # the direction of the representation transfers. Frames without a cached
                # tap contribute zero through the mask, which still keeps the adapter in
                # the autograd graph so DDP does not complain about unused parameters.
                if core.feat_adapter is not None:
                    sf = core.feat_adapter(core.last_backbone_feat.float())
                    tf = b["tvit"].cuda(non_blocking=True).permute(0, 1, 4, 2, 3)
                    tf = tf.reshape(-1, *tf.shape[2:])
                    hv = b["has_vit"].cuda().repeat_interleave(cfg.n_cams)
                    cos = F.cosine_similarity(sf, tf, dim=1)          # (B*N, 32, 56)
                    l_feat = ((1.0 - cos).mean((1, 2)) * hv).sum() / hv.sum().clamp_min(1.0)
                else:
                    l_feat = torch.zeros((), device=pc.device)
                loss = (det + args.traj_weight * l_traj
                        + args.occ_weight * l_occ + args.seg_weight * l_seg
                        + args.depth_weight * l_depth + args.feat_weight * l_feat)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:
                ema.update_parameters(model)
            step += 1
            if step % args.log_every == 0 and is_main:
                print(f"  step {step:6d}/{args.steps}  loss {loss.item():.4f}  "
                      f"cls {l_cls.item():.4f}  box {l_box.item():.4f}  "
                      f"traj {l_traj.item():.4f}  occ {l_occ.item():.4f}  "
                      f"seg {l_seg.item():.4f}  depth {l_depth.item():.4f}  "
                      f"feat {l_feat.item():.4f}  "
                      f"{step/(time.time()-t0):.2f} it/s", flush=True)
                hist.append({"step": step, "loss": float(loss.item()),
                             "cls": float(l_cls.item()), "box": float(l_box.item()),
                             "traj": float(l_traj.item()),
                             "occ": float(l_occ.item()), "seg": float(l_seg.item()),
                             "depth": float(l_depth.item())})
            if (step % 500 == 0 or step == args.steps) and is_main:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "ema": (ema.module.state_dict() if ema is not None else None),
                            "step": step, "cfg": cfg.to_dict()}, ckpt)
                (out / "history.json").write_text(json.dumps(hist, indent=1))
                # Late-training snapshots for checkpoint averaging (average_checkpoints.py):
                # every 5k steps over the last quarter of the schedule.
                if step % 5000 == 0 and step >= int(0.75 * args.steps):
                    shutil.copy2(ckpt, out / f"student_{step}.pt")
                # Periodic snapshots for per-step evaluation in the lane post (tags NAME-mid<step>):
                # r2_long peaked at 80k of 120k (61.0%) while its end (59.8%) and its late average
                # (60.1%) were lower, so the peak must be looked for, not assumed at the end.
                if args.snap_every and step % args.snap_every == 0 and not (out / f"snap_{step}.pt").exists():
                    shutil.copy2(ckpt, out / f"snap_{step}.pt")
    if is_main:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "ema": (ema.module.state_dict() if ema is not None else None),
                    "step": step, "cfg": cfg.to_dict()}, ckpt)
        (out / "history.json").write_text(json.dumps(hist, indent=1))
    if ddp:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()
    say(f"  trained {step} steps in {(time.time()-t0)/60:.1f} min -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
