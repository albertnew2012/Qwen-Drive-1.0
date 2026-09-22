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

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from local.distill.student import StudentConfig, StudentDetector
from local.distill.geometry import bev_indices


class DistillSet(Dataset):
    def __init__(self, frames: Path, teacher: Path, cfg: StudentConfig, tokens=None,
                 ego: Path | None = None):
        self.frames, self.teacher, self.cfg = frames, teacher, cfg
        self.ego = ego or (frames.parent / "ego")
        avail = sorted(p.stem for p in teacher.glob("*.npz"))
        self.tokens = [t for t in (tokens or avail) if (frames / t).is_dir()]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        from PIL import Image
        tok = self.tokens[i]
        d = np.load(self.teacher / f"{tok}.npz")
        fr = self.frames / tok
        cams = json.loads((fr / "frame.json").read_text())["cam_order"]
        w, h = self.cfg.image_size
        img = Image.open(fr / "images" / f"{cams[0]}.jpg").convert("RGB").resize(
            (w, h), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = (x.permute(2, 0, 1) - 0.5) / 0.5
        index, valid = bev_indices(d["lidar2img"], d["lidar2ego"], self.cfg)
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
            has_traj = np.float32(1.0)
        else:
            dim = self.cfg.hist_points * 3 + self.cfg.hist_points * 4 + 4
            state = np.zeros(dim, dtype=np.float32)
            future = np.zeros((self.cfg.traj_points, 3), dtype=np.float32)
            has_traj = np.float32(0.0)
        return {"image": x,
                "bev_index": torch.from_numpy(index),
                "valid": torch.from_numpy(valid),
                "cls": torch.from_numpy(d["cls"].astype(np.float32)),
                "box": torch.from_numpy(d["box"]),
                "pos": torch.from_numpy(pos),
                "ego": torch.from_numpy(state),
                "future": torch.from_numpy(future),
                "has_traj": torch.tensor(has_traj)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--teacher", default="data/distill/teacher")
    ap.add_argument("--out", default="outputs/distill/student")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--box-weight", type=float, default=2.0)
    ap.add_argument("--traj-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    os.chdir(_ROOT)

    cfg = StudentConfig()
    ds_all = DistillSet(Path(args.frames), Path(args.teacher), cfg)
    n_val = max(1, int(len(ds_all) * args.val_frac))
    val_tokens = ds_all.tokens[:n_val]
    train_tokens = ds_all.tokens[n_val:]
    train = DistillSet(Path(args.frames), Path(args.teacher), cfg, train_tokens)
    val = DistillSet(Path(args.frames), Path(args.teacher), cfg, val_tokens)
    print(f"  {len(train)} train / {len(val)} val frames", flush=True)
    if not len(train):
        print("  nothing cached yet")
        return 1

    dl = DataLoader(train, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)
    vdl = DataLoader(val, batch_size=args.batch, num_workers=2)

    model = StudentDetector(cfg).cuda()
    stats_f = Path("data/distill/box_stats.npz")
    if stats_f.exists():
        st_ = np.load(stats_f)
        model.set_box_stats(st_["mean"], st_["std"])
        box_std = torch.from_numpy(st_["std"]).cuda()
        print(f"  box stats installed (x std {float(st_['std'][0]):.2f} m)", flush=True)
    else:
        box_std = torch.ones(cfg.box_dim).cuda()
        print("  no box stats; box loss will be dominated by position", flush=True)
    print(f"  student {model.num_params()/1e6:.2f} M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "student.pt"
    step = 0
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cuda")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        step = st.get("step", 0)
        print(f"  resumed at step {step}", flush=True)

    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    hist = []
    while step < args.steps:
        for b in dl:
            if step >= args.steps:
                break
            img = b["image"].cuda(non_blocking=True)
            idx = b["bev_index"][0].cuda()
            val_m = b["valid"][0].cuda()
            tc = b["cls"].cuda(); tb = b["box"].cuda(); pos = b["pos"].cuda()
            ego = b["ego"].cuda(); fut = b["future"].cuda(); hw = b["has_traj"].cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pc, pb, pt = model(img, idx, val_m, ego)
                w = 1.0 + (args.pos_weight - 1.0) * pos.unsqueeze(-1)
                l_cls = (F.binary_cross_entropy_with_logits(
                    pc.float(), tc.sigmoid(), reduction="none") * w).mean()
                m = pos.unsqueeze(-1)
                # standardised, or the two position dimensions (std 24.8 and 17.2 m
                # against 0.3-1.9 for the rest) account for 94% of the gradient
                l_box = (((pb.float() - tb) / box_std).abs() * m).sum() \
                    / (m.sum().clamp_min(1.0) * 1.0)
                wt = hw.view(-1, 1, 1)
                l_traj = ((pt.float() - fut).abs() * wt).sum() / \
                    wt.expand_as(pt).sum().clamp_min(1.0)
                loss = l_cls + args.box_weight * l_box + args.traj_weight * l_traj
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % args.log_every == 0:
                print(f"  step {step:6d}/{args.steps}  loss {loss.item():.4f}  "
                      f"cls {l_cls.item():.4f}  box {l_box.item():.4f}  "
                      f"traj {l_traj.item():.4f}  "
                      f"{step/(time.time()-t0):.2f} it/s", flush=True)
                hist.append({"step": step, "loss": float(loss.item()),
                             "cls": float(l_cls.item()), "box": float(l_box.item()),
                             "traj": float(l_traj.item())})
            if step % 500 == 0 or step == args.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "cfg": cfg.to_dict()}, ckpt)
                (out / "history.json").write_text(json.dumps(hist, indent=1))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "cfg": cfg.to_dict()}, ckpt)
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"  trained {step} steps in {(time.time()-t0)/60:.1f} min -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
