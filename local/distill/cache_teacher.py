"""Record what the teacher sees and what it predicts, one file per frame.

10 Hz is 100 ms a frame. The current ONNX export is 907 ms, and the parts resist
pruning: the BEV head is 438 ms spread over 2,008 small kernels, and the planning
decoder cannot lose layers because the expert cross-attends to every full-attention
cache. Nothing incremental closes a 9x gap, so the student has to be a different,
compact model -- and a compact model needs a supervision signal.

The teacher provides it. For each frame this stores:

    front image        what a compact student would actually consume
    img_metas          the calibration its view transform needs
    teacher boxes      decoded detections from the full six-camera model
    teacher logits     the raw 900-query output, for soft targets

No ground truth is read. The teacher is the target, which is what makes this usable on
any driving imagery rather than only on annotated frames -- and it means the student is
trained to reproduce the model being replaced, not to solve nuScenes.

    python local/distill/cache_teacher.py --frames data/distill/frames --limit 0
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/distill/frames")
    ap.add_argument("--out", default="data/distill/teacher")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.25)
    args = ap.parse_args()
    os.chdir(_ROOT)

    from transformers import AutoTokenizer
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # A lock, so an orchestrator restart does not start a second cacher racing the first
    # over the same todo list. Stale locks are cleared by checking the pid.
    lock = out.parent / "cache.lock"
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
            if other != os.getpid() and Path(f"/proc/{other}").exists():
                print(f"  another cacher is running as pid {other}; exiting")
                return 0
        except (ValueError, OSError):
            pass
    lock.write_text(str(os.getpid()))
    dirs = sorted(p for p in Path(args.frames).iterdir() if p.is_dir())
    if args.limit:
        dirs = dirs[:args.limit]
    todo = [d for d in dirs if not (out / f"{d.name}.npz").exists()]
    print(f"  {len(dirs)} frames, {len(todo)} still to cache", flush=True)
    if not todo:
        return 0

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    head = QwenDrivePerception.from_pretrained(
        args.perception, dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, proc)

    vlm, lm = model.vlm, model.vlm.model.language_model
    types = list(vlm.config.text_config.layer_types)
    bev = head.bev_modeling
    dt = next(bev.parameters()).dtype
    from local.prune.measure_layer_influence_v1 import run_stack

    def teacher(frame):
        """Raw head tensors from the full six-camera model.

        ``infer()`` returns decoded boxes, occupancy and map segmentation; the soft
        targets a student needs are the 900-query logits underneath, which only
        ``bev_modeling`` returns.
        """
        with torch.no_grad():
            inputs, metas = proc(frame, device="cuda")
            grid = inputs["image_grid_thw"]
            n_cam = len(metas["cam_order"])
            gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
            tpi = gh // 2 * gw // 2
            emb = vlm.model.get_input_embeddings()(inputs["input_ids"])
            cap = {}
            hk = vlm.model.visual.merger.register_forward_hook(
                lambda m, a, o=None: cap.__setitem__("p", a[0]))
            vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
            hk.remove()
            vit = torch.stack(head._premerge_grids(
                vlm.model.visual.merger.norm(cap["p"]), grid)[-n_cam:], 0).to(dt)
            mt = vlm.model.visual.merger(cap["p"])
            mt = mt[0] if isinstance(mt, tuple) else mt
            mask = inputs["input_ids"][0] == vlm.config.image_token_id
            x = emb.clone()
            x[0, mask] = mt[-int(mask.sum()):].to(emb.dtype)
            pos = torch.arange(x.shape[1], device="cuda")[None].expand(3, 1, -1).contiguous()
            hidden, _ = run_stack(lm, types, x, pos)
            llm = hidden[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1).to(dt)
            o = bev(img_vit_feats=vit, img_llm_feats=llm, img_metas=[metas])
        return o, metas

    t0 = time.time()
    done = 0
    for d in todo:
        try:
            frame = PerceptionFrame(d)
            o, metas = teacher(frame)
            cls = o["all_cls_scores"][-1, 0].float().cpu().numpy()
            box = o["all_bbox_preds"][-1, 0].float().cpu().numpy()
            # keep only the queries that clear the threshold: 900x7 of logits per frame
            # is mostly background, and the student is trained on what the teacher
            # actually asserts
            prob = 1.0 / (1.0 + np.exp(-cls.max(-1)))
            keep = np.nonzero(prob >= args.thr)[0].astype(np.int32)
            np.savez_compressed(
                out / f"{d.name}.npz",
                cls=cls.astype(np.float16), box=box.astype(np.float32),
                keep=keep,
                lidar2img=np.asarray(metas["lidar2img"], dtype=np.float32),
                lidar2ego=np.asarray(metas["lidar2ego"], dtype=np.float32))
            done += 1
            if done % 25 == 0:
                rate = done / (time.time() - t0)
                left = (len(todo) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(todo)}  {rate:.2f} fps  "
                      f"{int(keep.size)} dets  eta {left/60:.0f} min", flush=True)
        except Exception as exc:
            print(f"  {d.name}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
        finally:
            torch.cuda.empty_cache()

    print(f"  cached {done} frames in {(time.time()-t0)/60:.1f} min -> {out}")
    try:
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
