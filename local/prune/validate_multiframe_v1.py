"""Does the front-camera 3 Hz configuration hold across frames, or was it one frame?

The configuration measured at 321 ms (3.12 Hz) was tuned and validated on a single
frame, which is not evidence. This reruns the shortlist over every demo frame and
reports the spread, against the unmodified six-camera model on each frame and
restricted to the sector a front camera can see.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

RANK_LEAST_FIRST = [13, 12, 16, 17, 4, 9, 15, 14, 20, 3, 8, 1, 21, 2, 18, 10,
                    25, 11, 29, 24, 5, 28, 30, 22, 26, 7, 6, 23, 19, 27, 31, 0]


def dets(cls, box, thr=0.3):
    p = 1 / (1 + np.exp(-cls.max(-1)))
    return p >= thr, cls.argmax(-1), box[:, :3]


def observable(frame, centres, max_range=60.0):
    """Which reference detections a single forward camera could actually see.

    Scoring a front-camera model against everything with x > 0 charges it for a
    180-degree sector when the camera covers about 64 degrees, which is measuring the
    rig rather than the model. The half-angle comes from the camera's own intrinsics,
    ``atan(width / 2 / fx)``, and the range cap matches the frustum's 60 m.
    """
    K = np.asarray(frame.cam_intrinsic[0], dtype=np.float64)
    fx = float(K[0, 0])
    width = float(frame.image(frame.cam_order[0]).size[0])
    half = np.arctan(width / 2.0 / fx)
    x, y = centres[:, 0], centres[:, 1]
    rng = np.linalg.norm(centres[:, :2], axis=-1)
    return (x > 0) & (np.abs(np.arctan2(y, x)) <= half) & (rng <= max_range), np.degrees(half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames-dir", default="data/demo/perception")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--skips", default="0,16,20,24")
    ap.add_argument("--out", default="outputs/prune/validate_multiframe_v1.json")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame
    from local.prune.eval_camera_subset_v1 import subset_frame
    from local.prune.measure_layer_influence_v1 import run_stack
    from local.prune.voxel_crop_v1 import install

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

    def prep(frame, cams):
        f = subset_frame(frame, cams)
        with torch.no_grad():
            inputs, metas = proc(f, device="cuda")
            grid = inputs["image_grid_thw"]; n_cam = len(metas["cam_order"])
            gh, gw = int(grid[-1, 1]), int(grid[-1, 2]); tpi = gh//2*gw//2
            emb = vlm.model.get_input_embeddings()(inputs["input_ids"])
            cap = {}
            hk = vlm.model.visual.merger.register_forward_hook(
                lambda m, a, o=None: cap.__setitem__("p", a[0]))
            torch.cuda.synchronize(); t = time.perf_counter()
            vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
            torch.cuda.synchronize(); vis = (time.perf_counter()-t)*1e3
            hk.remove()
            vit = torch.stack(head._premerge_grids(
                vlm.model.visual.merger.norm(cap["p"]), grid)[-n_cam:], 0).to(dt)
            mt = vlm.model.visual.merger(cap["p"])
            mt = mt[0] if isinstance(mt, tuple) else mt
            mask = inputs["input_ids"][0] == vlm.config.image_token_id
            x = emb.clone(); x[0, mask] = mt[-int(mask.sum()):].to(emb.dtype)
            pos = torch.arange(x.shape[1], device="cuda")[None].expand(3,1,-1).contiguous()
        return dict(x=x, pos=pos, mask=mask, vit=vit, metas=metas, n_cam=n_cam,
                    gh=gh, gw=gw, tpi=tpi, vis=vis)

    def once(s, skip):
        with torch.no_grad():
            torch.cuda.synchronize(); t = time.perf_counter()
            hidden, _ = run_stack(lm, types, s["x"], s["pos"], skip=set(skip))
            torch.cuda.synchronize(); dec = (time.perf_counter()-t)*1e3
            llm = hidden[0][s["mask"]][-s["n_cam"]*s["tpi"]:].view(
                s["n_cam"], s["gh"]//2, s["gw"]//2, -1).to(dt)
            torch.cuda.synchronize(); t = time.perf_counter()
            o = bev(img_vit_feats=s["vit"], img_llm_feats=llm, img_metas=[s["metas"]])
            torch.cuda.synchronize(); hd = (time.perf_counter()-t)*1e3
        return (o["all_cls_scores"][-1,0].float().cpu().numpy(),
                o["all_bbox_preds"][-1,0].float().cpu().numpy(), dec, hd)

    frames = sorted(p for p in Path(args.frames_dir).iterdir() if p.is_dir())
    skips = [int(s) for s in args.skips.split(",")]
    per = {n: {"ms": [], "kept": [], "of": [], "spur": []} for n in skips}
    print(f"  {len(frames)} frames, front camera only + bit-exact voxel crop\n")
    for fi, fd in enumerate(frames):
        frame = PerceptionFrame(fd)
        s6 = prep(frame, None)
        c6, b6, _, _ = once(s6, [])
        ref = dets(c6, b6, args.thr)
        front, half_deg = observable(frame, ref[2])
        fb = ref[0] & front
        del s6
        torch.cuda.empty_cache()
        restore = install(bev.view_trans)
        try:
            # the demo set mixes nuScenes (CAM_FRONT...) and nuPlan (CAM_F0...);
            # in both conventions cam_order starts with the forward camera
            s1 = prep(frame, [frame.cam_order[0]])
            line = [f"fov+-{half_deg:.0f}deg"]
            for n in skips:
                skip = RANK_LEAST_FIRST[:n]
                once(s1, skip)
                d, h = [], []
                for _ in range(2):
                    cls, box, dec, hd = once(s1, skip)
                    d.append(dec); h.append(hd)
                total = s1["vis"] + float(np.median(d)) + float(np.median(h))
                c = dets(cls, box, args.thr)
                both = fb & c[0]
                per[n]["ms"].append(total)
                per[n]["kept"].append(int(both.sum()))
                per[n]["of"].append(int(fb.sum()))
                per[n]["spur"].append(int((c[0] & ~ref[0]).sum()))
                line.append(f"skip{n}:{total:.0f}ms {int(both.sum())}/{int(fb.sum())}")
            print(f"  frame {fi+1}/{len(frames)} {fd.name[:14]}  " + "  ".join(line),
                  flush=True)
            del s1
        finally:
            restore()
        torch.cuda.empty_cache()

    print(f"\n  {'skip':>5s} {'ms median':>10s} {'ms max':>8s} {'Hz':>6s} "
          f"{'recall front':>13s} {'spurious':>9s}")
    rows = {}
    for n in skips:
        p = per[n]
        med = float(np.median(p["ms"])); mx = float(np.max(p["ms"]))
        kept, of = int(np.sum(p["kept"])), int(np.sum(p["of"]))
        rows[n] = {"ms_median": med, "ms_max": mx, "hz_median": 1000.0/med,
                   "front_kept": kept, "front_of": of,
                   "recall": kept/max(of, 1), "spurious_total": int(np.sum(p["spur"]))}
        print(f"  {n:5d} {med:10.0f} {mx:8.0f} {1000.0/med:6.2f} "
              f"{kept:4d}/{of:<4d} {kept/max(of,1):6.1%} {int(np.sum(p['spur'])):9d}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frames": len(frames), "rows": rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
