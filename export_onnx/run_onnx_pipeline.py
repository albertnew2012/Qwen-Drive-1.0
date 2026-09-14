"""Run Qwen-Drive-1.0 end to end from ONNX, and check it against PyTorch.

    vision.onnx ──┬─► vit_tap ──────────────────────────┐
                  └─► merged_tokens ──► (host: scatter) │
                                             │          │
                          layer_00..31.onnx ─┴─► hidden ─┴─► perception.onnx
                                             └─► 8 x (keys, values) ─► planner_step.onnx x10

Everything the graphs cannot express stays on the host and is named explicitly in
``host_*`` functions below: the embedding gather, the vision-token scatter, the
mRoPE position ids, and the Euler loop.

    python export_onnx/run_onnx_pipeline.py
    python export_onnx/run_onnx_pipeline.py --skip-planner     # perception only
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import onnxruntime as ort
from transformers import AutoTokenizer

from qwen_drive import QwenDriveForPlanning
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor


def session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def rel(a, b) -> float:
    """Relative max-abs difference. NaN/Inf anywhere is reported as inf.

    ``max(x, nan)`` returns x in Python, so a NaN output would otherwise be
    swallowed and the harness would print PASS over garbage - which it did.
    """
    b = b.detach().cpu().numpy() if torch.is_tensor(b) else b
    a = np.asarray(a)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-9)


# ────────────────────────────────────────────────────────── host-side steps

def host_embed(table: np.ndarray, input_ids: np.ndarray) -> np.ndarray:
    """Token embedding: a gather, not compute. Kept off the graph so the 2.4 GiB
    table is not duplicated into every export."""
    return table[input_ids]


def host_scatter_vision(embeds, merged, input_ids, image_token_id):
    """Replace the image-token rows with the vision tower's merged tokens."""
    out = embeds.copy()
    mask = input_ids[0] == image_token_id
    n = int(mask.sum())
    out[0, mask] = merged[-n:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--vision", default="outputs/onnx/vlm_vision/vision.onnx")
    ap.add_argument("--layers", default="outputs/onnx/vlm_layers")
    ap.add_argument("--perception", default="outputs/onnx/perception/perception.onnx")
    ap.add_argument("--skip-planner", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-3)
    ap.add_argument("--phase", choices=["run", "compare", "both"], default="both",
                    help="'run' executes the ONNX graphs and saves their outputs; "
                         "'compare' loads those and checks them against PyTorch. "
                         "Splitting them matters: holding the 17 GiB fp32 model "
                         "AND the 20 GiB perception forward at once gets the "
                         "process OOM-killed on a 78 GiB machine.")
    ap.add_argument("--save", default="outputs/onnx_run")
    args = ap.parse_args()

    rule = lambda t: print(f"\n{'='*88}\n  {t}\n{'='*88}")

    # ---- reference: the real model in PyTorch --------------------------------------
    save = Path(args.save); save.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.layers) / "manifest.json").read_text())
    layer_types = manifest["layer_types"]

    # ── shared setup: preprocessing only ────────────────────────────────────────────
    rule("SETUP")
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.float32, attn_implementation="eager")
    vlm = holder.vlm.eval()
    head = QwenDrivePerception.from_pretrained(args.model, dtype=torch.float32).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm, proc)
    frame = PerceptionFrame(Path(args.frames) / args.frame)
    inputs, metas = proc(frame, device="cpu")
    n_cam = len(metas["cam_order"])
    gh = inputs["image_grid_thw"][-1, 1].item()
    gw = inputs["image_grid_thw"][-1, 2].item()
    tpi = gh // 2 * gw // 2
    ids = inputs["input_ids"].numpy()
    img_tok = vlm.config.image_token_id
    pos = holder._rope_positions(inputs["input_ids"],
                                 inputs["image_grid_thw"]).numpy()
    print(f"  frame {frame.token[:16]}  {n_cam} cameras  {ids.shape[1]} tokens")
    from training.differentiable import enable_training_ops
    enable_training_ops()

    if args.phase in ("run", "both"):
        # The 17 GiB fp32 VLM is not needed to RUN the graphs - only to preprocess.
        # Dropping it here is what keeps this within memory.
        del holder, vlm
        import gc; gc.collect()

        rule("ONNX PIPELINE")
        t_all = time.time()
        print("  [1/4] vision tower")
        t0 = time.time()
        vis = session(Path(args.vision))
        px = inputs["pixel_values"].numpy().astype(np.float32)
        premerge, vit_tap, merged = vis.run(None, {"pixel_values": px})
        del vis
        print(f"        {time.time()-t0:5.1f}s   vit_tap {vit_tap.shape}")

        print("  [2/4] host: embedding gather, vision scatter")
        table = np.load(Path(args.layers) / "embed_tokens.npy", mmap_mode="r")
        embeds = host_embed(np.asarray(table), ids).astype(np.float32)
        del table
        embeds = host_scatter_vision(embeds, merged, ids, img_tok)
        print(f"        embeds {embeds.shape}")

        print(f"  [3/4] {len(layer_types)} decoder layers, one graph each")
        hidden = embeds
        kv = {}
        t0 = time.time()
        for i, kind in enumerate(layer_types):
            sess = session(Path(args.layers) / f"layer_{i:02d}.onnx")
            if kind == "linear_attention":
                hidden = sess.run(None, {"hidden_in": hidden})[0]
            else:
                hidden, k, v = sess.run(None, {"hidden_in": hidden,
                                               "position_ids": pos})
                kv[i] = (k, v)
            del sess
            if (i + 1) % 8 == 0:
                print(f"        {i+1:2d}/{len(layer_types)}   {time.time()-t0:5.1f}s")
        # THE FINAL NORM IS APPLIED TWICE, ON PURPOSE.
        # transformers appends the POST-norm output as hidden_states[-1]
        # (len(hidden_states) == 33 for 32 layers), and modeling_perception.py
        # then calls language_model.norm() on it again. The released heads were
        # fitted against that double-normed tensor, so reproducing the model
        # means reproducing the second norm. Applying it once differs from
        # PyTorch by 4.7e-01 - the tensors look plausible either way, which is
        # why only a comparison catches it.
        sess = session(Path(args.layers) / "final_norm.onnx")
        hidden = sess.run(None, {"hidden_in": hidden})[0]   # == hidden_states[-1]
        hidden = sess.run(None, {"hidden_in": hidden})[0]   # == what infer() feeds the head
        del sess
        print(f"        hidden_states {hidden.shape}   {time.time()-t0:.1f}s")

        print("  [4/4] perception head")
        mask = ids[0] == img_tok
        img_llm = hidden[0][mask][-n_cam * tpi:].reshape(n_cam, gh // 2, gw // 2, -1)
        vit_grids = QwenDrivePerception._premerge_grids(
            torch.from_numpy(vit_tap), inputs["image_grid_thw"])
        img_vit = torch.stack(vit_grids[-n_cam:], 0).numpy()
        t0 = time.time()
        perc = session(Path(args.perception))
        cls, box, occ, seg = perc.run(None, {"img_vit_feats": img_vit,
                                             "img_llm_feats": img_llm})
        del perc
        print(f"        {time.time()-t0:5.1f}s   cls {cls.shape}")
        print(f"\n  ONNX pipeline total {time.time()-t_all:.0f}s")

        np.savez(save / "onnx_outputs.npz", hidden=hidden, vit_tap=vit_tap,
                 merged=merged, img_vit=img_vit, img_llm=img_llm,
                 all_cls_scores=cls, all_bbox_preds=box, occ_pred=occ,
                 seg_preds=seg, **{f"key_{i}": kv[i][0] for i in kv},
                 **{f"value_{i}": kv[i][1] for i in kv})
        print(f"  saved -> {save/'onnx_outputs.npz'}")
        if args.phase == "run":
            print("\n  now run:  --phase compare")
            return 0

    # ── comparison ──────────────────────────────────────────────────────────────────
    rule("ONNX vs PYTORCH")
    got = np.load(save / "onnx_outputs.npz")
    if args.phase == "compare":
        holder = QwenDriveForPlanning.from_pretrained(
            args.vlm, dtype=torch.float32, attn_implementation="eager")
        vlm = holder.vlm.eval()
        head.attach(vlm, proc)
    print("  computing the PyTorch reference (this is the slow part)...")
    t0 = time.time()
    cap = {}
    hk = vlm.model.visual.merger.register_forward_hook(
        lambda m, a, o=None: cap.__setitem__("p", a[0]))
    try:
        with torch.no_grad():
            out = vlm(input_ids=inputs["input_ids"],
                      pixel_values=inputs["pixel_values"],
                      image_grid_thw=inputs["image_grid_thw"],
                      mm_token_type_ids=head._modality_ids(inputs["input_ids"]),
                      use_cache=False, output_hidden_states=True)
            ref_hidden = vlm.model.language_model.norm(out.hidden_states[-1])
            ref_vit = torch.stack(QwenDrivePerception._premerge_grids(
                vlm.model.visual.merger.norm(cap["p"]),
                inputs["image_grid_thw"])[-n_cam:], 0)
    finally:
        hk.remove()
    del out
    mask_t = torch.from_numpy(ids[0] == img_tok)
    ref_llm = ref_hidden[0][mask_t][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1)
    print(f"  VLM reference {time.time()-t0:.0f}s")

    rows = [("VLM hidden_states", rel(got["hidden"], ref_hidden)),
            ("VLM vit tap", rel(got["img_vit"], ref_vit)),
            ("VLM llm tap", rel(got["img_llm"], ref_llm))]

    # The 8 full-attention KV caches are what the Planning Expert cross-attends
    # to. They leave the layer graphs but were never checked, which left the
    # perception path validated and the planning path's input seam untested.
    full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    with torch.no_grad():
        cache_out = vlm(input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        image_grid_thw=inputs["image_grid_thw"],
                        mm_token_type_ids=head._modality_ids(inputs["input_ids"]),
                        use_cache=True)
        pkv = cache_out.past_key_values
    worst_k = worst_v = 0.0
    for i in full_idx:
        if f"key_{i}" not in got:
            continue
        worst_k = max(worst_k, rel(got[f"key_{i}"], pkv.layers[i].keys))
        worst_v = max(worst_v, rel(got[f"value_{i}"], pkv.layers[i].values))
    del cache_out, pkv
    rows.append((f"planner KV keys (x{len(full_idx)})", worst_k))
    rows.append((f"planner KV values (x{len(full_idx)})", worst_v))

    bev = head.bev_modeling
    del holder, vlm
    import gc; gc.collect()
    print("  perception reference (from PYTORCH taps, not the ONNX ones)...")
    t0 = time.time()
    with torch.no_grad():
        ref_perc = bev(img_vit_feats=ref_vit, img_llm_feats=ref_llm,
                       img_metas=[metas])
    print(f"  perception reference {time.time()-t0:.0f}s")
    for name in ("all_cls_scores", "all_bbox_preds", "occ_pred", "seg_preds"):
        rows.append((name, rel(got[name], ref_perc[name])))

    print(f"\n  {'tensor':22s} {'relative diff':>14}   verdict")
    worst = 0.0
    for name, r in rows:
        worst = r if (np.isnan(r) or r > worst) else worst
        tag = "NaN/Inf" if not np.isfinite(r) else ("ok" if r < args.tol else "MISMATCH")
        print(f"  {name:22s} {r:14.3e}   {tag}")
    ok = np.isfinite(worst) and worst < args.tol
    print(f"\n  PIPELINE {'PASS' if ok else 'FAIL'}  "
          f"(worst {worst:.2e}, tolerance {args.tol:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
