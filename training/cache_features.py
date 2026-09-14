"""Stage 1 and 3 freeze the VLM, so run it ONCE and cache its two taps.

Saves ~34 MiB per frame and removes the 9.1 GB VLM from the training loop
entirely, which is what makes head training fit comfortably on one 24 GB card.

    python training/cache_features.py --out data/train_cache
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer

from qwen_drive import QwenDriveForPlanning
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor


@torch.no_grad()
def extract_taps(head, vlm, inputs):
    """The two VLM taps, exactly as ``QwenDrivePerception.infer`` derives them."""
    captured = {}
    visual = vlm.model.visual
    handle = visual.merger.register_forward_hook(
        lambda m, a, o=None: captured.__setitem__("patches", a[0]))
    try:
        outputs = vlm(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=head._modality_ids(inputs["input_ids"]),
            use_cache=False,
            output_hidden_states=True,
        )
    finally:
        handle.remove()
    hidden = vlm.model.language_model.norm(outputs.hidden_states[-1])
    patches = visual.merger.norm(captured["patches"])
    vit_feats = head._premerge_grids(patches, inputs["image_grid_thw"])
    return hidden, vit_feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--out", default="data/train_cache")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation="sdpa")
    vlm = holder.vlm
    del holder.planning_expert
    head = QwenDrivePerception.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm.to(args.device).eval(), proc)

    frames = sorted(p for p in Path(args.frames).iterdir() if p.is_dir())
    print(f"caching {len(frames)} frames -> {out}")
    manifest = []
    for fp in frames:
        frame = PerceptionFrame(fp)
        inputs, metas = proc(frame, device=args.device)
        hidden, vit_feats = extract_taps(head, vlm, inputs)

        n_cam = len(metas["cam_order"])
        gh = inputs["image_grid_thw"][-1, 1].item()
        gw = inputs["image_grid_thw"][-1, 2].item()
        tokens_per_img = gh // 2 * gw // 2
        image_mask = inputs["input_ids"][0] == vlm.config.image_token_id
        llm_tokens = hidden[0][image_mask][-n_cam * tokens_per_img:]
        img_llm = llm_tokens.view(n_cam, gh // 2, gw // 2, -1)
        img_vit = torch.stack(vit_feats[-n_cam:], dim=0)

        gt = np.load(fp / "gt.npz")
        rec = out / f"{fp.name}.pt"
        torch.save({
            "img_vit_feats": img_vit.cpu(),
            "img_llm_feats": img_llm.cpu(),
            "img_metas": {k: v for k, v in metas.items()},
            "gt_boxes": torch.from_numpy(gt["boxes"]).float(),
            "gt_labels": torch.from_numpy(gt["labels"]).long(),
            "gt_occ": torch.from_numpy(gt["occ"]).long(),
            "gt_map": torch.from_numpy(gt["map"]).long(),
            "token": fp.name,
        }, rec)
        mb = rec.stat().st_size / 2**20
        print(f"  {fp.name[:16]}  vit {tuple(img_vit.shape)}  llm {tuple(img_llm.shape)}"
              f"  {len(gt['boxes']):3d} boxes   {mb:6.1f} MiB")
        manifest.append(rec.name)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
