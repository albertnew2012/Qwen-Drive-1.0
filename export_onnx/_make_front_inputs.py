"""Front-camera perception cache record and ONNX bench inputs.

``export_perception.py`` freezes the head's geometry from one cached record, so a
front-camera head needs a record whose ``img_metas`` carry exactly one camera --
otherwise the frozen calibration still describes six and the voxel indices cover a
volume no camera writes to.

Also writes the bench npz the runner consumes, with the front-camera prompt.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer


def main() -> int:
    os.chdir(_ROOT)
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame
    from local.prune.eval_camera_subset_v1 import subset_frame

    token = "90162f90eceb4ada9e595bc1adb71b5f"
    vlm_dir = "weights/Qwen-Drive-1.0-4B"
    model = QwenDriveForPlanning.from_pretrained(
        vlm_dir, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    head = QwenDrivePerception.from_pretrained(
        f"{vlm_dir}/perception", dtype=torch.bfloat16).to("cuda").eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(vlm_dir))
    head.attach(model.vlm, proc)
    vlm = model.vlm

    frame = PerceptionFrame(Path("data/demo/perception") / token)
    front = subset_frame(frame, [frame.cam_order[0]])
    with torch.no_grad():
        inputs, metas = proc(front, device="cuda")
        grid = inputs["image_grid_thw"]
        n_cam = len(metas["cam_order"])
        gh, gw = int(grid[-1, 1]), int(grid[-1, 2])
        tpi = gh // 2 * gw // 2
        embeds = vlm.model.get_input_embeddings()(inputs["input_ids"])
        cap = {}
        hk = vlm.model.visual.merger.register_forward_hook(
            lambda m, a, o=None: cap.__setitem__("p", a[0]))
        vlm.model.visual(inputs["pixel_values"], grid_thw=grid)
        hk.remove()
        vit = torch.stack(head._premerge_grids(
            vlm.model.visual.merger.norm(cap["p"]), grid)[-n_cam:], 0)
        merged = vlm.model.visual.merger(cap["p"])
        merged = merged[0] if isinstance(merged, tuple) else merged
        mask = inputs["input_ids"][0] == vlm.config.image_token_id
        x = embeds.clone()
        x[0, mask] = merged[-int(mask.sum()):].to(embeds.dtype)
        out = vlm.model.language_model(inputs_embeds=x,
                                       position_ids=model._rope_positions(
                                           inputs["input_ids"], grid).to("cuda"),
                                       use_cache=False, output_hidden_states=True)
        hidden = vlm.model.language_model.norm(out.hidden_states[-1])
        llm = hidden[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1)

    rec_dir = _ROOT / "data" / "train_cache_front"
    rec_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"img_vit_feats": vit.float().cpu(),
                "img_llm_feats": llm.float().cpu(),
                "img_metas": metas, "token": token}, rec_dir / "front.pt")
    print(f"  record: vit {tuple(vit.shape)}  llm {tuple(llm.shape)}  "
          f"cams {metas['cam_order']}")

    bench = _ROOT / "outputs" / "onnx_bench"
    bench.mkdir(parents=True, exist_ok=True)
    pos = model._rope_positions(inputs["input_ids"], grid)
    np.savez(bench / "front_perception_inputs.npz",
             pixel_values=inputs["pixel_values"].float().cpu().numpy(),
             input_ids=inputs["input_ids"].cpu().numpy(),
             position_ids=pos.cpu().numpy(),
             image_grid_thw=grid.cpu().numpy(),
             embeds=embeds.float().cpu().numpy(),
             img_tok=np.int64(vlm.config.image_token_id),
             n_cam=np.int64(n_cam))
    print(f"  bench inputs: {int(inputs['input_ids'].shape[1])} tokens, "
          f"pixels {tuple(inputs['pixel_values'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
