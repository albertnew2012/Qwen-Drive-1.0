"""Measure the real prompt layout of a demo planning scene and a perception frame."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from transformers import AutoTokenizer
from qwen_drive.configuration_qwen_drive import QwenDriveConfig
from qwen_drive.scene import QwenDriveProcessor
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.images import ImageArchive

ROOT = "weights/Qwen-Drive-1.0-4B"
cfg = QwenDriveConfig.from_pretrained(ROOT)
tok = AutoTokenizer.from_pretrained(ROOT)
proc = QwenDriveProcessor(tok, cfg)

print("="*96); print("PLANNING SCENE prompt layout"); print("="*96)
arch = ImageArchive.open("data/demo/frames.parquet")
samples = list(read_scene_file("data/demo/planning_scenes.jsonl", image_archive=arch,
                               num_history_points=cfg.num_history_points))
for s in samples:
    sc = s.scene
    for with_r in (False, True):
        inp = proc(sc, with_reasoning=with_r, device="cpu")
        ids = inp["input_ids"][0]
        n_img = int((ids == cfg.vlm_config.image_token_id).sum())
        grids = inp["image_grid_thw"].tolist()
        tag = "REASONING" if with_r else "DIRECT   "
        if not with_r:
            print(f"\n{s.token}  nav={sc.nav_command}  views={list(sc.views)}")
            per = [f"{t}x{h}x{w}->{h*w//4}tok" for t, h, w in grids]
            print(f"   {len(grids)} images: {per}")
            print(f"   pixel_values {tuple(inp['pixel_values'].shape)}")
        print(f"   [{tag}] total {len(ids):5d} tok = {n_img:5d} image + {len(ids)-n_img:4d} text")
    break

print()
print("="*96); print("PERCEPTION FRAME prompt layout"); print("="*96)
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
pp = PerceptionProcessor(tok)
for d in sorted(p for p in Path("data/demo/perception").iterdir() if p.is_dir()):
    f = PerceptionFrame(d)
    inp, metas = pp(f, device="cpu")
    ids = inp["input_ids"][0]
    n_img = int((ids == cfg.vlm_config.image_token_id).sum())
    g = inp["image_grid_thw"].tolist()
    print(f"{f.token[:16]:18s} {f.dataset_type:9s} {len(f.cam_order)} cams "
          f"grid {g[0][1]}x{g[0][2]} -> {g[0][1]*g[0][2]//4} tok/cam   "
          f"total {len(ids):5d} tok = {n_img} image + {len(ids)-n_img} text   "
          f"pixel_values {tuple(inp['pixel_values'].shape)}")
    print(f"                   cams: {f.cam_order}")
    print(f"                   gt keys: {list(f.gt.keys())}  lidar {None if f.lidar is None else f.lidar.shape}")
