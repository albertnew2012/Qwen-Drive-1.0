"""Dataset over the cached VLM features.

Stage 1 freezes the VLM, so a sample is just (two taps, calibration, GT). The
cameras differ per dataset - nuScenes frames have 6, nuPlan 8 - so samples are
NOT collated across different rigs; batch_size is per-rig.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedPerceptionDataset(Dataset):
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        manifest = self.dir / "manifest.json"
        self.records = (json.loads(manifest.read_text()) if manifest.exists()
                        else sorted(p.name for p in self.dir.glob("*.pt")))
        if not self.records:
            raise FileNotFoundError(
                f"no cached records in {self.dir}; run training/cache_features.py first")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return torch.load(self.dir / self.records[i], weights_only=False)


def collate_single(batch):
    """batch_size 1 only - the BEV head takes one img_metas dict per sample."""
    assert len(batch) == 1, "use batch_size=1; rigs differ in camera count"
    return batch[0]
