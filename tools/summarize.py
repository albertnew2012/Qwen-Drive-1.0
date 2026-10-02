#!/usr/bin/env python
"""Print what has been produced so far by the planning and perception runs."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES, OCC_CLASS_NAMES


def planning(d=Path("outputs/planning_demo")):
    npzs = sorted(d.glob("*.npz")) if d.exists() else []
    print("=" * 108)
    print(f"PLANNING  ({len(npzs)} scenes with results)")
    print("=" * 108)
    if not npzs:
        print("  (nothing yet)"); return
    print(f"  {'scene':<40} {'modes':<22} {'direct ADE/FDE':<18} {'reasoning ADE/FDE':<18}")
    print("  " + "-" * 104)
    for p in npzs:
        r = dict(np.load(p, allow_pickle=True))
        j = p.with_suffix(".json")
        meta = json.loads(j.read_text()) if j.exists() else {}
        modes = [m for m in ("direct", "reasoning") if m in r]
        if meta.get("vqa_answer"): modes.append("vqa")
        def fmt(k):
            a, f = meta.get(f"{k}_ade"), meta.get(f"{k}_fde")
            return f"{a:.3f} / {f:.3f}" if a is not None else "-"
        print(f"  {p.stem[:38]:<40} {','.join(modes):<22} {fmt('direct'):<18} {fmt('reasoning'):<18}")
    for p in npzs:
        meta_p = p.with_suffix(".json")
        if not meta_p.exists(): continue
        m = json.loads(meta_p.read_text())
        if m.get("reasoning_text") or m.get("vqa_answer"):
            print(f"\n  --- {p.stem} ---")
            if m.get("reasoning_text"): print(f"  reasoning: {m['reasoning_text']}")
            if m.get("vqa_answer"):     print(f"  vqa      : {m['vqa_answer']}")


def perception(d=Path("outputs/perception_demo")):
    npzs = sorted(d.glob("*.npz")) if d.exists() else []
    print("\n" + "=" * 108)
    print(f"PERCEPTION  ({len(npzs)} frames with results)")
    print("=" * 108)
    if not npzs:
        print("  (nothing yet)"); return
    for p in npzs:
        r = dict(np.load(p, allow_pickle=True))
        s, lab = r["scores"], r["labels"]
        keep = s > 0.3
        counts = {DET_CLASS_NAMES[i]: int((lab[keep] == i).sum()) for i in range(len(DET_CLASS_NAMES))}
        counts = {k: v for k, v in counts.items() if v}
        occ = r["occ"]; occupied = int((occ != len(OCC_CLASS_NAMES) - 1).sum())
        print(f"  {p.stem[:34]:<36} {int(keep.sum()):3d} boxes>0.3 / {len(s)}   "
              f"occ {occ.shape} {100*occupied/occ.size:5.2f}% non-empty   map {r['map'].shape}")
        print(f"      {counts}")


if __name__ == "__main__":
    planning(); perception()
