"""Average the late-training snapshots of one run (student_<step>.pt) into student_avg.pt.

Weight averaging over the tail of the schedule usually adds a fraction of a point over the
last checkpoint alone; lane.sh evaluates both and exports the better one.

    python local/distill/average_checkpoints.py outputs/distill/exp/<name>
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> int:
    out = Path(sys.argv[1])
    snaps = sorted(out.glob("student_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if len(snaps) < 2:
        print(f"  {len(snaps)} snapshot(s) in {out}; nothing to average"); return 1
    acc, n, last = None, 0, None
    for p in snaps:
        ck = torch.load(p, map_location="cpu"); sd = ck.get("ema") or ck["model"]; last = ck
        if acc is None:
            acc = {k: (v.double() if v.is_floating_point() else v.clone()) for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point(): acc[k] += v.double()
                else: acc[k] = v.clone()                     # e.g. num_batches_tracked: keep last
        n += 1
    avg = {k: (v.div(n).to(torch.float32) if v.is_floating_point() else v) for k, v in acc.items()}
    torch.save({"model": avg, "ema": avg, "step": last["step"], "cfg": last["cfg"],
                "averaged": [int(p.stem.split("_")[1]) for p in snaps]}, out / "student_avg.pt")
    print(f"  averaged {n} snapshots {[int(p.stem.split('_')[1]) for p in snaps]} -> {out / 'student_avg.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
