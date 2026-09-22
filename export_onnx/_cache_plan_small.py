"""Cache the planning scene KV with the side views shrunk.

``training/cache_planner_features.py`` prefills each scene once and stores the eight
post-rotary key/value pairs the expert reads. Reducing the token count changes those
shapes, so the cache has to be regenerated before the expert graph can be exported at
the new size.
"""
from __future__ import annotations

import argparse, os, runpy, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(_ROOT)

    from qwen_drive import benchmarks
    from local.prune.optimise_planning_v1 import shrink_side_views
    original = benchmarks.read_scene_file

    def patched(*a, **k):
        for sample in original(*a, **k):
            try:
                sample.scene = shrink_side_views(sample.scene, args.side)
            except Exception:
                pass
            yield sample

    benchmarks.read_scene_file = patched
    import training.cache_planner_features as mod
    mod.read_scene_file = patched
    sys.argv = ["cache_planner_features", "--out", str(args.out)]
    return mod.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
