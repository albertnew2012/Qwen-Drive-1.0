#!/usr/bin/env python
"""Copy benchmark frames off the read-only mount so evaluation is not I/O bound.

Keeps every CONFLICT frame (left-turn and through signals disagree - the only frames
where association and "report any light" differ) plus a random sample of agreement
frames as a control.
"""
import argparse, json, random, shutil, time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="outputs/tli_eval/bench.json")
    ap.add_argument("--dest", default="outputs/tli_eval/images")
    ap.add_argument("--controls", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/tli_eval/eval_set.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.bench).read_text())
    conflict = [r for r in rows if r["conflict"]]
    agree = [r for r in rows if not r["conflict"]]
    random.Random(args.seed).shuffle(agree)
    chosen = conflict + agree[:args.controls]
    print(f"  {len(conflict)} conflict + {len(chosen) - len(conflict)} control = {len(chosen)} frames")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    kept, missing, t0 = [], 0, time.time()
    for i, r in enumerate(chosen):
        src = Path(r["png"])
        if not src.exists():
            missing += 1
            continue
        name = f"{r['batch']}_{r['frame']:06d}.png"
        dst = dest / name
        if not dst.exists():
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"    copy failed {src}: {e}")
                missing += 1
                continue
        kept.append({**r, "local": str(dst), "name": name})
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(chosen)}  {time.time() - t0:.0f}s", flush=True)

    Path(args.out).write_text(json.dumps(kept, indent=1))
    mb = sum(Path(k["local"]).stat().st_size for k in kept) / 2**20
    print(f"\n  copied {len(kept)} ({missing} missing)  {mb:.0f} MB  in {time.time() - t0:.0f}s")
    print(f"  conflict kept: {sum(1 for k in kept if k['conflict'])}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
