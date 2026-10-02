#!/usr/bin/env python
"""Score the traffic-light association runs and break the errors down.

The headline number is accuracy on CONFLICT frames, where the left-turn and through
signals show different colours. Agreement frames are reported too, but they cannot
distinguish association from "name the obvious colour", so they are a floor rather than
a result.

`bias_to_through` counts, among wrong left-turn answers, how often the model instead
returned the through signal's colour. That is the signature of reading the wrong lamp
rather than misreading the right one.
"""
import argparse, json
from collections import Counter
from pathlib import Path


def load(p):
    f = Path(p)
    return json.loads(f.read_text()) if f.exists() else []


def block(rows, label):
    conf = [r for r in rows if r["conflict"]]
    agree = [r for r in rows if not r["conflict"]]
    print(f"\n  === {label}   ({len(rows)} frames: {len(conf)} conflict, {len(agree)} agree) ===")
    for kind in ("left", "through"):
        for name, sel in (("conflict", conf), ("agree", agree)):
            v = [r for r in sel if r.get(f"{kind}_gt") and r.get(f"{kind}_pred") is not None]
            if not v:
                continue
            ok = sum(r[f"{kind}_pred"] == r[f"{kind}_gt"] for r in v)
            print(f"    {kind:8s} {name:9s} {ok:4d}/{len(v):4d} = {ok/len(v):6.1%}")
    # the diagnostic: wrong left answers that echo the through colour
    v = [r for r in conf if r.get("left_gt") and r.get("left_pred")]
    wrong = [r for r in v if r["left_pred"] != r["left_gt"]]
    echo = [r for r in wrong if r["left_pred"] == r["through_gt"]]
    if wrong:
        print(f"    of {len(wrong)} wrong left answers, {len(echo)} ({len(echo)/len(wrong):.0%}) "
              f"returned the THROUGH colour instead")
    if v:
        print(f"    left  gt/pred pairs: {Counter((r['left_gt'], r['left_pred']) for r in v).most_common(6)}")
    return conf


def by_bucket(conf, field, edges, label):
    print(f"\n    left-turn accuracy by {label}:")
    for lo, hi in zip(edges, edges[1:] + [10**9]):
        v = [r for r in conf if r.get("left_gt") and r.get("left_pred") and lo <= r[field] < hi]
        if not v:
            continue
        ok = sum(r["left_pred"] == r["left_gt"] for r in v)
        bar = "#" * int(30 * ok / len(v))
        print(f"      {lo:5.0f}-{hi if hi < 10**8 else 999:<5.0f} n={len(v):4d}  {ok/len(v):6.1%}  {bar}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/tli_eval")
    args = ap.parse_args()
    d = Path(args.dir)

    runs = {}
    for f in sorted(d.glob("results_*.json")):
        rows = load(f)
        if rows:
            runs.setdefault(rows[0]["condition"], []).extend(rows)

    if not runs:
        print("  no results yet")
        return 0

    for cond in ("full", "hires", "crop"):
        if cond not in runs:
            continue
        conf = block(runs[cond], cond)
        if cond == "full" and conf:
            by_bucket(conf, "largest_box_px", [0, 40, 80, 160, 320], "largest head box (px, 4K)")
            by_bucket(conf, "nearest_m", [0, 15, 25, 40], "distance to nearest head (m)")
            by_bucket(conf, "n_heads", [1, 4, 8, 14], "number of signal heads in frame")

    if "full" in runs and "hires" in runs:
        a = {r["name"]: r for r in runs["full"]}
        b = {r["name"]: r for r in runs["hires"]}
        both = [n for n in a if n in b and a[n].get("left_gt")]
        if both:
            fa = sum(a[n]["left_pred"] == a[n]["left_gt"] for n in both) / len(both)
            fb = sum(b[n]["left_pred"] == b[n]["left_gt"] for n in both) / len(both)
            print(f"\n  === paired on {len(both)} frames seen by both conditions ===")
            print(f"    left-turn accuracy   full {fa:.1%}  ->  hires {fb:.1%}   ({fb-fa:+.1%})")
            fixed = [n for n in both if a[n]['left_pred'] != a[n]['left_gt'] and b[n]['left_pred'] == b[n]['left_gt']]
            broke = [n for n in both if a[n]['left_pred'] == a[n]['left_gt'] and b[n]['left_pred'] != b[n]['left_gt']]
            print(f"    hires fixed {len(fixed)}, broke {len(broke)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
