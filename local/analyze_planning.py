#!/usr/bin/env python
"""Per-scene error decomposition for the planning results.

Beyond ADE/FDE: is the model biased longitudinally (over/under-shooting distance),
laterally, or in heading? A consistent sign across scenes is a systematic bias; a
mixed sign is noise.

    PYTHONPATH=src python local/analyze_planning.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


SCALE = np.array([165.0, 25.0, 1.5703125])


def kinematics(d: Path):
    """Implied acceleration of predictions, with a bf16-quantisation control.

    The expert's out_proj is a bf16 Linear, so waypoints are quantised before the
    .float() cast. Quantisation is proportional to magnitude, and differencing twice
    at 10 Hz multiplies it by 100 - which manufactures accelerations that look like
    the model's fault. The control column pushes the GROUND TRUTH through the same
    quantisation, so you can see the floor the representation imposes.
    """
    import torch

    def amax(t):
        v = np.diff(t[:, :2], axis=0) * 10.0
        a = np.diff(v, axis=0) * 10.0
        return np.linalg.norm(a, axis=1).max()

    print(f"\n{'='*92}\nIMPLIED ACCELERATION (m/s^2), with the bf16 control\n{'='*92}")
    print(f"{'scene':<26} {'GT true':>9} {'GT->bf16':>10} {'pred':>9}   verdict")
    print("-" * 92)
    for f in sorted(d.glob("*.npz")):
        r = dict(np.load(f, allow_pickle=True))
        if "ground_truth" not in r or "direct" not in r:
            continue
        gt = r["ground_truth"]
        q = torch.tensor(gt / SCALE).to(torch.bfloat16).float().numpy() * SCALE
        g, qq, pp = amax(gt), amax(q), amax(r["direct"][0])
        verdict = ("dominated by quantisation" if pp <= 2.5 * qq
                   else "excess beyond quantisation")
        print(f"{f.stem[:24]:<26} {g:9.2f} {qq:10.2f} {pp:9.2f}   {verdict}")
    print("\n  Any comfort/jerk/feasibility metric computed from bf16 waypoints is measuring")
    print("  the number format. Re-run in fp32 before drawing kinematic conclusions.")


def collect(d: Path):
    """Return {(token, mode): row} for one results directory."""
    out = {}
    for r in _rows(d):
        out[(r["token"], r["mode"])] = r
    return out


def compare(a: Path, b: Path, label_a="A", label_b="B"):
    """Paired comparison of two checkpoints on the same scenes."""
    ra, rb = collect(a), collect(b)
    keys = sorted(set(ra) & set(rb))
    if not keys:
        print(f"no overlapping (scene, mode) between {a} and {b}"); return
    print(f"\n{'='*96}\n{label_a}  vs  {label_b}   paired on {len(keys)} (scene, mode) pairs\n{'='*96}")
    print(f"{'scene':<26} {'mode':<10} {label_a+' ADE':>10} {label_b+' ADE':>10} {'delta':>9} {'%':>8}")
    print("-" * 96)
    import numpy as _np
    deltas = []
    for k in keys:
        x, y = ra[k]["ade"], rb[k]["ade"]
        deltas.append(y - x)
        print(f"{k[0][:24]:<26} {k[1]:<10} {x:10.3f} {y:10.3f} {y-x:+9.3f} {100*(y-x)/x:+7.1f}%")
    d = _np.array(deltas)
    print(f"\n  mean {d.mean():+.3f} m   {label_b} better on {(d < 0).sum()}/{len(d)}")
    # within each checkpoint, does reasoning beat direct?
    for name, rr in ((label_a, ra), (label_b, rb)):
        pairs = [(rr[(t, 'direct')]['ade'], rr[(t, 'reasoning')]['ade'])
                 for t in {k[0] for k in rr}
                 if (t, 'direct') in rr and (t, 'reasoning') in rr]
        if pairs:
            dd = _np.array([r - dct for dct, r in pairs])
            print(f"  [{name}] reasoning vs direct: mean {dd.mean():+.3f} m, "
                  f"reasoning better on {(dd < 0).sum()}/{len(dd)}")


def _rows(d: Path):
    rows = []
    for p in sorted(d.glob("*.npz")):
        r = dict(np.load(p, allow_pickle=True))
        if "ground_truth" not in r:
            continue
        gt = r["ground_truth"]
        for mode in ("direct", "reasoning"):
            if mode not in r:
                continue
            tr = r[mode]                       # [S, T, 3]
            n = min(tr.shape[1], len(gt))
            err = np.linalg.norm(tr[:, :n, :2] - gt[None, :n, :2], axis=-1)
            ade_per_sample = err.mean(axis=1)
            best = int(ade_per_sample.argmin())
            s0 = tr[0, :n]
            rows.append(dict(
                token=p.stem, mode=mode,
                ade=float(ade_per_sample[0]), fde=float(err[0, -1]),
                min_ade=float(ade_per_sample.min()),
                spread=float(np.linalg.norm(tr[:, -1, :2] - tr[:, -1, :2].mean(0), axis=-1).max()),
                dlon=float(s0[-1, 0] - gt[n - 1, 0]),
                dlat=float(s0[-1, 1] - gt[n - 1, 1]),
                dhead=float(np.degrees(wrap(s0[-1, 2] - gt[n - 1, 2]))),
                # mean over the whole trajectory, not just the endpoint: guards against
                # reading a systematic bias off one waypoint
                dhead_mean=float(np.degrees(wrap(s0[:, 2] - gt[:n, 2]).mean())),
                dlon_mean=float((s0[:, 0] - gt[:n, 0]).mean()),
                gt_disp=float(np.linalg.norm(gt[n - 1, :2])),
                best=best,
            ))

    return rows


def main(d=Path("outputs/planning_demo")):
    rows = _rows(d)
    if not rows:
        print("no scored results yet"); return

    print(f"{'scene':<24} {'mode':<10} {'ADE':>6} {'FDE':>6} {'minADE':>7} {'spread':>7} "
          f"{'dlon_end':>8} {'dlon_avg':>8} {'dhd_end':>8} {'dhd_avg':>8} {'gtdisp':>7}")
    print("-" * 118)
    for r in rows:
        print(f"{r['token'][:22]:<24} {r['mode']:<10} {r['ade']:6.3f} {r['fde']:6.3f} "
              f"{r['min_ade']:7.3f} {r['spread']:7.3f} {r['dlon']:+8.2f} {r['dlon_mean']:+8.2f} "
              f"{r['dhead']:+8.1f} {r['dhead_mean']:+8.1f} {r['gt_disp']:7.2f}")

    print("\nSigns (negative dlon = under-shoots distance; dhead in degrees):")
    for mode in ("direct", "reasoning"):
        sel = [r for r in rows if r["mode"] == mode]
        if not sel:
            continue
        lon = np.array([r["dlon_mean"] for r in sel])
        head = np.array([r["dhead_mean"] for r in sel])
        ade = np.array([r["ade"] for r in sel])
        mn = np.array([r["min_ade"] for r in sel])
        print(f"  {mode:<10} n={len(sel)}  mean ADE {ade.mean():.3f}  mean minADE(6) {mn.mean():.3f}"
              f"   dlon_avg {lon.mean():+.2f} m (signs {''.join('+' if x>0 else '-' for x in lon)})"
              f"   dhead_avg {head.mean():+.1f}deg "
              f"(signs {''.join('+' if x>0 else '-' for x in head)})")

    both = {}
    for r in rows:
        both.setdefault(r["token"], {})[r["mode"]] = r
    pairs = [(v["direct"], v["reasoning"]) for v in both.values()
             if "direct" in v and "reasoning" in v]
    if pairs:
        print(f"\nreasoning vs direct, paired on {len(pairs)} scene(s):")
        for dcase, rcase in pairs:
            delta = rcase["ade"] - dcase["ade"]
            print(f"  {dcase['token'][:34]:<36} {dcase['ade']:.3f} -> {rcase['ade']:.3f}   "
                  f"{delta:+.3f} m ({100*delta/dcase['ade']:+.1f} %)")
        d = np.array([r["ade"] - dd["ade"] for dd, r in pairs])
        print(f"  mean change {d.mean():+.3f} m; reasoning better on {(d < 0).sum()}/{len(d)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/planning_demo")
    ap.add_argument("--compare-with", default=None,
                    help="second results dir, e.g. outputs/planning_sft")
    ap.add_argument("--label-a", default="planner-rl")
    ap.add_argument("--label-b", default="planner-sft")
    ap.add_argument("--kinematics", action="store_true",
                    help="implied accelerations with the bf16-quantisation control")
    a = ap.parse_args()
    main(Path(a.dir))
    if a.kinematics:
        kinematics(Path(a.dir))
    if a.compare_with and Path(a.compare_with).exists():
        compare(Path(a.dir), Path(a.compare_with), a.label_a, a.label_b)
