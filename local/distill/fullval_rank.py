"""Rank candidates on the full val set from lab-notebook records.

For every notebook tag family it finds either a single `<tag>-valall` record (all 4,452 cached val-scene
frames) or the pair `<tag>-valfoldA` + `<tag>-valfoldB` (the 150 val scenes split in two, run in parallel)
and merges the folds exactly: tp = recall * GT, fp = tp / precision - tp, fn = GT - tp, for student and
teacher alike. Prints a table sorted by retained % and writes outputs/distill/fullval_ranking.json.

    python local/distill/fullval_rank.py
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)


def counts(r, who):
    d = r["student_vs_gt_best_thr"] if who == "student" else r["teacher_vs_gt"]
    gt = sum(r["recall_by_class"]["gt_count"]); tp = d["recall"] * gt
    fp = tp / d["precision"] - tp if d["precision"] > 0 else 0.0
    return tp, fp, gt - tp, gt


def f1_of(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main():
    recs = json.load(open("outputs/distill/lab_notebook.json"))
    by_tag = {}
    for r in recs:
        by_tag[str(r.get("tag", ""))] = r                      # last record of a tag wins
    families = {}
    for t in by_tag:
        m = re.fullmatch(r"(.+)-(valall|valfoldA|valfoldB)", t)
        if m: families.setdefault(m.group(1), {})[m.group(2)] = by_tag[t]
    rows = []
    for fam, parts in families.items():
        use = [parts["valall"]] if "valall" in parts else ([parts["valfoldA"], parts["valfoldB"]] if {"valfoldA", "valfoldB"} <= parts.keys() else None)
        if not use: continue
        out = {}
        for who in ("student", "teacher"):
            tp = fp = fn = gt = 0.0
            for r in use:
                a, b, c, g = counts(r, who); tp += a; fp += b; fn += c; gt += g
            f1, p, rc = f1_of(tp, fp, fn); out[who] = dict(f1=f1, precision=p, recall=rc)
        traj = [r.get("trajectory_ade_m") for r in use if r.get("trajectory_ade_m")]
        occ = [(r.get("occ3d") or {}).get("student_miou") for r in use if r.get("occ3d")]
        rows.append(dict(candidate=fam, frames=sum(r["frames"] for r in use), gt_objects=int(gt), ckpt=use[0].get("ckpt"),
                         student_f1=out["student"]["f1"], teacher_f1=out["teacher"]["f1"], retained_pct=100 * out["student"]["f1"] / out["teacher"]["f1"],
                         precision=out["student"]["precision"], recall=out["student"]["recall"], source="valall" if "valall" in parts else "folds A+B",
                         traj_ade_m=(sum(traj) / len(traj) if traj else None), occ_miou=(sum(occ) / len(occ) if occ else None)))
    rows.sort(key=lambda x: -x["retained_pct"])
    print(f"Full-val ranking ({time.strftime('%Y-%m-%d %H:%M')}); teacher F1 on the same frames in each row")
    print(f"{'candidate':30s} {'frames':>6s} {'F1':>7s} {'teacher':>8s} {'retained':>9s} {'P':>6s} {'R':>6s} {'traj':>6s} {'occ':>6s}  source")
    for x in rows:
        print(f"{x['candidate']:30s} {x['frames']:6d} {100*x['student_f1']:6.2f}% {100*x['teacher_f1']:7.2f}% {x['retained_pct']:8.2f}% {x['precision']:6.3f} {x['recall']:6.3f} "
              f"{(x['traj_ade_m'] or 0):6.3f} {(x['occ_miou'] or 0):6.3f}  {x['source']}")
    json.dump(rows, open("outputs/distill/fullval_ranking.json", "w"), indent=1)


if __name__ == "__main__":
    main()
