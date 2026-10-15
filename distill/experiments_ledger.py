"""Experiment ledger: one place that says what each experiment changed, where its checkpoints
are, and how it scored -- regenerated from the lab notebook, so it never drifts from the data.

    python distill/experiments_ledger.py     -> distill/EXPERIMENTS.md

Naming: e0..e8 = the first series (frame-level split, later shown to leak ~12 F1 points);
final / r1 = the honest scene split at 896x512 / 1152x640; r2_* = corrected 1152 recipe
(depth targets at the right resolution, Occ3D occupancy GT); r3_*..r7_* = later rounds; r8_occ_* = the
occupancy rebuild on Occ3D's native grid (2026-10-01), scored as occfix-* records.
Every lane writes to its own outputs/distill/exp/<name>/ and its snapshots (snap_*.pt,
student_<step>.pt) are written once and never replaced; student.pt is the rolling resume
point of that same run. New experiments always get a new name.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# what changed, in order. key = notebook tag prefix (records whose tag == key or startswith key + "-")
EXPERIMENTS = [
    ("e0", "Baseline: 900-query DETR-style head distilled from the teacher's boxes, 896x512, six cameras, frame-level split (leaks neighbours)", "outputs/distill/student"),
    ("e1", "Query head variants (matching / objectness changes): the DETR head does not converge in this budget", "outputs/distill/exp/e1"),
    ("e1b", "Query head, short run", "outputs/distill/exp/e1b"),
    ("e2", "Query head with teacher-asserted query supervision", "outputs/distill/exp/e2"),
    ("e3", "Query head, 98k steps (first read 2% was diagnose.py using the default cfg; re-eval 28%)", "outputs/distill/exp/e3"),
    ("e4", "Query head, 98k steps, variant (re-eval 33%)", "outputs/distill/exp/e4"),
    ("e5", "First dense CenterPoint head (hidden 64, 2 blocks); e5b continued in place; the 0.0 reads are the heatmap-target bug (Gaussian on fractional coords, no positives)", "outputs/distill/exp/e5"),
    ("e5c", "Center head with the target fix, 154.5k steps: 55.2% (80.9% of the teacher on the leaked split)", "outputs/distill/exp/e5c"),
    ("e6", "Wider/deeper center head (hidden 128, 3 blocks) + min radius 2", "outputs/distill/exp/e6"),
    ("e7", "+ camera-flip augmentation (with the mirrored depth target)", "outputs/distill/exp/e7"),
    ("e8", "+ temporal BEV fusion (history 1), velocity, 10 nuScenes classes, CBGS, EMA: 94.1% of the teacher -- on the leaked split", "outputs/distill/exp/e8"),
    ("final", "e8 recipe from scratch on the official 700/150 SCENE split (scene-all), 896x512, 80k: the first honest number", "outputs/distill/exp/final"),
    ("r1", "final recipe at 1152x640, 80k -- poisoned by depth targets binned in 896 pixel space", "outputs/distill/exp/r1_hires"),
    ("r2_long", "1152x640 with corrected depth targets + Occ3D occupancy GT, 120k one-cycle, snapshots every 5k in the last quarter. 80k snapshot = deliverable (61.0%, 98.4%); end-of-run 59.8%, 7-snapshot average 60.1%. Graph v2 (de-contended scatter, same weights): 41 ms loop / 24.1 Hz idle H200 vs 93 ms / 10.7 Hz for v1", "outputs/distill/exp/r2_long"),
    ("r2_hist3", "r2_long recipe + three-frame BEV history (K=3), 60k. Final 60.6% (97.8%), 4-snapshot average 60.85% (98.2%): tied with r2_long@80k, so the deliverable did not change. Official mAP 0.342 / NDS 0.437 (avg ckpt). On ALL 4,452 val frames the final weights score 60.3% = 98.8% of the teacher, ahead of r2_long@80k (59.5%, 97.5%): the ranking now uses the full set. Graph (de-contended scatter, K=3): 55 ms real-frame loop / 18.1 Hz idle H200", "outputs/distill/exp/r2_hist3"),
    ("r3_final", "r2_hist3 recipe (1152x640, history 3), 75k one-cycle, GPUs 0-3; try 1 stopped at step 150 so the lane restarts with the de-contended scatter (try 2 launched 2026-09-27 01:56); snapshots every 10k are scored individually. 20k: 60.8% gate; 30k: 61.5% gate (99.3%) and 60.85% on ALL 4,452 val frames = 99.7% of the teacher, the leader of the full-val ranking (release v4); 40k 61.3, 50k 60.8, 60k 60.9, final 75k 60.9, avg(60-75k) 61.0 on the gate; student_avg30-50.pt = average of the 30/40/50k plateau snapshots: 61.4% on the gate (99.1%) and 61.2% on ALL val frames = 100.3% of the teacher, the best detector of the project", "outputs/distill/exp/r3_final"),
    ("r4_occfix", "r3_final 30k snapshot fine-tuned 30k->35k (annealing tail of a 35k one-cycle, ~1.4e-5 -> 0) with --occ-outside-empty 0.2: voxels outside the Occ3D camera-visible mask pulled toward empty so the exported occupancy stops painting the unobserved 88% of the grid. Result: gate 61.2% (98.7%, vs 61.5% for the seed), trajectory 1.514 m, occupancy mIoU 0.299 on the gate frames; raw occupancy outside the mask down from ~300k to 75-98k voxels per frame (contiguous extensions of the visible surfaces, no blanket)", "outputs/distill/exp/r4_occfix"),
    ("r7_head2", "Round 7 (24 h extension, 2026-09-28): the deliverable (r4_small avg) re-expressed EXACTLY as a 2x-grid detection head (transposed-conv entry = nearest 2x upsample of the trained 1x1, dilation-2 3x3 blocks; heatmap difference 0.00 at the transplant) and fine-tuned 10k annealing steps with the r4_small recipe on 0.256 m targets: sub-cell localisation and separation of adjacent pedestrians/cones without losing the trained large-object behaviour", "outputs/distill/exp/r7_head2"),
    ("r7_occ3d", "Round 7: the deliverable with a new 3D occupancy head (1x1 lift to 16 ch x 16 pillars, two 3x3x3 conv blocks, 1x1x1 classifier) trained ALONE (--train-only occ_head, backbone and other heads frozen, LR 1.8e-4 -> 0 over 10k steps): every other output stays bit-identical; target = occupancy mIoU 0.30 -> 0.35+ and sharper renders", "outputs/distill/exp/r7_occ3d"),
    ("r6a_avg_small", "LAST LANE (GPUs 0-3, from 2026-09-28 03:47): the r3_final 30/40/50k plateau average (100.3% full-val, best pure detector) fine-tuned with the r4_small recipe (heatmap loss x2 on pedestrian/motorcycle/bicycle/cone + occ-outside-empty 0.2), 10k annealing steps; Result: final 61.0% gate / 100.0% full-val, 35k/40k average 61.5% gate / 100.1% full-val, official mAP 0.370 / NDS 0.444, 27.1 Hz -- did not beat r4_small-avg (100.5%), v6 stands", "outputs/distill/exp/r6a_avg_small"),
    ("r5_avg_occfix", "r3_final 30/40/50k plateau average (100.3% full-val, the best detector) fine-tuned 5k annealing steps with --occ-outside-empty 0.2 (same schedule as r4_occfix; seed step relabelled 30000, fresh Adam): the best detector with the occupancy fix, launched 2026-09-28 01:50 on GPUs 4-7", "outputs/distill/exp/r5_avg_occfix"),
    ("r4_small", "r3_final 30k fine-tuned 30k->40k (anneal ~5.5e-5 -> 0, GPUs 0-3, 2026-09-27 23:19 - 09-28 02:00) with --hm-class-weight 5:2,6:2,7:2,8:2 (pedestrian, motorcycle, bicycle, traffic cone heatmap loss x2) + --occ-outside-empty 0.2. Gate 60.7% (97.9%, seed 61.5%): motorcycles 0.49 and bicycles 0.46 now ABOVE the teacher (0.46 / 0.42), barriers 0.65 vs 0.53, pedestrians 0.45 vs 0.51, cones 0.65 vs 0.68, cars equal; the class re-weighting trades ~0.5 pt of count-weighted F1 for small-class recall in the final weights. Its 35k/40k AVERAGE is the project winner: gate 61.6%, full val 61.35% = 100.5% of the teacher, official mAP 0.375 / NDS 0.444, 27.3 Hz real-frame loop, occupancy fix intact (release v6). Weight soups tried at the end: soup_small_occfix = mean(r4_small-avg, r4_occfix-final), soup_small_r5 = mean(r4_small-avg, r5_avg_occfix-final)", "outputs/distill/exp/r4_small"),
    ("occfix-baseline", "Round 8 baseline (2026-10-01): release v6 and the teacher re-scored on Occ3D's NATIVE grid (0.4 m, +-40 m, (x, y, z)), the teacher's own format. v6's occupancy was on the 0.512 m detection grid in (y, x) order: every render was transposed and 28% too large, and the teacher's old 0.03 mIoU was a grid-mismatch artefact (real: 27%). v6 visible mIoU 33.5% vs teacher 27.0%, but 116k occupied voxels per frame outside the camera-visible region vs the teacher's 18k (the blanket)", "outputs/student_release_v6"),
    ("occfix-r8_occ_a", "Round 8: v6 + a new NATIVE-grid occupancy head (grid_sample BEV->Occ3D grid, 3x3 ctx conv, lift to 16 slices x 16 ch, two 3x3x3 Conv3d, 1x1x1 classifier; 1.26 M params) trained alone 8k steps (frozen BN), class-weighted CE on visible voxels + 1.0 x CE on unobserved voxels vs their label. 33.8% visible mIoU, 36.7k unseen fill raw. Round 9: + 'empty' offset 0.75 (b075 records) 32.6% at teacher fill; + PER-CLASS logit offsets fitted on 200 fold-A frames outside eval250 (cb records): 35.3% on eval250, 34.9% / 34.3% on val folds A / B at the teacher's fill (v7 30.7 / 30.2) = RELEASE v8", "outputs/distill/exp/r8_occ_a"),
    ("occfix-r8_occ_b", "Round 8: as r8_occ_a with unseen weight 0.3; stopped after its 2k snapshot (37.0%, 93.5k unseen fill: too weak)", "outputs/distill/exp/r8_occ_b"),
    ("occfix-r8_occ_d", "Round 8: as r8_occ_a with unseen weight 3.0. Final (8k): 31.0% visible mIoU on eval250, 30.7% / 30.2% on val folds A / B (teacher 26.6 / 25.7), unseen fill 14.6k / 14.7k (teacher 16.9k / 17.7k); detection, map, trajectory bit-identical to v6. RELEASE v7", "outputs/distill/exp/r8_occ_d"),
    ("occfix-r8_occ_e", "Round 8: --occ-vis -- a learned camera-visibility channel gates unseen voxels to empty in-graph, classes trained on visible voxels only. Visibility too blunt from frozen features: at teacher-like fill ~5 points below r8_occ_d; stopped at ~5.4k", "outputs/distill/exp/r8_occ_e"),
    ("occfix-r8_occ_f", "Round 8: r8_occ_d's recipe with a 32-channel 3-D head: ahead at 2k (33.1%), converged to r8_occ_d's point by 4k (30.9% / 15.6k)", "outputs/distill/exp/r8_occ_f"),
    ("occfix-r8_occ_g", "Round 8: r8_occ_d's recipe with a 48-channel 3-D head (does capacity keep paying?)", "outputs/distill/exp/r8_occ_g"),
    ("occfix-r9_ray_a", "Round 9: native head + --occ-raymap (the lift's depth mass passing / stopping / sampled per BEV cell, splatted with the feature indices: the cameras' own visibility evidence), unobserved weight 1.0, 8k steps head-only", "outputs/distill/exp/r9_ray_a"),
    ("occfix-r10_joint", "Round 9: from r8_occ_a 8k, EVERYTHING but the backbone fine-tuned 4k steps (LR 5e-5, all losses, v6 recipe + native occupancy weight 1.0): detection 61.8% gate, map 59.0%, occupancy after per-class offsets = r8_occ_a's -- safe but no gain", "outputs/distill/exp/r10_joint"),
    ("occfix-r11_ray_joint", "Round 9: from r9_ray_a 4k, everything but the backbone with the raymap head, 4k steps (LR 5e-5, all losses)", "outputs/distill/exp/r11_ray_joint"),
    ("occfix-r12_ray_lovasz", "Round 9: r9_ray_a's recipe + Lovasz-softmax on visible voxels (--occ-lovasz 1.0), 8k steps head-only", "outputs/distill/exp/r12_ray_lovasz"),
    ("occfix-r13_ray_lovasz_c32", "Round 9: r12's recipe with a 32-channel 3-D head. Lovasz (r12/r13): raw 35.6-35.7% at 8k but only 32.1-32.8% after per-class offsets at the teacher's fill -- it rewards filling", "outputs/distill/exp/r13_ray_lovasz_c32"),
    ("occfix-r14_teacher_out", "Round 10: the TEACHER's occupancy as the target for unobserved voxels (--occ-outside-target teacher), weight 1.0: 25.9% calibrated at 2k, raw fill 52k, two classes the teacher never predicts collapse to IoU 0 -- stopped (r15_teacher_out_w3, weight 3.0, stopped with it)", "outputs/distill/exp/r14_teacher_out"),
    ("occfix-r16_w06", "Last check: v8's recipe (r8_occ_a) with unobserved weight 0.6, 6000 steps: 34.08% after per-class offsets on the calibration frames (v8's head 34.40%)", "outputs/distill/exp/r16_w06"),
    ("occfix-r17_w15", "Last check: v8's recipe (r8_occ_a) with unobserved weight 1.5, 6000 steps: 34.36% after per-class offsets on the calibration frames (v8's head 34.40%) -- the weight is flat around 1.0; v8 stands", "outputs/distill/exp/r17_w15"),
    ("r3_fine", "r2_hist3 recipe + 2x detection-head grid (head_upsample 2, 0.256 m heatmap) aimed at the pedestrian / cone / <20 m gap, 50k steps on GPUs 4-7 (2026-09-27 06:01-20:43). Gate: 54.0 / 59.5 / 60.1 / 59.9 / 59.6% at 10/20/30/40/50k (96-97% retained): the small-class recall it was built for did come (pedestrians 47-49%, cones 63-70%, bicycles 42-54%, barriers 58-65%) but cars and trucks stayed 2-5 points under r3_final, so it did not win overall", "outputs/distill/exp/r3_fine"),
]
TEACHER = {"f1 val scenes (eval250)": 0.620, "f1 frame split (sorted 250)": 0.682, "official mAP / NDS": "0.2155 / 0.222",
           "trajectory ADE (val scenes, 161 frames)": "1.629 m",
           "occupancy visible mIoU on Occ3D's native grid (eval250)": "0.270 (the 0.032 in older records is a grid-mismatch artefact)"}


def f(x, nd=3):
    return "" if x in (None, "") else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def main():
    recs = json.load(open("outputs/distill/lab_notebook.json"))
    md = [f"# Experiment ledger (generated {time.strftime('%Y-%m-%d %H:%M')} by experiments_ledger.py)", "",
          "Teacher references: " + "; ".join(f"{k} = {v}" for k, v in TEACHER.items()) + ".", "",
          "F1 = detection F1 at 2 m centre distance against nuScenes GT (student at its best threshold), retained % = student F1 / teacher F1 "
          "on the same frames. `tokens=scene` are frames from the 150 held-out val scenes (honest); `sorted`/blank are frame-split frames "
          "whose 0.5 s neighbours were in training (optimistic by ~12 points). Official = nuScenes DetectionEval on all 6,019 val keyframes.", ""]
    for key, what, d in EXPERIMENTS:
        mine = [r for r in recs if str(r.get("tag", "")) == key or str(r.get("tag", "")).startswith(key + "-")]
        md += [f"## {key}", "", what, "", f"Directory: `{d}`"]
        pts = sorted(Path(d).glob("*.pt")) if Path(d).is_dir() else []
        if pts:
            md += ["Checkpoints: " + ", ".join(f"`{p.name}` ({p.stat().st_size/2**20:.0f} MB, {time.strftime('%m-%d %H:%M', time.localtime(p.stat().st_mtime))})" for p in pts)]
        hz = []
        for jp in ("onnx_shape_h200.json", "onnx_h200.json"):
            q = Path(d) / jp
            if q.exists():
                try: hz.append(f"{jp}: {json.load(open(q)).get('hz', 0):.1f} Hz")
                except Exception: pass
        if hz: md += ["H200 speed: " + "; ".join(hz)]
        if mine:
            md += ["", "| tag | step | frames | tokens | F1 | retained % | official mAP | NDS | traj ADE m | occ mIoU (legacy grid) | occ mIoU native / teacher | unseen fill / teacher | checkpoint |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in mine:
                sb = r.get("student_vs_gt_best_thr") or {}; sg = r.get("student_vs_gt") or {}
                f1 = sb.get("f1", sg.get("f1"))
                occ = (r.get("occ3d") or {}).get("student_miou")
                on = r.get("occ_native") or {}
                if on.get("student_grid") == "native":
                    occ = None          # the legacy metric compares a native-grid student against student-grid labels
                onm = f"{on['student_miou']:.3f} / {on['teacher_miou']:.3f}" if on else ""
                onf = f"{on['student_occupied_outside']:,.0f} / {on['teacher_occupied_outside']:,.0f}" if on else ""
                md.append(f"| {r.get('tag')} | {r.get('step','')} | {r.get('frames','')} | {r.get('eval_tokens','')} | {f(f1)} | {f(r.get('retained_pct'),1)} | "
                          f"{f(r.get('mAP'),4)} | {f(r.get('NDS'),4)} | {f(r.get('trajectory_ade_m'))} | {f(occ)} | {onm} | {onf} | `{r.get('ckpt','')}` |")
        else:
            md += ["", "(no evaluation records yet)"]
        md += [""]
    Path("distill/EXPERIMENTS.md").write_text("\n".join(md) + "\n")
    print(f"wrote distill/EXPERIMENTS.md ({len(md)} lines, {len(recs)} notebook records)")


if __name__ == "__main__":
    main()
