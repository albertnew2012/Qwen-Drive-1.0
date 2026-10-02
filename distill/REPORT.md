# Student vs teacher -- live report (rewritten by orchestrate_95.py)

Updated 2026-09-27 23:08. Target: >= 95.0% of the teacher's F1 on the 150 held-out val scenes (250 frames, 2 m centre distance, 10 nuScenes classes, official GT filter) and >= 10 Hz on one H200 (all five outputs, one graph).

| lane | status | F1 | teacher F1 | retained % | official mAP | NDS | traj ADE m (teacher 1.629) | occ vs Occ3D GT (student / teacher) | occ agree | map agree | H200 Hz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| final | done | 0.572 | 0.620 | 92.3 | 0.304 | 0.394 | 1.540 | 0.050 / 0.032 | 0.075 | 0.583 | 24.6 |
| r1 | done | 0.566 | 0.620 | 91.4 | 0.289 | 0.370 | 1.514 | 0.049 / 0.032 | 0.075 | 0.564 | - |
| r2_long | done | 0.610 | 0.620 | 98.4 | - | - | 1.510 | 0.322 / 0.032 | 0.011 | 0.601 | 19.4 |
| r2_hist3 | done | 0.608 | 0.620 | 98.2 | 0.342 | 0.437 | 1.565 | 0.322 / 0.032 | 0.011 | 0.601 | 19.2 |
| r3_final | running | 0.615 | 0.620 | 99.3 | - | - | 1.515 | 0.299 / 0.032 | 0.011 | 0.586 | 17.5 |
| r3_fine | done | 0.601 | 0.620 | 97.0 | - | - | 1.516 | 0.305 / 0.032 | 0.011 | 0.597 | 13.8 |

H200 Hz = the graph's best timing on an idle GPU (launch speed gate or export); numbers taken while another lane was training on the same GPU are lower bounds and are not used. The deliverable's timing is re-measured on an idle GPU at the end.

**Best so far:** `r3_final` -- 99.3% retained (0.615 vs teacher 0.620), H200 17.5 Hz. Target MET.
Checkpoint `outputs/distill/exp/r3_final/snap_30000.pt`, ONNX `outputs/onnx/r3_final_snap30k/student.onnx`.
