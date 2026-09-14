"""Collect every measured result into one baseline, for regression checking.

Writes ``outputs/expected_results.json``. Re-run anything later and compare
against this file rather than against memory.

    PYTHONPATH=src python study/scripts/12_collect_results.py
"""
from __future__ import annotations

import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def grab(path, pattern, cast=float, group=1):
    p = ROOT / path
    if not p.exists():
        return None
    m = None
    for line in p.read_text(errors="ignore").splitlines():
        hit = re.search(pattern, line)
        if hit:
            m = hit
    return cast(m.group(group)) if m else None


def history_loss(path):
    p = ROOT / path
    if not p.exists():
        return None
    h = json.loads(p.read_text())
    if not h:
        return None
    key = "loss" if "loss" in h[0] else "L_perc"
    k = max(1, len(h) // 10)
    return {"steps": len(h),
            "first": round(sum(r[key] for r in h[:k]) / k, 5),
            "last": round(sum(r[key] for r in h[-k:]) / k, 5)}


def main() -> int:
    out = {
        "onnx": {
            "per_graph_rel_diff": {
                "vision_tower_vit_tap": 7.08e-05,
                "text_prefill_perception_shape": 1.36e-06,
                "text_decode_step_worst_of_65": 1.99e-06,
                "perception_head": 5.29e-04,
                "planner_step": 4.17e-07,
            },
            "end_to_end_perception": {
                "VLM hidden_states": 3.146e-04,
                "VLM vit tap": 7.162e-05,
                "VLM llm tap": 5.099e-04,
                "planner KV keys": 5.183e-04,
                "planner KV values": 2.248e-04,
                "all_cls_scores": 1.032e-03,
                "all_bbox_preds": 1.050e-03,
                "occ_pred": 1.307e-06,
                "seg_preds": 3.654e-06,
                "verdict": "PASS", "tolerance": 5e-03,
            },
            "end_to_end_planning": {
                "ADE_m": 1.64e-05, "FDE_m": 4e-05,
                "verdict": "PASS", "tolerance_m": 5e-02,
            },
            "ort_session_scaling": {
                "note": "quadratic in node count; this is why there is no single file",
                "1_layer": {"nodes": 14161, "load_s": 3.8},
                "2_layers": {"nodes": 28322, "load_s": 16.0},
                "4_layers": {"nodes": 56644, "load_s": 84.3},
            },
        },
        "training": {
            "gradient_test": {
                "without_patch": "NotImplementedError (expected)",
                "with_patch_params_with_grad": "682/682",
                "peak_gib": 19.1,
            },
            "stage1_overfit_one_frame": history_loss("outputs/sweep_s1/history.json"),
            "stage1_all_six_frames": history_loss("outputs/train_all6/history.json"),
            "stage2_joint_lora": history_loss("outputs/sweep_s2/history.json"),
            "stage3_planner_scratch": history_loss("outputs/sweep_s3/history.json"),
            "multi_gpu": {
                "hardware": "2x RTX 3090, 48 GiB, NO NVLink (P2P disabled)",
                "stage1_1gpu_samples_per_s": 0.217,
                "stage1_2gpu_ddp_samples_per_s": 0.435,
                "stage1_ddp_speedup": "2.0x",
                "stage3_1gpu_samples_per_s": 4.76,
                "stage3_2gpu_ddp_samples_per_s": 2.75,
                "stage3_ddp_speedup": "0.58x - DDP is SLOWER, run stage 3 on one card",
                "stage2_module_parallel": {
                    "vision_encoder": "TRAINABLE, 333.51 M (frozen on one card)",
                    "peak_cuda0_gib": 13.8, "peak_cuda1_gib": 8.5,
                    "single_gpu_peak_gib": 18.5,
                },
                "rule": "DDP pays when compute-per-step / bytes-reduced is large: "
                        "stage 1 needs 0.05 GB/s, stage 3 needs 10 GB/s",
            },
            "expectations": {
                "stage1": "loss must at least halve on an overfit run",
                "stage2": "L_perc must fall AND vlm_grad_abs_sum must be > 0",
                "stage3": "needs --scratch; from released weights loss starts at ~4e-5",
                "stage2_run_length": "judge over >=8 steps. L_perc spikes hard on "
                                     "step 1 (15 -> 39 observed) before falling; a "
                                     "5-step run reports a spurious FAIL",
            },
        },
    }
    dest = ROOT / "outputs" / "expected_results.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    for section, body in out.items():
        print(f"\n{section.upper()}")
        for k, v in body.items():
            if isinstance(v, dict) and "first" in v:
                print(f"  {k:34s} {v['first']:>9.4f} -> {v['last']:<9.4f} "
                      f"({v['steps']} steps)")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"      {kk:30s} {vv}")
            else:
                print(f"  {k:34s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
