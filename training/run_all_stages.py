"""Run every training stage in order and report PASS/FAIL for each.

One command to re-verify the whole training pipeline. Stages run sequentially on
purpose: two of them peak near 20 GiB and the machine OOM-kills concurrent runs.

    python training/run_all_stages.py
"""
from __future__ import annotations

import argparse, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_BIN = str(ROOT / ".venv" / "bin" / "python")

STAGES = [
    ("gradient test WITHOUT the patch (failing IS the pass)",
     ["training/test_gradients.py", "--skip-patch", "--dtype", "bfloat16"],
     "NotImplementedError", True),
    ("gradient test WITH the patch",
     ["training/test_gradients.py", "--dtype", "bfloat16"],
     "PASS: the perception head is differentiable", False),
    ("stage 1 - perception head, overfit one frame",
     ["training/train_perception.py", "--steps", "120", "--overfit",
      "--out", "outputs/sweep_s1"], "OVERFIT TEST: PASS", False),
    ("stage 3 - planning expert, overfit from scratch",
     ["training/train_planner.py", "--steps", "200", "--overfit", "--scratch",
      "--out", "outputs/sweep_s3"], "OVERFIT TEST: PASS", False),
    ("stage 2 - joint perception + VLM (LoRA)",
     ["training/train_joint.py", "--steps", "8", "--overfit",
      "--out", "outputs/sweep_s2"], "JOINT TEST: PASS", False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip stage 1 and 2 (the two multi-minute ones)")
    args = ap.parse_args()

    cuda_home = os.environ.get("CUDA_HOME", "/usr")
    env = {"PYTHONPATH": f"{ROOT/'src'}:{ROOT}", "CUDA_HOME": cuda_home,
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "PATH": f"{ROOT/'.venv'/'bin'}:{cuda_home}/bin:/usr/bin:/bin",
           "TOKENIZERS_PARALLELISM": "false", "HOME": str(Path.home())}
    results = []
    for label, cmd, marker, expect_failure in STAGES:
        if args.skip_slow and ("train_perception" in cmd[0] or "train_joint" in cmd[0]):
            print(f"\n=== SKIP  {label}")
            continue
        print(f"\n{'='*84}\n  {label}\n{'='*84}", flush=True)
        t0 = time.time()
        proc = subprocess.run([PY_BIN] + cmd, cwd=ROOT, env=env,
                              capture_output=True, text=True)
        blob = proc.stdout + proc.stderr
        ok = marker in blob
        for line in blob.splitlines():
            if re.search(r"loss .*->|OVERFIT|JOINT TEST|NON-ZERO|steps in|PASS|"
                         r"gradient reached|LoRA r=", line):
                print("   ", line.strip())
        results.append((label, ok, time.time() - t0))
        print(f"    -> {'PASS' if ok else 'FAIL'}  ({time.time()-t0:.0f}s)")

    print(f"\n{'='*84}\n  SUMMARY\n{'='*84}")
    for label, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:58s} {dt:6.0f}s")
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {n_ok}/{len(results)} stages passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
