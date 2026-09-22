"""Run the whole 10 Hz distillation plan unattended, and keep running.

Pruning is finished: it took ONNX from 29,761 ms to 907 ms and then hit a floor that
fifteen separate structural changes could not move. 10 Hz is 100 ms. The remaining 9x
has to come from a smaller model, so this trains one.

    1  frames          nuScenes samples -> packed frame directories (mini now,
                       trainval as the blobs land; re-run each cycle to pick them up)
    2  teacher cache   the full six-camera model's 900-query output per frame
    3  train           distil it into the 3.6 M-parameter student
    4  export          student -> ONNX, timed
    5  evaluate        student against teacher, on held-out frames

It loops: each cycle re-indexes any newly downloaded data, caches more teacher output,
trains further from the last checkpoint, and re-measures. Every stage is resumable and
skips work already done, so killing it at any point costs at most one stage.

    nohup python local/distill/orchestrate_10hz.py > outputs/distill.log 2>&1 &
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
PY = _ROOT / ".venv" / "bin" / "python"
STATE = _ROOT / "outputs" / "distill" / "state.json"
CACHER = None            # the teacher-caching process, kept alive across cycles


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def spawn(cmd, tag, gpu, logfile):
    """Start a stage detached and return the process, for work that should overlap.

    Teacher caching is hours of GPU 1 time and training is hours of GPU 0 time, and the
    second depends on the first only in the sense that more data is better. Running them
    in sequence leaves a card idle for the whole run; running them together means the
    student trains on what has been cached so far while the rest is still being cached.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_ROOT/'src'}:{_ROOT}"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    fh = open(logfile, "a")
    log(f"  [{tag}] spawned on gpu {gpu} -> {logfile}")
    return subprocess.Popen([str(c) for c in cmd], cwd=_ROOT, env=env,
                            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)


def run(cmd, tag, timeout=None, gpu="0"):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_ROOT/'src'}:{_ROOT}"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    log(f"  [{tag}] {' '.join(str(c) for c in cmd[-6:])}")
    t0 = time.time()
    try:
        r = subprocess.run([str(c) for c in cmd], cwd=_ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"  [{tag}] timed out after {timeout}s")
        return False
    keep = [l for l in r.stdout.splitlines()
            if l.strip() and "it/s]" not in l and "Loading weights" not in l]
    for l in keep[-18:]:
        print("    " + l, flush=True)
    if r.returncode != 0:
        log(f"  [{tag}] FAILED rc={r.returncode} after {time.time()-t0:.0f}s")
        for l in r.stderr.splitlines()[-18:]:
            print("    " + l, flush=True)
        return False
    log(f"  [{tag}] ok in {time.time()-t0:.0f}s")
    return True


def count(pattern) -> int:
    return len(list(Path(_ROOT).glob(pattern)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--steps-per-cycle", type=int, default=2000)
    ap.add_argument("--cache-per-cycle", type=int, default=0, help="0 = all available")
    args = ap.parse_args()
    os.chdir(_ROOT)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"cycles": []}

    for cycle in range(1, args.cycles + 1):
        log(f"================ cycle {cycle} ================")
        rec = {"cycle": cycle, "at": time.strftime("%Y-%m-%d %H:%M:%S")}

        # 1. index whatever data exists now, mini first then trainval as it arrives
        run([PY, _ROOT/"local/distill/nusc_frames.py",
             "--root", "data/nuscenes", "--version", "v1.0-mini"], "frames-mini")
        trainval = _ROOT / "data/nuscenes_trainval"
        if (trainval / "v1.0-trainval").is_dir() and (trainval / "samples").is_dir():
            run([PY, _ROOT/"local/distill/nusc_frames.py",
                 "--root", "data/nuscenes_trainval", "--version", "v1.0-trainval"],
                "frames-trainval", timeout=3600)
        # the ego future is what supervises planning; pure metadata, no GPU
        run([PY, _ROOT/"local/distill/nusc_ego.py",
             "--root", "data/nuscenes", "--version", "v1.0-mini"], "ego-mini")
        if (trainval / "v1.0-trainval").is_dir():
            run([PY, _ROOT/"local/distill/nusc_ego.py",
                 "--root", "data/nuscenes_trainval", "--version", "v1.0-trainval"],
                "ego-trainval", timeout=3600)
        rec["frames"] = count("data/distill/frames/*/frame.json")
        rec["ego"] = count("data/distill/ego/*.npz")

        # 2. teacher output, on the other card, overlapping everything below
        global CACHER
        if CACHER is None or CACHER.poll() is not None:
            cmd = [PY, _ROOT/"local/distill/cache_teacher.py"]
            if args.cache_per_cycle:
                cmd += ["--limit", args.cache_per_cycle]
            CACHER = spawn(cmd, "teacher", "1", _ROOT/"outputs/distill_cache.log")
        else:
            log("  [teacher] still caching on gpu 1")
        rec["cached"] = count("data/distill/teacher/*.npz")

        # 3. distil
        # box target scale, refreshed as more teacher output lands
        run([PY, _ROOT/"local/distill/box_stats.py"], "box-stats")

        total = int(state.get("train_steps", 0)) + args.steps_per_cycle
        if rec["cached"] >= 16:
            # Spawned with its own log rather than run() so step-by-step progress is
            # visible while it runs; run() only surfaces output once a stage ends, which
            # for a multi-hour training stage means no visibility at all.
            trainer = spawn([PY, _ROOT/"local/distill/train_student.py",
                             "--steps", total], "train", "0",
                            _ROOT/"outputs/distill_train.log")
            rc = trainer.wait()
            if rc == 0:
                state["train_steps"] = total
                log(f"  [train] reached step {total}")
            else:
                log(f"  [train] exited rc={rc}; see outputs/distill_train.log")
        else:
            log(f"  only {rec['cached']} frames cached; skipping training this cycle")
        rec["train_steps"] = state.get("train_steps", 0)

        # 4. export and time
        run([PY, _ROOT/"local/distill/export_student.py"], "export", gpu="0")
        res = _ROOT / "outputs/distill/student_onnx.json"
        if res.exists():
            rec["onnx"] = json.loads(res.read_text())
            hz = rec["onnx"].get("hz", 0)
            log(f"  student ONNX {rec['onnx'].get('ms', 0):.2f} ms -> {hz:.1f} Hz")

        # 5. quality against the teacher
        if (_ROOT/"local/distill/eval_student.py").exists():
            run([PY, _ROOT/"local/distill/eval_student.py"], "eval", gpu="0")
            ev = _ROOT / "outputs/distill/eval.json"
            if ev.exists():
                rec["eval"] = json.loads(ev.read_text())

        state["cycles"] = (state.get("cycles") or [])[-40:] + [rec]
        STATE.write_text(json.dumps(state, indent=1))
        log(f"  cycle {cycle} done: frames={rec.get('frames')} ego={rec.get('ego')} "
            f"cached={rec.get('cached')} steps={rec.get('train_steps')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
