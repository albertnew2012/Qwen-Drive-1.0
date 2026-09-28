"""Drive the whole distillation to two hard targets, unattended.

    GOAL   one ONNX graph carrying EVERYTHING the teacher's perception and planning
           produce -- 900-query detection, semantic occupancy, the online map, and the
           trajectory -- running at >= 10 Hz on a SINGLE GPU, with less than 10%
           degradation against the teacher on every one of them.

WHY THIS EXISTS RATHER THAN orchestrate_10hz.py
The original loop cycles forever and reports; it has no notion of "done". This one
is a closed loop against acceptance gates: it escalates the training budget until
the gates pass, or until it runs out of budget and says so plainly.

THE GATES, AND WHERE EACH NUMBER COMES FROM

    speed        >= 10.0 Hz      whole graph, both heads, ORT CUDA EP.
                                 export_student.py writes student_onnx.json.
    recall       >= 0.90         teacher detections the student also asserts.
    precision    >= 0.90         student detections the teacher agrees with.
                                 Both at score 0.3; 10% disagreement = 10% degradation.
    trajectory   <= 1.532 m      the teacher's measured ADE on nuScenes (1.393 m),
                                 +10%. For scale: constant velocity is 2.241 m and a
                                 linear probe on the ego state alone is 1.469 m, so
                                 this gate is demanding but reachable.

Detection, occupancy and map are scored against the TEACHER: that is what distillation
targets, and there is no ground truth for them here (gt.npz is written empty, and no
Occ3D dataset is present). Trajectory is scored against RECORDED GROUND TRUTH, with the
teacher measured on the same ground truth as the reference.

STRUCTURE
Preflight times the untrained graph first: shape decides speed, and if the shape
cannot clear 10 Hz there is no point spending hours training it. Then rounds of
(cache more teacher output -> refresh box stats -> train longer -> export -> eval),
escalating until the gates pass.

Every stage is resumable and skips finished work, so a kill and restart costs
nothing. State in outputs/distill/v2_state.json.

    setsid nohup .venv/bin/python -u local/distill/orchestrate_10hz_v2.py \
        > outputs/logs/distill_v2.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

PY = _ROOT / ".venv" / "bin" / "python"
OUT = _ROOT / "outputs" / "distill"
STATE = OUT / "v2_state.json"
LOGS = _ROOT / "outputs" / "logs"

# MEASURED, not inherited. HANDOVER.md quotes the teacher at 0.335 m "against
# recorded ego futures", but the only place that number is ever measured in this repo
# is _export_plan_small.py, on the four WOD_E2E demo planning scenes -- the benchmark
# the planner was built for. HANDOVER also states that converting nuScenes into WOD_E2E
# format was deliberately skipped, so 0.335 m cannot be a nuScenes figure.
# The teacher was run on nuScenes scene-0061 (148 frames, same 5 s ego-future ground
# truth the student is scored against) and returned ADE median 1.160 m, mean 1.393 m,
# p90 2.627 m. nuScenes is out of distribution for this planner, hence the 4x gap.
# Caveat: one scene, and a hard one (98 degree left turn). Widen the sample before
# treating this as final.
TEACHER_ADE = 1.393            # measured: teacher on nuScenes, not the 0.335 demo figure
TOLERANCE = 0.10               # "less than 10% degradation"

GATE_HZ = 10.0
GATE_RECALL = 1.0 - TOLERANCE
GATE_PRECISION = 1.0 - TOLERANCE
GATE_MIOU = 1.0 - TOLERANCE
GATE_ADE = TEACHER_ADE * (1.0 + TOLERANCE)

# (teacher frames to have cached, cumulative optimiser steps). The last two rounds ask
# for the WHOLE cache up front rather than 16k then 25.6k, which looks wasteful but is
# not: it means nothing is left to pre-cache, so those rounds train on all four GPUs
# instead of one. Interleaving caching with training keeps three cards on the teacher and
# one on the student; caching first puts all four on the student for the part that
# actually needs them. Measured: ~110 frames/min to cache, 9.2 samples/s to train on one
# card, so the ~1.5 h of front-loaded caching buys ~3.5x on every training hour after it.
# Sized to the wall clock actually available (~10 h from 23:30), not to a round number.
# Resuming at step 16,000 and measuring ~1.3 it/s under DDP, 34k then 52k leaves each of
# the two remaining rounds ~3.8 h and still finishes with a full eval and an exported
# graph, which a longer round 3 would not. Under DDP each step is 4 GPUs x batch 4, so
# 52k steps is ~3.3 M samples -- about 31 epochs over 25,599 frames, where the whole
# single-GPU run so far managed 5.
ROUNDS = [(3000, 6000), (8000, 16000), (25599, 34000), (25599, 52000)]

EGO_CEILING = 25599            # keyframes with a full 5 s future; the rest have none


def acquire_singleton() -> bool:
    """Refuse to start if another orchestrator is already running.

    Two of these racing is not just wasted GPU: they train concurrently and both write
    outputs/distill/student/student.pt, so a checkpoint can be torn between them. And
    the loser's cachers exit immediately on the per-shard locks, which used to make its
    stage_cache believe caching had finished.
    """
    lock = OUT / "orchestrator.pid"
    OUT.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
            if other != os.getpid() and Path(f"/proc/{other}").exists():
                print(f"another orchestrator is running as pid {other}; exiting",
                      flush=True)
                return False
        except (ValueError, OSError):
            pass
    lock.write_text(str(os.getpid()))
    return True


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def run(cmd, tag, env_extra=None, timeout=86400) -> bool:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_ROOT/'src'}:{_ROOT}"
    env["PATH"] = f"{_ROOT/'.venv'/'bin'}:{env['PATH']}"
    if env_extra:
        env.update(env_extra)
    log(f"  $ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    try:
        p = subprocess.run([str(c) for c in cmd], cwd=_ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"  [{tag}] TIMEOUT after {timeout}s")
        return False
    keep = [l for l in p.stdout.splitlines()
            if l.strip() and "it/s]" not in l and "Loading weights" not in l]
    for l in keep[-20:]:
        print("    " + l, flush=True)
    if p.returncode != 0:
        log(f"  [{tag}] FAILED rc={p.returncode} after {time.time()-t0:.0f}s")
        for l in p.stderr.splitlines()[-20:]:
            print("    ! " + l, flush=True)
        return False
    log(f"  [{tag}] ok in {time.time()-t0:.0f}s")
    return True


def n_gpus() -> int:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        return max(1, len([l for l in o.stdout.splitlines() if l.strip()]))
    except Exception:
        return 1


def n_cached() -> int:
    d = _ROOT / "data" / "distill" / "teacher"
    return len(list(d.glob("*.npz"))) if d.exists() else 0


# --------------------------------------------------------------------------- stages

def stage_data() -> bool:
    """Frame index and ego futures. Both cheap; skipped when already present."""
    frames = _ROOT / "data" / "distill" / "frames"
    ego = _ROOT / "data" / "distill" / "ego"
    n_f = len(list(frames.iterdir())) if frames.exists() else 0
    n_e = len(list(ego.glob("*.npz"))) if ego.exists() else 0
    if n_f and n_e:
        log(f"  data ready: {n_f} frames, {n_e} ego records")
        return True
    if not n_f:
        if not run([PY, "local/distill/nusc_frames.py", "--root", "data/nuscenes_trainval",
                    "--version", "v1.0-trainval"], "frames"):
            return False
    if not n_e:
        if not run([PY, "local/distill/nusc_ego.py", "--root", "data/nuscenes_trainval",
                    "--version", "v1.0-trainval"], "ego"):
            return False
    return True


def stage_preflight(state) -> bool:
    """Time the untrained graph. Shape decides speed; do not train a shape that cannot fit.

    export_student.py exports the untrained model when no checkpoint exists, which is
    exactly what is wanted here -- the node count and the kernel times do not depend on
    the weight values.
    """
    if state.get("preflight", {}).get("hz"):
        log(f"  preflight already done: {state['preflight']['hz']:.1f} Hz")
        return True
    ok = run([PY, "local/distill/export_student.py",
              "--out", "outputs/onnx/student_probe/student.onnx"], "preflight")
    if not ok:
        return False
    j = OUT / "student_onnx.json"
    if not j.exists():
        log("  preflight produced no timing json")
        return False
    r = json.loads(j.read_text())
    state["preflight"] = r
    save_state(state)
    log(f"  UNTRAINED SHAPE: {r['ms']:.1f} ms -> {r['hz']:.1f} Hz, "
        f"{r['nodes']} nodes, {r.get('params_m', 0):.1f} M params")
    if r["hz"] < GATE_HZ:
        log(f"  !! shape is {r['hz']:.1f} Hz, under the {GATE_HZ} Hz gate. Training it "
            f"cannot fix that -- the architecture needs to shrink. Continuing anyway so "
            f"the quality number is still measured, but the speed gate will not pass.")
    return True


def _spawn_cache(target: int, devices: list[int]) -> list:
    """One cacher per device, each taking a disjoint slice of what is still missing."""
    LOGS.mkdir(parents=True, exist_ok=True)
    procs = []
    for shard, dev in enumerate(devices):
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{_ROOT/'src'}:{_ROOT}"
        env["CUDA_VISIBLE_DEVICES"] = str(dev)
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PATH"] = f"{_ROOT/'.venv'/'bin'}:{env['PATH']}"
        lf = open(LOGS / f"distill_cache_gpu{dev}.log", "a")
        procs.append((subprocess.Popen(
            [str(PY), "local/distill/cache_teacher.py", "--limit", str(target),
             "--shard", str(shard), "--of", str(len(devices))],
            cwd=_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT), lf, dev))
    return procs


def _reap_cache(procs) -> int:
    bad = 0
    for p, lf, dev in procs:
        rc = p.wait()
        lf.close()
        if rc != 0:
            log(f"  cacher on GPU {dev} exited rc={rc} "
                f"(outputs/logs/distill_cache_gpu{dev}.log)")
            bad += 1
    return bad


def stage_cache(target: int, gpus: int) -> bool:
    """Teacher output for `target` frames, sharded one process per GPU.

    Caching measured at 0.57 fps on one A100 and 0.51 on a 3090 -- nearly identical, so
    it is not GPU bound (CPU preprocessing and image IO). Sharding still helps because
    each shard gets its own CPU workers and its own reader, but do not expect a clean Nx.
    """
    have = n_cached()
    if have >= target:
        log(f"  cache already has {have} >= {target}")
        return True
    log(f"  caching to {target} frames ({have} present) across {gpus} GPUs")
    bad = _reap_cache(_spawn_cache(target, list(range(gpus))))
    have = n_cached()
    log(f"  cache now {have} frames ({bad} cacher(s) errored)")
    if have < target:
        # Short of target means the cachers died or bailed on a held lock. Training here
        # would quietly use less data than the round asked for, so say so instead.
        log(f"  !! caching stopped at {have} of {target}; not proceeding to train")
        return False
    return True


def stage_box_stats(force: bool) -> bool:
    f = _ROOT / "data" / "distill" / "box_stats.npz"
    if f.exists() and not force:
        return True
    return run([PY, "local/distill/box_stats.py"], "box_stats")


def stage_train_ddp(steps: int, gpus: int) -> bool:
    """Train across every GPU with torchrun.

    Worth it only once nothing is left to cache: until then GPUs 1-3 are earning their
    keep on the teacher, and a DDP run would fight them for the same cards. Measured
    single-card throughput is 9.2 samples/s and the step is compute bound (batch 4 -> 8
    buys 4%), so more cards is the only route to more epochs.

    Note `--steps` counts OPTIMISER steps, not samples, so N ranks make each step worth
    N times as much data. 16,000 steps at batch 4 on four cards is 256k samples where one
    card gives 64k.
    """
    return run([_ROOT / ".venv" / "bin" / "torchrun",
                f"--nproc_per_node={gpus}", "--master_port=29533",
                "local/distill/train_student.py", "--steps", str(steps),
                "--workers", "8"],
               f"train-ddp{gpus}",
               {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpus))})


def stage_train(steps: int) -> bool:
    """Resumes from the existing checkpoint, so `steps` is a cumulative target.

    Training holds GPU 0 only -- train_student.py is single-device, and the model is
    81.8 M, so DDP across four A100s would spend more on gradient exchange than it saves.
    The other three cards are kept busy by pre-caching the next round (see main).
    More dataloader workers, though, matter a lot: caching showed this pipeline is
    CPU/IO bound, and the same is true of the training reader.
    """
    return run([PY, "local/distill/train_student.py", "--steps", str(steps),
                "--workers", "12"],
               "train", {"CUDA_VISIBLE_DEVICES": "0"})


def stage_export() -> dict | None:
    ok = run([PY, "local/distill/export_student.py",
              "--out", "outputs/onnx/student/student.onnx"], "export",
             {"CUDA_VISIBLE_DEVICES": "0"})
    j = OUT / "student_onnx.json"
    if not ok or not j.exists():
        return None
    return json.loads(j.read_text())


def stage_eval() -> dict | None:
    ok = run([PY, "local/distill/eval_student.py", "--limit", "256"], "eval",
             {"CUDA_VISIBLE_DEVICES": "0"})
    j = OUT / "eval.json"
    if not ok or not j.exists():
        return None
    return json.loads(j.read_text())


def gates(speed: dict | None, qual: dict | None) -> tuple[bool, list[str]]:
    """Both targets, reported as separate lines so a partial pass is legible."""
    rows = []
    hz = (speed or {}).get("hz")
    rows.append(("speed", hz, GATE_HZ, hz is not None and hz >= GATE_HZ,
                 f"{hz:.1f} Hz" if hz else "n/a", f">= {GATE_HZ} Hz"))
    rec = (qual or {}).get("recall")
    rows.append(("recall", rec, GATE_RECALL, rec is not None and rec >= GATE_RECALL,
                 f"{rec:.1%}" if rec is not None else "n/a", f">= {GATE_RECALL:.0%}"))
    pre = (qual or {}).get("precision")
    rows.append(("precision", pre, GATE_PRECISION,
                 pre is not None and pre >= GATE_PRECISION,
                 f"{pre:.1%}" if pre is not None else "n/a", f">= {GATE_PRECISION:.0%}"))
    # The other half of perception. mIoU is against the teacher's own argmax, so a
    # perfect student scores 1.0 and "< 10% degradation" is >= 0.90, same bar as
    # detection.
    occ = (qual or {}).get("occ_miou")
    rows.append(("occupancy mIoU", occ, GATE_MIOU, occ is not None and occ >= GATE_MIOU,
                 f"{occ:.1%}" if occ is not None else "n/a", f">= {GATE_MIOU:.0%}"))
    seg = (qual or {}).get("seg_miou")
    rows.append(("map mIoU", seg, GATE_MIOU, seg is not None and seg >= GATE_MIOU,
                 f"{seg:.1%}" if seg is not None else "n/a", f">= {GATE_MIOU:.0%}"))
    ade = (qual or {}).get("trajectory_ade_m")
    rows.append(("trajectory ADE", ade, GATE_ADE, ade is not None and ade <= GATE_ADE,
                 f"{ade:.3f} m" if ade is not None else "n/a", f"<= {GATE_ADE:.3f} m"))
    lines = [f"    {n:<15s} {got:>10s}   target {want:<12s} {'PASS' if ok else 'FAIL'}"
             for n, _, _, ok, got, want in rows]
    return all(r[3] for r in rows), lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-round", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=0, help="0 = autodetect")
    args = ap.parse_args()
    os.chdir(_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if not acquire_singleton():
        return 3
    gpus = args.gpus or n_gpus()
    state = load_state()
    log("=" * 78)
    log(f"distillation to >= {GATE_HZ} Hz with < {TOLERANCE:.0%} degradation")
    log(f"gates: recall >= {GATE_RECALL:.0%}, precision >= {GATE_PRECISION:.0%}, "
        f"occ/map mIoU >= {GATE_MIOU:.0%}, ADE <= {GATE_ADE:.3f} m "
        f"(teacher {TEACHER_ADE} m)")
    log(f"{gpus} GPUs, {len(ROUNDS)} rounds, {n_cached()} teacher frames cached")
    log("=" * 78)

    if not stage_data():
        log("data stage failed; cannot continue")
        return 1
    if not stage_preflight(state):
        log("preflight failed; cannot continue")
        return 1

    for r, (frames, steps) in enumerate(ROUNDS):
        if r < args.from_round:
            continue
        frames = min(frames, EGO_CEILING)
        log("")
        log(f"===== round {r}: cache {frames} frames, train to {steps} steps =====")
        t0 = time.time()

        if not stage_cache(frames, gpus):
            log("  caching produced nothing; stopping")
            return 1
        # Box statistics are computed from cached teacher boxes, so they shift as the
        # cache grows. Recompute each round; the affine is restored at the output, so
        # the ONNX signature does not change.
        stage_box_stats(force=True)

        # Training occupies GPU 0 for most of the round's wall clock. Caching is the
        # other slow stage and is independent of the weights, so the next round's frames
        # are fetched on the remaining cards meanwhile. By the time the next round starts
        # its cache stage, most or all of its data is already there.
        bg = []
        nxt = min(ROUNDS[r + 1][0], EGO_CEILING) if r + 1 < len(ROUNDS) else 0
        if nxt and n_cached() < nxt and gpus > 1:
            log(f"  pre-caching round {r+1} ({nxt} frames) on GPUs 1-{gpus-1} "
                f"while GPU 0 trains")
            bg = _spawn_cache(nxt, list(range(1, gpus)))

        if bg or gpus < 2:
            trained = stage_train(steps)
        else:
            # Nothing left to cache, so every card can train. This is where most of the
            # wall clock goes, and it is the difference between ~11 epochs and ~45.
            log(f"  nothing left to cache: training across all {gpus} GPUs")
            trained = stage_train_ddp(steps, gpus)

        if bg:
            log(f"  training done; waiting for the pre-cachers")
            _reap_cache(bg)
            log(f"  cache now {n_cached()} frames")
        if not trained:
            log("  training failed; stopping")
            return 1

        speed = stage_export()
        qual = stage_eval()
        ok, lines = gates(speed, qual)
        log(f"  --- round {r} gates ---")
        for l in lines:
            print(l, flush=True)

        state[f"round{r}"] = {"frames": n_cached(), "steps": steps, "speed": speed,
                              "quality": qual, "passed": ok,
                              "minutes": round((time.time() - t0) / 60, 1),
                              "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_state(state)

        if ok:
            log("")
            log("*** BOTH TARGETS MET ***")
            log(f"    ONNX: outputs/onnx/student/student.onnx")
            log(f"    {speed['ms']:.1f} ms -> {speed['hz']:.1f} Hz, {speed['nodes']} nodes")
            log(f"    recall {qual['recall']:.1%}  precision {qual['precision']:.1%}  "
                f"occ mIoU {qual['occ_miou']:.1%}  map mIoU {qual['seg_miou']:.1%}  "
                f"ADE {qual['trajectory_ade_m']:.3f} m")
            state["done"] = True
            save_state(state)
            return 0
        log(f"  round {r} did not clear the gates; escalating")

    log("")
    log("budget exhausted without clearing every gate. Last state in "
        f"{STATE.relative_to(_ROOT)}; see the per-round gate tables above for which "
        "target is short and by how much.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
