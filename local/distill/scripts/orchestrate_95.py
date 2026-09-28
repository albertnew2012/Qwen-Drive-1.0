"""Unattended plan toward: student ONNX >= 10 Hz on one A100, >= 95% of the teacher's F1 on
held-out val scenes. Rounds: 0 = FINAL (896) + R1 (1152) already running; 1 = lanes from
orchestrate_lanes.json (re-read every loop, so lanes not yet launched can be edited);
2 = one 8-GPU run combining what helped. Every lane runs through lane.sh (speed gate, smoke,
self-check, resume on early death, evaluation, official metrics, export, timing). Failures
retry up to 3 times. REPORT.md is rewritten after every finished lane. Loop: 5 min.
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]; os.chdir(ROOT)
S, LOGS, O = "local/distill/scripts", "outputs/logs", "outputs/distill/orchestrate"
NB, CFG = "outputs/distill/lab_notebook.json", f"{S}/orchestrate_lanes.json"
Path(O).mkdir(parents=True, exist_ok=True)
T0 = time.time()

def log(m): print(f"[{time.strftime('%m-%d %H:%M:%S')}] ORCH: {m}", flush=True)
def read(p):
    try: return Path(p).read_text(errors="replace")
    except Exception: return ""
def cfg(): return json.load(open(CFG))
def alive_script(name):
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try: c = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except Exception: continue
        if c.startswith("bash ") and name in c: return True
    return False

def nb():
    try: return json.load(open(NB))
    except Exception: return []
def best_eval(tags):
    """Best scene-split detection record over the given tag family (+ -nms2)."""
    fam = set(tags) | {t + "-nms2" for t in tags} | {t + "-avg" for t in tags}
    best = None
    for r in nb():
        tag = str(r.get("tag", ""))
        mid = any(tag.startswith(t + "-mid") and not tag.endswith("-official") for t in tags) and r.get("frames") == 250
        if (tag in fam or mid) and r.get("eval_tokens") == "scene" and not r.get("official"):
            f1 = r["student_vs_gt_best_thr"]["f1"]
            if best is None or f1 > best["f1"]:
                o3 = r.get("occ3d") or {}
                best = {"f1": f1, "teacher_f1": r["teacher_vs_gt"]["f1"], "retained": r.get("retained_pct"),
                        "ade": r.get("trajectory_ade_m"), "occ": r.get("occ_miou_occupied"), "map": r.get("map_miou"),
                        "occ3d_s": o3.get("student_miou"), "occ3d_t": o3.get("teacher_miou"),
                        "step": r.get("step"), "ckpt": r.get("ckpt"), "tag": r.get("tag")}
    return best
def official(tags, ckpt=None):
    """Official record for the lane; if the best detection record is a snapshot, its own official record."""
    out = None
    want = {t + "-official" for t in tags}
    if ckpt and "snap_" in str(ckpt):
        step = Path(ckpt).stem.split("_")[1]
        want = {t + f"-mid{step}-official" for t in tags}
    for r in nb():
        if r.get("official") and r.get("tag") in want and r.get("subset", "all") == "all":
            out = {"mAP": r["mAP"], "NDS": r["NDS"]}
    return out

def onnx_for(name, ckpt):
    """Where the ONNX of a record's checkpoint lives (lane export, or a snapshot export)."""
    special = {"final": "outputs/onnx/student_final/student.onnx", "r1": "outputs/onnx/student_r1/student.onnx"}
    if ckpt and "snap_" in str(ckpt):
        step = int(Path(ckpt).stem.split("_")[1])
        base = f"outputs/onnx/{name}_snap{step // 1000}k"
    else:
        base = special.get(name, f"outputs/onnx/{name}/student.onnx").rsplit("/", 1)[0]
    # A re-export of the same weights with a faster graph (e.g. the de-contended scatter of
    # 2026-09-27) lives next to the original as <dir>_v2, _v3 ...: newest version wins, and the
    # original file is never touched.
    cands = sorted(Path(base).parent.glob(Path(base).name + "_v[0-9]*"), key=lambda q: int(q.name.rsplit("_v", 1)[1]))
    return f"{cands[-1]}/student.onnx" if cands and (cands[-1] / "student.onnx").exists() else f"{base}/student.onnx"
def hz(name, gpu):
    """Best available timing of the lane's graph shape on an idle H200: the launch speed gate
    (untrained shape, GPU idle), the lane's export, or any snapshot export. Timings taken while
    the GPU was training another lane are lower bounds and must not decide the ranking."""
    vals = []
    for p in [Path(f"outputs/distill/exp/{name}/onnx_shape_{gpu}.json"), Path(f"outputs/distill/exp/{name}/onnx_{gpu}.json")] + \
             sorted(Path("outputs/onnx").glob(f"{name}_snap*/onnx_{gpu}.json")):
        if p.exists():
            try: vals.append(float(json.load(open(p)).get("hz")))
            except Exception: pass
    return max(vals) if vals else None

# ---------------- lane bookkeeping ----------------
lanes = {}   # name -> dict(status, tries, gpus, log, external)
def add_external(name, gpus, done_marker, fail_markers, logf, script, retry):
    lanes[name] = dict(status="running", tries=0, gpus=gpus, log=logf, external=True, done_marker=done_marker,
                       fail_markers=fail_markers, script=script, retry=retry, tags=[name] + ([retry["name"]] if retry else []))
add_external("final", "0,1,2,3", "FINAL_DONE", ["TRAINING_ENDED_EARLY", "FINAL_LAUNCH_FAILED"], f"{LOGS}/final_post_8gpu.log",
             "final_post_8gpu.sh", dict(name="final", gpus="0,1,2,3", nproc=4, steps=80000, extra="--image-size 896 512"))
add_external("r1", "4,5,6,7", "R1_DONE", ["R1_SMOKE_FAILED", "R1_LAUNCH_FAILED"], f"{LOGS}/r1.log",
             "r1_hires.sh", dict(name="r1_hires", gpus="4,5,6,7", nproc=4, steps=80000, extra="--image-size 1152 640"))
lanes["r1"]["tags"] = ["r1", "r1_hires"]; lanes["final"]["tags"] = ["final"]
# Reload persisted state: a restart must re-attach to lanes that are already running (their
# drivers are independent nohup processes) instead of launching them again.
_status = Path(O) / "status.json"
if _status.exists():
    try:
        for name, st in json.load(open(_status)).items():
            if name in ("final", "r1"):
                for k in ("status", "tries", "log", "external"): lanes[name][k] = st.get(k, lanes[name].get(k))
            else:
                lanes[name] = st
        log(f"reloaded state for {sorted(lanes)}: " + ", ".join(f"{n}={l['status']}" for n, l in lanes.items()))
    except Exception as ex:
        log(f"could not reload status.json ({ex}); starting clean")

def launch(spec, resolved_extra):
    name = spec["name"]; ln = lanes.setdefault(name, dict(status="pending", tries=0, gpus=spec["gpus"], external=False, tags=[name]))
    ln["tries"] += 1; ln["status"] = "running"; ln["gpus"] = spec["gpus"]; ln["extra"] = resolved_extra; ln["steps"] = spec["steps"]
    ln["log"] = f"{LOGS}/lane_{name}_try{ln['tries']}.log"
    cmd = ["bash", f"{S}/lane.sh", name, spec["gpus"], str(spec["nproc"]), str(spec["steps"]), resolved_extra, ln["log"]]
    subprocess.Popen(["nohup"] + cmd, stdout=open(ln["log"], "a"), stderr=subprocess.STDOUT, start_new_session=True)
    log(f"launched {name} (try {ln['tries']}) on GPUs {spec['gpus']}: steps {spec['steps']} extra '{resolved_extra}' -> {ln['log']}")

def refresh(name):
    ln = lanes[name]
    if ln["status"] != "running": return
    t = read(ln["log"])
    if ln.get("external"):
        if ln["done_marker"] in t: ln["status"] = "done"; log(f"{name}: DONE"); return
        failed = any(m in t for m in ln["fail_markers"]) or re.search(r"TRAIN_RC=[1-9]", t)
        gone = not alive_script(ln["script"]) and not alive_script("lane.sh " + ln["retry"]["name"]) and "LANE_DONE" not in t
        if "LANE_DONE" in t: ln["status"] = "done"; log(f"{name}: DONE (via retry lane)"); return
        if failed or gone:
            if ln["tries"] >= 3: ln["status"] = "failed"; log(f"{name}: FAILED permanently"); return
            r = ln["retry"]; ln["tries"] += 1
            ln["log"] = f"{LOGS}/lane_{r['name']}_try{ln['tries']}.log"; ln["external"] = False
            cmd = ["bash", f"{S}/lane.sh", r["name"], r["gpus"], str(r["nproc"]), str(r["steps"]), r["extra"], ln["log"]]
            subprocess.Popen(["nohup"] + cmd, stdout=open(ln["log"], "a"), stderr=subprocess.STDOUT, start_new_session=True)
            log(f"{name}: {'failed' if failed else 'driver gone'} -> relaunched as lane {r['name']} (resume-capable), try {ln['tries']}")
        return
    if "LANE_DONE" in t: ln["status"] = "done"; log(f"{name}: DONE")
    elif "LANE_SKIPPED" in t: ln["status"] = "skipped"; log(f"{name}: SKIPPED (speed gate)")
    elif "LANE_FAILED" in t or "LANE_INCOMPLETE" in t or not alive_script(f"lane.sh {name}") and "LANE_DONE" not in t:
        if ln["tries"] >= 3: ln["status"] = "failed"; log(f"{name}: FAILED permanently")
        else:
            spec = next((s for s in cfg()["round1"] + cfg().get("backlog", []) + [cfg()["round2"]] if s["name"] == name), None)
            if spec: log(f"{name}: failed/incomplete -> retry"); launch(spec, ln["extra"])
            else: ln["status"] = "failed"

def finished(name): return name in lanes and lanes[name]["status"] in ("done", "skipped", "failed")
def gpus_free(gpus):
    want = set(gpus.split(","))
    return all(not (set(l["gpus"].split(",")) & want) for n, l in lanes.items() if l["status"] == "running")

def resolution():
    if cfg().get("force_res"):
        return cfg()["force_res"]
    f, r = best_eval(["final"]), best_eval(["r1", "r1_hires"])
    if f and r and r["f1"] >= f["f1"] + 0.01: return "1152 640"
    return "896 512"

def report():
    rows = []
    for name, ln in lanes.items():
        e = best_eval(ln["tags"]); o = official(ln["tags"], e and e["ckpt"])
        rows.append(dict(name=name, status=ln["status"], f1=e and e["f1"], teacher=e and e["teacher_f1"], retained=e and e["retained"],
                         mAP=o and o["mAP"], NDS=o and o["NDS"], ade=e and e["ade"], occ=e and e["occ"], map=e and e["map"],
                         occ3d_s=e and e.get("occ3d_s"), occ3d_t=e and e.get("occ3d_t"),
                         a100=None, h200=(hz(ln["tags"][-1], "h200") or hz(name, "h200")),
                         ckpt=e and e["ckpt"], step=e and e["step"]))
    ok = [r for r in rows if r["retained"] is not None and (r["h200"] is None or r["h200"] >= 10)]
    best = max(ok, key=lambda r: r["retained"]) if ok else None
    tgt = cfg()["target_retained_pct"]
    t_ade = None
    for r in nb():
        if r.get("tag") == "teacher-traj-valscenes-egoframes": t_ade = r.get("trajectory_ade_m")
    f = lambda v, s="{:.1f}": "-" if v is None else s.format(v)
    md = ["# Student vs teacher -- live report (rewritten by orchestrate_95.py)", "",
          f"Updated {time.strftime('%Y-%m-%d %H:%M')}. Target: >= {tgt}% of the teacher's F1 on the 150 held-out val scenes "
          "(250 frames, 2 m centre distance, 10 nuScenes classes, official GT filter) and >= 10 Hz on one H200 (all five outputs, one graph).", "",
          f"| lane | status | F1 | teacher F1 | retained % | official mAP | NDS | traj ADE m (teacher {f(t_ade, '{:.3f}')}) | occ vs Occ3D GT (student / teacher) | occ agree | map agree | H200 Hz |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['status']} | {f(r['f1'], '{:.3f}')} | {f(r['teacher'], '{:.3f}')} | {f(r['retained'])} | {f(r['mAP'], '{:.3f}')} | "
                  f"{f(r['NDS'], '{:.3f}')} | {f(r['ade'], '{:.3f}')} | {f(r['occ3d_s'], '{:.3f}')} / {f(r['occ3d_t'], '{:.3f}')} | {f(r['occ'], '{:.3f}')} | {f(r['map'], '{:.3f}')} | {f(r['h200'])} |")
    md += ["", "H200 Hz = the graph's best timing on an idle GPU (launch speed gate or export); numbers taken while "
           "another lane was training on the same GPU are lower bounds and are not used. The deliverable's timing is "
           "re-measured on an idle GPU at the end."]
    if best:
        hit = best["retained"] >= tgt
        md += ["", f"**Best so far:** `{best['name']}` -- {best['retained']:.1f}% retained ({best['f1']:.3f} vs teacher {best['teacher']:.3f}), "
               f"H200 {f(best['h200'])} Hz. Target {'MET' if hit else 'not met'}.",
               f"Checkpoint `{best['ckpt']}`, ONNX `{onnx_for(best['name'], best['ckpt'])}`."]
        src = onnx_for(best["name"], best["ckpt"])
        if Path(src).exists():
            # Versioned copy per (lane, step): nothing already written is ever replaced, so every
            # deliverable that was ever "best" stays inspectable; deliverable.json points at the
            # current one and deliverable_history.jsonl keeps the sequence.
            Path("outputs/onnx/deliverable").mkdir(parents=True, exist_ok=True)
            step = "".join(ch for ch in str(best.get("step", "")) if ch.isdigit()) or "final"
            ver = ("_" + Path(src).parent.name.rsplit("_", 1)[1]) if "_v" in Path(src).parent.name else ""
            dst = Path(f"outputs/onnx/deliverable/student_{best['name']}_{step}{ver}.onnx")
            if not dst.exists():
                shutil.copy(src, dst)
                with open("outputs/onnx/deliverable/deliverable_history.jsonl", "a") as fh:
                    fh.write(json.dumps(dict(best, onnx=src, copy=str(dst), at=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
            json.dump(dict(best, onnx=src, copy=str(dst), target_pct=tgt, target_met=hit), open("outputs/onnx/deliverable/deliverable.json", "w"), indent=1)
    Path("local/distill/REPORT.md").write_text("\n".join(md) + "\n")
    try:   # keep the per-experiment ledger in step with the notebook (never raises into the loop)
        subprocess.run([".venv/bin/python", "local/distill/experiments_ledger.py"], capture_output=True, timeout=120)
    except Exception as e:
        print(f"ledger: {e}", flush=True)
    json.dump({n: {k: v for k, v in l.items() if k != "proc"} for n, l in lanes.items()}, open(f"{O}/status.json", "w"), indent=1)
    return best

# ---------------- main loop ----------------
log("started; round 0 = FINAL (896, GPUs 0-3) + R1 (1152, GPUs 4-7) already running")
round2_launched = False; last_report = 0
while True:
    for n in list(lanes): refresh(n)
    c = cfg()
    # round 1: launch lanes whose prerequisites finished and GPUs are free
    for spec in c["round1"]:
        name = spec["name"]
        if name in lanes: continue
        if not all(finished(a) for a in spec["after"]): continue
        if not gpus_free(spec["gpus"]): continue
        res = resolution() if finished("r1") else "896 512"
        extra = spec["extra"].replace("RES", res)
        dup = any(l.get("extra") == extra and l.get("steps") == spec["steps"] for l in lanes.values())
        if dup:
            lanes[name] = dict(status="skipped", tries=0, gpus=spec["gpus"], external=False, tags=[name], extra=extra, steps=spec["steps"], log="")
            log(f"{name}: skipped, identical to a launched lane ({extra}, {spec['steps']} steps)"); continue
        launch(spec, extra)
    # backlog: keep every free half busy while round 1 is still going
    r1_names = [s["name"] for s in c["round1"]]
    r1_all_done = all(finished(n) for n in ["final", "r1"] + r1_names)
    if not r1_all_done and not round2_launched:
        for spec in c.get("backlog", []):
            name = spec["name"]
            if name in lanes: continue
            pending_here = any(s["name"] not in lanes and set(s["gpus"].split(",")) & set(spec["gpus"].split(",")) for s in c["round1"])
            if pending_here or not gpus_free(spec["gpus"]): continue
            if not all(finished(a) for a in spec.get("after", [])): continue
            extra = spec["extra"].replace("RES", resolution() if finished("r1") else "896 512")
            launch(spec, extra)
    # round 2: after everything in round 1 (and round 0) finished; takes whatever half is free
    if not round2_launched and r1_all_done:
        free = [g for g in "01234567" if gpus_free(g)]
        if len(free) >= 4:
            c["round2"] = dict(c["round2"], gpus=",".join(free), nproc=len(free))
        else:
            free = []
    if not round2_launched and r1_all_done and len(free) >= 4 and c["round2"].get("steps", 0) > 0:
        base = best_eval(["final", "r1", "r1_hires"]); base_f1 = base["f1"] if base else 0
        res = resolution(); flags = [f"--image-size {res}"]
        for s in c["round1"] + c.get("backlog", []):
            e = best_eval([s["name"]])
            if e and e["f1"] >= base_f1 + 0.01:
                for tok in ("--arch", "--history"):
                    m = re.search(tok + r" (\S+)", lanes[s["name"]].get("extra", ""))
                    if m and tok not in " ".join(flags): flags += [tok, m.group(1)]
                log(f"round 2 keeps from {s['name']}: {lanes[s['name']].get('extra')} (F1 {e['f1']:.3f} vs base {base_f1:.3f})")
        spec = dict(c["round2"]); spec["extra"] = " ".join(flags)
        hours_left = (time.mktime(time.strptime(c["deadline"], "%Y-%m-%d %H:%M")) - time.time()) / 3600
        rate = 0.9 if "--history" in spec["extra"] else 1.3        # it/s at 1152 on 4 H200 (K=3 pays 3 extra no-grad passes)
        spec["steps"] = int(min(spec["steps"], max(20000, (hours_left - 4) * 3600 * rate)))
        log(f"round 2: {hours_left:.1f} h left -> {spec['steps']} steps, flags '{spec['extra']}'")
        launch(spec, spec["extra"]); round2_launched = True
    best = report()
    if best and best["retained"] >= c["target_retained_pct"] and (best["h200"] or 0) >= 10 and time.time() - last_report > 3600:
        log(f"TARGET MET by {best['name']}: {best['retained']:.1f}% retained, H200 {best['h200']:.1f} Hz (plan continues for margin)"); last_report = time.time()
    if (round2_launched and finished(c["round2"]["name"])) or \
       (c["round2"].get("steps", 0) == 0 and r1_all_done and all(finished(s["name"]) for s in c.get("backlog", []) if s["name"] in lanes)
        and all(s["name"] in lanes for s in c.get("backlog", []))):
        report(); log("ORCH_DONE"); break
    time.sleep(300)
