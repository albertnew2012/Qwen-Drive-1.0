#!/usr/bin/env python
"""Compose one video showing every Qwen-Drive-1.0 output.

Three segments:
  1. PLANNING  — per demo scene: the camera ring, the 6 sampled trajectories drawn
                 progressively against the ground truth, the generated reasoning and
                 the VQA answer.
  2. PERCEPTION— per demo frame: the repo's own summary render (camera ring + BEV
                 detections + occupancy pair + map pair).
  3. title/section cards.

    PYTHONPATH=src python local/make_video.py --output outputs/qwen_drive_demo.mp4
"""
from __future__ import annotations

import argparse, json, sys, textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (244, 245, 247)          # light, to match the repo's own renders
PANEL = (255, 255, 255)
FG = (32, 36, 42)
DIM = (104, 112, 124)
ACCENT = (198, 104, 8)
GOOD = (22, 122, 72)
RULE = (206, 211, 218)


# ----------------------------------------------------------------- text helpers

def _font(size: int, bold: bool = False):
    for name in (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/" + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def card(lines, sub=None):
    """A full-frame title card, on the same light ground as the panels."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H // 2 - 55 * len(lines)
    for i, line in enumerate(lines):
        f = _font(76 if i == 0 else 40, bold=(i == 0))
        w = d.textbbox((0, 0), line, font=f)[2]
        d.text(((W - w) // 2, y), line, font=f, fill=FG if i == 0 else DIM)
        y += 104 if i == 0 else 58
    d.line([(W // 2 - 190, y + 12), (W // 2 + 190, y + 12)], fill=RULE, width=3)
    if sub:
        f = _font(31)
        w = d.textbbox((0, 0), sub, font=f)[2]
        d.text(((W - w) // 2, y + 40), sub, font=f, fill=ACCENT)
    return np.array(img)


def brighten(img: np.ndarray, target: float = 108.0) -> np.ndarray:
    """Display-only gamma lift for dark frames (the demo has a night scene).

    Model inputs are untouched; this only affects what the video shows.
    """
    a = np.asarray(img, dtype=np.float32)
    mean = a.mean()
    if mean >= target:
        return np.asarray(img, dtype=np.uint8)
    gamma = float(np.clip(np.log(target / 255.0) / np.log(max(mean, 1.0) / 255.0), 0.30, 1.0))
    return np.clip(255.0 * (a / 255.0) ** gamma, 0, 255).astype(np.uint8)


def fit(img: np.ndarray, box_w: int, box_h: int) -> Image.Image:
    """Resize preserving aspect to fit inside the box."""
    pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    scale = min(box_w / pil.width, box_h / pil.height)
    return pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                      Image.LANCZOS)


# ----------------------------------------------------------------- planning

def bev_panel(hist, gt, trajs, upto, px, py, dpi=110):
    """The ego-frame trajectory plot, with predictions drawn up to `upto` points."""
    fig = plt.figure(figsize=(px / dpi, py / dpi), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_color("#b0b7bf")
    ax.tick_params(colors="#5c6470", labelsize=8)
    ax.xaxis.label.set_color("#5c6470"); ax.yaxis.label.set_color("#5c6470")

    if hist is not None:
        ax.plot(hist[:, 1], hist[:, 0], color="#9aa2ae", lw=3.0, label="history", zorder=2)
    if gt is not None and upto > 0:
        n = max(1, min(upto, len(gt)))
        # full path faintly for context, and the matched-horizon part solid, so the
        # prediction is compared against the same number of elapsed seconds
        ax.plot(gt[:, 1], gt[:, 0], color="#1b1f24", lw=1.0, ls=":", alpha=0.30, zorder=2)
        ax.plot(gt[:n, 1], gt[:n, 0], color="#1b1f24", lw=2.6, ls="--",
                label="ground truth", zorder=3)
        ax.scatter([gt[n - 1, 1]], [gt[n - 1, 0]], s=30, color="#1b1f24", zorder=6)
    if upto > 0:
        for i, t in enumerate(trajs):
            n = max(1, upto)
            ax.plot(t[:n, 1], t[:n, 0], lw=3.0 if i == 0 else 1.5,
                    color="#e8590c" if i == 0 else "#4c93d8",
                    alpha=1.0 if i == 0 else 0.45, zorder=5 if i == 0 else 4,
                    label=("sample 0" if i == 0 else ("samples 1-5" if i == 1 else None)))
            ax.scatter([t[n - 1, 1]], [t[n - 1, 0]], s=42 if i == 0 else 14,
                       color="#e8590c" if i == 0 else "#4c93d8", zorder=6)
    ax.scatter([0], [0], marker="s", s=80, color="#d6336c", zorder=8, label="ego")

    allx = [np.zeros(1)]
    if hist is not None: allx.append(hist[:, 1])
    if gt is not None: allx.append(gt[:, 1])
    allx.append(trajs[:, :, 1].ravel())
    lat = np.concatenate(allx)
    c = 0.5 * (lat.min() + lat.max()); half = max(6.0, 0.62 * (lat.max() - lat.min()) * 1.15)
    ax.set_xlim(c - half, c + half)
    fwd = [np.zeros(1), trajs[:, :, 0].ravel()]
    if gt is not None: fwd.append(gt[:, 0])
    if hist is not None: fwd.append(hist[:, 0])
    f = np.concatenate(fwd)
    ax.set_ylim(min(-4.0, f.min() - 2.0), max(10.0, f.max() * 1.10))
    ax.set_xlabel("lateral y [m]   (left +)", fontsize=9)
    ax.set_ylabel("longitudinal x [m]", fontsize=9)
    ax.grid(alpha=0.30, color="#c8ced6")
    lg = ax.legend(loc="upper left", fontsize=9, facecolor="#ffffff", edgecolor="#b0b7bf")
    for t in lg.get_texts(): t.set_color("#32363f")
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def camera_strip(scene, width, height, frame_idx=-1, label=None):
    """Front-left / front / front-right at ONE timestep, side by side.

    The model sees four timesteps per view (t-1.5 s .. t=0). Showing only the last one
    hides three quarters of its input, so the video plays them in sequence.
    """
    from qwen_drive.scene import CAMERA_VIEWS
    order = ["<FRONT LEFT VIEW>", "<FRONT VIEW>", "<FRONT RIGHT VIEW>"]
    order = [v for v in order if v in scene.views] or list(CAMERA_VIEWS)
    imgs = [scene.views[v][frame_idx].load() for v in order]
    cw = width // len(imgs)
    strip = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(strip)
    for i, (v, im) in enumerate(zip(order, imgs)):
        r = fit(brighten(np.array(im)), cw - 10, height - 30)
        x = i * cw + (cw - r.width) // 2
        y = 28 + (height - 30 - r.height) // 2
        d.rectangle([x - 2, y - 2, x + r.width + 1, y + r.height + 1], outline=RULE, width=2)
        strip.paste(r, (x, y))
        lbl = v.strip("<>").replace(" VIEW", "").title()
        d.text((x, 3), lbl, font=_font(21, True), fill=DIM)
    if label:
        w = d.textbbox((0, 0), label, font=_font(22, True))[2]
        d.text((width - w - 6, 3), label, font=_font(22, True), fill=ACCENT)
    return np.array(strip)


def planning_frames(sample, record, fps, hold_s=1.6, draw_s=3.4, hist_s=2.4):
    """Yield frames animating one planning scene.

    Three phases:
      A  play the camera history, t-1.5 s -> t=0, one panel per timestep (the model sees
         four, and showing only the last hides three quarters of its input)
      B  hold the current frame and draw the predicted trajectory forward in time
      C  hold the completed trajectory
    """
    scene = sample.scene
    if "reasoning" in record:
        trajs, mode_label = record["reasoning"], "reasoning-conditioned"
    elif "direct" in record:
        trajs, mode_label = record["direct"], "direct (no rationale)"
    else:
        return
    trajs = np.asarray(trajs)
    hist = np.asarray(record["history"]) if "history" in record else None
    gt = np.asarray(record["ground_truth"]) if "ground_truth" in record else None

    strip_h = 392
    n_frames_in = scene.num_camera_frames
    dt_hist = 1.5 / max(n_frames_in - 1, 1)          # 4 frames over 1.5 s at 2 Hz
    strips = [
        camera_strip(scene, W - 56, strip_h, frame_idx=k,
                     label=("t = 0   (current)" if k == n_frames_in - 1
                            else f"t \u2212 {(n_frames_in - 1 - k) * dt_hist:.1f} s"))
        for k in range(n_frames_in)
    ]
    bev_w, bev_h = 830, 566
    text_x = 56 + bev_w + 44
    n_pts = trajs.shape[1]

    reasoning = record.get("reasoning_text") or ""
    vqa = record.get("vqa_answer") or ""
    key = "reasoning" if "reasoning" in record else "direct"
    ade, fde = record.get(f"{key}_ade"), record.get(f"{key}_fde")
    token = record.get("token", "")

    n_hist = int(hist_s * fps)
    n_draw = int(draw_s * fps)
    n_hold = int(hold_s * fps)
    cache = {}
    for k in range(n_hist + n_draw + n_hold):
        if k < n_hist:                                  # phase A: play the input history
            strip_idx = min(n_frames_in - 1, k * n_frames_in // n_hist)
            upto = 0
        else:                                           # phase B/C: predict
            strip_idx = n_frames_in - 1
            j = k - n_hist
            upto = n_pts if j >= n_draw else max(1, int(round(n_pts * (j + 1) / n_draw)))
        strip = strips[strip_idx]
        if upto not in cache:
            cache[upto] = bev_panel(hist, gt, trajs, upto, bev_w, bev_h)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((60, 16), f"PLANNING   \u00b7   {mode_label}, {len(trajs)} samples",
               font=_font(31, True), fill=ACCENT)
        d.text((60, 56), f"scene {token}   \u00b7   planner-rl   \u00b7   "
                         f"5 s horizon @ 10 Hz", font=_font(20), fill=DIM)
        img.paste(Image.fromarray(strip), (28, 86))
        img.paste(Image.fromarray(cache[upto]), (56, 86 + strip_h + 14))

        y = 86 + strip_h + 18
        if upto == 0:
            back = (n_frames_in - 1 - strip_idx) * dt_hist
            d.text((text_x, y), (f"input:  t \u2212 {back:.1f} s" if back
                                 else "input:  t = 0   (current)"),
                   font=_font(34, True), fill=FG); y += 60
            d.text((text_x, y), f"{n_frames_in} timesteps x {len(scene.views)} views "
                                f"= {n_frames_in*len(scene.views)} images",
                   font=_font(25), fill=DIM); y += 54
        else:
            t = (upto / n_pts) * 5.0
            d.text((text_x, y), f"t + {t:4.1f} s   ({upto}/{n_pts} waypoints)",
                   font=_font(34, True), fill=FG); y += 60
        if ade is not None and upto > 0:
            d.text((text_x, y), f"ADE {ade:.3f} m      FDE {fde:.3f} m",
                   font=_font(27, True), fill=GOOD); y += 54
        if reasoning:
            d.text((text_x, y), "MODEL REASONING", font=_font(21, True), fill=ACCENT); y += 32
            for line in textwrap.wrap(reasoning, width=46)[:7]:
                d.text((text_x, y), line, font=_font(23), fill=FG); y += 32
            y += 16
        if vqa:
            d.text((text_x, y), "VQA", font=_font(21, True), fill=ACCENT); y += 32
            for line in textwrap.wrap(vqa, width=46)[:9]:
                d.text((text_x, y), line, font=_font(21), fill=(70, 78, 90)); y += 29
        yield np.array(img)


# ----------------------------------------------------------------- perception

def _trim(img: np.ndarray, tol: int = 247) -> np.ndarray:
    """Crop the uniform near-white border matplotlib leaves around the figure."""
    mask = (img < tol).any(axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    pad = 6
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    return img[y0:y1, x0:x1]


def _hsplit(img: np.ndarray, min_gap: int = 24):
    """Split the square summary render at its widest all-white horizontal gap.

    render_frame stacks a camera ring + BEV block on top of an occupancy/map row.
    Fitting that near-square figure into 16:9 wastes half the frame, so it is cut
    into two shots that each fill the width.
    """
    row_is_blank = (img > 246).all(axis=(1, 2))
    runs, start = [], None
    for i, blank in enumerate(row_is_blank):
        if blank and start is None:
            start = i
        elif not blank and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(row_is_blank)))
    lo, hi = int(0.30 * len(row_is_blank)), int(0.80 * len(row_is_blank))
    inner = [r for r in runs if r[1] - r[0] >= min_gap and lo < (r[0] + r[1]) // 2 < hi]
    if not inner:
        return [img]
    a, b = max(inner, key=lambda r: r[1] - r[0])
    cut = (a + b) // 2
    return [_trim(img[:cut]), _trim(img[cut:])]


def perception_frames(frame, result, fps, hold_s=3.2):
    from qwen_drive_perception.visualize import render_frame
    full = _trim(render_frame(frame, result))
    parts = _hsplit(full)
    captions = ["camera ring + BEV detections",
                "semantic occupancy  ·  BEV map segmentation"]
    n = int((result["scores"] > 0.3).sum()) if len(result["scores"]) else 0
    for k, part in enumerate(parts):
        panel = fit(part, W - 60, H - 130)
        base = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(base)
        d.text((60, 16), "3D PERCEPTION   \u00b7   detection + occupancy + BEV map",
               font=_font(31, True), fill=ACCENT)
        sub = (f"frame {frame.token}   ({frame.dataset_type}, {len(frame.cam_order)} cameras)"
               f"   \u00b7   {n} detections over 0.3")
        if k < len(captions):
            sub += f"   \u00b7   {captions[k]}"
        d.text((60, 54), sub, font=_font(20), fill=DIM)
        base.paste(panel, ((W - panel.width) // 2,
                            92 + min(24, (H - 130 - panel.height) // 2)))
        arr = np.array(base)
        for _ in range(int((hold_s if k == 0 else hold_s * 0.85) * fps)):
            yield arr


def _clean_markdown(text: str) -> str:
    """Flatten the model's markdown so it reads as plain text in a video frame."""
    import re
    out = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        line = re.sub(r"^#{1,6}\s*", "\u25b8 ", line)          # headings -> arrow
        line = re.sub(r"^\s*[-*]\s+", "\u2022 ", line)          # bullets
        line = line.replace("**", "").replace("`", "")
        line = line.replace("\u2705", "").replace("\u274c", "").strip()
        out.append(line)
    # collapse runs of blank lines
    cleaned, prev_blank = [], False
    for line in out:
        blank = not line
        if blank and prev_blank:
            continue
        cleaned.append(line); prev_blank = blank
    return "\n".join(cleaned).strip()


# ----------------------------------------------------------------- vqa probe

def vqa_frames(scene, entry, fps, hold_s=5.0):
    """One question/answer about a single camera frame."""
    view = entry["view"]
    if view not in scene.views:
        return
    img = brighten(np.array(scene.views[view][-1].load()))
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    d.text((60, 16), f"VISUAL QUESTION ANSWERING   \u00b7   {entry['kind']}",
           font=_font(31, True), fill=ACCENT)
    d.text((60, 54), f"{view.strip('<>').replace(' VIEW','').title()} camera, current frame"
                     f"   \u00b7   VLM only, no planning expert   \u00b7   {entry['seconds']}s on CPU",
           font=_font(20), fill=DIM)

    panel = fit(img, 900, H - 190)
    px, py = 60, 104
    d.rectangle([px - 2, py - 2, px + panel.width + 1, py + panel.height + 1],
                outline=RULE, width=2)
    base.paste(panel, (px, py))

    tx = px + panel.width + 46
    y = py + 4
    d.text((tx, y), "QUESTION", font=_font(21, True), fill=ACCENT); y += 31
    for line in textwrap.wrap(entry["question"], width=48)[:4]:
        d.text((tx, y), line, font=_font(23), fill=FG); y += 31
    y += 16
    d.text((tx, y), "ANSWER", font=_font(21, True), fill=ACCENT); y += 31

    # The model answers in markdown; render it as plain text but keep the structure.
    body, avail = [], (H - 40 - y) // 26
    for raw in _clean_markdown(entry["answer"]).split("\n"):
        if not raw.strip():
            body.append(("", False)); continue
        head = raw.startswith("\u25b8 ")
        for i, line in enumerate(textwrap.wrap(raw, width=54) or [""]):
            body.append((line if i == 0 else "   " + line, head))
    note = entry.get("note")
    if note:
        avail -= 2 + len(textwrap.wrap(note, width=56))
    for line, head in body[:max(1, avail)]:
        if line:
            d.text((tx, y), line, font=_font(20, head), fill=FG if head else (58, 64, 74))
        y += 26
    if note:
        y += 12
        d.line([(tx, y), (tx + 700, y)], fill=(200, 120, 40), width=2); y += 12
        for line in textwrap.wrap(note, width=56):
            d.text((tx, y), line, font=_font(19, True), fill=(176, 96, 16)); y += 26

    arr = np.array(base)
    for _ in range(int(hold_s * fps)):
        yield arr


def summary_card(lines, fps, hold_s=6.0):
    """Closing card: what was actually measured."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((110, 70), "What this run measured", font=_font(52, True), fill=FG)
    d.line([(110, 148), (W - 110, 148)], fill=RULE, width=3)
    y = 186
    for kind, left, right in lines:
        if kind == "h":
            y += 14
            d.text((110, y), left, font=_font(27, True), fill=ACCENT); y += 46
        elif kind == "r":
            d.text((140, y), left, font=_font(24), fill=(58, 64, 74))
            d.text((1060, y), right, font=_font(24, True), fill=FG); y += 36
        else:
            y += 10
    d.text((110, H - 74), "CPU only, no GPU  \u00b7  AMD Ryzen 9 3900X  \u00b7  "
                          "reproduce: bash local/run_all.sh",
           font=_font(21), fill=DIM)
    arr = np.array(img)
    for _ in range(int(hold_s * fps)):
        yield arr


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planning", default="outputs/planning_demo")
    ap.add_argument("--perception", default="outputs/perception_demo")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--vqa", default="outputs/vqa_probe.json")
    ap.add_argument("--output", default="outputs/qwen_drive_demo.mp4")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    import subprocess
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def emit(arr, n=1):
        b = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
        for _ in range(n):
            proc.stdin.write(b)

    n_frames = 0
    lead = int(args.fps * 2.0)
    emit(card(["Qwen-Drive-1.0-4B", "one VLM \u00b7 three heads"],
              "3D perception  \u00b7  driving VQA  \u00b7  motion planning"), lead)
    n_frames += lead

    # ---- planning
    pl = Path(args.planning)
    npzs = sorted(pl.glob("*.npz")) if pl.exists() else []
    if npzs:
        seg = int(args.fps * 1.3)
        emit(card(["Motion planning"],
                  "reasoning-conditioned  \u00b7  6 samples  \u00b7  5 s @ 10 Hz"), seg)
        n_frames += seg
        from qwen_drive.benchmarks import read_scene_file
        from qwen_drive.images import ImageArchive
        archive = ImageArchive.open(args.image_archive) if args.image_archive else None
        ordered = list(read_scene_file(args.scenes, image_archive=archive))
        samples = {s.token: s for s in ordered}
        # play scenes in scene-file order (night intersection, left, right, cruise),
        # not alphabetical token order
        rank = {s.token: i for i, s in enumerate(ordered)}
        npzs = sorted(npzs, key=lambda q: rank.get(
            json.loads(q.with_suffix(".json").read_text()).get("token", q.stem)
            if q.with_suffix(".json").exists() else q.stem, 999))
        for p in npzs:
            rec = dict(np.load(p, allow_pickle=True))
            rec = {k: (v.item() if v.shape == () else v) for k, v in rec.items()}
            j = p.with_suffix(".json")
            if j.exists():
                rec.update(json.loads(j.read_text()))
            token = rec.get("token", p.stem)
            if token not in samples:
                print(f"  skip {token}: no scene"); continue
            print(f"  planning segment {token}", flush=True)
            for f in planning_frames(samples[token], rec, args.fps):
                emit(f); n_frames += 1

    # ---- vqa probe
    vq = Path(args.vqa)
    if vq.exists():
        probe = json.loads(vq.read_text())
        if probe.get("answers"):
            seg = int(args.fps * 1.3)
            emit(card(["Visual question answering"],
                      "the VLM alone  \u00b7  driving understanding + retained general ability"), seg)
            n_frames += seg
            from qwen_drive.benchmarks import read_scene_file as _rsf
            from qwen_drive.images import ImageArchive as _IA
            _arch = _IA.open(args.image_archive) if args.image_archive else None
            _scenes = {s.token: s for s in _rsf(args.scenes, image_archive=_arch)}
            sample = _scenes.get(probe.get("token"))
            if sample is not None:
                for entry in probe["answers"]:
                    print(f"  vqa segment [{entry['kind']}]", flush=True)
                    for f in vqa_frames(sample.scene, entry, args.fps):
                        emit(f); n_frames += 1

    # ---- perception
    pc = Path(args.perception)
    pnpz = sorted(pc.glob("*.npz")) if pc.exists() else []
    if pnpz:
        seg = int(args.fps * 1.3)
        emit(card(["3D perception"],
                  "300 boxes / 7 classes  \u00b7  200\u00d7200\u00d716 occupancy  \u00b7  60\u00d730 m BEV map"),
             seg)
        n_frames += seg
        from qwen_drive_perception.dataset import PerceptionFrame
        for p in pnpz:
            fd = Path(args.frames) / p.stem
            if not fd.exists():
                print(f"  skip {p.stem}: no frame dir"); continue
            print(f"  perception segment {p.stem}", flush=True)
            res = dict(np.load(p, allow_pickle=True))
            for f in perception_frames(PerceptionFrame(fd), res, args.fps):
                emit(f); n_frames += 1

    # ---- closing summary, built from whatever actually ran
    planner = Path(args.planning).name.replace("planning_demo", "planner-rl") \
                    .replace("planning_sft", "planner-sft")
    lines = [("h", f"Planning  ({planner}, 5 s @ 10 Hz, 6 samples)", "")]
    for p in npzs:
        rec = dict(np.load(p, allow_pickle=True))
        j = p.with_suffix(".json")
        meta = json.loads(j.read_text()) if j.exists() else {}
        key = "reasoning" if "reasoning_ade" in meta else "direct"
        if f"{key}_ade" in meta:
            lines.append(("r", f"{p.stem[:30]}   ({key})",
                          f"ADE {meta[f'{key}_ade']:.3f} m   FDE {meta[f'{key}_fde']:.3f} m"))
    if pnpz:
        lines.append(("h", "Perception  (boxes over 0.3 / ground truth)", ""))
        from qwen_drive_perception.dataset import PerceptionFrame as _PF
        for p in pnpz:
            fd = Path(args.frames) / p.stem
            if not fd.exists():
                continue
            r = dict(np.load(p, allow_pickle=True))
            gt = len(_PF(fd).gt["labels"])
            lines.append(("r", f"{p.stem[:30]}", f"{int((r['scores'] > 0.3).sum())} / {gt}"))
    if vq.exists() and json.loads(vq.read_text()).get("answers"):
        lines.append(("h", "Visual question answering", ""))
        for e in json.loads(vq.read_text())["answers"]:
            lines.append(("r", e["kind"], f"{e['seconds']:.0f} s on CPU"))
    if len(lines) > 1:
        for f in summary_card(lines, args.fps):
            emit(f); n_frames += 1

    proc.stdin.close(); proc.wait()
    print(f"wrote {out}  ({n_frames} frames, {n_frames/args.fps:.1f}s @ {args.fps} fps)")


if __name__ == "__main__":
    main()
