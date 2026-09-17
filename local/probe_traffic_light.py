#!/usr/bin/env python
"""Does the model associate a traffic light with the ego lane, or just report colours?

Five questions per frame, over nuScenes frames chosen so that the *correct* answer
differs between them. Two frames show a red light with the ego stopped; one shows a
green light with the ego crossing at 12 m/s while a red light for another direction
is also in view. A model that pattern-matches "intersection -> red -> stop" passes
the first two and fails the third, which is the point of including it.

    .venv/bin/python local/probe_traffic_light.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from qwen_drive import QwenDriveForPlanning

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# (file stem, what is actually true, ego speed m/s)
FRAMES = [
    ("scene-0553_f20_CAM_FRONT",
     "ego STOPPED at stop line. Two RED lamps on the overhead mast arm face ego. "
     "A cluster of unlit/side-facing lights sits upper-right. 'NO TURN ON RED' sign "
     "on the right pole. Pedestrians crossing in front.", 0.0),
    ("scene-0757_f24_CAM_FRONT",
     "ego STOPPED at stop line. Ego's lights are RED. Cross traffic (white truck, "
     "red Coca-Cola van) is moving through the junction, so the cross street is green.", 0.0),
    ("scene-0796_f16_CAM_FRONT",
     "ego CROSSING at 12 m/s. Ego's signal ahead is GREEN. A RED lamp for a "
     "different approach is also visible to the right. Control case.", 12.0),
]

QUESTIONS = [
    ("canonical",
     "What is the state of the traffic light ahead, and what should the ego vehicle do?"),
    ("association",
     "Several traffic lights may be visible in this image. Which one governs the lane "
     "the ego vehicle is travelling in, and what colour is it? Ignore any traffic "
     "lights that face a different direction."),
    ("enumerate",
     "List every traffic light you can see. For each one, give its colour and say "
     "whether it controls the ego vehicle's direction of travel or a different direction."),
    ("regulatory",
     "Is the ego vehicle allowed to turn right at this intersection at this moment? "
     "Quote any sign or signal that supports your answer."),
    ("action",
     "Should the ego vehicle proceed or stay stopped? Start your answer with either "
     "PROCEED or STOP, then give one sentence of justification."),
]

# Same association questions, but on 3x-upscaled crops around the signal cluster.
# If the answers improve, the limit is sensor resolution rather than reasoning.
ZOOM_FRAMES = [
    ("scene-0796_f16_CROP",
     "3x crop of the control frame. GREEN on the mast arm over ego's carriageway; "
     "a RED lamp for a different approach on the right.", 12.0),
    ("scene-0553_f20_CROP",
     "3x crop. The two RED lamps on the mast arm that face ego.", 0.0),
]

ZOOM_QUESTIONS = [q for q in QUESTIONS if q[0] in ("canonical", "association", "enumerate")]

# bbox_3d appears nowhere in this repo - the model volunteered it unprompted, so
# ask for it directly to see whether traffic lights are a grounded class.
GROUNDING_QUESTIONS = [
    ("ground_all",
     "Detect every traffic light in this image and return them as JSON with a "
     "bbox_3d field and a label field."),
    ("ground_relevant",
     "Detect only the traffic lights that control the ego vehicle's lane. Return "
     "JSON with a bbox_3d field, a label field, and a colour field."),
]

# scene-0796 kf17-25 is the one frame in the whole mini split where two lanes carry
# different signal states at once: ego's lane is marked with a straight arrow and is
# governed by a GREEN mast-arm head, while the lanes to the right are marked with
# right-turn arrows and are governed by two RED heads on the right. Asking which light
# applies to which movement separates real lane binding from "report the nearest lamp".
LANE_TRUTH = ("ego lane carries a painted STRAIGHT arrow and is governed by the GREEN "
              "head on the mast arm ahead. The lanes to the right carry painted RIGHT-TURN "
              "arrows and are governed by two RED heads on the right. Double yellow on the left.")
LANE_FRAMES = [(f"scene-0796_f{kf:02d}_CAM_FRONT", LANE_TRUTH, 12.0) for kf in (17, 21, 25)]

LANE_QUESTIONS = [
    ("lanes",
     "How many lanes are there in the ego vehicle's direction of travel, which one is the "
     "ego vehicle in, and what marking is painted on the road in that lane?"),
    ("which_light",
     "A green traffic light and a red traffic light are both visible. Which one applies to "
     "the ego vehicle's lane and which applies to a different lane? Explain how the painted "
     "lane markings tell you."),
    ("turn_right",
     "If the ego vehicle wanted to turn right here instead of going straight, which traffic "
     "light would it have to obey, and what colour is that light right now?"),
    ("both_movements",
     "Answer both: (a) may the ego vehicle proceed straight, and under which light? "
     "(b) may it turn right, and under which light?"),
]

# The lane battery above asserts a red light exists. It does not: HSV blob detection and
# an 8x crop both show the right-hand head at x1260 y477 is GREEN, and the red pixels
# nearby are a red car's roof. Two of three frames accepted that false premise and
# invented a lane assignment for the non-existent red. These questions assert nothing,
# so anything the model volunteers about lane binding is its own.
NEUTRAL_TRUTH = ("ego lane carries a painted STRAIGHT arrow. Every lit signal head in "
                 "frame is GREEN - the mast-arm head ahead and the head on the right. "
                 "There is NO red signal anywhere in this frame.")
NEUTRAL_FRAMES = [(f"scene-0796_f{kf:02d}_CAM_FRONT", NEUTRAL_TRUTH, 12.0) for kf in (17, 21)]

NEUTRAL_QUESTIONS = [
    ("count_colour",
     "What colour is each traffic light visible in this image? If there is more than one, "
     "say where each is."),
    ("same_colour",
     "Are all the traffic lights visible in this image showing the same colour?"),
    ("governs",
     "Which traffic light governs the lane the ego vehicle is in, and how do you know?"),
    ("false_red",
     "Where is the red traffic light in this image?"),
]

BATTERIES = {
    "relevance": (FRAMES, QUESTIONS),
    "zoom": (ZOOM_FRAMES, ZOOM_QUESTIONS),
    "grounding": (FRAMES, GROUNDING_QUESTIONS),
    "lane": (LANE_FRAMES, LANE_QUESTIONS),
    "neutral": (NEUTRAL_FRAMES, NEUTRAL_QUESTIONS),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--frames", default="outputs/traffic_light/frames")
    ap.add_argument("--battery", default="relevance", choices=sorted(BATTERIES))
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    # 1.0 is the released default but makes short factual answers loop.
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    args = ap.parse_args()

    frames, questions = BATTERIES[args.battery]
    out = Path(args.out or f"outputs/traffic_light/vqa_{args.battery}.json")

    frames_dir = Path(args.frames)
    missing = [s for s, _, _ in frames if not (frames_dir / f"{s}.jpg").exists()]
    if missing:
        raise SystemExit(f"missing frames in {frames_dir}: {missing}")

    t0 = time.time()
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=DTYPES[args.dtype], attn_implementation="sdpa"
    ).to(args.device).eval()
    print(f"model loaded in {time.time() - t0:.0f}s\n", flush=True)

    results = []
    for stem, truth, speed in frames:
        path = frames_dir / f"{stem}.jpg"
        print("=" * 88)
        print(f"{stem}   ego speed {speed:.1f} m/s")
        print(f"  GROUND TRUTH: {truth}")
        print("=" * 88, flush=True)
        for kind, question in questions:
            t = time.time()
            res = model.generate_text(
                [str(path)], question,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
            )
            dt = time.time() - t
            print(f"\n[{kind}]  ({dt:.1f}s)\n  Q: {question}\n  A: {res.text}", flush=True)
            results.append({"frame": stem, "ego_speed": speed, "ground_truth": truth,
                            "kind": kind, "question": question, "answer": res.text,
                            "seconds": round(dt, 1)})
        print(flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model,
                               "battery": args.battery,
                               "repetition_penalty": args.repetition_penalty,
                               "results": results}, indent=2))
    print(f"\nwrote {out}  ({len(results)} answers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
