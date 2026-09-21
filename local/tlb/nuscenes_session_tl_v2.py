#!/usr/bin/env python
"""Run BOTH heads over a real nuScenes keyframe sequence and render it as a video.

Every video frame is laid out like `perception_walkthrough.png` - the camera ring wrapping
a BEV panel, with occupancy and map underneath - but played across a driving session, and
with the PREDICTED TRAJECTORY drawn into the BEV.

Why nuScenes: its keyframes are 2 Hz, which is exactly the history rate the planner expects
(4 frames over 1.5 s), and its 6-camera rig is one of the two the perception head was trained
on. So a single sequence feeds both heads in-distribution. The bundled demo frames cannot do
this - they are six unrelated scenes, not a sequence.

    PYTHONPATH=src python local/nuscenes_session.py --scene 0 --out outputs/session
"""
from __future__ import annotations

import argparse, json, sys, textwrap
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "local" / "tlb"))

import torch
from PIL import Image
from pyquaternion import Quaternion

from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES
from qwen_drive_perception.visualize import _camera_image as V_camera_image

# nuScenes category -> the model's 7-class taxonomy
CAT = {
    "vehicle.car": "vehicle", "vehicle.truck": "vehicle", "vehicle.bus.bendy": "vehicle",
    "vehicle.bus.rigid": "vehicle", "vehicle.trailer": "vehicle",
    "vehicle.construction": "vehicle", "vehicle.emergency.ambulance": "vehicle",
    "vehicle.emergency.police": "vehicle",
    "vehicle.bicycle": "bicycle", "vehicle.motorcycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone", "movable_object.barrier": "barrier",
    "movable_object.debris": "generic_object",
    "movable_object.pushable_pullable": "generic_object",
    "static_object.bicycle_rack": "generic_object",
}
CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
# The prompt layout the released frames use, verbatim: a view tag then its image, clockwise
# from the front, then the instruction. Reproduced exactly so the evaluated prompt matches.
VIEW_TAG = {"CAM_FRONT": "<FRONT VIEW>", "CAM_FRONT_RIGHT": "<FRONT RIGHT VIEW>",
            "CAM_BACK_RIGHT": "<BACK RIGHT VIEW>", "CAM_BACK": "<BACK VIEW>",
            "CAM_BACK_LEFT": "<BACK LEFT VIEW>", "CAM_FRONT_LEFT": "<FRONT LEFT VIEW>"}
PROMPT_TAIL = "Analyze the scene."
VIEW_OF = {"<FRONT VIEW>": "CAM_FRONT", "<FRONT LEFT VIEW>": "CAM_FRONT_LEFT",
           "<FRONT RIGHT VIEW>": "CAM_FRONT_RIGHT"}


# ---------------------------------------------------------------------------
# traffic-light stack, added in this version
# ---------------------------------------------------------------------------
_TL = {}


def tl_init(device="cuda"):
    """Load the traffic-light heads once. All of them read the SAME frozen ViT tap the
    perception head uses, so nothing here perturbs detection, mapping or occupancy."""
    import torch as _t
    from det_model_v2 import TLDetHead3D
    from selector import Selector
    from train_colour import build as build_colour
    base = ROOT_DIR
    ck = _t.load(base / "outputs/tlb/det_head3d_v2.pt", map_location="cpu")
    sck = _t.load(base / "outputs/tlb/selector_v2.pt", map_location="cpu")
    cck = _t.load(base / "outputs/tlb/colour_head_v2.pt", map_location="cpu")
    sel = Selector(d=sck["args"]["d"], layers=sck["args"]["layers"]).to(device)
    sel.load_state_dict(sck["model"]); sel.eval()
    col = build_colour().to(device)
    col.load_state_dict(cck["model"]); col.eval()
    _TL.update(ck=ck, sel=sel, col=col, head=None, up=ck["up"], patch=ck["patch"],
               device=device)


def tl_infer(vlm, img_path, device="cuda"):
    """Detections with colour, P(governs ego), range and height for one front image."""
    import torch as _t, numpy as _np
    import torch.nn.functional as _F
    from PIL import Image as _I
    from det_model_v2 import TLDetHead3D
    from det_model import decode as _decode
    from train_det import ViTTap
    from extract_crops import crop_one
    from cache_roi import roi_pool, GH, GW, MAXL
    if "tap" not in _TL:
        _TL["tap"] = ViTTap(vlm)
    tap = _TL["tap"]
    feat, gr, gc, W0, H0 = tap.features(img_path, device)
    if _TL["head"] is None:
        h = TLDetHead3D(in_dim=feat.shape[0], hid=_TL["ck"]["args"].get("hid", 256),
                        up=_TL["up"]).to(device).float()
        h.load_state_dict(_TL["ck"]["head"]); h.eval()
        _TL["head"] = h
    head = _TL["head"]
    with _t.no_grad():
        o = head(feat.float().unsqueeze(0))
        cell = _TL["patch"] / _TL["up"]
        dets = _decode(o, cell, thr=0.30)[0][:MAXL]
        sx, sy = (gc * _TL["patch"]) / W0, (gr * _TL["patch"]) / H0
        boxes, r3 = [], []
        for d_ in dets:
            boxes.append([d_["box"][0]/sx, d_["box"][1]/sy, d_["box"][2]/sx, d_["box"][3]/sy])
            cx = int(_np.clip((d_["box"][0]+d_["box"][2])/2/cell, 0, o["logrange"].shape[3]-1))
            cy = int(_np.clip((d_["box"][1]+d_["box"][3])/2/cell, 0, o["logrange"].shape[2]-1))
            r3.append((float(o["logrange"][0,0,cy,cx].exp()), float(o["height"][0,0,cy,cx])))
        if not boxes:
            return {"boxes": [], "pred": None}
        rf = roi_pool(feat, boxes, W0, H0).float()
        L = len(boxes)
        roi = _t.zeros(1, MAXL, rf.shape[1], device=device); roi[0,:L] = rf
        bx = _t.zeros(1, MAXL, 4, device=device)
        for j, b in enumerate(boxes):
            bx[0,j] = _t.tensor([(b[0]+b[2])/2/W0, (b[1]+b[3])/2/H0,
                                 (b[2]-b[0])/W0, (b[3]-b[1])/H0], device=device)
        ctx = _F.adaptive_avg_pool2d(feat.unsqueeze(0).float(), (GH, GW))
        mask = _t.zeros(1, MAXL, dtype=_t.bool, device=device); mask[0,:L] = True
        lg = _TL["sel"](roi, bx, ctx, mask).masked_fill(~mask, -1e4)
        gov = _t.sigmoid(lg[0,:L]).cpu().numpy()
        im = _I.open(img_path).convert("RGB")
        ts = [((_t.from_numpy(_np.asarray(crop_one(im, b), _np.uint8).copy())
                .permute(2,0,1).float()/255.) - .5)/.5 for b in boxes]
        with _t.amp.autocast("cuda", dtype=_t.bfloat16):
            P = _t.softmax(_TL["col"](_t.stack(ts).to(device)).float(), 1).cpu().numpy()
        order = _np.argsort(-gov)[:3]
        rgy = (gov[order][:,None] * P[order][:,1:4]).sum(0)
    pred = ["red","green","yellow"][int(rgy.argmax())] if rgy.sum() > 0 else None
    col3 = [["red","green","yellow"][int(p.argmax())] for p in P[:, 1:4]]
    raw = [int(i) for i in _np.nonzero(gov >= 0.35)[0]] or [int(gov.argmax())]
    # keep only the lights agreeing with the answer: several green lamps on one gantry are
    # fine, a red left-arrow alongside a green through-arrow is not
    ego = [i for i in raw if col3[i] == pred] or [int(max(raw, key=lambda i: gov[i]))]
    return {"boxes": boxes, "gov": gov.tolist(), "r3": r3,
            "colour": [["unknown","red","green","yellow"][int(p.argmax())] for p in P],
            "colour3": col3, "chosen": int(order[0]),
            "ego_set": ego, "n_ego_raw": len(raw),
            "inconsistent": int(len({col3[i] for i in raw}) > 1),
            "pred": pred}


def tl_draw(ax, tl, K, img_w, img_h):
    """Predicted 3D cuboids over the front camera, the ego's light picked out."""
    import numpy as _np
    from matplotlib.patches import Polygon as _Poly
    RGBF = {"red": "#ff4646", "green": "#3ceb6e", "yellow": "#ffd23c",
            "unknown": "#8aa0aa", None: "#8aa0aa"}
    # imshow set the axis to the image extent; plotting a cuboid whose corners fall
    # outside it makes matplotlib autoscale and the panel collapses. Pin the limits.
    _xlim, _ylim = ax.get_xlim(), ax.get_ylim()
    # The panel is the 896x512 image the perception head consumes, while cam_intrinsic is
    # for the 1600x900 original the detector ran on. Project with the true intrinsics,
    # then scale into panel pixels -- otherwise the cuboids land in the wrong place.
    _pw = abs(_xlim[1] - _xlim[0])
    _ph = abs(_ylim[0] - _ylim[1])
    _kx = _pw / float(img_w or 1600.0)
    _ky = _ph / float(img_h or 900.0)
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    _ego = set(tl.get("ego_set", [tl.get("chosen", -1)]))
    for j, b in enumerate(tl["boxes"]):
        rng, _h = tl["r3"][j]
        pick = (j in _ego)
        u, v = (b[0]+b[2])/2, (b[1]+b[3])/2
        Z = max(rng, 1.0)
        X, Y = (u-cx)*Z/fx, (v-cy)*Z/fy
        W = (b[2]-b[0])*Z/fx; H = (b[3]-b[1])*Z/fy; D = 0.30
        pts = [(X+sx_*W/2, Y+sy_*H/2, Z+sz_*D/2)
               for sz_ in (-1, 1) for sy_ in (-1, 1) for sx_ in (-1, 1)]
        p2 = [((fx*px/max(pz,.1)+cx) * _kx, (fy*py/max(pz,.1)+cy) * _ky)
              for (px, py, pz) in pts]
        col = RGBF[tl["colour"][j]] if pick else "#00d2ff"
        lw = 2.0 if pick else 0.8
        for a_, b_ in ((0,1),(1,3),(3,2),(2,0)):
            ax.plot([p2[a_][0], p2[b_][0]], [p2[a_][1], p2[b_][1]], color=col, lw=lw, zorder=6)
        for a_, b_ in ((4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)):
            ax.plot([p2[a_][0], p2[b_][0]], [p2[a_][1], p2[b_][1]], color=col,
                    lw=max(lw*0.5, 0.5), alpha=0.55, zorder=6)
        if pick:
            ax.text(p2[0][0], p2[0][1]-6, f"ego {rng:.0f} m", color=col, fontsize=5.5,
                    zorder=7)
    ax.set_xlim(*_xlim); ax.set_ylim(*_ylim)


def pose_matrix(rec) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
    m[:3, 3] = rec["translation"]
    return m


class SessionFrame:
    """Duck-types PerceptionFrame so PerceptionProcessor and the renderers accept it."""

    def __init__(self, nusc, sample, root: Path, cam_sd=None, lidar_sd=None,
                 gt_global=None, token=None):
        """`sample` is the owning keyframe. Passing `cam_sd`/`lidar_sd` renders an
        arbitrary sweep instead of the keyframe itself; `gt_global` then carries
        annotations interpolated to that instant."""
        self.nusc, self.sample, self.root = nusc, sample, Path(root)
        self.token = token or sample["token"]
        self.dataset_type = "nuscenes"
        self.cam_order = list(CAM_ORDER)
        self._gt_global = gt_global
        self.content = ([it for cam in CAM_ORDER
                         for it in ({"text": VIEW_TAG[cam]}, {"image": cam})]
                        + [{"text": PROMPT_TAIL}])

        lidar_sd = lidar_sd or nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        self.lidar2ego = pose_matrix(lidar_cs).astype(np.float32)
        ego2global_lidar = pose_matrix(nusc.get("ego_pose", lidar_sd["ego_pose_token"]))
        self.global_from_lidar = ego2global_lidar @ self.lidar2ego

        K, R, T, self._paths = [], [], [], {}
        for cam in self.cam_order:
            sd = cam_sd[cam] if cam_sd else nusc.get("sample_data", sample["data"][cam])
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            self._paths[cam] = self.root / sd["filename"]
            K.append(np.asarray(cs["camera_intrinsic"], np.float32))
            # camera -> lidar, going through both ego poses (the sensors are not synchronous)
            cam2ego = pose_matrix(cs)
            ego2global_c = pose_matrix(nusc.get("ego_pose", sd["ego_pose_token"]))
            cam2lidar = np.linalg.inv(self.global_from_lidar) @ ego2global_c @ cam2ego
            R.append(cam2lidar[:3, :3].astype(np.float32))
            T.append(cam2lidar[:3, 3].astype(np.float32))
        self.cam_intrinsic = np.stack(K)
        self.sensor2lidar_rotation = np.stack(R)
        self.sensor2lidar_translation = np.stack(T)

        pts = np.fromfile(self.root / lidar_sd["filename"], dtype=np.float32)
        self.lidar = pts.reshape(-1, 5)[:, :3]
        self.gt = {"boxes": self._gt_boxes(), "labels": self._gt_labels()}

    def _annotations(self):
        out = []
        for tok in self.sample["anns"]:
            a = self.nusc.get("sample_annotation", tok)
            name = CAT.get(a["category_name"])
            if name is None or a["num_lidar_pts"] + a["num_radar_pts"] == 0:
                continue
            out.append((a, name))
        return out

    def _global_anns(self):
        """(translation, quaternion, size, class name) in the GLOBAL frame."""
        if self._gt_global is not None:
            return self._gt_global
        return [(np.asarray(a["translation"]), Quaternion(a["rotation"]), a["size"], n)
                for a, n in self._annotations()]

    def _gt_boxes(self):
        rows = []
        inv = np.linalg.inv(self.global_from_lidar)
        for tr, q, size, _ in self._global_anns():
            c = inv @ np.append(np.asarray(tr), 1.0)
            yaw = Quaternion(matrix=inv[:3, :3] @ q.rotation_matrix).yaw_pitch_roll[0]
            w, l, h = size                            # nuScenes stores (w, l, h)
            rows.append([c[0], c[1], c[2] - h / 2, l, w, h, yaw, 0.0, 0.0])
        return np.asarray(rows, np.float32).reshape(-1, 9)

    def _gt_labels(self):
        return np.asarray([DET_CLASS_NAMES.index(n) for *_, n in self._global_anns()],
                          np.int64)

    def image(self, cam) -> Image.Image:
        return Image.open(self._paths[cam]).convert("RGB")

    def img_metas(self, image_size=(896, 512)):
        from qwen_drive_perception import geometry
        tw, th = image_size
        l2i = np.stack([
            geometry.apply_image_scale(geometry.build_lidar2img(K, R, t), tw / w, th / h)
            for K, R, t, (w, h) in zip(self.cam_intrinsic, self.sensor2lidar_rotation,
                                       self.sensor2lidar_translation,
                                       [self.image(c).size for c in self.cam_order])
        ])
        return {"sample_token": self.token, "dataset_type": self.dataset_type,
                "cam_order": self.cam_order, "lidar2img": l2i.astype(np.float32),
                "lidar2ego": np.repeat(self.lidar2ego[None], len(self.cam_order), 0),
                "img_shape": [(th, tw)] * len(self.cam_order), "box_coord_system": "ego"}


def ego_track(nusc, sample):
    """Dense ego poses (~20 Hz from the lidar sweeps) around a keyframe, in global frame."""
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    # walk back then forward along the sweep chain
    first = sd
    while first["prev"]:
        first = nusc.get("sample_data", first["prev"])
    out = []
    cur = first
    while True:
        p = nusc.get("ego_pose", cur["ego_pose_token"])
        out.append((p["timestamp"] * 1e-6, np.asarray(p["translation"]),
                    Quaternion(p["rotation"])))
        if not cur["next"]:
            break
        cur = nusc.get("sample_data", cur["next"])
    return out


def resample(track, t0, offsets, ref_inv, ref_yaw):
    """Poses at t0 + offsets, expressed in the reference (current) ego frame."""
    ts = np.asarray([t for t, _, _ in track])
    xy = np.stack([p[:2] for _, p, _ in track])
    yaw = np.unwrap(np.asarray([q.yaw_pitch_roll[0] for _, _, q in track]))
    out = []
    for dt in offsets:
        t = t0 + dt
        x = np.interp(t, ts, xy[:, 0]); y = np.interp(t, ts, xy[:, 1])
        h = np.interp(t, ts, yaw)
        local = ref_inv @ np.array([x, y, 0.0, 1.0])
        out.append([local[0], local[1], (h - ref_yaw + np.pi) % (2 * np.pi) - np.pi])
    return np.asarray(out, np.float32)


# ─────────────────────── trajectory projected into the camera views ────────────────────
# Alpamayo's perception_demo.mp4 draws the predicted path and the driven path as ground
# curves in every camera that sees them. Same idea here: lift the ego-frame waypoints to
# the ground plane (ego z = 0 is the road in nuScenes), take them to the lidar frame, and
# push them through the same lidar2img the boxes use.

TRAJ_PRED = (0, 200, 255)      # cyan, as in the Alpamayo demo
TRAJ_GT = (255, 176, 32)       # amber
CAM_W, CAM_H = 896, 512


def project_ground_path(xy, lidar2ego, lidar2img, z_ego=0.0):
    """Ego-frame (x, y) -> pixel uv in one camera, with a behind-camera mask."""
    n = len(xy)
    ego = np.concatenate([np.asarray(xy, np.float64)[:, :2],
                          np.full((n, 1), z_ego), np.ones((n, 1))], axis=1)
    lidar = (np.linalg.inv(np.asarray(lidar2ego, np.float64)) @ ego.T).T
    proj = (np.asarray(lidar2img, np.float64) @ lidar.T).T
    depth = proj[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = proj[:, :2] / depth[:, None]
    return uv, depth > 0.5


def draw_path_on_camera(canvas, xy, lidar2ego, lidar2img, colour, width=4, dots=5,
                        dashed=False):
    """Draw one ground path, breaking the polyline wherever it passes behind the camera."""
    import cv2
    uv, ok = project_ground_path(xy, lidar2ego, lidar2img)
    h, w = canvas.shape[:2]
    margin = 4 * max(h, w)
    drawn = 0
    for i in range(len(uv) - 1):
        if dashed and (i // 3) % 2:          # 3 on, 3 off - visible under the prediction
            continue
        if not (ok[i] and ok[i + 1]):
            continue
        a, b = uv[i], uv[i + 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        if max(abs(a[0]), abs(b[0])) > margin or max(abs(a[1]), abs(b[1])) > margin:
            continue
        cv2.line(canvas, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                 colour, width, cv2.LINE_AA)
        drawn += 1
    if dots:
        for i in range(0, len(uv), dots):
            if ok[i] and np.isfinite(uv[i]).all() and abs(uv[i]).max() < margin:
                cv2.circle(canvas, tuple(np.rint(uv[i]).astype(int)), width + 2,
                           (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(np.rint(uv[i]).astype(int)), width, colour,
                           -1, cv2.LINE_AA)
    return drawn


def camera_with_paths(frame, index, cam, boxes, labels, traj, gt_traj, lidar2img):
    """The repo's camera panel, plus the two ground paths."""
    canvas = V_camera_image(frame, index, cam, boxes, labels).copy()
    if gt_traj is not None:
        draw_path_on_camera(canvas, gt_traj[:, :2], frame.lidar2ego, lidar2img,
                            TRAJ_GT, width=5, dots=0, dashed=True)
    if traj is not None:
        draw_path_on_camera(canvas, traj[0][:, :2], frame.lidar2ego, lidar2img,
                            TRAJ_PRED, width=4, dots=5)
    return canvas


def build_scene(nusc, samples, idx, track):
    """A DrivingScene from keyframe `idx` and the three before it (2 Hz = the model's rate)."""
    from qwen_drive import CameraFrame, DrivingScene
    cur = samples[idx]
    window = [samples[idx - 3], samples[idx - 2], samples[idx - 1], cur]

    sd = nusc.get("sample_data", cur["data"]["LIDAR_TOP"])
    p = nusc.get("ego_pose", sd["ego_pose_token"])
    t0 = p["timestamp"] * 1e-6
    ref = np.eye(4)
    ref[:3, :3] = Quaternion(p["rotation"]).rotation_matrix
    ref[:3, 3] = p["translation"]
    ref_inv, ref_yaw = np.linalg.inv(ref), Quaternion(p["rotation"]).yaw_pitch_roll[0]

    hist = resample(track, t0, np.arange(-1.5, 0.0001, 0.1), ref_inv, ref_yaw)   # [16,3]
    fut = resample(track, t0, np.arange(0.1, 5.0001, 0.1), ref_inv, ref_yaw)     # [50,3]
    vel = np.gradient(hist[:, :2], 0.1, axis=0).astype(np.float32)
    acc = np.gradient(vel, 0.1, axis=0).astype(np.float32)

    lat = float(fut[-1, 1])
    nav = 1 if lat > 4.0 else (2 if lat < -4.0 else 0)
    cmd = [int(nav == 1), int(nav == 0), int(nav == 2), 0]

    views = {}
    for view, cam in VIEW_OF.items():
        views[view] = [CameraFrame(Path(nusc.dataroot) /
                                   nusc.get("sample_data", s["data"][cam])["filename"])
                       for s in window]
    scene = DrivingScene(views=views, history=hist, history_velocity=vel,
                         history_acceleration=acc,
                         ego_velocity=tuple(vel[-1]), ego_acceleration=tuple(acc[-1]),
                         driving_command=cmd, nav_command=nav, token=cur["token"])
    return scene, fut


# ───────────────────────────── sweep-rate rendering ────────────────────────────────────
# nuScenes keyframes are 2 Hz, which reads as a slideshow. Every sensor also has
# non-keyframe *sweeps* (~12 Hz for cameras, ~20 Hz for lidar) with their own calibration
# and ego pose, so the whole pipeline can be run at the sweep rate instead. Annotations
# only exist at keyframes, so ground truth is interpolated between the two that bracket
# each sweep - constant velocity over <=0.5 s, which is what the boxes do anyway.

def sensor_chain(nusc, sample, channel):
    """Every sample_data for one sensor in this scene, ordered in time."""
    sd = nusc.get("sample_data", sample["data"][channel])
    while sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])
    out = [sd]
    while sd["next"]:
        sd = nusc.get("sample_data", sd["next"])
        out.append(sd)
    return out


def _nearest(chain, ts, t):
    return chain[int(np.abs(ts - t).argmin())]


def interp_global_anns(nusc, samples, sample_ts, t):
    """Annotations at time `t`, interpolated between the bracketing keyframes.

    Instances are matched through sample_annotation's `next` pointer, so a box only
    interpolates against itself. The earlier keyframe's annotation set defines which
    boxes exist, so an object first labelled in the later keyframe appears only once
    that keyframe is reached - at most 0.5 s late.
    """
    k = int(np.searchsorted(sample_ts, t) - 1)
    k = max(0, min(k, len(samples) - 1))
    a_s = samples[k]
    if k + 1 >= len(samples):
        alpha, b_s = 0.0, a_s
    else:
        b_s = samples[k + 1]
        span = sample_ts[k + 1] - sample_ts[k]
        alpha = float(np.clip((t - sample_ts[k]) / max(span, 1e-6), 0.0, 1.0))

    out = []
    for tok in a_s["anns"]:
        a = nusc.get("sample_annotation", tok)
        name = CAT.get(a["category_name"])
        if name is None or a["num_lidar_pts"] + a["num_radar_pts"] == 0:
            continue
        tr_a, q_a = np.asarray(a["translation"]), Quaternion(a["rotation"])
        if alpha > 0 and a["next"]:
            b = nusc.get("sample_annotation", a["next"])
            tr = tr_a + alpha * (np.asarray(b["translation"]) - tr_a)
            q = Quaternion.slerp(q_a, Quaternion(b["rotation"]), alpha)
        else:
            tr, q = tr_a, q_a
        out.append((tr, q, a["size"], name))
    return out


def build_scene_at(nusc, cam_chains, cam_ts, track, t0, root):
    """A DrivingScene ending at an arbitrary time `t0`, history at the model's 2 Hz."""
    from qwen_drive import CameraFrame, DrivingScene
    ts_track = np.asarray([x[0] for x in track])
    i = int(np.abs(ts_track - t0).argmin())
    _, pos, quat = track[i]
    ref = np.eye(4)
    ref[:3, :3] = quat.rotation_matrix
    ref[:3, 3] = pos
    ref_inv, ref_yaw = np.linalg.inv(ref), quat.yaw_pitch_roll[0]

    hist = resample(track, t0, np.arange(-1.5, 0.0001, 0.1), ref_inv, ref_yaw)
    fut = resample(track, t0, np.arange(0.1, 5.0001, 0.1), ref_inv, ref_yaw)
    vel = np.gradient(hist[:, :2], 0.1, axis=0).astype(np.float32)
    acc = np.gradient(vel, 0.1, axis=0).astype(np.float32)

    lat = float(fut[-1, 1])
    nav = 1 if lat > 4.0 else (2 if lat < -4.0 else 0)
    cmd = [int(nav == 1), int(nav == 0), int(nav == 2), 0]

    views = {}
    for view, cam in VIEW_OF.items():
        views[view] = [
            CameraFrame(Path(root) / _nearest(cam_chains[cam], cam_ts[cam], t0 + dt)["filename"])
            for dt in (-1.5, -1.0, -0.5, 0.0)
        ]
    scene = DrivingScene(views=views, history=hist, history_velocity=vel,
                         history_acceleration=acc, ego_velocity=tuple(vel[-1]),
                         ego_acceleration=tuple(acc[-1]), driving_command=cmd,
                         nav_command=nav, token=f"t{t0:.3f}")
    return scene, fut


BEV_R = 46.0          # metres shown; the 5 s path reaches ~40 m so this keeps it in frame


def bev_panel(ax, frame, pred_boxes, pred_labels, traj, gt_traj):
    """A BEV tuned for dense urban scenes.

    The repo's `_panel_bev` is built for the demo frames (~30 boxes). A nuScenes keyframe
    here has 135 predictions + 76 ground-truth boxes and 35 k lidar points, which at this
    panel size is solid colour. So: fewer, fainter lidar points; ground-truth as thin
    outlines rather than heavy ones; predictions filled so they read at a glance.
    """
    from matplotlib import patches
    from qwen_drive_perception import geometry
    from qwen_drive_perception.visualize import _class_rgb

    ax.set_xlim(-BEV_R, BEV_R); ax.set_ylim(-BEV_R, BEV_R)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#b0b7bf"); sp.set_linewidth(0.8)
    for r in (10.0, 20.0, 30.0, 40.0):
        ax.add_patch(patches.Circle((0, 0), r, fill=False, edgecolor="#e6e9ed",
                                    lw=0.6, zorder=1))

    pts = frame.lidar @ np.asarray(frame.lidar2ego)[:3, :3].T + np.asarray(frame.lidar2ego)[:3, 3]
    pts = pts[np.abs(pts[:, :2]).max(axis=1) < BEV_R]
    if len(pts) > 14000:                       # was 80 k - far too dense to see through
        pts = pts[:: len(pts) // 14000]
    ax.scatter(-pts[:, 1], pts[:, 0], s=0.45, c="#9fb2c4", alpha=0.55,
               linewidths=0, zorder=2, rasterized=True)

    def draw(boxes, colour_of, lw, zorder, fill_alpha):
        if not len(boxes):
            return
        corners = geometry.box_corners(np.asarray(boxes, np.float64))
        for i in range(len(boxes)):
            fp = corners[i][:4, :2]
            xy = np.stack([-fp[:, 1], fp[:, 0]], axis=1)
            c = colour_of(i)
            ax.add_patch(patches.Polygon(xy, closed=True, facecolor=c, edgecolor=c,
                                         alpha=fill_alpha, lw=lw, zorder=zorder))

    gt_ego = geometry.lidar_to_ego_boxes(torch.as_tensor(np.asarray(frame.gt["boxes"], np.float32)),
                                         torch.as_tensor(frame.lidar2ego)).numpy() \
        if len(frame.gt["boxes"]) else np.zeros((0, 9), np.float32)
    draw(gt_ego, lambda i: "#3aaf4a", 0.9, 3, 0.0)          # ground truth: outline only

    pe = geometry.lidar_to_ego_boxes(torch.as_tensor(np.asarray(pred_boxes, np.float32)),
                                     torch.as_tensor(frame.lidar2ego)).numpy() \
        if len(pred_boxes) else np.zeros((0, 9), np.float32)
    draw(pe, lambda i: np.asarray(_class_rgb(pred_labels[i])) / 255.0, 0.7, 4, 0.55)

    if gt_traj is not None:
        ax.plot(-gt_traj[:, 1], gt_traj[:, 0], color="#ffb020", lw=3.2, ls=(0, (4, 3)),
                zorder=9, label="actual path")
    if traj is not None:
        ax.plot(-traj[0][:, 1], traj[0][:, 0], color="#00c8ff", lw=3.4, zorder=10,
                solid_capstyle="round", label="predicted path")
        ax.scatter(-traj[0][::10, 1], traj[0][::10, 0], s=22, color="#00c8ff",
                   edgecolors="white", linewidths=0.8, zorder=11)
    ax.add_patch(patches.Circle((0, 0), 1.1, facecolor="#d6336c", edgecolor="white",
                                lw=0.8, zorder=12))
    lg = ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    lg.set_zorder(13)



MAP_XBOUND = (-30.0, 30.0, 0.15)          # forward, 400 cells
MAP_YBOUND = (-15.0, 15.0, 0.15)          # left,    200 cells


def _map_xy_to_pixel(x, y):
    """Ego metres -> pixel (col, row) in the rendered map panel.

    ``visualize._map_rgb`` does ``transpose(1, 0, 2)[::-1, ::-1]`` on a (Y, X)
    grid, so the displayed image is (400, 200) with **forward up** and **+y
    (left) on the left**. Inverting that:

        row = 399 - (x + 30) / 0.15          col = 199 - (y + 15) / 0.15

    The map is in the EGO frame, so anything drawn here must be in ego metres.
    Detections come out of the head in the LIDAR frame and have to be converted
    first - mixing the two silently shifts every box by the sensor offset.
    """
    x0, _, xs = MAP_XBOUND
    y0, _, ys = MAP_YBOUND
    nx = int(round((MAP_XBOUND[1] - x0) / xs))
    ny = int(round((MAP_YBOUND[1] - y0) / ys))
    col = (ny - 1) - (np.asarray(y) - y0) / ys
    row = (nx - 1) - (np.asarray(x) - x0) / xs
    return col, row


def map_panel(ax, map_grid, frame, pred_boxes, pred_labels, traj, gt_traj):
    """The BEV map segmentation with the detections and the trajectory on top.

    The bare segmentation answers "where is the road"; adding the objects and the
    path answers "where is the road, what is on it, and where are we going" - the
    three outputs that actually have to agree with each other, in one picture.
    """
    import torch
    from matplotlib import patches
    from qwen_drive_perception import geometry, visualize as V
    from qwen_drive_perception.visualize import _class_rgb

    rgb = V._map_rgb(map_grid)
    ax.imshow(rgb, interpolation="antialiased")
    h, w = rgb.shape[:2]
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(V.FRAME_COLOR); sp.set_linewidth(0.8)

    # detections: lidar -> ego, then ego -> pixels
    if len(pred_boxes):
        ego = geometry.lidar_to_ego_boxes(
            torch.as_tensor(np.asarray(pred_boxes, np.float32)),
            torch.as_tensor(frame.lidar2ego)).numpy()
        corners = geometry.box_corners(np.asarray(ego, np.float64))
        for i in range(len(ego)):
            fp = corners[i][:4, :2]
            col, row = _map_xy_to_pixel(fp[:, 0], fp[:, 1])
            if col.max() < 0 or col.min() > w or row.max() < 0 or row.min() > h:
                continue                                   # fully outside the 60x30 m window
            c = np.asarray(_class_rgb(pred_labels[i])) / 255.0
            ax.add_patch(patches.Polygon(np.stack([col, row], 1), closed=True,
                                         facecolor=c, edgecolor=c, alpha=0.55,
                                         lw=0.9, zorder=4))

    if gt_traj is not None and len(gt_traj):
        col, row = _map_xy_to_pixel(gt_traj[:, 0], gt_traj[:, 1])
        ax.plot(col, row, color="#ffb020", lw=2.4, ls=(0, (4, 3)), zorder=9)
    if traj is not None and len(traj):
        col, row = _map_xy_to_pixel(traj[0][:, 0], traj[0][:, 1])
        ax.plot(col, row, color="#00c8ff", lw=2.8, zorder=10, solid_capstyle="round")
        ax.scatter(col[::10], row[::10], s=16, color="#00c8ff",
                   edgecolors="white", linewidths=0.7, zorder=11)

    col0, row0 = _map_xy_to_pixel(0.0, 0.0)
    ax.add_patch(patches.Circle((float(col0), float(row0)), 5.0, facecolor="#d6336c",
                                edgecolor="white", lw=0.9, zorder=12))


def render_session_frame(frame, result, traj, gt_traj, reasoning=None,
                         stats=None, score_threshold=0.3, dpi=100, tl=None):
    """The perception_walkthrough.png layout, with the trajectory drawn into the BEV."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from qwen_drive_perception import visualize as V

    keep = result["scores"] >= score_threshold
    pb, pl = result["boxes"][keep], result["labels"][keep]

    layout = V._ring_layout(frame)
    rows = V._row_heights(layout)
    width = (2.0 * V.SIDE_WIDTH + V.CENTER_WIDTH) * V.UNIT_INCHES
    ring_h = sum(rows) * V.UNIT_INCHES
    # Taller bottom row: the map is a 1:2 portrait raster, so its on-screen size
    # is set by ROW HEIGHT, not by the column width - widening the cell only adds
    # whitespace beside it.
    bottom_h, legend_h = 1.62 * V.UNIT_INCHES, 0.62
    height = ring_h + bottom_h + legend_h

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(2, 1, height_ratios=[ring_h, bottom_h], left=0.006, right=0.994,
                             top=0.975, bottom=legend_h / height, hspace=0.02)
    ring = outer[0].subgridspec(3, 3, width_ratios=[V.SIDE_WIDTH, V.CENTER_WIDTH, V.SIDE_WIDTH],
                                height_ratios=rows, wspace=0.02, hspace=0.06)
    l2i = frame.img_metas()["lidar2img"]
    for i, cam in enumerate(frame.cam_order):
        r, c = layout[cam]
        _ax = fig.add_subplot(ring[r, c])
        V._panel_image(_ax, camera_with_paths(frame, i, cam, pb, pl, traj, gt_traj, l2i[i]),
                       V._camera_title(cam))
        # the traffic-light stack only ever looks at CAM_FRONT
        if tl is not None and cam == "CAM_FRONT" and tl.get("boxes"):
            tl_draw(_ax, tl, frame.cam_intrinsic[i], 1600.0, 900.0)

    bev = fig.add_subplot(ring[1, 1])
    bev_panel(bev, frame, pb, pl, traj, gt_traj)

    # The 6-camera ring leaves two cells of the 3x3 empty. Alpamayo's perception_demo gives
    # its chain-of-causation a panel of its own; do the same with the cells we already have.
    used = set(layout.values()) | {(1, 1)}
    free = [(r, c) for r in range(3) for c in range(3) if (r, c) not in used]

    if reasoning and free:
        ax = fig.add_subplot(ring[free[0]])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                                   facecolor="#f2f4f7", edgecolor=V.FRAME_COLOR, lw=0.8))
        ax.text(0.5, 0.955, "chain of thought", transform=ax.transAxes, ha="center",
                va="top", fontsize=10, color="#8a9099", family="monospace")
        ax.plot([0.10, 0.90], [0.885, 0.885], transform=ax.transAxes, color="#d8dde3",
                lw=0.8, clip_on=False)
        # Anchored to the TOP so a long rationale grows downward into free space instead of
        # upward into the header. Font shrinks once it needs more than three lines.
        lines = textwrap.wrap(reasoning, 30)
        size = 14.0 if len(lines) <= 2 else (12.5 if len(lines) == 3 else 11.0)
        ax.text(0.5, 0.80, "\n".join(lines), transform=ax.transAxes, ha="center",
                va="top", fontsize=size, color="#1b1f24", linespacing=1.55)
        ax.text(0.5, 0.05, "generated BEFORE the trajectory;\nthe expert reads the cache "
                           "that wrote it", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8, color="#8a9099", linespacing=1.4)

    if stats and len(free) > 1:
        ax = fig.add_subplot(ring[free[1]])
        ax.axis("off")
        y = 0.86
        for k, v in stats:
            ax.text(0.06, y, k, transform=ax.transAxes, ha="left", va="center",
                    fontsize=9.5, color="#8a9099", family="monospace")
            ax.text(0.94, y, v, transform=ax.transAxes, ha="right", va="center",
                    fontsize=11.5, color="#1b1f24", family="monospace")
            y -= 0.115

    # Two panels, not three. The separate trajectory chart was redundant once the
    # path is drawn onto the map itself, and dropping it lets both survivors grow.
    bottom = outer[1].subgridspec(2, 2, width_ratios=[1.30, 0.78],
                                  height_ratios=[0.07, 1.0], wspace=0.02, hspace=0.0)
    occ_ax = fig.add_subplot(bottom[1, 0], projection="3d")
    V._panel_occupancy(occ_ax, result["occ"], V._occupancy_window(result["occ"]))
    map_panel(fig.add_subplot(bottom[1, 1]), result["map"], frame, pb, pl,
              traj, gt_traj)


    for col, title in enumerate(("occupancy prediction",
                                 "map prediction  +  objects  +  trajectory")):
        cell = bottom[1, col].get_position(fig)
        fig.text(cell.x0 + cell.width / 2, cell.y1 + 0.004, title, ha="center", va="bottom",
                 fontsize=9.5, color=V.TITLE_COLOR)
    V._grow(occ_ax, 1.0, 1.35)
    fig.text(0.006, 0.996, f"{frame.token}   {getattr(frame, 'kind', 'nuScenes keyframe')}",
             fontsize=9,
             color=V.TITLE_COLOR, va="top")
    V._legend(fig)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return _append_path_legend(img)


def _append_path_legend(img, strip=44):
    """A strip under the figure explaining the two ground paths (the repo legend is classes)."""
    from PIL import ImageDraw, ImageFont
    out = Image.new("RGB", (img.shape[1], img.shape[0] + strip), (255, 255, 255))
    out.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(out)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        f = ImageFont.load_default()
    y = img.shape[0] + strip // 2
    x = 16
    d.line([(x, y), (x + 46, y)], fill=TRAJ_PRED, width=5)
    d.ellipse([x + 20, y - 4, x + 28, y + 4], fill=(255, 255, 255), outline=TRAJ_PRED)
    x += 56
    d.text((x, y - 9), "predicted trajectory  (5 s @ 10 Hz)", font=f, fill=(47, 52, 55)); x += 300
    for k in range(4):
        d.line([(x + k * 14, y), (x + k * 14 + 8, y)], fill=TRAJ_GT, width=5)
    x += 62
    d.text((x, y - 9), "actual path driven  (ground truth)", font=f, fill=(47, 52, 55)); x += 300
    d.text((x, y - 9), "projected onto the ground plane in every camera",
           font=f, fill=(120, 126, 132))
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--perception", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--around", type=int, default=None,
                    help="nuScenes timestamp in microseconds to centre the window on")
    ap.add_argument("--window", type=float, default=2.0,
                    help="seconds either side of --around")
    ap.add_argument("--future", type=float, default=5.0,
                    help="seconds of future the last frame needs. Below 5 the GT path is "
                         "clamped at the scene end and nav_command may degrade")
    ap.add_argument("--out", default="outputs/session")
    ap.add_argument("--rate", default="sweep", choices=["keyframe", "sweep"],
                    help="keyframe = 2 Hz (a slideshow); sweep = the ~12 Hz camera cadence")
    ap.add_argument("--resume", action="store_true",
                    help="skip frames whose PNG already exists")
    ap.add_argument("--speed", type=float, default=0.75,
                    help="playback speed relative to real time")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from transformers import AutoTokenizer
    from qwen_drive import InferenceMode, QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor

    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]
    samples, tok = [], scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok); samples.append(s); tok = s["next"]
    print(f"scene {args.scene}: {scene['name']}  {len(samples)} keyframes @ 2 Hz "
          f"({scene['description'][:60]})")

    model = QwenDriveForPlanning.from_pretrained(
        args.vlm, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    head = QwenDrivePerception.from_pretrained(args.perception,
                                               dtype=torch.bfloat16).to("cuda").eval()
    pproc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(model.vlm, pproc)

    # Load the traffic-light heads once, after the VLM is up. They read the same frozen
    # ViT tap the perception head uses; nothing here touches the perception weights.
    try:
        tl_init("cuda")
        print("  traffic-light stack loaded (shares the frozen ViT tap, read-only)",
              flush=True)
    except Exception as _e:
        print(f"  traffic-light stack unavailable: {_e}", flush=True)

    track = ego_track(nusc, samples[0])
    sample_ts = np.asarray([s["timestamp"] * 1e-6 for s in samples])

    # every sensor's full chain, keyframes and sweeps alike
    cam_chains = {c: sensor_chain(nusc, samples[0], c) for c in CAM_ORDER}
    cam_ts = {c: np.asarray([x["timestamp"] * 1e-6 for x in cam_chains[c]])
              for c in CAM_ORDER}
    lidar_chain = sensor_chain(nusc, samples[0], "LIDAR_TOP")
    lidar_ts = np.asarray([x["timestamp"] * 1e-6 for x in lidar_chain])

    # the first renderable instant needs 1.5 s of history behind it
    t_start = sample_ts[0] + 1.5
    t_end = sample_ts[-1] - args.future            # and 5 s of future for the GT path
    if args.future < 5.0:
        print(f"  note: --future {args.future:g}s < 5s, so the last frames' ground-truth path "
              f"is clamped at the scene end (np.interp does not extrapolate)")
    if args.rate == "sweep":
        drive = cam_ts["CAM_FRONT"]
        times = drive[(drive >= t_start) & (drive <= t_end)]
        src_hz = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 2.0
    else:
        times = sample_ts[(sample_ts >= t_start) & (sample_ts <= t_end)]
        src_hz = 2.0
    if args.limit:
        times = times[: args.limit]
    if args.around:
        centre = args.around * 1e-6
        keep = (times >= centre - args.window) & (times <= centre + args.window)
        if not keep.any():
            raise SystemExit(
                f"--around {args.around} lands outside the renderable span "
                f"[{times[0]:.1f}, {times[-1]:.1f}]; a frame needs 1.5 s of history "
                f"behind it and 5 s of future ahead")
        times = times[keep]
        print(f"  window {args.window:g}s either side of {args.around} "
              f"-> {len(times)} frames, {times[0] - centre:+.1f}s to {times[-1] - centre:+.1f}s")
    out_fps = src_hz * args.speed
    print(f"  rendering {len(times)} frames at {src_hz:.1f} Hz source "
          f"-> {out_fps:.2f} fps for {args.speed}x real speed "
          f"({len(times)/src_hz:.1f}s real -> {len(times)/out_fps:.1f}s video)")

    meta = []
    for n, t in enumerate(times):
        if args.resume and (out / "frames" / f"{n:04d}.png").exists():
            continue
        owner = samples[max(0, min(int(np.searchsorted(sample_ts, t) - 1), len(samples) - 1))]
        cam_sd = {c: _nearest(cam_chains[c], cam_ts[c], t) for c in CAM_ORDER}
        lidar_sd = _nearest(lidar_chain, lidar_ts, t)
        gt_global = interp_global_anns(nusc, samples, sample_ts, t)
        frame = SessionFrame(nusc, owner, Path(args.dataroot), cam_sd=cam_sd,
                             lidar_sd=lidar_sd, gt_global=gt_global,
                             token=f"{owner['token'][:12]}{(t-owner['timestamp']*1e-6)*1000:+.0f}ms")
        dt_ms = abs(t - owner["timestamp"] * 1e-6) * 1000
        frame.kind = ("nuScenes keyframe" if dt_ms < 1
                      else f"nuScenes sweep  ({src_hz:.0f} Hz, {dt_ms:+.0f} ms from keyframe)")
        pin, pmeta = pproc(frame, device="cuda")
        with torch.no_grad():
            result = head.infer(pin, pmeta)
        sc, gt_fut = build_scene_at(nusc, cam_chains, cam_ts, track, float(t), args.dataroot)
        with torch.no_grad():
            plan = model.run(InferenceMode.REASONING_PLANNING, scene=sc,
                             num_samples=args.num_samples)
        traj = plan.trajectories
        err = np.linalg.norm(traj[0][:, :2] - gt_fut[:, :2], axis=-1)
        speed = float(np.linalg.norm(sc.ego_velocity))
        path_len = float(np.linalg.norm(np.diff(traj[0][:, :2], axis=0), axis=1).sum())
        tl = None
        try:
            tl = tl_infer(model, frame._paths["CAM_FRONT"], device="cuda")
        except Exception as _e:                      # a demo overlay must never stop the run
            print(f"    [tl] skipped: {_e}", flush=True)
        stats = [("frame", f"{n + 1}/{len(times)}"),
                 ("speed", f"{speed * 3.6:.1f} km/h"),
                 ("path", f"{path_len:.1f} m / 5 s"),
                 ("ADE", f"{err.mean():.2f} m"),
                 ("FDE", f"{err[-1]:.2f} m"),
                 ("boxes >0.3", f"{int((result['scores'] > 0.3).sum())} / {len(frame.gt['labels'])} GT")]
        if tl is not None:
            _n = len(tl.get("boxes", []))
            stats.append(("lights", f"{_n} detected"))
            if tl.get("pred"):
                _e = tl.get("ego_set", [tl.get("chosen", 0)])
                _r = min(tl["r3"][i][0] for i in _e) if tl.get("r3") else 0.0
                stats.append(("ego lights", f"{len(_e)} of {_n}"))
                stats.append(("EGO LIGHT", f"{tl['pred'].upper()}  @ {_r:.0f} m"))
        img = render_session_frame(frame, result, traj, gt_fut,
                                   reasoning=plan.reasoning, stats=stats, tl=tl)
        Image.fromarray(img).save(out / "frames" / f"{n:04d}.png")
        keep = int((result["scores"] > 0.3).sum())
        meta.append({"i": n, "t": float(t), "token": frame.token, "boxes": keep,
                     "gt_boxes": int(len(frame.gt["labels"])), "ade": float(err.mean()),
                     "fde": float(err[-1]), "reasoning": plan.reasoning})
        print(f"[{n+1}/{len(times)}] {frame.token[:20]}  {keep:3d} boxes "
              f"(GT {len(frame.gt['labels']):3d})  ADE {err.mean():.3f}  "
              f"FDE {err[-1]:.3f}  | {plan.reasoning}", flush=True)
        prev = []
        sj = out / "session.json"
        if args.resume and sj.exists():
            try:
                prev = [r for r in json.loads(sj.read_text())
                        if r["i"] not in {m["i"] for m in meta}]
            except Exception:
                prev = []
        sj.write_text(json.dumps(sorted(prev + meta, key=lambda r: r["i"]), indent=2))
    (out / "encode.json").write_text(json.dumps(
        {"frames": len(times), "rendered_this_run": len(meta),
         "source_hz": float(src_hz), "speed": args.speed,
         "out_fps": float(out_fps)}, indent=2))
    print(f"\nwrote {len(meta)} frames -> {out}/frames")
    print(f"encode with:  ffmpeg -y -framerate {out_fps:.4f} -i {out}/frames/%04d.png "
          f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30' "
          f"-c:v libx264 -pix_fmt yuv420p -crf 20 <output>.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
