"""
07_qwen_drive_pipeline.py — Qwen-Drive-1.0 planning, unrolled. Our code, step by step.

The counterpart to alpamayo1.5/study/scripts/07_alpamayo_pipeline.py, for the other model.
Every tensor between the DrivingScene and the 50 waypoints is produced by a function in this
file. The VLM itself is called as a module (it is stock Qwen3.5 from HuggingFace, and the
point of this file is the *planning expert*, which is the part that is not stock).

Read it top to bottom. F10 to walk the steps, F11 to descend into any unrolled_* helper.

    STEP 1  scene -> prompt                12 images, 3385 tokens
    STEP 2  VLM prefill -> KV cache        ONLY the 8 full-attention layers leave one
    STEP 3  mRoPE anchor                   where the waypoints sit in rotary space
    STEP 4  history -> conditioning        re-reference, drop origin, normalize, 3 encoders
    STEP 5  waypoint queries               7 signals concatenated and fused
    STEP 6  adaLN condition                flow time + nav command + ego status
    STEP 7  flow-matching Euler loop       10 steps, CLEAN-ENDPOINT parameterization
    STEP 8  denormalize -> metres          and the figure

WHAT MAKES THIS MODEL DIFFERENT, and where to put your breakpoints:

  * STEP 2. The VLM is a 3:1 hybrid — 24 of its 32 layers are Gated DeltaNet linear
    attention and keep a recurrent *state*, not keys and values. Only layers
    [3,7,11,15,19,23,27,31] leave a KV cache, so the expert can read 8 of 32 layers, and
    four consecutive expert layers share each one (`layers_per_kv = 4`).

  * STEP 5. A waypoint token is SEVEN things concatenated: the noisy waypoint, its Fourier
    features, the flow time, the encoded history poses, a learned waypoint-index embedding,
    the encoded history velocity and the encoded history acceleration.

  * STEP 7. The network predicts the FINISHED trajectory (x1), not a velocity field. The
    velocity is derived: v = (x1_hat - x_t) / max(1 - t, 0.1). Alpamayo does the opposite.

  * unrolled_rotary(). Phases are computed in **bfloat16**, deliberately — training held the
    inverse-frequency table in bf16, which rounds positions above 256 onto a coarser grid,
    and the weights were fitted against those exact phases. Running this in fp32 is not a
    free precision upgrade; it is a distribution shift. See study/03_LOCAL_SETUP.md.

Run (CPU-only, safe beside a GPU job):

    PYTHONPATH=src python study/scripts/07_qwen_drive_pipeline.py
    PYTHONPATH=src python study/scripts/07_qwen_drive_pipeline.py --verify
    PYTHONPATH=src python study/scripts/07_qwen_drive_pipeline.py --mode reasoning_planning
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qwen_drive import InferenceMode, QwenDriveForPlanning  # noqa: E402
from qwen_drive.benchmarks import read_scene_file  # noqa: E402
from qwen_drive.images import ImageArchive  # noqa: E402
from qwen_drive.trajectory import denormalize_trajectory, normalize_history  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 96}\n  {title}\n{'=' * 96}")


def show(name: str, t) -> None:
    if torch.is_tensor(t):
        print(f"    {name:<38} {str(tuple(t.shape)):<26} {str(t.dtype).replace('torch.',''):<10}"
              f" |x| mean {t.float().abs().mean():.4f}")
    else:
        print(f"    {name:<38} {t}")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  KERNELS  — spelled out, so a breakpoint here shows the arithmetic
# ══════════════════════════════════════════════════════════════════════════════════════════
def rms_norm(module, x: torch.Tensor) -> torch.Tensor:
    """RMSNorm exactly as qwen_drive.planning_expert.RMSNorm does it.

    Note the ordering: reduce in fp32, scale by the weight IN fp32, and only then cast back.
    (`x * self.weight.float()` then `.to(dtype)`.) Doing the multiply after the cast is a
    different number in bf16.
    """
    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + module.eps)
    return (x32 * module.weight.float()).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = torch.chunk(x, 2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate only the leading `partial_rotary_factor * head_dim` channels (64 of 256)."""
    rotary_dim = cos.shape[-1]
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    rotated = rotated * cos + rotate_half(rotated) * sin
    return torch.cat([rotated, passthrough], dim=-1)


def attention(q, k, v, n_heads, n_kv) -> torch.Tensor:
    """Non-causal grouped-query attention. q/k/v arrive as [B, L, H, D]."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=(n_heads != n_kv)
    )
    return out.transpose(1, 2)


# ══════════════════════════════════════════════════════════════════════════════════════════
#  ROTARY  —  interleaved multi-section mRoPE, in bf16 on purpose
# ══════════════════════════════════════════════════════════════════════════════════════════
def unrolled_rotary(rotary, position_ids: torch.Tensor, dtype):
    """position_ids [3, B, L] -> cos/sin [B, L, 1, rotary_dim].

    Three mRoPE sections (11, 11, 10 frequency pairs) are INTERLEAVED, not concatenated:
    section s owns every third frequency starting at s. All three sections carry the same
    positions here, because the last prefix token is text, so the anchor is shared.
    """
    device = position_ids.device
    exponents = torch.arange(0, rotary.rotary_dim, 2, dtype=torch.float32, device=device)
    inv_freq = (1.0 / (rotary.rope_theta ** (exponents / rotary.rotary_dim))).to(dtype)
    angles = position_ids.to(dtype).unsqueeze(-1) * inv_freq          # [3, B, L, pairs]
    merged = angles[0].clone()
    for offset, length in enumerate(rotary.sections[1:], start=1):
        merged[..., offset : length * 3 : 3] = angles[offset][..., offset : length * 3 : 3]
    emb = torch.cat([merged, merged], dim=-1).unsqueeze(2)            # [B, L, 1, dim]
    return emb.cos(), emb.sin()


# ══════════════════════════════════════════════════════════════════════════════════════════
#  ONE EXPERT LAYER  —  adaLN + joint attention over [scene KV ; waypoint KV] + SwiGLU
# ══════════════════════════════════════════════════════════════════════════════════════════
def unrolled_expert_layer(layer, hidden, scene_key, scene_value, cos, sin, condition):
    batch, length, _ = hidden.shape

    # adaLN: one Linear(1024 -> 6*1024) produces shift/scale/gate for attention and for FFN.
    # This is 6.291 M of the layer's 31.990 M — a fifth of the expert is conditioning.
    modulation = layer.adaln_modulation(condition).chunk(6, dim=-1)
    shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
        m.unsqueeze(1) for m in modulation
    )

    # ── attention ──────────────────────────────────────────────────────────────────────
    residual = hidden
    x = rms_norm(layer.input_layernorm, hidden) * (1 + scale_attn) + shift_attn

    # Fused QKV in the VLM's layout: per KV group, [4 query heads, 4 output gates, K, V].
    fused = layer.qkv_proj(x)
    head_dim, groups, per_group = layer.head_dim, layer.num_kv_heads, layer.heads_per_group
    fused = fused.view(batch, length, groups, (per_group * 2 + 2) * head_dim)
    gated_query, key, value = torch.split(
        fused, [per_group * 2 * head_dim, head_dim, head_dim], dim=3
    )
    query, gate = torch.chunk(gated_query, 2, dim=-1)
    query = query.reshape(batch, length, layer.num_heads, head_dim)
    gate = gate.reshape(batch, length, layer.num_heads, head_dim)
    key = key.reshape(batch, length, groups, head_dim)
    value = value.reshape(batch, length, groups, head_dim)

    # per-head RMSNorm on the head_dim, then partial rotary
    query = apply_rotary(rms_norm(layer.q_norm, query), cos, sin)
    key = apply_rotary(rms_norm(layer.k_norm, key), cos, sin)

    # JOINT attention: the waypoints see the frozen scene AND each other, in one softmax,
    # non-causally. There is no separate self-attention block in this expert.
    attn = attention(
        query,
        torch.cat([scene_key, key], dim=1),
        torch.cat([scene_value, value], dim=1),
        layer.num_heads, layer.num_kv_heads,
    ).reshape(batch, length, -1)

    attn = attn * torch.sigmoid(gate.reshape(batch, length, -1))   # per-head output gate
    hidden = residual + (1 + gate_attn) * layer.o_proj(attn)       # note: (1 + gate), not gate

    # ── feed-forward (SwiGLU) ──────────────────────────────────────────────────────────
    residual = hidden
    x = rms_norm(layer.post_attention_layernorm, hidden) * (1 + scale_ffn) + shift_ffn
    swiglu_gate, swiglu_up = layer.gate_up_proj(x).chunk(2, dim=-1)
    return residual + (1 + gate_ffn) * layer.down_proj(F.silu(swiglu_gate) * swiglu_up)


# ══════════════════════════════════════════════════════════════════════════════════════════
#  THE SEVEN-WAY WAYPOINT QUERY  +  ONE DENOISING STEP
# ══════════════════════════════════════════════════════════════════════════════════════════
def unrolled_fourier(enc, waypoints: torch.Tensor) -> torch.Tensor:
    """Per-channel Fourier features: 16 log-spaced frequencies to 16.0, sin and cos.

    The frequency table is rebuilt in the module's dtype every call — training stored it in
    bf16 and its rounding is part of the features the weights expect.
    """
    weight = enc.net[0].weight
    freqs = torch.logspace(0, math.log10(enc.max_frequency), steps=enc.num_features,
                           device=weight.device, dtype=weight.dtype)
    angles = waypoints.float().unsqueeze(-1) * freqs * (2 * math.pi)
    features = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)   # 3*16*2 = 96
    return enc.net(features.to(waypoints.dtype))


def unrolled_predict_endpoint(expert, waypoints, flow_time, history_queries, scene_cache,
                              anchor, nav_onehot, ego_status, *, trace=False):
    """One forward of the expert: noisy trajectory -> predicted CLEAN trajectory."""
    dtype = expert.dtype
    waypoints = waypoints.to(dtype)
    batch, length, _ = waypoints.shape
    pose_q, vel_q, acc_q = history_queries

    time_condition = expert.time_mlp(expert.time_embed(flow_time).to(dtype))
    index = torch.arange(length, device=waypoints.device)

    seven = [
        expert.trajectory_proj(waypoints),                                    # 1 noisy waypoint
        unrolled_fourier(expert.fourier_encoder, waypoints),                  # 2 Fourier feats
        time_condition.unsqueeze(1).expand(-1, length, -1),                   # 3 flow time
        pose_q.unsqueeze(1).expand(-1, length, -1),                           # 4 history poses
        expert.waypoint_embed(index).unsqueeze(0).expand(batch, -1, -1).to(dtype),  # 5 index
        vel_q.unsqueeze(1).expand(-1, length, -1),                            # 6 history vel
        acc_q.unsqueeze(1).expand(-1, length, -1),                            # 7 history acc
    ]
    if trace:
        for i, part in enumerate(seven, 1):
            show(f"query part {i}", part)
    hidden = expert.query_fusion(torch.cat(seven, dim=-1))                    # 7*1024 -> 1024

    # adaLN condition: three summed signals, injected into every layer
    condition = (time_condition
                 + expert.nav_mlp(nav_onehot)
                 + expert.ego_mlp(ego_status.to(dtype)))

    positions = anchor.unsqueeze(-1) + torch.arange(1, length + 1, device=anchor.device,
                                                    dtype=anchor.dtype)
    cos, sin = unrolled_rotary(expert.rotary_emb, positions, dtype)

    per_kv = expert.config.layers_per_kv
    for i, layer in enumerate(expert.layers):
        scene_key, scene_value = scene_cache[i // per_kv]     # 4 layers share one VLM cache
        hidden = unrolled_expert_layer(
            layer, hidden, scene_key.expand(batch, -1, -1, -1),
            scene_value.expand(batch, -1, -1, -1), cos, sin, condition,
        )
    return expert.out_proj(rms_norm(expert.final_layernorm, hidden)).float()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--mode", default="direct_planning",
                    choices=["direct_planning", "reasoning_planning"])
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--verify", action="store_true",
                    help="also run the model's own sample() and diff the trajectories")
    ap.add_argument("--plot", default=None, help="write a PNG here")
    ap.add_argument("--sweep-steps", default=None,
                    help="comma-separated Euler step counts to compare, e.g. 1,2,5,10,20. "
                         "Reuses the one expensive prefill, so this is nearly free.")
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    rule(f"LOAD   device={args.device}  planner={Path(args.planner).name}  mode={args.mode}")
    model = QwenDriveForPlanning.from_pretrained(
        args.model, planner=args.planner, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    expert = model.planning_expert
    cfg = model.config
    print(f"    VLM     {sum(p.numel() for p in model.vlm.parameters())/1e9:.4f} B")
    print(f"    expert  {sum(p.numel() for p in expert.parameters())/1e9:.4f} B  "
          f"{cfg.expert_config.num_hidden_layers} layers x width {cfg.expert_config.hidden_size}")
    print(f"    VLM layers leaving a KV cache: {cfg.full_attention_layers}")
    print(f"    layers_per_kv = {cfg.expert_config.layers_per_kv}  -> "
          f"{cfg.expert_config.num_kv_sources} caches for "
          f"{cfg.expert_config.num_hidden_layers} expert layers")

    # ── STEP 1  scene -> prompt ────────────────────────────────────────────────────────
    rule("STEP 1  scene -> prompt")
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    sample = list(read_scene_file(args.scenes, image_archive=archive,
                                  num_history_points=cfg.num_history_points,
                                  limit=args.index + 1))[args.index]
    scene = sample.scene
    mode = InferenceMode(args.mode)
    inputs = model.processor(scene, with_reasoning=(mode is InferenceMode.REASONING_PLANNING),
                             device=args.device)
    ids = inputs["input_ids"]
    n_img = int((ids[0] == cfg.vlm_config.image_token_id).sum())
    print(f"    token {sample.token}   nav_command {scene.nav_command}")
    print(f"    views {list(scene.views)}  x {scene.num_camera_frames} timestamps")
    show("input_ids", ids)
    print(f"    {n_img} image tokens + {ids.shape[1]-n_img} text = {ids.shape[1]}")
    show("pixel_values", inputs["pixel_values"])
    print(f"    image_grid_thw {inputs['image_grid_thw'].tolist()}")

    # ── STEP 2  VLM prefill -> the 8 caches the expert can read ────────────────────────
    rule("STEP 2  VLM prefill -> KV cache (only full_attention layers leave one)")
    with torch.no_grad():
        if mode is InferenceMode.REASONING_PLANNING:
            scene_cache, anchor, reasoning = model._prefill_with_reasoning(
                inputs, cfg.max_reasoning_tokens)
            print(f"    reasoning: {reasoning!r}")
        else:
            scene_cache, anchor = model._prefill(inputs)
            reasoning = None
    print(f"    {len(scene_cache)} caches, one per full-attention layer")
    show("scene_cache[0] keys", scene_cache[0][0])
    show("scene_cache[0] values", scene_cache[0][1])
    print("    layout [batch, sequence, kv_heads, head_dim] — post-rotary, no projection:")
    print("    the expert's head_dim/num_key_value_heads were COPIED from the VLM so these")
    print("    tensors can be consumed as-is.")

    # ── STEP 3  where the waypoints sit in rotary space ────────────────────────────────
    rule("STEP 3  mRoPE anchor")
    show("anchor (3 mRoPE sections)", anchor)
    print(f"    waypoints occupy positions {int(anchor[0,0])+1} .. "
          f"{int(anchor[0,0])+cfg.num_future_points}, continuing the prefix")
    print(f"    rotary_dim {expert.rotary_emb.rotary_dim} of head_dim "
          f"{cfg.expert_config.head_dim}  (partial_rotary_factor "
          f"{cfg.expert_config.partial_rotary_factor})")
    print(f"    sections {expert.rotary_emb.sections}  theta {expert.rotary_emb.rope_theta:g}")

    # ── STEP 4  history -> conditioning ────────────────────────────────────────────────
    rule("STEP 4  history -> conditioning queries")
    scale = model.trajectory_scale(torch.device(args.device))
    print(f"    trajectory_scale {scale.tolist()}   (heading = pi/2 rounded to bf16)")
    raw_hist = inputs["history"].float()
    hist = normalize_history(raw_hist, scale)
    show("history raw [16,3]", raw_hist)
    show("history normalized", hist)
    print(f"    re-referenced to the OLDEST pose, origin row dropped -> "
          f"{hist.shape[-2]} poses of {hist.shape[-1]}")

    n = args.num_samples
    tile = lambda t: t.repeat_interleave(n, dim=0)
    nav = tile(inputs["nav_command"])
    ego = tile(inputs["ego_status"].float())
    nav_onehot = torch.nn.functional.one_hot(
        nav.clamp(0, cfg.expert_config.nav_command_classes - 1).long(),
        cfg.expert_config.nav_command_classes).to(expert.dtype)
    with torch.no_grad():
        history_queries = expert.encode_history(
            tile(hist), nav, tile(inputs["history_velocity"].float()),
            tile(inputs["history_acceleration"].float()))
    for nm, q in zip(("pose_query", "velocity_query", "acceleration_query"), history_queries):
        show(nm, q)
    print(f"    ego_status = [vx, vy, ax, ay, *driving_command] = "
          f"{ego[0].tolist()}")

    # ── STEP 5-7  the flow-matching loop ───────────────────────────────────────────────
    rule("STEP 5+6+7  waypoint queries, adaLN condition, flow-matching Euler loop")
    steps = cfg.num_inference_steps
    waypoints = (cfg.noise_init_std
                 * model._initial_noise(n, cfg.num_future_points, cfg.noise_seed,
                                        torch.device(args.device))).float()
    show("x_0  (gaussian noise)", waypoints)
    print(f"    {steps} Euler steps, dt = {1.0/steps:.2f}, "
          f"min_one_minus_t = {cfg.min_one_minus_t}")
    print()
    print("      step   t      remaining   |x_t|     |x1_hat|   ||dx||")
    print("      " + "-" * 62)
    step = 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((n,), i * step, device=waypoints.device, dtype=torch.float32)
            endpoint = unrolled_predict_endpoint(
                expert, waypoints, t, history_queries, scene_cache,
                anchor.expand(-1, n), nav_onehot, ego, trace=(i == 0 and args.verify))
            remaining = max(1.0 - i * step, cfg.min_one_minus_t)
            delta = (endpoint - waypoints) / remaining * step
            print(f"      {i:>3}   {i*step:4.2f}   {remaining:6.2f}    "
                  f"{waypoints.abs().mean():7.4f}   {endpoint.abs().mean():7.4f}  "
                  f"{delta.norm():7.4f}")
            waypoints = waypoints + delta
    print("\n    the network predicts the FINISHED trajectory (x1); the velocity is derived.")
    print("    the 0.1 floor on `remaining` stops the last step dividing by ~0.")

    # ── STEP 8  denormalize ────────────────────────────────────────────────────────────
    rule("STEP 8  denormalize -> metres and radians")
    traj = denormalize_trajectory(waypoints, scale).cpu().numpy()
    show("trajectory", torch.tensor(traj))
    print(f"    {traj.shape[1]} waypoints @ {cfg.trajectory_hz:g} Hz = "
          f"{traj.shape[1]/cfg.trajectory_hz:.1f} s")
    print(f"    endpoint (x, y, heading) = {np.round(traj[0,-1],3).tolist()}")
    if sample.future_trajectory is not None:
        gt = sample.future_trajectory
        err = np.linalg.norm(traj[0][:, :2] - gt[: traj.shape[1], :2], axis=-1)
        print(f"    ground truth endpoint     = {np.round(gt[-1],3).tolist()}")
        print(f"    ADE {err.mean():.3f} m   FDE {err[-1]:.3f} m")

    # ── optional: prove the unrolled path matches the library one ──────────────────────
    if args.verify:
        rule("VERIFY  unrolled vs. the model's own PlanningExpert.sample()")
        with torch.no_grad():
            reference = model._plan_from_cache(
                scene_cache, anchor, inputs, num_samples=n,
                num_steps=steps, seed=cfg.noise_seed)
        diff = np.abs(reference - traj)
        print(f"    max abs diff  {diff.max():.3e} m")
        print(f"    mean abs diff {diff.mean():.3e} m")
        print("    (expect 0.000e+00 — same seeds, same kernels, same order of operations)"
              if diff.max() == 0 else "    NON-ZERO: the unrolled path has diverged.")

    # ── how many Euler steps do you actually need? ─────────────────────────────────────
    if args.sweep_steps:
        rule("SWEEP  Euler step count  (one prefill, reused)")
        print("    the endpoint prediction barely moves across the default 10 steps, so the")
        print("    loop is close to a straight interpolation from noise to a fixed target.")
        print()
        print("      steps   endpoint (x, y, heading)              ADE      FDE    vs 10-step")
        print("      " + "-" * 72)
        base = None
        for k in sorted({int(x) for x in args.sweep_steps.split(",")}):
            w = (cfg.noise_init_std
                 * model._initial_noise(n, cfg.num_future_points, cfg.noise_seed,
                                        torch.device(args.device))).float()
            st = 1.0 / k
            with torch.no_grad():
                for i in range(k):
                    tt = torch.full((n,), i * st, device=w.device, dtype=torch.float32)
                    ep = unrolled_predict_endpoint(expert, w, tt, history_queries, scene_cache,
                                                   anchor.expand(-1, n), nav_onehot, ego)
                    w = w + (ep - w) / max(1.0 - i * st, cfg.min_one_minus_t) * st
            tj = denormalize_trajectory(w, scale).cpu().numpy()
            if k == 10:
                base = tj
            ade = fde = float("nan")
            if sample.future_trajectory is not None:
                gtx = sample.future_trajectory
                e = np.linalg.norm(tj[0][:, :2] - gtx[: tj.shape[1], :2], axis=-1)
                ade, fde = e.mean(), e[-1]
            delta = ("" if base is None
                     else f"   {np.abs(tj - base).max():.4f} m max")
            print(f"      {k:>5}   {np.round(tj[0,-1],3).tolist()!s:<36} "
                  f"{ade:6.3f}  {fde:6.3f}{delta}")
        print()
        print("    if 2-3 steps land within a few cm of 10, the sampler is doing far less")
        print("    work than its step count suggests, and the expert cost drops accordingly.")

    if args.plot:
        from qwen_drive.visualize import plot_scene_summary
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot_scene_summary(scene, traj, history=scene.history,
                           ground_truth=sample.future_trajectory, reasoning=reasoning,
                           title=f"{sample.token} ({args.mode})", output=args.plot)
        print(f"\n    wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
