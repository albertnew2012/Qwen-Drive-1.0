"""Run the PLANNING path end to end from ONNX, and check the trajectory.

    vision.onnx ──► merged ──► (host: embed + scatter + mRoPE)
                                        │
                     layer_00..31.onnx ─┴──► 8 x (keys, values)
                                                   │
                        planner_step.onnx x10 Euler ──► normalized waypoints
                                                   │
                                    (host: denormalize) ──► trajectory [50, 3]

The Euler loop stays on the host on purpose: ``PlanningExpert.sample`` is a
Python loop around one network, so exporting the single step lets the step count
change without re-exporting.

Shapes are frozen per task, so this uses the PLANNING graphs (3385 tokens),
not the perception ones (2744).

    python export_onnx/run_onnx_planner.py --phase run
    python export_onnx/run_onnx_planner.py --phase compare
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

import onnxruntime as ort

from export_onnx.run_onnx_pipeline import host_embed, host_scatter_vision, rel, session


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-root", default="data/demo")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--vision", default="outputs/onnx/vlm_vision_plan/vision.onnx")
    ap.add_argument("--layers", default="outputs/onnx/vlm_layers_plan")
    ap.add_argument("--step", default="outputs/onnx/planner/planner_step.onnx")
    ap.add_argument("--save", default="outputs/onnx_run_plan")
    ap.add_argument("--phase", choices=["run", "compare", "both"], default="both")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=5e-2,
                    help="trajectory tolerance in NORMALISED units")
    args = ap.parse_args()

    rule = lambda t: print(f"\n{'='*88}\n  {t}\n{'='*88}")
    save = Path(args.save); save.mkdir(parents=True, exist_ok=True)

    rule("SETUP")
    from export_onnx.scene_inputs import planning_inputs
    ctx = planning_inputs(args.vlm, args.planner, args.scenes, args.image_root,
                          args.image_archive, index=args.index)
    model, vlm, inputs = ctx["holder"], ctx["vlm"], ctx["inputs"]
    pos = ctx["position_ids"]
    ids = inputs["input_ids"].numpy()
    img_tok = vlm.config.image_token_id
    manifest = json.loads((Path(args.layers) / "manifest.json").read_text())
    layer_types = manifest["layer_types"]
    cfg_p = model.planning_expert.config
    n_points = model.config.num_future_points
    scale = model.trajectory_scale(torch.device("cpu"))
    print(f"  scene {str(ctx['token'])[:26]}   {ids.shape[1]} tokens   "
          f"{n_points} waypoints   {args.num_steps} Euler steps")

    from qwen_drive.trajectory import denormalize_trajectory, normalize_history
    anchor = model._rope_positions(inputs["input_ids"],
                                   inputs["image_grid_thw"])[:, :, -1]
    hist = normalize_history(inputs["history"].float(), scale)
    noise = (model.config.noise_init_std *
             model._initial_noise(1, n_points, args.seed, torch.device("cpu")))
    # capture the sampler constants BEFORE the model is freed for the ONNX run
    min_one_minus_t = float(model.config.min_one_minus_t)
    n_nav = cfg_p.nav_command_classes
    nav_onehot = torch.eye(n_nav)[inputs["nav_command"].reshape(-1).long()].reshape(
        *inputs["nav_command"].shape, n_nav)

    if args.phase in ("run", "both"):
        del model, vlm
        import gc; gc.collect()

        rule("ONNX PLANNING PIPELINE")
        t_all = time.time()
        print("  [1/4] vision tower")
        t0 = time.time()
        vis = session(Path(args.vision))
        _, vit_tap, merged = vis.run(
            None, {"pixel_values": inputs["pixel_values"].numpy().astype(np.float32)})
        del vis
        print(f"        {time.time()-t0:5.1f}s   merged {merged.shape}")

        print("  [2/4] host: embedding gather + vision scatter")
        table = np.load(Path(args.layers) / "embed_tokens.npy", mmap_mode="r")
        embeds = host_embed(np.asarray(table), ids).astype(np.float32)
        del table
        embeds = host_scatter_vision(embeds, merged, ids, img_tok)

        print(f"  [3/4] {len(layer_types)} decoder layers -> the 8 KV caches")
        hidden, kv = embeds, {}
        pos_np = pos.numpy()
        t0 = time.time()
        for i, kind in enumerate(layer_types):
            s = session(Path(args.layers) / f"layer_{i:02d}.onnx")
            if kind == "linear_attention":
                hidden = s.run(None, {"hidden_in": hidden})[0]
            else:
                hidden, k, v = s.run(None, {"hidden_in": hidden,
                                            "position_ids": pos_np})
                # _scene_cache returns [B, S, kv_heads, head_dim]; the layer graph
                # emits the cache layout [B, kv_heads, S, head_dim]
                kv[i] = (k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))
            del s
            if (i + 1) % 8 == 0:
                print(f"        {i+1:2d}/{len(layer_types)}   {time.time()-t0:5.1f}s")
        order = sorted(kv)
        print(f"        {len(order)} KV caches, each {kv[order[0]][0].shape}")

        print(f"  [4/4] planner: {args.num_steps} Euler steps")
        t0 = time.time()
        step_sess = session(Path(args.step))
        flat = [t for i in order for t in kv[i]]
        names = [x.name for x in step_sess.get_inputs()]
        way = noise.numpy().astype(np.float32)
        dt = 1.0 / args.num_steps
        for it in range(args.num_steps):
            feed = {"waypoints": way,
                    "flow_time": np.full((way.shape[0],), it * dt, np.float32),
                    "history": hist.numpy().astype(np.float32),
                    "history_velocity": inputs["history_velocity"].numpy().astype(np.float32),
                    "history_acceleration": inputs["history_acceleration"].numpy().astype(np.float32),
                    "nav_onehot": nav_onehot.numpy().astype(np.float32),
                    "ego_status": inputs["ego_status"].numpy().astype(np.float32),
                    "position_anchor": anchor.numpy()}
            for nm, t in zip(names[8:], flat):
                feed[nm] = t.astype(np.float32)
            endpoint = step_sess.run(None, feed)[0]
            remaining = max(1.0 - it * dt, min_one_minus_t)
            way = way + (endpoint - way) / remaining * dt
        del step_sess
        print(f"        {time.time()-t0:5.1f}s   normalized waypoints {way.shape}")
        traj = denormalize_trajectory(torch.from_numpy(way), scale).numpy()
        print(f"\n  ONNX planning total {time.time()-t_all:.0f}s")
        np.savez(save / "onnx_plan.npz", waypoints_norm=way, trajectory=traj,
                 hidden=hidden)
        print(f"  saved -> {save/'onnx_plan.npz'}   endpoint {traj[0,-1].round(3)}")
        if args.phase == "run":
            print("\n  now run:  --phase compare")
            return 0

    rule("ONNX vs PYTORCH")
    got = np.load(save / "onnx_plan.npz")
    if args.phase == "compare":
        ctx = planning_inputs(args.vlm, args.planner, args.scenes, args.image_root,
                              args.image_archive, index=args.index)
        model = ctx["holder"]
    print("  PyTorch reference (prefill + 10 Euler steps)...")
    t0 = time.time()
    with torch.no_grad():
        ref_traj = model.plan_from_inputs(inputs, num_samples=1,
                                          num_steps=args.num_steps, seed=args.seed)
    print(f"  {time.time()-t0:.0f}s")
    # plan_from_inputs returns a QwenDriveOutput; .trajectory is the first sample
    ref = np.asarray(ref_traj.trajectory if hasattr(ref_traj, "trajectory")
                     else ref_traj)
    if ref.ndim == 3:
        ref = ref[0]
    onx = got["trajectory"][0]
    d = np.abs(onx - ref)
    print(f"\n  trajectory {onx.shape}   PyTorch endpoint {ref[-1].round(3)}")
    print(f"                      ONNX endpoint {onx[-1].round(3)}")
    print(f"  max |diff| per axis (metres): x {d[:,0].max():.4f}  "
          f"y {d[:,1].max():.4f}  heading {d[:,2].max():.4f}")
    ade = float(np.linalg.norm(onx[:, :2] - ref[:, :2], axis=-1).mean())
    fde = float(np.linalg.norm(onx[-1, :2] - ref[-1, :2]))
    print(f"  ADE(onnx, pytorch) {ade:.5f} m      FDE {fde:.5f} m")
    ok = ade < args.tol
    print(f"\n  PLANNER {'PASS' if ok else 'FAIL'}  "
          f"(ADE {ade:.2e} m, tolerance {args.tol:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
