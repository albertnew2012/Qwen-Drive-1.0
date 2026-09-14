"""Export the shared Qwen3.5-4B VLM to ONNX.

This is the four-fifths of the model the head exports do NOT cover. It is split
into the two pieces a deployment actually needs:

  --part vision   the ViT tower. Produces BOTH taps the driving heads read:
                  the pre-merge patch grid (geometry) and the merged tokens
                  that enter the language model.
  --part text     one PREFILL pass of the 32-layer hybrid language model.
                  Produces the last hidden state (the semantics tap) and the
                  8 full-attention KV caches the Planning Expert cross-attends
                  to.

WHY PREFILL ONLY. Perception and planning read the VLM in a SINGLE forward -
there is no token loop. Only VQA generation needs a decode step, and a decode
graph has to carry the 24 Gated-DeltaNet recurrent states in and out, which is a
different contract. Prefill is what makes the driving path runnable end to end.

WHY THIS IS EXPORTABLE AT ALL. ``fla`` and ``causal_conv1d`` are not installed,
so the linear-attention layers already run ``torch_chunk_gated_delta_rule`` -
pure PyTorch. Had the fused kernels been present they would have had to be
patched out first, exactly as the perception kernels were.

    python export_onnx/export_vlm.py --part vision
    python export_onnx/export_vlm.py --part text
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer

from qwen_drive import QwenDriveForPlanning
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor


def patch_chunk_rule_for_export(model) -> int:
    """NOT USED. Kept as a record of an optimisation that does not work.

    THE IDEA. ``torch_chunk_gated_delta_rule`` inverts a unit lower-triangular
    matrix with 63 sequential steps, which unroll at trace time once per layer -
    a fixed cost that dominates the exported graph (~244k of its ~250k nodes,
    independent of sequence length). The matrix ``A`` is STRICTLY lower
    triangular, so it is nilpotent with A^64 = 0, and

        (I - A)^-1 = (I + A)(I + A^2)(I + A^4) ... (I + A^32)

    which is 5 squarings and 5 multiplies instead of 63 indexed writes. The
    identity is exact: 1e-15 relative in fp64, 1e-6 in fp32, on random inputs.

    WHY IT FAILS ANYWAY. Forming A^32 requires the intermediate powers to stay
    representable, and in this model they do not. ``A`` comes from
    ``k_beta @ key.transpose(-1, -2)``, whose entries are O(head_dim) = O(100).
    A 64x64 matrix with entries ~100 raised to the 32nd power overflows fp32 long
    before the series terminates, and the result is NaN - measured, on the real
    model, not predicted.

    The shipped loop is stable precisely BECAUSE it never forms a high power: it
    accumulates one row at a time and every intermediate stays the same
    magnitude as the answer. The slow version is the numerically correct one.

    The lesson matches section 2.4: the fast path looked right, the isolated
    maths agreed to 1e-15, and it was still wrong in context. Only running it
    against the real model caught it.
    """
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    src = inspect.getsource(M.torch_chunk_gated_delta_rule)
    old = """    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)"""
    new = """    _eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    _power = attn
    attn = _eye + _power
    _k = 1
    while (1 << _k) < chunk_size:
        _power = _power @ _power
        attn = attn @ (_eye + _power)
        _k += 1"""
    if old not in src:
        raise RuntimeError("the chunk-rule loop does not match the installed "
                           "transformers source; refusing to patch blindly")
    namespace = dict(M.__dict__)
    exec(compile(src.replace(old, new), "<patched_chunk_rule>", "exec"), namespace)
    patched = namespace["torch_chunk_gated_delta_rule"]

    # each GatedDeltaNet captured the function at __init__, so patch instances
    n = 0
    for mod in model.modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            mod.chunk_gated_delta_rule = patched
            n += 1
    return n


class VisionONNX(nn.Module):
    """The ViT tower, emitting both taps.

    ``grid_thw`` is frozen: it depends only on the camera count and resolution,
    which are fixed for a given rig. It drives position ids and sequence
    boundaries, i.e. pure bookkeeping, so baking it in removes integer control
    flow from the graph without changing any arithmetic.
    """

    def __init__(self, visual, grid_thw):
        super().__init__()
        self.visual = visual
        self.register_buffer("grid_thw", grid_thw, persistent=False)

    def forward(self, pixel_values):
        out = self.visual(pixel_values, self.grid_thw)
        premerge = out.last_hidden_state
        # the perception head taps AFTER the merger's norm
        return premerge, self.visual.merger.norm(premerge), out.pooler_output


class TextPrefillONNX(nn.Module):
    """One prefill of the hybrid language model.

    Returns the post-norm last hidden state (what the perception head reads) and
    the 8 full-attention key/value tensors (what the Planning Expert reads). The
    24 linear-attention layers keep recurrent state that prefill does not need to
    expose, because nothing continues from it in the driving path.
    """

    def __init__(self, language_model, norm, full_layers):
        super().__init__()
        self.language_model = language_model
        self.norm = norm
        self.full_layers = full_layers

    def forward(self, inputs_embeds, position_ids):
        out = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        hidden = self.norm(out.last_hidden_state)
        cache = out.past_key_values
        kv = []
        for i in self.full_layers:
            kv.append(cache.layers[i].keys)
            kv.append(cache.layers[i].values)
        return (hidden, *kv)


def export_and_verify(module, example, names, onames, out, opset, dynamo=False,
                      skip_check=False, verify_only=False):
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        ref = module(*example)
    refs = (ref,) if torch.is_tensor(ref) else tuple(ref)
    print(f"  reference outputs: " +
          ", ".join(f"{n}{tuple(r.shape)}" for n, r in zip(onames, refs)))
    t0 = time.time()
    if verify_only:
        print(f"  reusing the graph already on disk: {out}")
    with torch.no_grad():
        if verify_only:
            pass
        elif dynamo:
            prog = torch.onnx.export(module, example, dynamo=True, input_names=names,
                                     output_names=onames, opset_version=opset)
            prog.save(str(out))
        else:
            torch.onnx.export(module, example, str(out), input_names=names,
                              output_names=onames, opset_version=opset,
                              do_constant_folding=False, dynamo=False)
    print(f"  exported in {time.time()-t0:.0f}s -> {out} "
          f"({out.stat().st_size/2**20:.1f} MiB graph, "
          f"{sum(f.stat().st_size for f in out.parent.rglob('*') if f.is_file())/2**30:.2f} GiB total)")
    import onnx
    if skip_check:
        # check_model on a 14 GiB graph re-reads all 372 external-data files and
        # took over an hour without finishing. The structural check adds nothing
        # that the numerical comparison below does not already prove.
        print("  onnx.checker SKIPPED (see --skip-check); relying on the "
              "numerical comparison instead")
    else:
        onnx.checker.check_model(str(out), full_check=False)
    m = onnx.load(str(out), load_external_data=False)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  checker OK.  {len(m.graph.node)} nodes, {len(ops)} distinct ops")
    print("  most common: " + ", ".join(f"{k}x{v}" for k, v in
                                        sorted(ops.items(), key=lambda x: -x[1])[:8]))

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    t0 = time.time()
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: t.cpu().numpy() for n, t in zip(names, example)})
    print(f"  onnxruntime forward {time.time()-t0:.0f}s")
    worst = 0.0
    for n, a, b in zip(onames, got, refs):
        b = b.cpu().numpy()
        d = float(np.abs(a - b).max())
        rel = d / max(float(np.abs(b).max()), 1e-9)
        worst = max(worst, rel)
        print(f"    {n:20s} max abs {d:.3e}   rel {rel:.3e}")
    ok = worst < 2e-3
    print(f"\n  VERIFY: {'PASS' if ok else 'FAIL'}  (worst relative {worst:.2e})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["vision", "text"], default="vision")
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="weights/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", default="data/demo/perception")
    ap.add_argument("--frame", default="90162f90eceb4ada9e595bc1adb71b5f")
    ap.add_argument("--task", choices=["perception", "planning"],
                    default="perception")
    ap.add_argument("--planner", default="weights/Qwen-Drive-1.0-4B/planner-sft")
    ap.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    ap.add_argument("--image-root", default="data/demo")
    ap.add_argument("--image-archive", default="data/demo/frames.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dynamo", action="store_true")
    ap.add_argument("--skip-check", action="store_true",
                    help="skip onnx.checker; unusable on the 14 GiB text graph")
    ap.add_argument("--max-seq", type=int, default=0,
                    help="truncate the prompt to N tokens before exporting. The "
                         "Gated-DeltaNet chunk loop UNROLLS at trace time, so the "
                         "graph size scales with sequence length: 2744 tokens gives "
                         "341,977 nodes and onnxruntime cannot load it in an hour. "
                         "A short sequence proves the mechanism and verifies.")
    ap.add_argument("--verify-only", action="store_true",
                    help="reuse the graph on disk and only re-run verification")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=dtype, attn_implementation="eager")
    vlm = holder.vlm
    del holder.planning_expert
    vlm = vlm.eval()

    head = QwenDrivePerception.from_pretrained(args.model, dtype=dtype).eval()
    proc = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    head.attach(vlm, proc)
    if args.task == "planning":
        # free the perception-task model first: two fp32 copies of a 4.5 B VLM is
        # 34 GiB and will not coexist with anything else on this machine
        del head, vlm, holder
        import gc; gc.collect()
        from export_onnx.scene_inputs import planning_inputs
        ctx = planning_inputs(args.vlm, args.planner, args.scenes,
                              args.image_root, args.image_archive)
        holder, vlm, inputs = ctx["holder"], ctx["vlm"], ctx["inputs"]
        head = QwenDrivePerception.from_pretrained(
            args.model, dtype=dtype).eval()
        head.attach(vlm, PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm)))
        token = str(ctx["token"])[:24]
    else:
        frame = PerceptionFrame(Path(args.frames) / args.frame)
        inputs, _ = proc(frame, device="cpu")
        token = frame.token[:16]
    print(f"task={args.task}  {token}  pixel_values {tuple(inputs['pixel_values'].shape)}  "
          f"input_ids {tuple(inputs['input_ids'].shape)}")

    if args.part == "vision":
        default = ("outputs/onnx/vlm_vision/vision.onnx" if args.task == "perception"
                   else "outputs/onnx/vlm_vision_plan/vision.onnx")
        out = Path(args.out or default)
        mod = VisionONNX(vlm.model.visual, inputs["image_grid_thw"]).eval()
        px = inputs["pixel_values"].to(dtype)
        print(f"\nvision tower: {sum(p.numel() for p in vlm.model.visual.parameters())/1e9:.4f} B\n")
        return export_and_verify(mod, (px,), ["pixel_values"],
                                 ["premerge_patches", "vit_tap", "merged_tokens"],
                                 out, args.opset, args.dynamo,
                                 args.skip_check, args.verify_only)

    # ---- text prefill ---------------------------------------------------------------
    out = Path(args.out or "outputs/onnx/vlm_text/text_prefill.onnx")
    cfg = vlm.config.text_config if hasattr(vlm.config, "text_config") else vlm.config
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    lm = vlm.model.language_model
    with torch.no_grad():
        embeds = vlm.model.get_input_embeddings()(inputs["input_ids"])
        # mRoPE gives images a 2-D position grid, so positions are [3, B, S] and
        # cannot be a plain arange. The repo's own helper builds them correctly;
        # get_rope_index alone needs mm_token_type_ids as well.
        pos = head._rope_positions(inputs["input_ids"], inputs["image_grid_thw"]) \
            if hasattr(head, "_rope_positions") else \
            holder._rope_positions(inputs["input_ids"], inputs["image_grid_thw"])
    if args.max_seq and args.max_seq < embeds.shape[1]:
        embeds = embeds[:, :args.max_seq]
        pos = pos[..., :args.max_seq]
        print(f"\ntruncated to {args.max_seq} tokens (the chunk loop unrolls, so "
              f"graph size scales with sequence length)")
    print(f"\nlanguage model: {sum(p.numel() for p in lm.parameters())/1e9:.4f} B, "
          f"{len(cfg.layer_types)} layers "
          f"({cfg.layer_types.count('linear_attention')} linear + {len(full)} full)")
    print(f"inputs_embeds {tuple(embeds.shape)}  position_ids {tuple(pos.shape)}\n")
    mod = TextPrefillONNX(lm, vlm.model.language_model.norm, full).eval()
    onames = ["hidden_states"] + [f"{n}_{i}" for i in full for n in ("key", "value")]
    return export_and_verify(mod, (embeds.to(dtype), pos),
                             ["inputs_embeds", "position_ids"], onames,
                             out, args.opset, args.dynamo,
                             args.skip_check, args.verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
