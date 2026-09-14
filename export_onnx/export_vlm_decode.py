"""Export ONE token-generation (decode) step of the hybrid language model.

This is the piece prefill does not cover. VQA generation runs a loop, and each
iteration must carry state forward:

  * 24 linear-attention layers each hold a ``conv_state`` (kernel 4) and a
    ``recurrent_state`` - the Gated-DeltaNet memory,
  * 8 full-attention layers each hold a growing key/value cache.

WHY THIS GRAPH IS SMALL. At decode the sequence length is 1, so
``torch_recurrent_gated_delta_rule`` loops once instead of over 43 chunks, and
none of the forward-substitution machinery that made prefill 341,977 nodes is
reached. The decode step is a few thousand nodes.

WHAT IS FROZEN. The past KV length. ONNX graphs are traced at one shape, so a
decode graph is valid for exactly one cache length. A real deployment exports
once at the maximum length and masks, or re-exports per length; this validates
the mechanism at a fixed length.

    python export_onnx/export_vlm_decode.py --prefill 64
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

from qwen_drive import QwenDriveForPlanning


CONV_STATES: list = []


def patch_causal_conv1d_update():
    """Make the conv1d state update FUNCTIONAL.

    ``torch_causal_conv1d_update`` ends with

        conv_state.copy_(hidden_states_new[:, :, -state_len:])

    and returns only the convolution output - the caller relies on that in-place
    write to advance the state. ONNX cannot write into a graph input; the tracer
    reports

        aten::copy_ on block input: 'conv_state'. This changes graph semantics.

    and the export fails. This replacement computes the same new state but
    RECORDS it in ``CONV_STATES`` instead of mutating, so it becomes a graph
    value the decode step can return. Layers execute in order, so the recording
    order is the layer order.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    import torch.nn.functional as F

    def functional_update(hidden_states, conv_state, weight, bias=None,
                          activation=None):
        _, hidden_size, seq_len = hidden_states.shape
        state_len = conv_state.shape[-1]
        hidden_states_new = torch.cat([conv_state, hidden_states],
                                      dim=-1).to(weight.dtype)
        CONV_STATES.append(hidden_states_new[:, :, -state_len:])
        out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias,
                       padding=0, groups=hidden_size)
        return F.silu(out[:, :, -seq_len:]).to(hidden_states.dtype)

    M.torch_causal_conv1d_update = functional_update
    # the layer captured it at __init__ as well
    return functional_update


def patch_conv_state_update():
    """Make the conv-state update out-of-place so it can be traced.

    ``update_conv_state`` ends with

        self.conv_states[state_idx].copy_(full_conv_states[..., -k:])

    and the comment says why: "Copy instead of assigning to keep the static
    address", which matters for CUDA-graph capture. ONNX has no way to express a
    write into an existing buffer, and the export dies with

        UnsupportedOperatorError: Exporting the operator 'aten::copy'

    Assigning the slice instead computes the same value; only the storage
    identity differs, and nothing downstream of the export depends on that.
    """
    import inspect, textwrap
    from transformers import cache_utils as C

    # Every in-place cache write that the DECODE path reaches. Each is
    # "copy into the existing buffer to keep the address stable", which ONNX
    # cannot express; assigning computes the same value.
    rewrites = {
        "update_conv_state": (
            "self.conv_states[state_idx].copy_("
            "full_conv_states[..., -self.conv_kernel_size[state_idx] :])",
            "self.conv_states[state_idx] = "
            "full_conv_states[..., -self.conv_kernel_size[state_idx] :]"),
        "update_recurrent_state": (
            "self.recurrent_states[state_idx].copy_(recurrent_states)",
            "self.recurrent_states[state_idx] = recurrent_states"),
    }
    # Several layer classes carry their own copy of this method; patch every one
    # that actually contains the in-place write.
    patched = []
    for name in dir(C):
        obj = getattr(C, name)
        if not isinstance(obj, type):
            continue
        for meth, (old, new) in rewrites.items():
            if meth not in vars(obj):
                continue
            try:
                src = inspect.getsource(getattr(obj, meth))
            except (OSError, TypeError):
                continue
            if old not in src:
                continue
            ns = dict(vars(C))
            # dedent, not cleandoc: cleandoc is docstring-specific and
            # destroys a method body's indentation
            exec(compile(textwrap.dedent(src.replace(old, new)),
                         "<patched_cache>", "exec"), ns)
            setattr(obj, meth, ns[meth])
            patched.append(f"{name}.{meth}")
    if not patched:
        raise RuntimeError("no cache layer class had an in-place state write")
    return f"{len(patched)} method(s)"


class DecodeStepONNX(nn.Module):
    """One decode step: embedding + all past state in, hidden + new state out.

    States arrive as a flat tuple because ONNX has no notion of a Cache object.
    The order is fixed by ``state_names()`` and must be kept by the host loop.
    """

    def __init__(self, language_model, norm, cfg, template_cache):
        super().__init__()
        self.language_model = language_model
        self.norm = norm
        self.cfg = cfg
        # A cache built by a REAL prefill. Constructing one by hand leaves
        # metadata unset (conv_kernel_size comes out None), so the template is
        # reused and only its tensors are swapped for the graph inputs.
        self._template = template_cache
        self.linear_idx = [i for i, t in enumerate(cfg.layer_types)
                           if t == "linear_attention"]
        self.full_idx = [i for i, t in enumerate(cfg.layer_types)
                         if t == "full_attention"]

    def _build_cache(self, states):
        cache = self._template
        n_lin = len(self.linear_idx)
        conv = states[:n_lin]
        rec = states[n_lin:2 * n_lin]
        kv = states[2 * n_lin:]
        for j, i in enumerate(self.linear_idx):
            layer = cache.layers[i]
            layer.conv_states = [conv[j]]
            layer.recurrent_states = [rec[j]]
        for j, i in enumerate(self.full_idx):
            cache.layers[i].keys = kv[2 * j]
            cache.layers[i].values = kv[2 * j + 1]
        return cache

    def forward(self, inputs_embeds, position_ids, cache_position, *states):
        CONV_STATES.clear()
        cache = self._build_cache(states)
        out = self.language_model(inputs_embeds=inputs_embeds,
                                  position_ids=position_ids,
                                  cache_position=cache_position,
                                  past_key_values=cache, use_cache=True)
        hidden = self.norm(out.last_hidden_state)
        new = []
        # conv states come from the functional recorder, not the cache: the
        # in-place write that would have updated the cache is exactly what had
        # to be removed for the graph to be exportable.
        assert len(CONV_STATES) == len(self.linear_idx), (
            f"recorded {len(CONV_STATES)} conv states for "
            f"{len(self.linear_idx)} linear layers")
        new.extend(CONV_STATES)
        for i in self.linear_idx:
            new.append(cache.layers[i].recurrent_states[0])
        for i in self.full_idx:
            new.append(cache.layers[i].keys)
            new.append(cache.layers[i].values)
        return (hidden, *new)

    def state_names(self, prefix="in"):
        n = [f"{prefix}_conv_{i}" for i in self.linear_idx]
        n += [f"{prefix}_recur_{i}" for i in self.linear_idx]
        for i in self.full_idx:
            n += [f"{prefix}_key_{i}", f"{prefix}_value_{i}"]
        return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="weights/Qwen-Drive-1.0-4B")
    ap.add_argument("--out", default="outputs/onnx/vlm_decode/decode_step.onnx")
    ap.add_argument("--prefill", type=int, default=64,
                    help="past length to freeze the graph at")
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--dynamo", action="store_true",
                    help="the dynamo exporter functionalises in-place ops, "
                         "which the cache update is full of")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.float32, attn_implementation="eager")
    vlm = holder.vlm.eval()
    lm = vlm.model.language_model
    cfg = vlm.config.text_config if hasattr(vlm.config, "text_config") else vlm.config
    D = cfg.hidden_size

    # a real prefill, to produce real state
    from transformers.cache_utils import DynamicCache
    torch.manual_seed(0)
    pre = torch.randn(1, args.prefill, D)
    cache = DynamicCache(config=cfg)
    pos_pre = torch.arange(args.prefill).view(1, 1, -1).expand(3, 1, -1)
    with torch.no_grad():
        lm(inputs_embeds=pre, position_ids=pos_pre, use_cache=True,
           past_key_values=cache)

    fn = patch_causal_conv1d_update()
    for mod_ in lm.modules():
        if hasattr(mod_, "causal_conv1d_update"):
            mod_.causal_conv1d_update = fn
    cls = patch_conv_state_update()
    print(f"  conv-state update made out-of-place on {cls}")
    mod = DecodeStepONNX(lm, lm.norm, cfg, cache).eval()
    states = []
    for i in mod.linear_idx:
        states.append(cache.layers[i].conv_states[0].clone())
    for i in mod.linear_idx:
        states.append(cache.layers[i].recurrent_states[0].clone())
    for i in mod.full_idx:
        states.append(cache.layers[i].keys.clone())
        states.append(cache.layers[i].values.clone())
    print(f"  prefill {args.prefill} tokens -> {len(states)} state tensors "
          f"({len(mod.linear_idx)} conv + {len(mod.linear_idx)} recurrent + "
          f"{2*len(mod.full_idx)} kv)")
    print(f"    conv_state      {tuple(states[0].shape)}")
    print(f"    recurrent_state {tuple(states[len(mod.linear_idx)].shape)}")
    print(f"    kv              {tuple(states[-1].shape)}")

    tok = torch.randn(1, 1, D)
    pos = torch.full((3, 1, 1), args.prefill, dtype=torch.long)
    cpos = torch.tensor([args.prefill])
    example = (tok, pos, cpos, *states)
    in_names = ["inputs_embeds", "position_ids", "cache_position"] + mod.state_names("in")
    out_names = ["hidden_states"] + mod.state_names("out")

    with torch.no_grad():
        ref = mod(*[e.clone() if torch.is_tensor(e) else e for e in example])
    print(f"\n  reference: hidden {tuple(ref[0].shape)}, {len(ref)-1} new states")

    print(f"\nexporting decode step (opset {args.opset})...")
    t0 = time.time()
    with torch.no_grad():
        if args.dynamo:
            # No opset_version: requesting one runs onnxscript's version
            # converter, which fails on this graph ("ConvertVersionPass").
            # Letting dynamo emit its native opset skips that pass entirely.
            kw = {} if args.opset <= 0 else {"opset_version": args.opset}
            prog = torch.onnx.export(mod, tuple(e.clone() for e in example),
                                     dynamo=True, input_names=in_names,
                                     output_names=out_names, **kw)
            prog.save(str(out))
        else:
            torch.onnx.export(mod, tuple(e.clone() for e in example), str(out),
                              input_names=in_names, output_names=out_names,
                              opset_version=args.opset, do_constant_folding=False,
                              dynamo=False)
    print(f"  {time.time()-t0:.0f}s -> {out} "
          f"({out.stat().st_size/2**20:.1f} MiB graph)")
    import onnx
    onnx.checker.check_model(str(out), full_check=False)
    m = onnx.load(str(out), load_external_data=False)
    print(f"  checker OK.  {len(m.graph.node)} nodes  "
          f"(prefill was 341,977 - decode is far smaller because seq_len is 1)")
    if args.skip_verify:
        return 0

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    t0 = time.time()
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    print(f"  ORT session {time.time()-t0:.1f}s")
    # Feed only what the graph actually declares: inputs that turned out to be
    # constant (cache_position here) are folded away and ORT rejects them.
    declared = {i.name for i in sess.get_inputs()}
    dropped = [n for n in in_names if n not in declared]
    if dropped:
        print(f"  {len(dropped)} input(s) constant-folded away: {dropped[:4]}"
              f"{' ...' if len(dropped) > 4 else ''}")
    feed = {n: t.numpy() for n, t in zip(in_names, example) if n in declared}
    got = sess.run(None, feed)
    worst, worst_name = 0.0, ""
    for n, a, b in zip(out_names, got, ref):
        b = b.detach().numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            r = float("inf")
        else:
            r = float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-9)
        if r > worst:
            worst, worst_name = r, n
    print(f"\n  hidden_states rel "
          f"{np.abs(got[0]-ref[0].detach().numpy()).max()/max(np.abs(ref[0].detach().numpy()).max(),1e-9):.3e}")
    print(f"  worst over all {len(out_names)} outputs: {worst:.3e}  ({worst_name})")
    ok = np.isfinite(worst) and worst < 2e-3
    print(f"\n  DECODE STEP {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
