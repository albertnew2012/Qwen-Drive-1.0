"""Re-express a trained 1x-grid CenterPoint head as a 2x-grid head with dilated 3x3 blocks, exactly.

The 1x head starts with a 1x1 conv (in -> hidden). The 2x head starts with a ConvTranspose2d(k=2, s=2);
filling all four kernel positions with the 1x1 weights makes it a nearest 2x upsample of the old features,
and dilation-2 3x3 blocks on that map equal the upsampled 3x3 blocks of the original. So the new model's
heatmap at every (2i, 2j) equals the old heatmap at (i, j) -- verified below -- and a fine-tune starts
from the trained function.

    python local/distill/warmstart_head2.py IN.pt OUT.pt [--step 30000]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "src"))
from local.distill.student import StudentConfig, StudentDetector


def convert(sd):
    out = dict(sd)
    w = sd["center.shared.0.weight"]                 # (hidden, in, 1, 1)
    hidden, cin = w.shape[:2]
    t = torch.zeros(cin, hidden, 2, 2, dtype=w.dtype)
    for a in range(2):
        for b in range(2):
            t[:, :, a, b] = w[:, :, 0, 0].t()
    out["center.shared.0.weight"] = t
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--step", type=int, default=30000)
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu")
    sd = ck.get("ema") or ck["model"]
    cfg = dict(ck["cfg"]); assert int(cfg.get("head_upsample", 1)) == 1, "source must be a 1x head"
    new_cfg = dict(cfg, head_upsample=2, head_dilate=True)
    sd2 = convert(sd)
    # exactness check on a random BEV: heatmap at (2i, 2j) must equal the old heatmap at (i, j)
    kw = lambda c: {k: v for k, v in c.items() if k in StudentConfig.__init__.__code__.co_varnames}
    m1 = StudentDetector(StudentConfig(**kw(cfg))).eval(); m1.load_state_dict(sd, strict=True)
    m2 = StudentDetector(StudentConfig(**kw(new_cfg))).eval(); m2.load_state_dict(sd2, strict=True)
    with torch.no_grad():
        bev = torch.randn(1, m1.cfg.d_model, 200, 200)
        h1, r1 = m1.center(bev); h2, r2 = m2.center(bev)
    d_hm = (h2[:, :, ::2, ::2] - h1).abs().max().item(); d_rg = (r2[:, :, ::2, ::2] - r1).abs().max().item()
    print(f"  exactness: max|hm2[::2,::2]-hm1| = {d_hm:.2e}, max|reg diff| = {d_rg:.2e}  (hm max |value| {h1.abs().max():.2f})")
    assert d_hm < 1e-3 and d_rg < 1e-3, "warm start is not exact"
    assert not Path(a.dst).exists()
    torch.save({"model": sd2, "ema": sd2, "step": a.step, "cfg": new_cfg,
                "note": f"warm start: {a.src} re-expressed as a 2x dilated head (exact); step relabelled {a.step}"}, a.dst)
    print(f"  wrote {a.dst}")


if __name__ == "__main__":
    main()
