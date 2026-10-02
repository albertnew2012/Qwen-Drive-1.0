"""Check _voxel_pool_depth_torch against a naive transcription of the CUDA kernel.

The reference below is a literal loop over voxel_pool_depth_forward_all_kernel:
  out[batch, camera, x, y, z, c] += img_feats[image, c, h, w] * img_depth[image, d, h, w]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from qwen_drive_perception.ops import voxel_pool_depth


def reference(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z):
    _, n_images, C, H, W = img_feats.shape
    _, N_sweep, N_cam, D, _, _ = mask.shape
    out = torch.zeros(B, N_cam, X, Y, Z, C, dtype=torch.float32)
    feats = img_feats.reshape(-1, C, H, W).float()
    depth = img_depth.reshape(-1, D, H, W).float()
    for b in range(B):
        for s in range(N_sweep):
            for cam in range(N_cam):
                img = (b * N_sweep + s) * N_cam + cam
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            if not mask[b, s, cam, d, h, w]:
                                continue
                            _, xi, yi, zi = voxel_coords[b, s, cam, d, h, w].tolist()
                            out[b, cam, xi, yi, zi] += feats[img, :, h, w] * depth[img, d, h, w]
    return out


def main():
    torch.manual_seed(0)
    B, N_sweep, N_cam, C, H, W, D = 1, 1, 3, 5, 4, 6, 3
    X = Y = 7; Z = 2
    feats = torch.randn(B, N_sweep * N_cam, C, H, W)
    depth = torch.rand(B * N_sweep * N_cam, D, H, W)
    coords = torch.stack([
        torch.zeros(B, N_sweep, N_cam, D, H, W, dtype=torch.long),
        torch.randint(-2, X + 2, (B, N_sweep, N_cam, D, H, W)),
        torch.randint(-2, Y + 2, (B, N_sweep, N_cam, D, H, W)),
        torch.randint(-1, Z + 1, (B, N_sweep, N_cam, D, H, W)),
    ], dim=-1)
    mask = ((coords[..., 1] >= 0) & (coords[..., 1] < X)
            & (coords[..., 2] >= 0) & (coords[..., 2] < Y)
            & (coords[..., 3] >= 0) & (coords[..., 3] < Z))
    print(f"valid points: {int(mask.sum())} / {mask.numel()}")

    got = voxel_pool_depth(feats, depth, coords, mask, B, X, Y, Z)
    want = reference(feats, depth, coords, mask, B, X, Y, Z)
    err = (got.float() - want).abs().max().item()
    print(f"shape got {tuple(got.shape)}  want {tuple(want.shape)}  dtype {got.dtype}")
    print(f"max abs diff: {err:.3e}   nonzero voxels: {int((want.abs() > 0).any(-1).sum())}")
    assert got.shape == want.shape, "shape mismatch"
    assert err < 1e-4, f"numeric mismatch {err}"

    # multiple points landing in one voxel must accumulate, not overwrite
    coll = torch.zeros_like(coords); coll[..., 0] = 0
    m2 = torch.ones_like(mask)
    g2 = voxel_pool_depth(feats, depth, coll, m2, B, X, Y, Z).float()
    w2 = reference(feats, depth, coll, m2, B, X, Y, Z)
    e2 = (g2 - w2).abs().max().item()
    print(f"all-collide accumulation: max abs diff {e2:.3e}  "
          f"(sum={float(w2.sum()):.3f}, {int(m2.sum())} points into 1 voxel/cam)")
    assert e2 < 1e-3, f"accumulation mismatch {e2}"
    print("\nPASS - CPU fallback matches the kernel semantics")


if __name__ == "__main__":
    main()
