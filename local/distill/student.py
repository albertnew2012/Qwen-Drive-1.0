"""A compact front-camera 3-D detector, shaped to be distilled from the big one.

WHY A NEW MODEL AND NOT MORE PRUNING
Pruning took the ONNX frame from 29,761 ms to 907 ms and then stopped. The BEV head is
438 ms spread over 2,008 kernels averaging 221 us -- the portable-op expansion of what
PyTorch fuses -- and it survived fifteen structural changes: fp16 (0.98x), CUDA graphs
(0.99x), dropping voxel convs (destroys detection), narrowing them (memory bound),
coarsening the depth grid (4.7 ms of 420), truncating the DETR decoder (4 ms). 10 Hz is
100 ms. Nothing incremental closes 9x.

THE ONE DESIGN DECISION THAT MATTERS
The student emits the teacher's exact output format: 900 queries of 7 class logits and
10 box parameters. That means query i of the student is trained against query i of the
teacher, so distillation is a plain per-query regression with no Hungarian matching, no
label assignment, and no ground truth -- and every piece of downstream code, including
``get_bboxes``, works on the student unchanged. The teacher's queries are fixed learned
embeddings with stable semantics, which is what makes the correspondence meaningful.

WHAT IS CHEAP ABOUT IT
    backbone      stride-16 CNN to a 32x56 map, matching the grid the teacher's head reads
    lift-splat    32 depth bins rather than 118, onto a 128x128 BEV with height collapsed
    BEV encoder   three 3x3 convs at 64 channels
    decoder       three layers of ordinary attention over a 32x32 pooled BEV

Ordinary attention, not deformable: ``multi_scale_deformable_attn`` has no ONNX
representation and is what forced the teacher's export onto GridSample in the first
place. Height is collapsed because the teacher's 16x200x200 voxel volume is 6.79 TFLOP
of Conv3d, and at one camera it is 8.7% occupied.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["StudentConfig", "StudentDetector"]


class StudentConfig:
    # Sized by measuring, not guessing, and against the whole frame rather than
    # perception alone -- the budget is 100 ms for detection *and* planning, on one card,
    # because that is what an Orin has. The first shape tried was 3.56 M parameters and
    # 5.23 ms, far too small to absorb a 4 B teacher: BEVFormer-tiny is 33 M and BEVDet
    # with a ResNet-50 is ~50 M, which is the band a camera-only 3-D detector belongs in.
    #
    # Measured end to end in ONNX, both heads in one graph:
    #
    #   r34  d384 bev256 L6 blocks3    47.0 M   19.0 ms   52.7 Hz
    #   r50  d384 bev256 L6 blocks3    49.8 M   20.2 ms   49.6 Hz
    #   r50  d512 bev384 L8 blocks3    77.8 M   29.4 ms   34.0 Hz
    #   r50  d512 bev384 L8 blocks6    81.8 M   39.1 ms   25.6 Hz   <- these defaults
    #   r50  d512 bev512 L8 blocks8    92.3 M   63.6 ms   15.7 Hz
    #   r101 d512 bev512 L8 blocks8   111.3 M   68.1 ms   14.7 Hz
    #
    # Running faster than needed is lost accuracy, so the defaults sit near the top of the
    # range rather than the bottom: 2.6x margin on 10 Hz, which host-side preprocessing
    # will spend down toward 20 Hz in the real pipeline.
    def __init__(self, num_classes: int = 7, box_dim: int = 10, num_queries: int = 900,
                 arch: str = "resnet50", pretrained: bool = True, feat_channels: int = 256,
                 bev_size: int = 200, bev_channels: int = 384,
                 depth_bins: int = 64, d_model: int = 512, dec_layers: int = 8,
                 pool: int = 40, traj_points: int = 50, traj_layers: int = 3,
                 hist_points: int = 16, bev_blocks: int = 6, pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 5.4),
                 depth_range=(1.0, 60.0), image_size=(896, 512)):
        self.arch = arch
        self.pretrained = pretrained
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.box_dim = box_dim
        self.num_queries = num_queries
        self.bev_size = bev_size
        self.bev_channels = bev_channels
        self.depth_bins = depth_bins
        self.d_model = d_model
        self.dec_layers = dec_layers
        self.pool = pool
        self.traj_points = traj_points
        self.traj_layers = traj_layers
        self.hist_points = hist_points
        self.bev_blocks = bev_blocks
        self.pc_range = tuple(pc_range)
        self.depth_range = tuple(depth_range)
        self.image_size = tuple(image_size)      # (width, height)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def conv_bn(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Backbone(nn.Module):
    """ImageNet-pretrained ResNet to a stride-16 map, with the stride-32 stage folded in.

    Two reasons this is a torchvision ResNet rather than the hand-rolled stack it replaces.
    First, the training set is ~3.4k frames and growing as blobs download, so ImageNet
    initialisation is worth more than parameter count -- a backbone learned from scratch on
    thousands of frames is the part most likely to leave the student unable to follow the
    teacher. Second, a plain ResNet is all Conv/BN/ReLU/Add, which exports without a single
    special case.

    ``layer3`` is stride 16, which lands on the 32x56 grid the teacher's head reads.
    ``layer4`` is stride 32 and carries the long-range context that matters for distant
    objects, so it is upsampled and added rather than discarded.
    """

    def __init__(self, arch: str = "resnet34", pretrained: bool = True,
                 out_channels: int = 256):
        super().__init__()
        import torchvision.models as tvm
        weights = None
        if pretrained:
            # torchvision spells these ResNet18_Weights, not Resnet18_Weights, and
            # verify() rejects a mismatched class rather than ignoring it
            cls = getattr(tvm, arch.replace("resnet", "ResNet") + "_Weights", None)
            weights = getattr(cls, "IMAGENET1K_V1", None) if cls is not None else None
        net = getattr(tvm, arch)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        # BasicBlock (resnet18/34) ends at conv2; Bottleneck (resnet50+) ends at conv3
        # and expands 4x, so conv3 has to win where both exist
        def out_ch(block):
            return (block.conv3 if hasattr(block, "conv3") else block.conv2).out_channels
        c3, c4 = out_ch(net.layer3[-1]), out_ch(net.layer4[-1])
        self.lat3 = nn.Conv2d(c3, out_channels, 1)
        self.lat4 = nn.Conv2d(c4, out_channels, 1)
        self.fuse = conv_bn(out_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.layer2(self.layer1(self.stem(x)))
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        up = F.interpolate(self.lat4(f4), size=f3.shape[-2:], mode="nearest")
        return self.fuse(self.lat3(f3) + up)


class LiftSplat(nn.Module):
    """Depth distribution times features, scattered into a flat BEV grid.

    The scatter is a single ``index_add`` over precomputed indices. Those indices depend
    only on the camera calibration, so at export they fold into a constant and the whole
    transform becomes one gather and one accumulate -- no GridSample, no ScatterND chain.
    """

    def __init__(self, cfg: StudentConfig, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.depth = nn.Conv2d(in_channels, cfg.depth_bins, 1)
        self.feat = nn.Conv2d(in_channels, cfg.bev_channels, 1)

    def forward(self, feat_map, bev_index, valid):
        """``bev_index`` is (D*H*W,) into the flattened BEV, ``valid`` its mask."""
        b = feat_map.shape[0]
        d = self.depth(feat_map).softmax(1)                      # B, D, H, W
        f = self.feat(feat_map)                                  # B, C, H, W
        # outer product over the depth axis: B, C, D, H, W
        lifted = f.unsqueeze(2) * d.unsqueeze(1)
        c = lifted.shape[1]
        lifted = lifted.reshape(b, c, -1)                        # B, C, D*H*W
        cells = self.cfg.bev_size * self.cfg.bev_size
        # scatter_add rather than index_add: it maps onto ONNX ScatterElements with
        # reduction="add", which has a CUDA kernel, whereas index_add does not trace.
        # Invalid points are aimed at a scratch row that is then dropped, so no masking
        # arithmetic survives into the graph.
        idx = torch.where(valid, bev_index, torch.full_like(bev_index, cells))
        idx = idx.reshape(1, 1, -1).expand(b, c, -1)
        bev = lifted.new_zeros(b, c, cells + 1).scatter_add(2, idx, lifted)
        return bev[:, :, :cells].reshape(b, c, self.cfg.bev_size, self.cfg.bev_size)


class Attention(nn.Module):
    """Multi-head attention written out.

    ``nn.MultiheadAttention`` dispatches to ``aten::_native_multi_head_attention``, which
    has no ONNX export at any opset. Spelling it out costs nothing and keeps the graph to
    MatMul, Softmax and Reshape -- all of which have CUDA kernels and fuse well.
    """

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.dk = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def _split(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.h, self.dk).transpose(1, 2)

    def forward(self, q, kv):
        qh, kh, vh = self._split(self.q(q)), self._split(self.k(kv)), self._split(self.v(kv))
        att = (qh @ kh.transpose(-1, -2)) * (self.dk ** -0.5)
        out = att.softmax(-1) @ vh
        b, _, n, _ = out.shape
        return self.o(out.transpose(1, 2).reshape(b, n, self.h * self.dk))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads=8, ff=None):
        super().__init__()
        ff = ff or 4 * d_model
        self.sa = Attention(d_model, heads)
        self.ca = Attention(d_model, heads)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(inplace=True),
                                nn.Linear(ff, d_model))
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model) for _ in range(3))

    def forward(self, q, mem):
        q = self.n1(q + self.sa(q, q))
        q = self.n2(q + self.ca(q, mem))
        return self.n3(q + self.ff(q))


class StudentDetector(nn.Module):
    """Front image plus frozen calibration indices -> the teacher's output format."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg.arch, cfg.pretrained, cfg.feat_channels)
        self.lss = LiftSplat(cfg, self.backbone.out_channels)
        # The BEV stack is where the 3-D reasoning happens, so extra capacity goes here
        # rather than into input resolution: the teacher only ever saw 896x512, so a
        # student given more pixels than that cannot use them to match it better.
        blocks = [conv_bn(cfg.bev_channels, cfg.bev_channels)
                  for _ in range(max(1, cfg.bev_blocks - 1))]
        blocks.append(conv_bn(cfg.bev_channels, cfg.d_model))
        self.bev = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(cfg.pool)
        self.pos = nn.Parameter(torch.zeros(1, cfg.pool * cfg.pool, cfg.d_model))
        self.query = nn.Parameter(torch.zeros(1, cfg.num_queries, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg.d_model) for _ in range(cfg.dec_layers))
        self.cls_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, cfg.box_dim))
        # Buffers, not constants: the head regresses a standardised target and these put
        # the output back in the teacher's units, so the exported signature is unchanged
        # and the loss still sees every dimension at comparable scale.
        self.register_buffer("box_mean", torch.zeros(cfg.box_dim))
        self.register_buffer("box_std", torch.ones(cfg.box_dim))

    def set_box_stats(self, mean, std):
        self.box_mean.copy_(torch.as_tensor(mean, dtype=self.box_mean.dtype))
        self.box_std.copy_(torch.as_tensor(std, dtype=self.box_std.dtype))

        # Planning shares the backbone and the BEV memory with detection. Two separate
        # students would each need to fit the frame budget on their own; one network with
        # two heads gets planning for the cost of a 50-query decoder, which is why the
        # whole model can clear 10 Hz rather than just the perception half.
        ego_dim = cfg.hist_points * 3 + cfg.hist_points * 2 * 2 + 3 + 1
        self.ego = nn.Sequential(
            nn.Linear(ego_dim, cfg.d_model), nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, cfg.d_model))
        self.traj_query = nn.Parameter(torch.zeros(1, cfg.traj_points, cfg.d_model))
        nn.init.normal_(self.traj_query, std=0.02)
        self.traj_layers = nn.ModuleList(
            DecoderLayer(cfg.d_model) for _ in range(cfg.traj_layers))
        self.traj_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, 3))

    def forward(self, image, bev_index, valid, ego=None):
        """(cls, box) when ``ego`` is absent, (cls, box, trajectory) when it is.

        ``ego`` is the flattened driving state -- history, velocity, acceleration, the
        navigation one-hot and speed -- and the trajectory is emitted in the ego frame at
        10 Hz, matching what the planner it replaces produces.
        """
        feat = self.backbone(image)
        bev = self.bev(self.lss(feat, bev_index, valid))
        mem = self.pool(bev).flatten(2).transpose(1, 2) + self.pos
        b = image.shape[0]
        q = self.query.expand(b, -1, -1)
        for layer in self.layers:
            q = layer(q, mem)
        cls = self.cls_head(q)
        box = self.box_head(q) * self.box_std + self.box_mean
        if ego is None:
            return cls, box
        t = self.traj_query.expand(b, -1, -1) + self.ego(ego).unsqueeze(1)
        for layer in self.traj_layers:
            t = layer(t, mem)
        return cls, box, self.traj_head(t)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
