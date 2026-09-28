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

import numpy as np
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
                 depth_range=(1.0, 60.0), image_size=(896, 512), n_cams: int = 6,
                 occ_pillar_h: int = 16, occ_num_classes: int = 10,
                 map_num_classes: int = 6, map_size=(200, 400),
                 map_range=(-30.0, -15.0, 30.0, 15.0), ref_points: bool = True,
                 det_head: str = "query", center_hidden: int = 64,
                 center_blocks: int = 2, center_min_radius: int = 1, head_upsample: int = 1,
                 head_dilate: bool = False, occ_head: str = "conv1x1", occ_3d_channels: int = 16,
                 temporal: bool = False, velocity: bool = False, history: int = 1,
                 feat_distill: bool = False, teacher_dim: int = 1024):
        self.arch = arch
        self.n_cams = n_cams
        # From weights/Qwen-Drive-1.0-4B/perception/config.json: occ_pillar_h 16,
        # occ_num_classes 10, map_num_classes 6, and a map grid of 200x400 cells
        # (map_ybound [-15,15,0.15] x map_xbound [-30,30,0.15]).
        self.occ_pillar_h = occ_pillar_h
        self.occ_num_classes = occ_num_classes
        self.map_num_classes = map_num_classes
        self.map_size = tuple(map_size)
        # feature-level distillation against the teacher's ViT tap. Training only: the
        # adapter is never called in forward, so it stays out of the exported graph.
        self.feat_distill = feat_distill
        self.teacher_dim = teacher_dim
        # map_xbound [-30,30] x map_ybound [-15,15]; the BEV covers +/-51.2 m, so the
        # map is a WINDOW inside it, not a rescaling of the whole thing.
        self.map_range = tuple(map_range)
        # Reference points break query symmetry from scratch. A checkpoint distilled
        # per-query already has spread boxes (25.35 m measured) and an absolute box
        # head, so warm-starting from one needs them OFF or the reference is added on
        # top of an already-absolute prediction.
        self.ref_points = ref_points
        # 'query' = the original 900-query DETR decoder. 'center' = a dense CenterPoint
        # heatmap at the BEV's native 0.512 m/cell; see center_head.py for why.
        self.det_head = det_head
        # Center-head shape and target radius floor. Defaults are what E5c trained with;
        # config fields so a saved checkpoint rebuilds its own head exactly.
        self.center_hidden = center_hidden
        self.center_blocks = center_blocks
        self.center_min_radius = center_min_radius
        # Detection-head grid factor. 1 = the BEV's native 0.512 m/cell. 2 = a learned 2x
        # transposed conv in front of the head, so heatmap and regression live on a
        # 400x400, 0.256 m grid. Motivated by the r2_long@80k per-class read: the student
        # trails the teacher mostly on pedestrians (42.8 vs 51.0 recall), traffic cones
        # (54.9 vs 68.0) and bicycles, and mostly within 20 m -- objects of 0.3-0.7 m that
        # occupy less than one native cell and collide in crowds. Costs ~1-2 ms per frame.
        self.head_upsample = head_upsample
        # head_dilate: 3x3 blocks of the 2x-grid head use dilation 2 (exact warm start from a 1x head).
        self.head_dilate = head_dilate
        # occ_head: "conv1x1" = the original single 1x1 conv over the BEV (0.30 mIoU on Occ3D);
        # "conv3d" = 1x1 lift to occ_3d_channels x 16 pillars, two 3x3x3 conv blocks, 1x1x1 classifier:
        # real vertical context for the voxels, a few ms per frame.
        self.occ_head = occ_head
        self.occ_3d_channels = occ_3d_channels
        # Temporal fusion of the previous keyframe's BEV (see temporal.py).
        self.temporal = temporal
        self.history = history      # previous keyframes fused (1 = BEVDet4D, K = long history)
        # Regress ego-frame velocity in the center head (needs gt_boxes10.npz); the official
        # NDS scores velocity and the attributes derived from it.
        self.velocity = velocity
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
        """``feat_map`` is (B*N, C, H, W); ``bev_index``/``valid`` are (N*D*H*W,).

        Every camera scatters into the SAME BEV grid, so one accumulate covers the full
        360 degrees. That is what makes this comparable to the six-camera teacher: only
        ~24% of the teacher's detections fall inside the front camera alone.
        """
        n = self.cfg.n_cams
        b = feat_map.shape[0] // n
        depth_logits = self.depth(feat_map)                      # B*N, D, H, W
        # Stashed rather than returned: training needs these to supervise depth against
        # lidar, but adding a return value would add an output to the exported ONNX and
        # change the graph's signature for every consumer. An attribute is invisible to
        # the tracer.
        self.last_depth_logits = depth_logits
        d = depth_logits.softmax(1)
        f = self.feat(feat_map)                                  # B*N, C, H, W
        # outer product over the depth axis: B*N, C, D, H, W
        lifted = f.unsqueeze(2) * d.unsqueeze(1)
        c = lifted.shape[1]
        # (B*N, C, DHW) -> (B, C, N*DHW), camera-major so it lines up with the index
        # blocks stacked by geometry.bev_indices_all.
        lifted = lifted.reshape(b, n, c, -1).permute(0, 2, 1, 3).reshape(b, c, -1)
        cells = self.cfg.bev_size * self.cfg.bev_size
        # scatter_add rather than index_add: it maps onto ONNX ScatterElements with
        # reduction="add", which has a CUDA kernel, whereas index_add does not trace.
        # Invalid points are aimed at a scratch row that is then dropped, so no masking
        # arithmetic survives into the graph.
        # Invalid rays used to share ONE scratch row. With real calibration roughly half of the
        # 1.1M rays are invalid, so ~500k atomic adds serialised on that row's 384 addresses:
        # the graph ran 90 ms on real inputs against 60 ms on the exporter's all-valid dummies
        # (measured 2026-09-27 on an idle H200). Spread them over `cells` scratch rows with a
        # static arange -- a constant in the graph -- and drop those rows after the scatter.
        # Same numbers out, no contention.
        scratch = cells + torch.arange(bev_index.shape[-1], device=bev_index.device) % cells
        idx = torch.where(valid, bev_index, scratch.expand_as(bev_index) if bev_index.dim() > 1 else scratch)
        # Per-sample indices (B, N*DHW) -- calibration, and the camera-flip augmentation,
        # differ per frame, so a batch cannot share sample 0's scatter. A 1-D (N*DHW,)
        # index (the export path) is broadcast to the batch as before.
        idx = idx.reshape(-1, 1, idx.shape[-1])
        if idx.shape[0] == 1:
            idx = idx.expand(b, -1, -1)
        idx = idx.expand(-1, c, -1)
        bev = lifted.new_zeros(b, c, 2 * cells).scatter_add(2, idx, lifted)
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
    """Six camera images plus frozen calibration indices -> the teacher's format."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg.arch, cfg.pretrained, cfg.feat_channels)
        self.lss = LiftSplat(cfg, self.backbone.out_channels)
        if cfg.temporal:
            from local.distill.temporal import TemporalFuse
            self.temporal_fuse = TemporalFuse(cfg.bev_channels, history=cfg.history)
        # The BEV stack is where the 3-D reasoning happens, so extra capacity goes here
        # rather than into input resolution: the teacher only ever saw 896x512, so a
        # student given more pixels than that cannot use them to match it better.
        blocks = [conv_bn(cfg.bev_channels, cfg.bev_channels)
                  for _ in range(max(1, cfg.bev_blocks - 1))]
        blocks.append(conv_bn(cfg.bev_channels, cfg.d_model))
        self.bev = nn.Sequential(*blocks)
        # Fixed AvgPool2d, not AdaptiveAvgPool2d: bev_size divides pool exactly
        # (200 / 40 = 5), so this is numerically identical, and adaptive pooling refuses
        # to export ("input size not accessible") as soon as the BEV's spatial dims are
        # not statically known -- which they stopped being when the camera axis was
        # folded into the batch.
        # The other half of the teacher's perception output. Both read the same BEV
        # feature the detection decoder attends to, so they cost one 1x1 convolution
        # each rather than a second backbone.
        #   occupancy: 200x200 pillars x 16 height bins x 10 classes
        #   map: a different grid entirely (200x400 at 0.15 m), so the BEV is resized
        #        into it -- Resize has a CUDA kernel at opset 20, which was verified.
        # d_model, not bev_channels: the BEV stack's last block widens to d_model,
        # which is what both heads and the decoder's memory actually see.
        # 1x1 to the teacher's channel width. Built only when asked, because a module
        # that never receives a gradient trips DDP's find_unused_parameters=False.
        self.feat_adapter = (nn.Conv2d(self.backbone.out_channels, cfg.teacher_dim, 1)
                             if cfg.feat_distill else None)

        if cfg.det_head == "center":
            from local.distill.center_head import CenterHead
            self.center = CenterHead(cfg.d_model, cfg.num_classes, upsample=cfg.head_upsample,
                                     hidden=cfg.center_hidden, blocks=cfg.center_blocks,
                                     reg_dims=(10 if cfg.velocity else 8),
                                     dilation=(cfg.head_upsample if getattr(cfg, "head_dilate", False) else 1))
        if getattr(cfg, "occ_head", "conv1x1") == "conv3d":
            c3 = cfg.occ_3d_channels
            self.occ_head = nn.ModuleDict({
                "lift": nn.Conv2d(cfg.d_model, cfg.occ_pillar_h * c3, 1),
                "body": nn.Sequential(
                    nn.Conv3d(c3, c3, 3, padding=1, bias=False), nn.BatchNorm3d(c3), nn.ReLU(inplace=True),
                    nn.Conv3d(c3, c3, 3, padding=1, bias=False), nn.BatchNorm3d(c3), nn.ReLU(inplace=True),
                    nn.Conv3d(c3, cfg.occ_num_classes, 1))})
        else:
            self.occ_head = nn.Conv2d(cfg.d_model,
                                      cfg.occ_pillar_h * cfg.occ_num_classes, 1)
        self.seg_head = nn.Sequential(
            conv_bn(cfg.d_model, cfg.d_model // 2),
            nn.Conv2d(cfg.d_model // 2, cfg.map_num_classes, 1))
        # Where the map window sits inside the BEV. Resizing the whole 200x200 into the
        # map grid stretches +/-51.2 m onto a 30x60 m window -- 3.4x wrong in y, 1.7x in
        # x -- so the head is asked to predict map content at the wrong place and its
        # loss sits flat at ~1.25 forever. Crop first, then resize.
        x0, y0, _, x1, y1, _ = cfg.pc_range
        mx0, my0, mx1, my1 = cfg.map_range
        n = cfg.bev_size
        self.mx0 = int(round((mx0 - x0) / (x1 - x0) * n))
        self.mx1 = int(round((mx1 - x0) / (x1 - x0) * n))
        self.my0 = int(round((my0 - y0) / (y1 - y0) * n))
        self.my1 = int(round((my1 - y0) / (y1 - y0) * n))

        assert cfg.bev_size % cfg.pool == 0, (
            f"bev_size {cfg.bev_size} must be divisible by pool {cfg.pool}")
        k = cfg.bev_size // cfg.pool
        self.pool = nn.AvgPool2d(k, k)
        self.pos = nn.Parameter(torch.zeros(1, cfg.pool * cfg.pool, cfg.d_model))
        self.query = nn.Parameter(torch.zeros(1, cfg.num_queries, cfg.d_model))
        # LEARNED REFERENCE POINTS -- each query is anchored to its own place in the BEV
        # and the box head predicts an OFFSET from it.
        #
        # Without this the 900 queries have no spatial prior, so at initialisation they
        # all predict the same box. Measured: after 2k steps of Hungarian training the
        # box centres had a spread of 0.90 m across a +/-51.2 m BEV -- every query on one
        # point. Matching is then arbitrary, a different query set is assigned each step,
        # and the symmetry never breaks. Per-query distillation hid this because the
        # teacher handed each query a distinct consistent target from step one (spread
        # 25.35 m), which is why it worked at all.
        #
        # This is the DETR3D / Deformable-DETR construction, and it is the reason those
        # converge in ~24 epochs where vanilla DETR needs ~500.
        g = int(round(cfg.num_queries ** 0.5))
        _ = g  # grid built below only when ref_points is on
        ys, xs = torch.meshgrid(torch.linspace(0.04, 0.96, g),
                                torch.linspace(0.04, 0.96, g), indexing="ij")
        grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)[:cfg.num_queries]
        if len(grid) < cfg.num_queries:                    # pad if Q is not a square
            grid = torch.cat([grid, torch.rand(cfg.num_queries - len(grid), 2)], 0)
        # store as logits so the points stay inside the range under sigmoid
        self.ref_logit = (nn.Parameter(torch.log(grid / (1 - grid)).unsqueeze(0))
                          if cfg.ref_points else None)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg.d_model) for _ in range(cfg.dec_layers))
        self.cls_head = nn.Linear(cfg.d_model, cfg.num_classes)
        # Prior-probability bias, the companion to focal loss that RetinaNet documents
        # and that is not optional. With bias 0 every one of 900x7 outputs starts at
        # p=0.5, so ~6,288 background terms outweigh ~12 positives by roughly 1500:1 and
        # the fastest descent is to drive every logit negative. Measured: 20k steps of
        # Hungarian+focal with default init left cls loss flat at 0.48-0.54 and the model
        # predicting ZERO detections at every threshold. Starting at p=0.01 puts the
        # background term near its minimum so the positives own the gradient.
        PRIOR = 0.01
        nn.init.constant_(self.cls_head.bias, -float(np.log((1 - PRIOR) / PRIOR)))
        self.box_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, cfg.box_dim))
        # Buffers, not constants: the head regresses a standardised target and these put
        # the output back in the teacher's units, so the exported signature is unchanged
        # and the loss still sees every dimension at comparable scale.
        self.register_buffer("box_mean", torch.zeros(cfg.box_dim))
        self.register_buffer("box_std", torch.ones(cfg.box_dim))
        if cfg.det_head == "center":
            # The 900-query decoder is dead weight under the center head: no loss term
            # touches it, so its parameters get no gradient and DDP aborts on the first
            # step. This has to come AFTER the assignments above -- an earlier attempt
            # placed it before them and was silently overwritten (param count never
            # moved from 83.17 M). Nulling it also frees ~34 M params and eight attention
            # layers; `mem`/`pos` stay because the trajectory head cross-attends to them.
            self.layers = nn.ModuleList()
            self.cls_head = nn.Identity()
            self.box_head = nn.Identity()
            self.query = None

        # Planning shares the backbone and the BEV memory with detection. Two separate
        # students would each need to fit the frame budget on their own; one network with
        # two heads gets planning for the cost of a 50-query decoder, which is why the
        # whole model can clear 10 Hz rather than just the perception half.
        #
        # Built here rather than in set_box_stats: the trajectory head is part of the
        # architecture, not part of the box standardisation. Constructing it in the setter
        # meant a model that had never been given box statistics -- an untrained export,
        # or eval on a checkpoint loaded before the call -- had no self.traj_query at all
        # and raised AttributeError on the first forward that passed `ego`.
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
        # A direct linear path from the ego state to all 50 waypoints, zero-initialised.
        #
        # WHY: with only the anchor plus a BEV-conditioned MLP, the learned residual
        # converged to 1e-4 m -- the model returned the constant-velocity anchor exactly,
        # and trajectory ADE was byte-identical across two very different checkpoints.
        # That is not a dead head, it is L1 doing its job: the optimal CONSTANT under L1
        # is the median of the residual, and median(future - constant_velocity) is ~0.
        # A head that cannot condition well therefore converges to precisely zero.
        # A plain least-squares probe on this same ego vector reaches 1.469 m against the
        # anchor's 2.21 m, so the signal is linearly extractable -- the head just could
        # not find it through 3 cross-attention layers. This gives it that path directly;
        # zero-init means training still starts exactly at the anchor.
        self.ego_direct = nn.Linear(ego_dim, cfg.traj_points * 3)
        nn.init.zeros_(self.ego_direct.weight); nn.init.zeros_(self.ego_direct.bias)

    def set_box_stats(self, mean, std):
        """Install the affine that restores teacher units at the output."""
        self.box_mean.copy_(torch.as_tensor(mean, dtype=self.box_mean.dtype))
        self.box_std.copy_(torch.as_tensor(std, dtype=self.box_std.dtype))

    def bev_from(self, image, bev_index, valid):
        """Lift-splat BEV of one frame before the BEV encoder: the temporal state."""
        b, n = image.shape[0], image.shape[1]
        feat = self.backbone(image.reshape(b * n, *image.shape[2:]))
        return self.lss(feat, bev_index, valid)

    def forward(self, image, bev_index, valid, ego=None, prev_bev=None, warp_grid=None):
        """(cls, box, occ, seg) without ``ego``; + trajectory with it.

        All four perception outputs match the teacher's layout exactly, so every
        downstream consumer -- ``get_bboxes``, the occupancy and map decoders -- works
        on the student unchanged.

        ``ego`` is the flattened driving state -- history, velocity, acceleration, the
        navigation one-hot and speed -- and the trajectory is emitted in the ego frame at
        10 Hz, matching what the planner it replaces produces.
        """
        # image is (B, N, 3, H, W): the backbone runs per view with shared weights, so
        # the camera axis folds into the batch and costs nothing in the graph.
        b, n = image.shape[0], image.shape[1]
        feat = self.backbone(image.reshape(b * n, *image.shape[2:]))
        # Stashed for feature-level distillation against the teacher's ViT tap, which
        # lives on the same (32, 56) grid. An attribute keeps it out of the traced graph.
        self.last_backbone_feat = feat
        bev_raw = self.lss(feat, bev_index, valid)
        # bev_state is the UNFUSED lift-splat BEV: at training the previous frame's BEV is
        # bev_from(prev images) = its lss output, so feeding this frame's lss output back
        # as prev_bev at deployment reproduces training exactly (returning the fused BEV
        # would make the state recurrent over all history, which was never trained).
        bev_state = bev_raw
        if self.cfg.temporal:
            bev_raw = self.temporal_fuse(bev_raw, prev_bev, warp_grid)
        bev = self.bev(bev_raw)
        if self.cfg.det_head == "center":
            hm, rg = self.center(bev)
        mem = self.pool(bev).flatten(2).transpose(1, 2) + self.pos
        if self.cfg.det_head != "center":
            q = self.query.expand(b, -1, -1)
            for layer in self.layers:
                q = layer(q, mem)
        if self.cfg.det_head == "center":
            from local.distill.center_head import decode_centers
            self._hm, self._reg = hm, rg          # retained for the training loss
            cls, box = decode_centers(hm, rg, self.cfg, k=self.cfg.num_queries)
        else:
            cls = self.cls_head(q)
            # xy is reference + offset; the other eight dims are absolute as before.
            raw = self.box_head(q) * self.box_std + self.box_mean
            if self.ref_logit is not None:
                x0, y0, _, x1, y1, _ = self.cfg.pc_range
                ref = torch.sigmoid(self.ref_logit)
                ref = torch.cat([x0 + ref[..., :1] * (x1 - x0),
                                 y0 + ref[..., 1:] * (y1 - y0)], -1).expand(b, -1, -1)
                box = torch.cat([raw[..., :2] + ref, raw[..., 2:]], -1)
            else:
                box = raw

        # occupancy: (B, H*C, 200, 200) -> (B, 200, 200, H, C), the teacher's layout
        if isinstance(self.occ_head, nn.ModuleDict):                # conv3d head
            n = self.cfg.bev_size
            v = self.occ_head["lift"](bev).reshape(b, self.cfg.occ_3d_channels, self.cfg.occ_pillar_h, n, n)
            oc = self.occ_head["body"](v)                              # (B, C, H, y, x)
            occ = oc.permute(0, 3, 4, 2, 1)                            # (B, y, x, H, C)
        else:
            oc = self.occ_head(bev)
            oc = oc.reshape(b, self.cfg.occ_pillar_h, self.cfg.occ_num_classes,
                            self.cfg.bev_size, self.cfg.bev_size)
            occ = oc.permute(0, 3, 4, 1, 2)
        # map: resize the BEV into the map grid, then classify
        mh, mw = self.cfg.map_size
        # BEV is (B, C, y, x) -- geometry.bev_indices writes iy * n + ix
        bev_win = bev[:, :, self.my0:self.my1, self.mx0:self.mx1]
        seg = self.seg_head(F.interpolate(bev_win, size=(mh, mw), mode="bilinear",
                                          align_corners=False))
        if ego is None:
            return (cls, box, occ, seg, bev_state) if self.cfg.temporal else (cls, box, occ, seg)
        t = self.traj_query.expand(b, -1, -1) + self.ego(ego).unsqueeze(1)
        for layer in self.traj_layers:
            t = layer(t, mem)
        # Anchor on constant velocity and learn the residual. Regressing absolute metres
        # from a near-zero init means starting at the all-zeros trajectory, which is
        # 12.83 m ADE, and crawling down -- round 0 reached only 8.22 m after 6k steps,
        # four times WORSE than extrapolating the ego's own velocity (2.16 m). This is
        # the same conditioning problem box_stats fixes for boxes, and it was never
        # applied to the trajectory. The anchor costs no parameters and starts the head
        # at the 2.16 m baseline instead of 12.83 m.
        v0 = self.cfg.hist_points * 3
        v1 = v0 + self.cfg.hist_points * 2
        vel = ego[:, v1 - 2:v1]                                   # latest (vx, vy), m/s
        dt = torch.arange(1, self.cfg.traj_points + 1, device=ego.device,
                          dtype=vel.dtype) * 0.1                  # 50 points at 10 Hz
        xy = vel.unsqueeze(1) * dt.reshape(1, -1, 1)              # (B, 50, 2)
        anchor = torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)
        direct = self.ego_direct(ego).reshape(b, self.cfg.traj_points, 3)
        traj = anchor + direct + self.traj_head(t)
        # bev_state: this frame's unfused lift-splat BEV, fed back as prev_bev next step
        return (cls, box, occ, seg, traj, bev_state) if self.cfg.temporal else (cls, box, occ, seg, traj)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
