# Concrete model structure — Qwen-Drive-1.0 (4B) with all three heads

> The counterpart to [`alpamayo1.5/study/07_MODEL_STRUCTURE_WITH_PERCEPTION.md`](/home/albert/Desktop/alpamayo1.5/study/07_MODEL_STRUCTURE_WITH_PERCEPTION.md),
> for the other model. The structural difference to keep in mind while reading: in
> Alpamayo the perception head is **yours**, bolted onto a frozen checkpoint through a
> forward hook. Here the perception head is **shipped by the authors**, it is a first-class
> released artifact, and it reads *two* taps instead of one.
>
> Every number below was dumped from the **actual downloaded checkpoint**
> (`Qwen/Qwen-Drive-1.0-4B`, 4 safetensors files, 1908 tensors) by
> [`local/anatomy.py`](../local/anatomy.py) and [`local/token_layout.py`](../local/token_layout.py)
> — not from the README and not from memory.
> Reproduce with: `python local/anatomy.py --per-layer` and `PYTHONPATH=src python local/token_layout.py`

Code: [`QwenDriveForPlanning`](../src/qwen_drive/modeling_qwen_drive.py) holds the VLM and
the planning expert; [`QwenDrivePerception`](../src/qwen_drive_perception/modeling_perception.py)
is a *separate* `PreTrainedModel` that is handed the same VLM through `.attach()`.

---

## 0. End-to-end data flow — all three modes on one page

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                          THE SHARED VLM — Qwen3.5-4B                                  ║
║                    4.5393 B params · one copy · never modified                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝

  PLANNING INPUT                             │        PERCEPTION INPUT
  3 views × 4 timesteps = 12 images          │        the whole camera ring, 1 timestep
  <FRONT> <FRONT LEFT> <FRONT RIGHT>         │        6 cams (nuScenes) or 8 cams (nuPlan)
  history 26×24 grid → 156 tok  (×9)         │        each 896×512 → 32×56 grid → 448 tok
  current 50×44 grid → 550 tok  (×3)         │
        └─ 3054 image + 331 text = 3385 tok  │        └─ 3584 image + 70 text = 3654 tok (8 cam)
                                             │           2688 image + 56 text = 2744 tok (6 cam)
                    │                        │                     │
                    ▼                        │                     ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │  VISION TOWER  model.visual   333.514 M  (7.35 %)                  │
        │  24 blocks · width 1024 · 16 heads · patch 16 · temporal patch 2   │
        │  merge 2×2 → out_hidden 2560 ·  deepstack_visual_indexes = []      │
        │  pixel_values row = 3 ch × 2 temporal × 16 × 16 = 1536             │
        └──────────┬──────────────────────────────────┬─────────────────────┘
                   │ post-merge tokens                │ PRE-MERGE patches, after
                   │                                  │ merger.norm  ── PERCEPTION TAP 1
                   ▼                                  │  [N_cam, 32, 56, 1024]
        ┌───────────────────────────────────────────────────────────────────┐
        │  LANGUAGE MODEL  4205.751 M  (92.65 %)   32 layers · width 2560   │
        │  vocab 248 320 · embed 635.70 M · tie_word_embeddings = True      │
        │  mRoPE interleaved, sections [11, 11, 10], θ = 1e7, partial 0.25  │
        │                                                                   │
        │  HYBRID STACK — layer_types alternate 3:1                         │
        │   ├ 24 × linear_attention  (Gated DeltaNet)     112.9 M each      │
        │   │     in_proj_qkv 8192 = q 16×128 ⊕ k 16×128 ⊕ v 32×128        │
        │   │     in_proj_z 4096 gate · conv1d (8192,1,4) · A_log/dt_bias   │
        │   │     ► NO KV CACHE — a recurrent state, not keys and values    │
        │   │                                                               │
        │   └  8 × full_attention  at layers [3,7,11,15,19,23,27,31]        │
        │         q_proj 8192 = query 16×256 ⊕ OUTPUT GATE 16×256           │
        │         k_proj/v_proj 1024 = 4 kv heads × 256   (GQA 4:1)         │
        │         q_norm/k_norm per-head RMSNorm over head_dim 256          │
        │         ► these 8 are the ONLY layers that leave a KV cache       │
        └───────┬───────────────────────────────┬───────────────────────────┘
                │                               │ final norm applied explicitly,
                │                               │ image-token rows only
                │                               │  ── PERCEPTION TAP 2
                │                               │  [N_cam, 16, 28, 2560]
                │                               ▼
                │              ┌──────────────────────────────────┐
                │              │  BEV PERCEPTION HEAD             │
                │              │  125.064 M · fp32 · §3           │
                │              └──────────────────────────────────┘
                │
        ┌───────┴────────────────────────────────────────────────┐
        │                                                        │
        ▼ VQA mode                                               ▼ PLANNING modes
   the LLM decoder, unchanged.                        8 KV caches, one per full-
   greedy-equivalent decoding                         attention layer, each
   (temp 0.01, top_k 1)                               [1, 3385, 4 heads, 256]
   → text                                                        │
                                                                 ▼
                              ┌────────────────────────────────────────────────────┐
                              │  PLANNING EXPERT  1.0398 B  ·  §2                  │
                              │  32 layers × 31.990 M · width 1024 · mlp 3584      │
                              │  10 Euler steps of flow matching                   │
                              └────────────────────────────────────────────────────┘
                                                                 │
                                          OUTPUT ►  [num_samples, 50, 3]
                                          (x, y, heading) · 5 s @ 10 Hz · ego frame
```

The load-bearing structural fact, and the one that differs most from Alpamayo:
**three heads, one VLM, and the VLM is byte-identical in all three modes.** The perception
head does not feed the planner; the planner does not feed perception; the LLM decoder is
untouched. They are three readers of one representation. The authors say so explicitly —
the perception head "serves as a **probe** of the 3D information accessible from the shared
representations".

---

## 0b. Parameter tree (real numbers, from `local/anatomy.py`)

```
Qwen-Drive-1.0 release directory ─────────────── 13.78 GB on disk
│
├── model.safetensors ······················   4.5393 B  (4,539,265,536)   9.079 GB  bf16
│   THE SHARED VLM — Qwen3_5ForConditionalGeneration
│   ├── vlm.model.visual   24 × Qwen3_5VisionBlock ......   333.514 M  ( 7.35 %)  297 tensors
│   └── vlm.model.language_model ........................  4205.751 M  (92.65 %)  426 tensors
│       ├── embed_tokens  248 320 × 2560 ................   635.699 M  (14.01 %)  ◄ tied to lm_head
│       ├── 24 × linear_attention layer  @ 112.92 M .....  2710.172 M  (59.71 %)
│       │      linear_attn 42.14 M  +  mlp 70.78 M
│       ├──  8 × full_attention   layer  @ 107.48 M .....   859.877 M  (18.94 %)  ═╗
│       └── final norm ..................................     0.003 M              ║
│              self_attn 36.70 M  +  mlp 70.78 M                                   ║
│                                                                                  ║
├── planner-rl/model.safetensors ···········   1.0398 B  (1,039,848,451)  2.080 GB bf16
├── planner-sft/model.safetensors ··········   1.0398 B  (identical shape)  2.080 GB bf16
│   PLANNING EXPERT — pick exactly one at load time                                ║
│   ├── 32 × PlanningExpertLayer  @ 31.990 M ............  1023.689 M  (98.45 %)  ═╝
│   │     qkv_proj (10240,1024)  10.486 M   ◄ 4 groups × (4 q ⊕ 4 gate ⊕ k ⊕ v) × 256
│   │     adaln_modulation.1     6.291 M    ◄ the DiT conditioning, 6 × 1024
│   │     gate_up_proj           7.340 M
│   │     down_proj              3.670 M
│   │     o_proj                 4.194 M
│   ├── query_fusion (7×1024 → 1024) ......................    8.391 M  ( 0.81 %)
│   ├── 6 × conditioning encoders (time/nav/ego/hist×3) ...    6.561 M  ( 0.63 %)
│   ├── fourier_encoder.net  (96 → 1024 → 1024) ...........    1.149 M  ( 0.11 %)
│   ├── waypoint_embed  Embedding(50, 1024) ...............    0.051 M  ( 0.00 %)
│   ├── trajectory_proj  Linear(3 → 1024) .................    0.003 M
│   └── out_proj         Linear(1024 → 3) .................    0.003 M  ◄ the whole output
│
└── perception/model.safetensors ···········   0.1251 B  (125,063,986)  0.500 GB  fp32
    BEV PERCEPTION HEAD — 827 tensors
    ├── head.transformer  (6 enc + 6 dec, deformable) ......   61.624 M  (49.27 %)
    ├── adaptor  SimpleFPN over the LLM tap ................   33.663 M  (26.92 %)
    ├── head.bev_embedding  200×200 × 256 ..................   10.240 M  ( 8.19 %)
    ├── view_trans.conv_layer  (3D convs) ..................    5.312 M  ( 4.25 %)
    ├── head.seg_decoder  (resnet18 trunk + U-Net up) ......    5.200 M  ( 4.16 %)
    ├── depth_net  (reduce 0.591 M + depth_conv 4.409 M) ...    5.000 M  ( 4.00 %)
    ├── uvtr_query_proj  Conv2d(256×16 → 256, k=1) .........    1.049 M  ( 0.84 %)
    ├── vit_neck  SimpleFPN over the ViT tap ...............    0.853 M  ( 0.68 %)
    ├── cls_branches (60 t) / reg_branches (36 t) ..........    1.611 M  ( 1.29 %)
    ├── head.query_embedding  900 × 512 ....................    0.461 M  ( 0.37 %)
    └── head.positional_encoding ...........................    0.051 M  ( 0.04 %)
```

| configuration | params | bf16 resident |
|---|---|---|
| VLM only — VQA | **4.5393 B** | 9.08 GB |
| VLM + planning expert | **5.5791 B** | 11.16 GB |
| VLM + perception head | **4.6643 B** | 9.58 GB (head stays fp32) |
| everything at once | **5.7042 B** | 11.66 GB |

**Read this:** the model is *small*. Everything at once is 5.70 B — **half** of
Alpamayo 1.5's 11.08 B, and it fits on a 12 GB card with room to spare where Alpamayo
needs 24 GB and still gets pushed into CPU offload on this machine.

Three accounting facts worth pausing on:

- **`tie_word_embeddings = True`.** One 635.70 M tensor serves as both `embed_tokens` and
  `lm_head` — there is **no `lm_head` tensor in the checkpoint at all** (verified: no key
  matching `lm_head`). Alpamayo carries *two* untied 637.73 M tensors — 1.28 B, 11.5 % of that model —
  because its 4000 discrete trajectory bins inflated the vocabulary. Qwen-Drive has no
  trajectory vocabulary at all (§2), so it pays for embeddings once.
- **59.71 % of the VLM is linear-attention layers that produce no KV cache.** This is the
  single most consequential architectural fact in the whole model, and §2 is about what it
  forces the expert to do.
- **The perception head ships in fp32** while everything else is bf16. 125.06 M × 4 bytes =
  0.500 GB. It is the only part of the release that was not cast down.

---

## 1. Config, from the checkpoint's `config.json`

| field | value | meaning |
|---|---|---|
| `vlm_config.architectures` | `Qwen3_5ForConditionalGeneration` | stock, unmodified |
| `text_config.hidden_size` | 2560 | |
| `text_config.num_hidden_layers` | 32 | |
| `text_config.num_attention_heads` / `num_key_value_heads` | 16 / 4 | GQA 4:1 |
| `text_config.head_dim` | **256** | wide heads; 16 × 256 = 4096 ≠ 2560 |
| `text_config.intermediate_size` | 9216 | |
| `text_config.vocab_size` | 248 320 | no trajectory bins |
| `text_config.attn_output_gate` | **True** | `q_proj` is 8192 = query ⊕ per-head sigmoid gate |
| `text_config.full_attention_interval` | 4 | every 4th layer, i.e. `[3,7,…,31]` |
| `text_config.linear_num_key_heads` × `linear_key_head_dim` | 16 × 128 | |
| `text_config.linear_num_value_heads` × `linear_value_head_dim` | 32 × 128 | |
| `text_config.linear_conv_kernel_dim` | 4 | short causal depthwise conv before the recurrence |
| `text_config.tie_word_embeddings` | True | |
| `rope_parameters` | `mrope_interleaved`, sections `[11,11,10]`, θ 1e7, partial 0.25 | only 64 of 256 head channels rotate |
| `vision_config.depth` / `hidden_size` | 24 / 1024 | out_hidden 2560 |
| `vision_config.patch_size` / `spatial_merge_size` / `temporal_patch_size` | 16 / 2 / 2 | |
| `expert_config.hidden_size` | 1024 | |
| `expert_config.num_hidden_layers` | 32 | same depth as the VLM |
| `expert_config.head_dim` / `num_key_value_heads` | **256 / 4** | **must** equal the VLM's |
| `expert_config.layers_per_kv` | 4 | 32 expert layers ÷ 4 = 8 caches |
| `num_future_points` / `trajectory_hz` | 50 / 10.0 | **5.0 s** horizon |
| `trajectory_scale` | `[165.0, 25.0, 1.5703125]` | per-channel normalizer; the heading value is `pi/2` rounded to bf16 |
| `num_inference_steps` | 10 | Euler steps |
| `noise_seed` | 42 | sample *k* uses seed `42 + k` |
| `min_one_minus_t` | 0.1 | floor on the remaining-time divisor |
| `max_reasoning_tokens` / `min_reasoning_tokens` | 256 / 10 | |

---

## 2. The planning expert: what it reads, and why it can

### 2.1 The KV geometry is not a coincidence

```
VLM full-attention layer        expert layer
  k_proj  (1024, 2560)            qkv_proj (10240, 1024)
  → 4 kv heads × 256              → 4 groups × (4 q ⊕ 4 gate ⊕ 1 k ⊕ 1 v) × 256
      ▲                                                        ▲
      └──────────── same 4 × 256 ──────────────────────────────┘
```

`_scene_cache()` takes the VLM's **post-rotary** keys and values straight out of the cache
and hands them to the expert with no projection at all:

```python
for index in self.config.full_attention_layers:      # [3, 7, 11, 15, 19, 23, 27, 31]
    layer = past_key_values.layers[index]
    cache.append((layer.keys.transpose(1, 2), layer.values.transpose(1, 2)))
```

Each entry is `[1, 3385, 4, 256]`. The docs put it plainly: *"the scene enters the
trajectory network without any extra projection."* This is only possible because
`expert_config.head_dim` and `num_key_value_heads` were **copied from** the VLM. The config
docstring says so: they "have to match the VLM exactly".

### 2.2 The 3:1 hybrid forces the 4:1 sharing

24 of the VLM's 32 layers are Gated DeltaNet. A linear-attention layer keeps a *recurrent
state*, not a sequence of keys and values, so there is nothing for a cross-attending expert
to read. Only the 8 `full_attention` layers leave a cache — and that is exactly why
`layers_per_kv = 4`:

```
expert layers  0  1  2  3   4  5  6  7   8  9 10 11  …  28 29 30 31
                └──┬──┘       └──┬──┘      └──┬──┘        └──┬──┘
cache index        0             1            2      …        7
VLM layer          3             7           11      …       31
```

`scene_cache[index // 4]`. Four consecutive expert layers re-read the same VLM cache. The
expert's depth (32) was chosen so that 32 ÷ 4 lands exactly on the 8 caches the hybrid
stack happens to produce.

> **This is the sharpest architectural contrast with Alpamayo.** Alpamayo's expert is a
> 36-layer stack reading a 36-layer dense VLM — one cache per layer, a clean 1:1. Qwen-Drive's
> VLM is 75 % linear-attention, so only a quarter of its layers are legible to a
> cross-attending expert at all, and each one has to serve four expert layers.

### 2.3 Joint attention, not cross-attention

Each layer attends over the **concatenation** of the scene cache and the waypoints' own
keys and values:

```python
attn = self._attend(query,
                    torch.cat([scene_key, key], dim=1),      # 3385 + 50 = 3435 keys
                    torch.cat([scene_value, value], dim=1))
```

so the 50 waypoints read the scene *and each other* in one softmax, non-causally. There is
no separate self-attention block.

### 2.4 What a waypoint token is made of

`query_fusion` is `Linear(7 × 1024 → 1024)` — literally seven signals concatenated:

| # | signal | producer |
|---|---|---|
| 1 | the current **noisy waypoint** | `trajectory_proj` `Linear(3 → 1024)` |
| 2 | its **Fourier features** | `fourier_encoder`, 16 log-spaced freqs to 16.0, 3 ch × 16 × 2 = 96 |
| 3 | the **flow time** *t* | `time_embed` (sinusoidal, dim 128, scale 1000) → `time_mlp` |
| 4 | the **ego history poses** | `history_encoder`, 15 × 3 ⊕ 3 nav one-hot = 48 |
| 5 | the **waypoint index** | `waypoint_embed` `Embedding(50, 1024)` |
| 6 | the **history velocity** | `history_velocity_encoder`, 16 × 2 = 32 |
| 7 | the **history acceleration** | `history_acceleration_encoder`, 16 × 2 = 32 |

And separately, the **adaLN condition** is a sum of three, injected into every layer:

```python
condition = time_condition + self.nav_mlp(nav_onehot) + self.ego_mlp(ego_status)
```

`ego_status` is the eight numbers `[vx, vy, ax, ay, *driving_command]`. Each layer's
`adaln_modulation` turns that into 6 × 1024 — shift/scale/gate around both the attention and
the feed-forward. That single `Linear(1024 → 6144)` is **6.291 M of the 31.990 M per layer,
19.7 %**: a fifth of the expert is spent on conditioning modulation.

### 2.5 Where the waypoints sit in rotary space

The waypoints are given mRoPE positions `anchor + 1 … anchor + 50`, continuing straight
after the VLM prefix. All three mRoPE sections share one anchor because the last prefix
token is always text. In `REASONING_PLANNING` the anchor is advanced past the generated
rationale *and* past the tokens that close the turn (`<|im_end|>` + newline), which
`_prefill_with_reasoning` appends by hand — otherwise the waypoints would sit at positions
the model never saw in training.

Two deliberate bf16 reproductions live here, both documented in the source:

- `WaypointRotaryEmbedding` computes phases **in bf16, not fp32**, because training held the
  inverse-frequency table in bf16 — which rounds positions above 256 onto a coarser grid.
  **Measured:** the anchor for a demo scene is **522** (not the 3385 token count — mRoPE gives
  images a 2-D position grid, so positions advance far more slowly than tokens), and the 50
  waypoint positions `523 … 572` collapse onto **13 distinct bf16 values**, stepping by 4. So
  roughly every four consecutive waypoints share a rotary phase.
- `FourierFeatureEncoder` rebuilds its frequency table in the module dtype every call, for
  the same reason.

> Practical consequence: **running the planner in fp32 is not a free accuracy upgrade.** It
> changes the rotary phases and Fourier features away from the ones the weights were fitted
> against. See [`03_LOCAL_SETUP.md`](03_LOCAL_SETUP.md).

### 2.6 The sampler

Flow matching with a **clean-endpoint (`x1`) parameterization** — the network predicts the
finished trajectory, and the velocity is derived:

```python
endpoint  = predict_endpoint(waypoints, t, …)          # the FINISHED trajectory
remaining = max(1.0 - index * step, min_one_minus_t)   # floored at 0.1
waypoints = waypoints + (endpoint - waypoints) / remaining * step
```

10 steps, `dt = 0.1`. The floor stops the last step from dividing by ~0 and amplifying
prediction error.

> **`num_inference_steps` and `min_one_minus_t` are coupled, and the defaults sit on the
> boundary.** The last step lands exactly on the prediction only while `dt >= min_one_minus_t`,
> i.e. `num_inference_steps <= 10`. Raise the step count without lowering the floor and the
> final step under-shoots (0.5x at 20 steps), leaving residual noise: measured ADE goes
> 0.275 -> 3.529. See [`04_RESULTS.md`](04_RESULTS.md) §1.6.

> **Contrast with Alpamayo**, which predicts the **velocity field directly**
> (`action_out_proj` → `v`, then `x += dt·v`). Same 10 Euler steps, opposite
> parameterization. Predicting the endpoint means every intermediate step is already a
> complete, inspectable trajectory.

`num_samples=N` tiles only the expert's conditioning; the VLM runs **once** and its cache is
shared. Sample *k* draws from seed `42 + k`, so a sample is identical alone or in a batch.

---

## 3. The BEV perception head — the part Alpamayo does not have

125.064 M parameters, fp32, released by the authors. One forward pass produces **three**
outputs.

| output | shape | classes |
|---|---|---|
| 3D detection | up to **300** boxes `[x,y,z,w,l,h,yaw,vx,vy]`, lidar frame | 7 |
| semantic occupancy | **200 × 200 × 16** voxels, ego frame | 10 |
| BEV map segmentation | **200 × 400** array `(y, x)` — 30 m lateral × 60 m longitudinal @ 0.15 m | 6 |

```
DET_CLASS_NAMES = vehicle · czone_sign · bicycle · generic_object · pedestrian ·
                  traffic_cone · barrier
OCC_CLASS_NAMES = the 7 above + driveable · background · empty
MAP_CLASS_NAMES = background · driveable_surface · road_line · road_edge ·
                  crosswalk · walkway
```

### 3.1 Two taps, not one

This is the design detail worth stealing. The head does **not** read a single hidden layer.

```
TAP 1 — the ViT stream (geometry)              TAP 2 — the LLM stream (semantics)
  a forward hook on visual.merger, then          the LAST decoder layer's hidden states,
  merger.norm run explicitly                     with language_model.norm applied by hand
  PRE-merge patches [N, 32, 56, 1024]            image-token rows only [N, 16, 28, 2560]
        │                                                    │
        ▼                                                    ▼
  vit_neck  SimpleFPN, scale (1.0,)              adaptor  SimpleFPN, scales (4.0, 2.0, 1.0, 0.5)
  0.853 M                                        33.663 M  ── 27 % of the whole head
        │                                                    │
        ▼                                                    │  4 levels of multi-scale
  DepthNet  → 118-bin depth distribution                     │  features for the deformable
  frustum 896×512×[1,60] m, cell 16×16×0.5 m                 │  spatial cross-attention
        │                                                    │
        ▼  voxel_pool_depth  (fused scatter)                 │
  UVTR voxel volume [B, 256, 16, 200, 200]                   │
        │                                                    │
        ├──────────────► uvtr_query_proj (256×16 → 256) ─────┤
        │                     BEV init tokens                │
        ▼                                                    ▼
   occupancy branch                          ┌────────────────────────────────┐
                                             │  BEVFormer encoder ×6          │
                                             │  bev_embedding 200×200×256     │
                                             │  temporal self-attn (degenerate│
                                             │    at 1 frame) + spatial cross │
                                             │    attn, 8 pts, 4 pillars      │
                                             └───────────┬────────────────────┘
                                                         │  BEV [40000, 256]
                        ┌────────────────────────────────┼────────────────────────┐
                        ▼                                ▼                        ▼
              detection decoder ×6              occ_refiner 3D U-Net       seg_decoder
              900 queries, NMS-free             + UVTR fusion              resnet18 + U-Net
              → 300 boxes, 7 cls                → 200×200×16, 10 cls       → 200×400, 6 cls
```

**Why two taps.** The ViT patches are pre-merge, so they keep the full 32 × 56 spatial
resolution and have not yet been through 32 layers of language modelling — they are the
right thing to unproject into a frustum. The LLM hidden states are 2×-downsampled and
semantically abstract — the right thing to attend into from BEV queries. The head takes
geometry from one and meaning from the other.

> Compare: the Alpamayo detection head reads **one** tap, layer 24 of 36, image-token rows
> only, 720 × 4096 — and it was found empirically that layer 24 beat both 12 and 36. Qwen-Drive
> sidesteps that hyperparameter entirely by taking the *last* layer for semantics and going
> around the language model altogether for geometry.

### 3.2 Two CUDA kernels, one of which had no CPU path

`ops/` ships two hand-written kernels, JIT-compiled by `torch.utils.cpp_extension.load` on
first use:

| kernel | what it does | CPU fallback |
|---|---|---|
| `ms_deform_attn_bf16` | multi-scale deformable attention, bf16 storage / fp32 math | **shipped** — `attention.py` dispatches to `multi_scale_deformable_attn_pytorch` when `not value.is_cuda` |
| `voxel_pool_depth` | fuses the depth distribution and the feature scatter into one pass | **was missing** — added locally, see below |

`voxel_pool_depth` computes, for every in-range frustum point,

```
out[batch, camera, x, y, z, c] += img_feats[image, c, h, w] * img_depth[image, d, h, w]
```

accumulated in fp32 into a `[B, N_cam, X, Y, Z, C]` volume, where `ranks` is already the
flattened output index. That is a weighted scatter-add, so the torch equivalent is an
`index_add_`. It is implemented in [`ops/__init__.py`](../src/qwen_drive_perception/ops/__init__.py)
as `_voxel_pool_depth_torch`, dispatched exactly the way the repo already dispatches the
other kernel, and it casts its fp32 accumulator back to `promote_types(feats, depth)` to
match the CUDA path's return dtype. **Nothing else in the repository is modified.**

### 3.3 Coordinate conventions (the part that bites)

* Predicted boxes come back in the **lidar** frame; packed ground truth is in the **ego**
  frame. `geometry.lidar_to_ego_boxes` converts, which is what the visualization uses.
* Occupancy and the map raster are indexed in **ego** coordinates, X forward, Y left, Z up.
* Box format is `[x, y, z, w, l, h, yaw, vx, vy]` with **z at the box bottom** and `w` along
  the heading.
* `lidar2img` folds the 896 × 512 resize into the projection, so it maps lidar points
  directly onto the model-resolution image plane.

### 3.4 The hard constraint

The head was trained on nuScenes (6 cameras) and OpenScene/nuPlan (8 cameras) at a **fixed
896 × 512** per camera, with fixed rigs. `configuration_perception.py` calls its constants
"frozen". A different camera layout or resolution "is not covered by the released weights
and will likely degrade results". Inference is strictly single-frame, so the temporal
self-attention degenerates to plain BEV self-attention — exactly as during evaluation.

---

## 4. The three inference modes

| mode | VLM passes | expert reads the cache of | output |
|---|---|---|---|
| `VQA` | 1 (generate) | — | text |
| `DIRECT_PLANNING` | 1 (prefill only) | a prompt-only pass, 3385 tok | 50 waypoints |
| `REASONING_PLANNING` | 1 (generate) + 1 (close the turn) | a pass that **includes the generated rationale** | 50 waypoints + text |

The network is identical in all three. Only two things change: whether the user turn asks
for a rationale, and whose cache the expert reads.

`planner-rl` was reward-optimized **only** on reasoning-conditioned rollouts, so it must be
run in `REASONING_PLANNING`. `planner-sft` covers both.

Measured prompt lengths (`local/token_layout.py`, scene `a53176b0…-149`):

```
DIRECT     3385 tok = 3054 image + 331 text
REASONING  3400 tok = 3054 image + 346 text     (+15 tokens of reasoning request)
```

The 3054 image tokens break down as 9 history frames × 156 + 3 current frames × 550. The
history/current asymmetry comes from the pixel budgets: `history_image_pixels` 174 080 vs
`current_image_pixels` 921 600 — roughly 320p for history, 720p for now.

---

## 5. Trajectory representation

50 waypoints of `(x, y, heading)`, 5 s at 10 Hz, ego frame of the current timestamp,
x forward, y left, heading positive for a left turn.

Normalization divides each channel by a fixed scale:

| channel | scale | note |
|---|---|---|
| `x` | 165 m | 33 m/s × 5 s |
| `y` | 25 m | |
| `heading` | 1.5703125 rad | `pi/2` **rounded to bf16** |

History is handled differently: the 16 poses are re-referenced to the **oldest** one — so
history and future both progress in the driving direction — that origin row is dropped as
uninformative, and the remaining **15** poses use the same scale. Hence
`history_encoder` takes 15 × 3 ⊕ 3 = 48 inputs.

> **Contrast with Alpamayo.** Alpamayo does not predict waypoints at all. It predicts
> 64 × (accel, curvature) and integrates a **unicycle model** to get poses, which makes every
> output kinematically feasible by construction. Qwen-Drive predicts `(x, y, heading)`
> directly, which is simpler and strictly more expressive but carries no feasibility
> guarantee. It also predicts heading explicitly, which Alpamayo gets for free from the
> integration.

---

## 6. Reproducing every number here

```bash
python local/anatomy.py --root weights/Qwen-Drive-1.0-4B --per-layer
PYTHONPATH=src python local/token_layout.py
```
