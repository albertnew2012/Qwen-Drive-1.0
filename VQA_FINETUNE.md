# Fine-tuning Qwen-Drive's VQA mode to read its own lane

Companion to [TRAFFIC_LIGHT.md](TRAFFIC_LIGHT.md), which covers the perception pipeline and
the ground truth. This document is only about the **VQA path**: the model answering in
words, from the image alone, with no boxes and no head attached.

Two capabilities were trained, both on OpenLane-V2 subset B (nuScenes imagery):

| | base | fine-tuned |
|---|---|---|
| ego-lane traffic light colour (1300 frames with a light) | 92.6% | **94.7%** |
| ego-lane traffic light, all 6019 val frames | 80.1% | **88.5%** |
| ego lane type — left / through / right (exact set) | 44.9% | **73.8%** |
| ego lane type — macro F1 | 53.0% | **85.6%** |

---

## 1. Before training anything: does the base model associate?

It does not, and finding that out first changed what was worth training.

Twelve question phrasings were scored on **dev** (46 train segments, disjoint from val), in
three groups: association prompts ("the light controlling the lane the ego vehicle is in"),
format controls (reversed option order, A/B/C multiple choice), and **association-free
controls** ("what colour is the traffic light ahead?", which never mentions the ego lane).

```
paired vs controls on discriminative frames, macro-recall difference, episode-bootstrapped
  ego_jargon    - salient   +3.7%  [-1.5%, +12.2%]
  exclude_cross - salient   +3.2%  [-2.9%, +11.9%]
  cot           - salient   +1.5%  [-7.1%, +11.6%]
  action        - salient   -1.7%  [-7.4%,  +2.7%]
```

No association prompt beats the control with significance, and `salient` had the **highest
raw accuracy of all twelve**. Reversing the answer options alone moved discriminative
macro-recall 85.9% -> 73.7%.

Confirmed at scale: on the 1192 val frames where traffic lights are visible but none
governs the ego lane, the base model **names a colour anyway 77% of the time**. It reports
the salient lamp, and the label agrees often enough to look like understanding.

Two phrasings I expected to help were the worst: chain-of-thought (90.9%) and a prompt
forcing the model to name the light's position before its colour (91.6%, macro-recall
68.3%). The plain question won.

---

## 2. How the fine-tune works

LoRA on the attention projections only — `q_proj`, `k_proj`, `v_proj`, `o_proj` and the
vision tower's `qkv`/`proj` — reusing `training/lora.py` from the stage-2 recipe.

```
rank 16, alpha 32        5.51 M trainable / 4.539 B frozen  (0.12%)
loss                     next-token prediction, MASKED to the answer tokens
                         the prompt and all image tokens are context, never supervised
answer format            "<think>\n\n</think>\n\nred"
optimiser                AdamW, lr 1e-4, OneCycle, grad-accum 8
image budget             921600 px (the model's pretraining default)
```

Three details that mattered more than they look:

- **Keep the model's own `<think></think>` prefix.** Training on the bare colour teaches
  it to drop its own output format and starts the loss at ~13.5 instead of ~0.005.
- **`model.eval()` silently disables gradient checkpointing.** HF gates it on
  `self.training`, so the model must be in `train()` mode — the difference between 13 GiB
  and an OOM on a 24 GB card.
- **The composite `gradient_checkpointing_enable()` only reaches the vision tower.** The
  32-layer language model keeps every activation unless it is enabled explicitly, and that
  is what actually blows the card.

Checkpoints are written every 25 steps and the trainer resumes in place: one card threw
both an `Xid 31` MMU fault and an autograd segfault during this work, and a crash at step
180 of a 2.5 h run should cost two minutes, not the run.

---

## 3. The training mix decided everything

Four mixes, same architecture and hyperparameters:

| run | training data | all 6019 | frames with a light | phantom | missed |
|---|---|---|---|---|---|
| v1 | governed frames only, 3-way question | 80.1% | **94.7%** | 1109 | 33 |
| v2 | + hard negatives, 4-way with `none` | 87.9% | 74.3% | 397 | 326 |
| **v3** | rebalanced toward positives | **88.5%** | 86.6% | 517 | 161 |

*phantom* = named a colour where none governs. *missed* = said "none" where one did.

v1 is never shown a frame whose answer is "none", so it cannot produce one — it invents a
colour on 77% of the `not_ego` frames, exactly like the base model. v2 added 1100 hard
negatives (lights visible, none the ego's) and over-corrected into abstaining. v3 kept the
negatives but nearly doubled the positive evidence.

**No single model wins both columns.** v1 reads colour best; v3 handles the whole split
best. Which to deploy depends on whether a missed red or a phantom red is worse.

---

## 4. Hyperparameters were exhausted

| variant | resolution | rank | epochs | all 6019 | frames with a light |
|---|---|---|---|---|---|
| **v3** | 921600 px | 16 | 1 | **88.5%** | **86.6%** |
| v4 | native 1600x900 | 32 | 1 | 86.9% | 82.6% |
| v5 | 921600 px | 32 | 1 | 88.3% | 85.2% |
| v6 | 921600 px | 16 | 2 | 88.4% | 79.7% |

v4-vs-v5 isolates resolution (**-1.4**), v3-vs-v5 isolates capacity (**-0.2, noise**),
v3-vs-v6 isolates epochs (**-7** where a light exists).

Native resolution was my hypothesis — the lamp is ~12 px after the default 0.8x downscale,
so more pixels should help. It hurt: 1400 image tokens is outside the pretraining
distribution, and the extra resolution costs more than it gains. Three independent levers,
none beating v3.

A decomposition also failed: splitting "is there a light?" (a trained yes/no gate) from
"what colour?" scored **88.2%**, below the single model, because the gate recognises a real
ego light only 72.9% of the time and discards a quarter of the true positives before colour
is read. (An earlier 93.9% for this was a measurement leak — the colour file covered
exactly the frames that have a light, so the rest fell through to "none" for free.)

---

## 5. Lane type: the same model, a different bias

The base model scores 44.9% exact-set, and **every** top error is the same shape:

```
truth                model said     count
left+straight   ->   straight        44
left            ->   straight        24
left+right+str  ->   straight        17
right+straight  ->   straight        15
```

Left recall 18.2% at **75% precision** — when it says "left" it is usually right, it just
almost never says it. That is a through-lane prior, not blindness, so the training mix was
balanced across all seven lane types (700 frames each) and trained at native resolution,
which *does* help here because a road arrow is large and on the ground plane.

| | exact set | macro F1 | left F1 | straight F1 | right F1 |
|---|---|---|---|---|---|
| base | 44.9% | 53.0% | 29.3% (R 18.2%) | 87.0% | 42.6% |
| **fine-tuned** | **73.8%** | **85.6%** | **83.3%** (R 87.1%) | 94.0% | **79.5%** |

Composed with the pipeline's colour, the explicit answer — *"I am in a left+right+straight
lane and its light is green"* — is right **76.7%** of the time (lane type 79.9%, colour
95.6%), with lane-type errors accounting for 32 of the 37 failures.

---

## 6. Reproducing

```bash
# traffic light, the v3 mix
.venv/bin/python local/tlb/train_vqa_v2.py --train data/tlb/ft_full_train_v3.jsonl \
    --cond full --rank 16 --epochs 1 --save outputs/tlb/lora_vqa_v3.pt
.venv/bin/python local/tlb/eval_vqa.py --src data/tlb/full_val.jsonl --cond full \
    --question full --lora outputs/tlb/lora_vqa_v3.pt

# lane type
.venv/bin/python local/tlb/train_lanetype_v1.py --train data/tlb/ft_lanetype_train.jsonl \
    --cond hires --rank 32 --epochs 1 --save outputs/tlb/lora_lanetype_v1.pt
.venv/bin/python local/tlb/eval_lanetype.py --question arrow --lora outputs/tlb/lora_lanetype_v1.pt

# the control that matters: does any association prompt beat an association-free one?
.venv/bin/python local/tlb/prompt_study.py --cond hires
.venv/bin/python local/tlb/compare_prompts.py
```

Every prompt, mix and hyperparameter was chosen on dev or on a holdout of train segments;
val was read once per configuration.
