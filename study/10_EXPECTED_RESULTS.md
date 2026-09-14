# Expected results

Every number below was measured on this machine (RTX 3090, 78 GiB RAM, fp32 on
CPU for ONNX). `outputs/expected_results.json` is the machine-readable copy;
regenerate it with **`4c`** and diff against it rather than trusting memory.

---

## 1. ONNX export

### Per-graph, against PyTorch

| graph | nodes | expect |
|---|---|---|
| vision tower (the tap the heads read) | 3,747 | **7.1e-05** |
| text prefill, 33 graphs (perception, 2744 tok) | 344,127 | **1.4e-06** |
| text prefill, 33 graphs (planning, 3385 tok) | 366,983 | ~1e-06 |
| decode step, 64 state tensors | 10,233 | **2.0e-06** worst of 65 outputs |
| perception head | 12,436 | **5.3e-04** |
| planner step | 9,315 | **4.2e-07** |

### End to end

**Perception** - config **7a** then **7b**:

| tensor | expect | tolerance |
|---|---|---|
| VLM hidden_states | 3.1e-04 | 5e-03 |
| VLM vit tap | 7.2e-05 | |
| VLM llm tap | 5.1e-04 | |
| planner KV keys / values | 5.2e-04 / 2.2e-04 | |
| all_cls_scores / all_bbox_preds | 1.0e-03 / 1.1e-03 | |
| occ_pred / seg_preds | 1.3e-06 / 3.7e-06 | |
| **verdict** | **PASS** | |

**Planning** - config **7c** then **7d**:

```
ADE 0.00002 m     FDE 0.00004 m     PASS  (tolerance 0.05 m)
endpoints identical to 3 decimals
```

### Runtime, so you know what "normal" looks like

| stage | time |
|---|---|
| vision tower | ~30 s |
| 32 decoder layers (33 ORT sessions) | ~190 s |
| perception head | ~80 s |
| planner, 10 Euler steps | ~26 s |
| **ONNX pipeline total** | **~300 s** |
| PyTorch reference (the comparison half) | ~150 s |

If a comparison run dies silently around the 5-minute mark it is the OOM killer:
the fp32 VLM is 17 GiB and the perception forward peaks near 20 GiB. That is why
the runners have `--phase run` and `--phase compare`; do not merge them.

---

## 2. Training

Run the whole set with **`5s`**. Expected, on one 3090:

| stage | config | steps | loss | time |
|---|---|---|---|---|
| gradient test, no patch | **5z** | - | `NotImplementedError` **(this failing is the pass condition)** | - |
| gradient test, patched | **5b** | - | **682/682** params with gradient, 19.1 GiB | ~30 s |
| 1, overfit one frame | **5c** | 120 | 15.5 -> **3.4** | 4.7 s/step |
| 1, all six frames | **5d** | 180 | 15.5 -> **3.8** | 4.6 s/step, 20.7 GiB |
| 2, joint LoRA | **5g** | 8 | 15.4 -> **9.5**, VLM grad > 0 | 16.6 s/step, 18.5 GiB |
| 3, planner from scratch | **5f** | 200 | 0.0079 -> **0.0001** | 0.22 s/step |

### What counts as a regression

* **Stage 1** must at least halve on an overfit run. If it plateaus above ~8, the
  differentiable-ops patch is probably not active - check `5b` first.
* **Stage 2** must show `gradient reached the VLM: True`. A falling loss alone is
  not enough; the head can improve while the VLM is effectively frozen.
* **Stage 3 needs `--scratch`.** From the released `planner-sft` weights the loss
  starts at **4e-5** - the model already fits the demo scenes, so an overfit test
  from there measures nothing and will "pass" vacuously.
* Losses are noisy across frames by design: the six demo frames carry 3 to 40
  ground-truth boxes, so a single step's loss varies 2x either way. Compare the
  10% averages in `expected_results.json`, not individual steps.

### Not implemented

**Stage 4 (RL).** It needs closed-loop PDMS/RFS scoring with G=8 rollouts, which
requires a simulator that is not part of this repo and not public. Losses and
stage structure for 1-3 are implemented from the report; stage 4 is absent, not
stubbed.

---

## 3. Re-verifying from scratch

```
5a   cache the VLM feature taps          (do this first)
5s   the whole training sweep
6a   export perception head              6e / 6f  VLM vision / text
6g   export decode step                  6c       planner step
7a + 7b   perception pipeline, run then validate
7c + 7d   planning pipeline, run then validate
4c   regenerate outputs/expected_results.json
```
