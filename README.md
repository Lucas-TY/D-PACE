# D-PACE

**Dynamic Position-Aware Cross-Entropy for DFlash speculative drafting.**

This repository is based on [SGLang SpecForge](https://github.com/sgl-project/SpecForge/tree/main) and adds a focused D-PACE training loss for DFlash models. D-PACE changes the training objective only: the drafter architecture, target model interface, and inference pipeline stay unchanged.

<p align="center">
  <img src="./assets/dpace_results.svg" alt="D-PACE headline results" width="880">
</p>

## What D-PACE changes

DFlash trains parallel block drafters with a fixed position-decay cross-entropy schedule. D-PACE instead derives per-position cross-entropy weights from a smooth accepted-length surrogate, so the loss can shift learning signal toward the draft positions that currently limit accepted length.

For a draft block with target-token draft confidence `q_j`:

```text
q_tilde_j = (1 - alpha) * q_j + alpha
P_j       = prod_{i <= j} q_tilde_i
w_j       = sum_{m >= j} P_m
L_D-PACE  = sum_j stop_gradient(w_j) * CE_j
```

The implementation uses the optimized prefix-product / suffix-sum form and normalizes D-PACE losses by the local per-GPU batch size. The standard DFlash loss remains available for compatibility.

<p align="center">
  <img src="./assets/dpace_weight_dynamics.svg" alt="D-PACE dynamic position weights" width="880">
</p>

## Results from the paper

On Qwen3-4B DFlash drafts, D-PACE improves both wall-clock decoding speedup (SR) and average emitted length (`tau`) over the DFlash decayed-CE baseline across the main settings:

| Target / setting | DFlash avg. SR | D-PACE avg. SR | SR gain | DFlash avg. `tau` | D-PACE avg. `tau` | `tau` gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B, 3L, T=0 | 2.93 | 3.20 | +9.2% | 4.10 | 4.52 | +10.2% |
| Qwen3-4B, 5L, T=0 | 2.98 | 3.27 | +9.7% | 4.38 | 4.85 | +10.7% |
| Qwen3-4B, 3L, T=1 | 2.74 | 2.96 | +8.0% | 3.86 | 4.19 | +8.5% |
| Qwen3-4B, 5L, T=1 | 2.77 | 3.03 | +9.4% | 4.10 | 4.50 | +9.8% |

Additional paper highlights:

- Up to **4.47x** speedup on MATH-500 with the 5L Qwen3-4B drafter.
- Cross-target average emitted length gains of about **+12.5%** on Llama-3.1-8B-Instruct and **+12.8%** on Qwen3-8B.

## Training

Use the existing SpecForge DFlash training entrypoint and select D-PACE explicitly:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node 8 \
  scripts/train_dflash.py \
  --target-model-path Qwen/Qwen3-8B \
  --target-model-backend sglang \
  --draft-config-path configs/qwen3-8b-dflash.json \
  --train-data-path cache/dataset/perfectblend_qwen3-8b_regen.jsonl \
  --output-dir outputs/qwen3-8b-dpace \
  --num-epochs 6 \
  --batch-size 4 \
  --learning-rate 6e-4 \
  --warmup-ratio 0.04 \
  --max-grad-norm 1.0 \
  --max-length 3072 \
  --chat-template qwen \
  --attention-backend flex_attention \
  --block-size 16 \
  --num-anchors 512 \
  --loss-type dpace \
  --dpace-alpha 0.5
```

Or start from the included example:

```bash
NUM_GPUS=8 DPACE_ALPHA=0.5 bash examples/run_qwen3_8b_dpace_online.sh
```

### Loss options

| `--loss-type` | Use |
| --- | --- |
| `dflash` | Existing DFlash decayed-CE path. Keeps `--loss-decay-gamma` compatibility. |
| `dpace` | Main D-PACE objective. |
| `dpace_p` | Cumulative-confidence-only component ablation. |
| `dpace_f` | Continuation-value-only component ablation. |

## Notes

- D-PACE is draft-only after target-generated training tokens / hidden states are available; it does not require target-probability hooks.
- This release intentionally keeps the public surface focused on the D-PACE method family.
- General SpecForge data preparation and training details still apply; see the upstream SpecForge documentation for broader framework usage.

## Acknowledgement

This codebase is adapted from SGLang's SpecForge project. We thank the SpecForge and SGLang contributors for the DFlash training framework that this implementation builds on.
