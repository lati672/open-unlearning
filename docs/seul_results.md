# SeUL on TOFU Forget05: fixed operating-point results

Paper-based SeUL TOFU adaptation, learning rate `1e-5`, exactly **2 optimizer updates**, seeds **0/1/2**. This is a method-level comparison; it is not a count-matched selector ablation or an official reproduction of the original benchmark.

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| ES | 0.56221655 | 0.52326521 | 0.54287275 | 0.54278484 | 0.01947582 |
| forget_probability | 0.81158686 | 0.79726776 | 0.80301871 | 0.80395778 | 0.00720559 |
| forget_rouge | 0.70175982 | 0.68662579 | 0.70385230 | 0.69741264 | 0.00940009 |
| forget_ratio | 0.48210455 | 0.48376791 | 0.48338156 | 0.48308467 | 0.00087052 |
| forget_quality | 2.56563e-14 | 2.96235e-13 | 2.96235e-13 | 2.06042e-13 | 1.56219e-13 |
| utility | 0.58883058 | 0.58122558 | 0.58813410 | 0.58606342 | 0.00420414 |
| abs_privleak | 99.65103512 | 99.01584064 | 98.75705770 | 99.14131115 | 0.46000657 |

All three seeds reduce ES, forget probability and forget ROUGE relative to the freshly evaluated original target. Mean ES decreases by 24.03% relative and Utility by 2.04%. Absolute PrivLeak remains 99.14 and mean Min-K AUC remains 0.994525; these results do not establish adequate privacy or retain-only equivalence.

Retain ROUGE declines from 0.779874 to a three-seed mean of 0.691285 (11.36% relative), close to the 12.24% relative Forget ROUGE decrease. Aggregate Utility therefore obscures a larger generation-overlap change on the retain subset. ROUGE changes alone do not establish loss of knowledge.

## Protocol and interpretation

- Full 200-example Forget05 training dataset; effective batch 32, microbatch 2, accumulation 16. Two updates expose 64 examples (0.32 epoch), not all 200 examples.
- BF16, paged AdamW32, constant learning rate, no warmup/weight decay, gradient clipping 1. No retain loss is added to the SeUL objective.
- Frozen original-target mean-confidence selection plus one-token gap filling selects 1762/7165 eligible answer-label positions (24.59%). All 200 masks were recomputed before and after training for each seed and stayed unchanged.
- Original-target, tokenizer and dataset revisions and official TOFU protocol match the calibration. The full evaluation uses 200 forget / 400 retain / 100 real-author / 117 world-fact / 200 holdout examples, Min-K=0.4. Evaluation seed is fixed at 0 across all training seeds.
- Seven official metrics were independently recomputed from per-example outputs. PrivLeak is made absolute per run before calculating its mean and sample SD (n-1 denominator).
- The 11 SeUL unit tests pass. Losses and gradients are finite for exactly two updates per seed. Existing LowConfidence artifacts and implementation remain unchanged.
- LR/stopping were fixed after seed0 calibration over three LRs and steps 1/2/4/8. The two predeclared NLL screening guards both selected step4; a documented amendment added step2 before viewing candidate official results. The user subsequently selected step2 for this run.
- Seed0 was trained afresh and exactly reproduces the calibration checkpoint weight hash and all official metrics. It participated in tuning; seeds1/2 are new confirmation runs. The three-seed summary is descriptive and is not independent held-out model selection.
- Method comparisons change masks, objectives and optimization budgets. These points do not establish selector superiority, a globally optimal setting, or failure of the original SeUL method.

## Saved artifacts

Implementation and run instructions: [SeUL details](seul_paper.md), [calibration](seul_calibration.md), `scripts/run_seul_formal.py`. Machine-readable results: `docs/seul/results/`.

Full server results: `saves/experiments/tofu_forget05_seul_lr1e-5_steps2_multiseed_v1/`. Three final checkpoint directories remain under `seed{0,1,2}/checkpoint-2/`.

Lightweight exports (locally backed up and manifest-verified; no model weights):

- `tofu_forget05_seul_paper_smoke_v3.zip`: SHA-256 `4f882ad3c2e0960fff0814d52766f5e04b9d2727f43ea02e2990a975013ec0ea`.
- `tofu_forget05_seul_calibration_seed0_v1.zip`: SHA-256 `f21c01a090d19dfde9c09d152d40a643b301bb176e617c5b63cb6f9b74c9e076`.
- `tofu_forget05_seul_lr1e-5_steps2_multiseed_v1.zip`: SHA-256 `26fc6558850204180a308f7c19d22769be2d9c2700d700e32065de85c7503512`.

Paper and reviewer responses were not edited for these experiments.
