# SU-NGram (paper-based reimplementation)

Wan et al. (2025), [paper §3, §4.3 and Appendix B](https://arxiv.org/html/2506.00876v1).

## Paper-derived choices

- Use 3-gram assistants for TOFU. Compare an assistant fit on full data against one fit on retain data.
- Score an observed next token by the absolute difference in assistant conditional probabilities. The strict threshold is 0.9.
- Apply selective gradient ascent on forget data and ordinary next-token gradient descent on retain data. The paper describes a five-token neighborhood around selected tokens.

## Reimplementation choices

- The pinned experiment uses Forget05 and Retain95. The full corpus is their concatenation. The paper's TOFU experiment uses Forget10 and Retain90. Dataset and model revisions are pinned in the shared protocol config.
- Use the exact tokenizer and chat template used by the target model, including its date string. Count all attended IDs of the tokenized question and response, including real control tokens, capped at the training length of 512. Each example resets the history. There is no cross-example context.
- Each target uses up to two preceding IDs. The initial real BOS is context and is unscored. The second position uses one observed preceding ID. No virtual start/end tokens are inserted. Real EOS/EOT can be counted and selected when supervised and attended. A pad position is ineligible even when its token ID equals EOS/EOT.
- The default estimator is unsmoothed maximum likelihood, count of context and target divided by context total. MLE returns zero for an unseen target in a seen context and for an unseen context. Optional add-alpha uses `(count + alpha) / (total + alpha * V)` and yields `1/V` for an unseen context. `V` is the complete target tokenizer size including added tokens. The configured alpha is 0.1 and applies only with add-alpha. Threshold sensitivity keeps the configured estimator fixed.
- For the paper variant, the threshold applies to all scored attended positions by default, including prompt centers. An eligible-target center scope is also supported in config. Radius two windows are clipped at example boundaries, and overlapping windows are unioned. This follows the requested interpretation of the paper's prose. Algorithm 1 also shows a removal operation and does not uniquely specify a pure window union. Only supervised, attended targets with an attended predecessor survive the final intersection. Position `j` uses logits `j-1`. Report eligible-center and all-attended-center rates separately.
- SU-Score-Matched is a selection-criterion control. It ranks eligible positions by descending score and ascending token position for ties, takes the exact frozen Adaptive count, and applies no expansion. The frozen Random-Adaptive count is checked for every sample.
- The forget loss is negative summed selected NLL per example, then averaged over examples with a nonempty mask. Empty examples contribute no forget gradient. An all-empty batch returns graph-connected zero forget loss while retaining the retain term. DDP uses the repository's global nonempty-sample reduction.
- Retain CE averages all valid next-token labels within each rank microbatch, with DDP averaging rank gradients and gradient accumulation averaging microbatch gradients. Retain remains active for empty forget masks. The total is `gamma_forget * forget_loss + alpha_retain * retain_loss`, with both coefficients set to one. Mask selection uses its own threshold.
- Count artifacts and masks are fit once before gradient updates. Training reads an indexed frozen mask using the repository's collator, with stable dataset indices and exact input-ID verification. Assistants are not loaded by the trainer.
- The smoke and future formal runs use the repository's forget-anchored dataset and sample retain examples with replacement. The optimizer, effective batch size and 120-step schedule are declared separately from the paper method. Smoke uses one warmup optimizer step, four optimizer steps and one checkpoint. Its reloaded-model evaluation is eight raw QA examples per split with greedy generation capped at 16 tokens.
- Token category figures reuse the frozen TOFU audit's lexical heuristics, including its author-name matches, function words and attribute proxies. The categories are descriptive. Precision and recall use the reference mask as the positive label, and Jaccard excludes empty unions in its macro average.

The two variant configs and shared config under `configs/protocols/` make these choices explicit. The prepared output saves both resolved configurations and checksums for the code, frozen reference inputs, assistant artifacts and final masks.
