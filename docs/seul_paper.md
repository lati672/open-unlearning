# SeUL-Paper: TOFU implementation and bounded smoke test

Source: Lingzhi Wang et al., *Selective Forgetting: Advancing Machine
Unlearning Techniques and Evaluation in Language Models*, AAAI 2025,
DOI 10.1609/aaai.v39i1.32068, https://arxiv.org/html/2402.05813v2 .
This is a paper-based implementation, not the authors' official code.

## Algorithm retained from the paper

- Eq. 2 scores tokens using the original model **before unlearning**. The
  scorer is a frozen, evaluation-mode deep copy of the initialized target.
  It never becomes the evolving unlearned model or the pretraining base.
  Scoring uses canonical unpadded single-example inputs; selections are cached
  by sample ID with content/label checks. This avoids BF16 batch-shape effects
  at the threshold and is caching of Eq. 2, not oracle LLM annotation. The smoke
  test recomputes all selections from the frozen scorer after training.
- Select strictly below the sequence's mean original-model log probability,
  or use an explicitly configured fixed log-probability threshold.
- Include the eligible token between two original selected positions at
  distance two; do not recursively grow spans. Merge adjacent positions into
  disjoint spans, counting each token once.
- Minimize Eq. 1's **sum** of selected log probabilities per example (selective
  gradient ascent on NLL). No NPO transform, retain CE, or KL term is added.
- Offline LLM annotations are an oracle/evaluation variant in the paper and
  are not required for its main online-selection baseline. No new annotation
  is generated here.

## Explicit interpretations and benchmark adaptations

- The prose calls alpha the average log probability, but its displayed formula
  omits division by the token count. `threshold: mean` follows the prose;
  it does not silently implement the displayed sum.
- TOFU adaptation selects only supervised answer labels, including actual
  supervised EOS/EOT, and excludes prompts, padding and position zero. Label j
  uses logits j-1. Gap filling cannot cross ineligible positions.
- The paper does not fully specify optimizer or batch reduction. This
  implementation averages per-example sums over nonempty examples, using the
  project's globally correct DDP reduction. All-empty forget batches return a
  graph-connected zero. These are disclosed choices, not recovered official
  code details.
- The main paper studies pretrained models on Pile extraction examples and
  stops according to EL/MA thresholds. This TOFU adaptation does not claim to
  reproduce its reported benchmark numbers or its original stopping rule.

## Smoke only

`scripts/run_seul_smoke.py` is restricted to seed 0, one process, at most 32
examples and 8 optimizer steps (default 4). Default IDs span all ten authors.
It audits the selector on all 200 Forget05 examples before training and
rechecks all masks afterward. It uses the frozen v4 model, tokenizer, data
and chat-template revisions, 512-token preprocessing, bfloat16, paged AdamW,
constant 5e-5 learning rate, zero weight decay/warmup, effective batch 32,
and gradient clipping at 1. AdamW/clipping are explicit adaptations; the
5e-5 learning rate and d=32 effective batch follow the paper's main setting.

The pilot checks finite loss/gradients, frozen scorer, checkpoint save/reload,
and answer NLL on a small forget/retain sample. These are implementation
diagnostics, not official TOFU metrics or formal multi-seed results.

Example (after confirming GPU 0 is free):

```sh
source /workspace/open-unlearning/open_unlearning/bin/activate
cd /workspace/open-unlearning
PYTHONPATH=src python -m unittest discover -s tests -p test_seul.py -v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_seul_smoke.py \
  --output saves/experiments/tofu_forget05_seul_paper_smoke_v1
```

Existing LowConfidence scripts, masks, results and configs are not replaced.
