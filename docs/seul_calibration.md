# SeUL seed-0 learning-rate/early-stop calibration

This is exploratory calibration, not formal multi-seed evidence. The SeUL
objective, frozen original-model selector and LowConfidence comparator are
unchanged. Reuse the 200 frozen SeUL masks from smoke v3 (1762/7165 positions).

Fixed grid: learning rates 1e-6, 3e-6, 1e-5; checkpoints 1, 2, 4, 8 updates;
seed 0 only, one GPU/process per run. Use all 200 Forget05 training examples,
nominal effective batch 32 (microbatch 2, accumulation 16), constant LR, no
warmup or weight decay, bfloat16, paged AdamW and max-grad-norm 1. Record actual
epochs, sample exposures and an order hash because the final accumulation
window at an epoch boundary may be partial. All LRs must share sample order.
These runs use 200 training examples, unlike the earlier 32-example smoke;
their NLL changes must not be presented as a controlled comparison with that
smoke. No original-paper EL/MA stopping policy or official reproduction is
claimed.

Probe all 200 forget examples and 128 fixed, evenly spaced Retain95 examples
at initialization and each checkpoint. The retain subset is used for tuning;
it is not an independent held-out confirmation set. Per-token answer NLL is
a screening diagnostic, not official TOFU utility or proof of data removal.

Before running the grid, freeze these candidate rules:

1. Maximize forget-NLL increase among points with retain-NLL increase <=0.1 nat.
2. Also maximize forget-NLL increase with a looser <=0.3-nat retain constraint.
3. Deduplicate those two points. If neither constraint yields a point, report
   that failure and choose the least-retention-damaging point with at least
   0.1-nat forget increase (or maximum forget increase if none).

The constraints are exploratory heuristics, not guarantees or reviewer-defined
success criteria. Evaluate at most two candidates and the original target
using the existing official TOFU suite, fixed data/model/tokenizer revisions,
all standard evaluation samples, pinned retain-only reference, Min-K=0.4,
and independent numerical replay/validation of the seven official metrics.
This official comparison remains part of tuning, not an unbiased final test.
Formal seed1/2 confirmation and any final operating-point decision require a
subsequent user instruction.

Entrypoint: `scripts/calibrate_seul.py` with phases `init`, `train`, `select`,
and `official`. The grid is hard-bounded to three LRs and eight steps, and has
no training-seed argument. Baseline/candidate evaluations never update weights.
Checkpoint weights remain preserved after evaluation.
