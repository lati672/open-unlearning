# Inventory

12 archived trajectories; 24 official evaluations at steps60/120 replayed from per-example outputs. No trained full/incremental weights, optimizer, scheduler or RNG state remain. Exact continuation and direct paraphrase evaluation of these archived models are impossible. Trainer-state JSON and training_args alone are insufficient.

12 fresh reproductions at fixed steps30/60/90/120 are authorized; never splice their checkpoints with archived curves. Original target cache currently lacks weight files and must be restored at its pinned revision. The original tokenizer and TOFU dataset caches are present.

See manifests/inventory.json, run_manifest.json and original_preservation.json for individual hashes and availability.

## Final availability after this experiment

The pinned original target full checkpoint is restored, upstream-hash verified and retained. Its original-question and paraphrase evaluation ran once. The pinned base tokenizer and all required raw datasets are present and hash verified. Base full weights and retain-only full weights were not required or independently verified; frozen masks and official retain-only evaluation logs were used unchanged.
All12 new trajectories finished exactly120 updates; all48 full-model checkpoints loaded successfully for official evaluation, and all12 step120 checkpoints also completed original/paraphrase evaluation. New weights were pruned only after those evaluations under the standing retention preference, with file sizes and hashes in per-run NEW_WEIGHTS_CLEANUP_RECEIPT.json. No old checkpoint/model/result was removed or overwritten. New runs retain metadata/configs/evaluations, not a resumable optimizer/scheduler/RNG state bundle.
The actual trained weights from the archived Table4 experiments remain unavailable. Exact old-trajectory continuation and direct new-question evaluation of those deleted old bytes are impossible; new results are explicitly fresh reproductions. No missing material prevents replaying the saved scores and matching analysis. See final_checkpoint_inventory.json and reproduction_vs_archived.csv.
