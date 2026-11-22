# Experiment checkpoint retention

User preference recorded on 2026-11-22 for this open-unlearning project:

- After all required evaluations of a checkpoint finish successfully, save and verify the metrics, per-example evaluation outputs, logs, masks, actual configuration and provenance. Then delete trained checkpoint weights and unnecessary optimizer/scheduler state when no further evaluation or training continuation needs them.
- Record the weight filenames, sizes and SHA-256 hashes before deletion, and save a cleanup receipt. Verify that evaluation artifacts remain intact afterward.
- Calibration checkpoints may be removed after their planned screening is complete and they are no longer selected for further evaluation.
- Preserve original target/base/reference/retrained input models, Hugging Face caches, datasets, environments, tokenizers, training_args.bin, configuration files, masks, logprobs and evaluation products.
- Do not remove checkpoints needed by active jobs, pending evaluations or an explicit user request to retain them. Check active processes before cleanup.
- Keep deletion scoped to identified training-output files. Old result archives may describe weights that were retained at the time; consult WEIGHTS_REMOVED.json and cleanup receipts for current availability.
