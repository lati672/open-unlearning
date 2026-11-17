import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_entity_attribute_ablation import (
    ARMS, ROOT, SEEDS, annotation_interval, digest, indexed_jsonl, span_labels, union_positions,
)
from trainer.masking import density_matched_random_mask, eligible_positions
from trainer.utils import compute_masked_batch_nll, compute_masked_dpo_loss


class EntityAttributeTests(unittest.TestCase):
    def test_offsets_require_nonempty_intersection_with_label_positions(self):
        offsets = [(0, 4), (5, 9), (9, 12), (12, 12), (12, 17)]
        self.assertEqual(annotation_interval("an engaging story", "engaging"), (3, 11))
        self.assertEqual(span_labels(offsets, 6, 12, {2, 3, 4}), {2})
        with self.assertRaises(ValueError):
            annotation_interval("name name", "name")

    def test_union_deduplicates_and_intersects_eligible_positions(self):
        self.assertEqual(union_positions({0, 2, 3}, {2, 4, 5}, {2, 3, 4}), {2, 3, 4})

    def test_random_matches_counts_and_empty_rows(self):
        ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        labels = ids.clone()
        labels[:, :2] = -100
        inputs = {"input_ids": ids, "labels": labels, "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])}
        allowed = eligible_positions(inputs)
        target = torch.tensor([[False, False, True, False], [False, False, False, False]])
        result = density_matched_random_mask(allowed, target, [23, 90], 1729)
        self.assertTrue(torch.equal(result.sum(dim=1), target.sum(dim=1)))
        self.assertFalse((result & ~allowed).any())

    def test_summed_npo_label_shift_and_empty_samples(self):
        inputs = {"input_ids": torch.tensor([[0, 0, 1, 1]] * 2),
                  "labels": torch.tensor([[-100, -100, 1, 1]] * 2),
                  "attention_mask": torch.ones(2, 4, dtype=torch.bool)}
        logits = torch.tensor([[[8., 0.], [0., math.log(3)], [math.log(3), 0.], [0., 8.]]] * 2,
                              requires_grad=True, dtype=torch.float64)
        reference = torch.zeros_like(logits)
        policy = lambda **_: SimpleNamespace(logits=logits)
        ref = lambda **_: SimpleNamespace(logits=reference)
        mask = torch.tensor([[False, False, True, True], [False, False, False, False]])
        sums, _ = compute_masked_batch_nll(policy, inputs, token_mask=mask)
        torch.testing.assert_close(sums, torch.tensor([-math.log(.75 * .25), 0.], dtype=torch.float64))
        loss, _ = compute_masked_dpo_loss(policy, ref, lose_inputs=inputs, lose_mask=mask, beta=.1)
        expected = 20 * math.log1p(math.exp(-.1 * math.log(.25 / (.75 * .25))))
        self.assertAlmostEqual(loss.item(), expected)
        grad, = torch.autograd.grad(loss, logits, retain_graph=True)
        self.assertTrue((grad[0, 1:3].abs().sum(dim=-1) > 0).all())
        self.assertEqual(grad[0, [0, 3]].count_nonzero().item(), 0)
        self.assertEqual(grad[1].count_nonzero().item(), 0)
        empty, _ = compute_masked_dpo_loss(policy, ref, lose_inputs=inputs, lose_mask=torch.zeros_like(mask), beta=.1)
        self.assertEqual(empty.item(), 0.)
        self.assertTrue(empty.requires_grad)
        self.assertEqual(torch.autograd.grad(empty, logits)[0].count_nonzero().item(), 0)

    def test_each_planned_seed_uses_same_verified_mask(self):
        manifest = json.loads((ROOT / "protocol_manifest.json").read_text())
        plan = json.loads((ROOT / "run_plan.json").read_text())
        for arm in ARMS:
            path = ROOT / "frozen_inputs" / f"{arm}.jsonl"
            self.assertEqual([r["seed"] for r in plan["new_runs"] if r["arm"] == arm], list(SEEDS))
            for seed in SEEDS:
                self.assertEqual(digest(path), manifest["mask_hashes"][arm], f"mask for {arm}/{seed}")
        ea, random = [indexed_jsonl(ROOT / "frozen_inputs" / f"{arm}.jsonl") for arm in ARMS]
        for a, b in zip(ea, random, strict=True):
            self.assertEqual((a["doc_id"], a["input_ids"], sum(a["mask"])),
                             (b["doc_id"], b["input_ids"], sum(b["mask"])))


if __name__ == "__main__":
    unittest.main()
