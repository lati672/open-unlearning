"""Tests for ESS matching, weighted causal NLL, and frozen replay."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_soft_adaptive import ess_weights
from run_soft_adaptive import FrozenTokenWeightCollator
from data.collators import DataCollatorForSupervisedDataset
from trainer.masking import eligible_positions
from trainer.unlearn.weighted_npo import compute_weighted_batch_nll, compute_weighted_npo_loss
from trainer.utils import compute_masked_batch_nll, compute_masked_dpo_loss


class ToyModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def forward(self, input_ids, labels, attention_mask=None):
        return SimpleNamespace(logits=self.logits)


class TestWeightedNPO(unittest.TestCase):
    def setUp(self):
        self.batch = {"input_ids": torch.tensor([[1, 2, 0, 0, 0], [1, 0, 2, 1, 0]]),
                      "labels": torch.tensor([[1, 2, 0, -100, 0], [1, -100, 2, 1, 0]]),
                      "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])}
        logits = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(73))
        self.model = ToyModel(logits)
        self.reference = ToyModel(logits.roll(1, dims=-1))

    def test_uniform_and_binary_match_existing_nll_and_npo(self):
        all_eligible = eligible_positions(self.batch)
        selective = all_eligible.clone()
        selective[1, 4] = False
        for mask in (all_eligible, selective):
            count = mask.sum(-1)
            weights = mask.float() / count[:, None]
            weighted_nll, valid, _ = compute_weighted_batch_nll(self.model, self.batch, weights)
            binary_nll, _ = compute_masked_batch_nll(self.model, self.batch, token_mask=mask)
            self.assertTrue(valid.all())
            torch.testing.assert_close(weighted_nll, binary_nll / count)
            weighted, _ = compute_weighted_npo_loss(self.model, self.reference, self.batch, weights, .1)
            binary, _ = compute_masked_dpo_loss(self.model, self.reference, lose_inputs=self.batch, lose_mask=mask,
                                                beta=.1, normalization="token_mean")
            torch.testing.assert_close(weighted, binary)

    def test_same_weights_shift_and_nonlinearity(self):
        weights = torch.tensor([[0, .2, .8, 0, 0], [0, 0, .5, .3, .2]])
        with patch("trainer.unlearn.weighted_npo.compute_weighted_batch_nll", wraps=compute_weighted_batch_nll) as weighted:
            loss, _ = compute_weighted_npo_loss(self.model, self.reference, self.batch, weights, .1)
        self.assertEqual(weighted.call_count, 2)
        self.assertIs(weighted.call_args_list[0].args[2], weighted.call_args_list[1].args[2])
        ratios = []
        for row in range(2):
            ratio = 0.
            for pos in weights[row].nonzero().flatten().tolist():
                y = self.batch["labels"][row, pos]
                ratio += weights[row, pos] * (self.reference.logits[row, pos - 1].log_softmax(-1)[y] -
                                              self.model.logits[row, pos - 1].log_softmax(-1)[y])
            ratios.append(ratio)
        torch.testing.assert_close(loss, -20 * F.logsigmoid(.1 * torch.stack(ratios)).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.logits.grad).all())
        self.assertIsNone(self.reference.logits.grad)

    def test_invalid_and_zero_weights(self):
        for pos in (0, 3, 4):
            w = torch.zeros(2, 5)
            w[0, pos] = 1
            with self.assertRaises(ValueError):
                compute_weighted_batch_nll(self.model, self.batch, w)
        w = torch.tensor([[0, .1, .3, 0, 0], [0, 0, .5, .3, .2]])
        with self.assertRaises(ValueError):
            compute_weighted_batch_nll(self.model, self.batch, w)
        loss, _ = compute_weighted_npo_loss(self.model, self.reference, self.batch, torch.zeros(2, 5), .1)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertFalse(self.model.logits.grad.any())

    def test_weight_replay_and_padding(self):
        sample = {"index": 2, "input_ids": torch.tensor([2, 1, 0]), "labels": torch.tensor([-100, 1, 0])}
        other = {"index": 4, "input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, 2])}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.jsonl"
            path.write_text("".join(json.dumps({"doc_id": row["index"], "input_ids": row["input_ids"].tolist(),
                "labels": row["labels"].tolist(), "weights": weights}) + "\n"
                for row, weights in ((sample, [0, .125, .875]), (other, [0, 1]))))
            for side in ("left", "right"):
                c = FrozenTokenWeightCollator(DataCollatorForSupervisedDataset(SimpleNamespace(pad_token_id=0),
                                                     index="index", padding_side=side), path)
                a, b = c([sample, other]), c([other, sample])
                torch.testing.assert_close(a["token_weights"], b["token_weights"].flip(0), rtol=0, atol=0)
                torch.testing.assert_close(a["token_weights"].sum(-1), torch.ones(2))
                self.assertFalse(a["token_weights"][~eligible_positions(a)].any())
                torch.testing.assert_close(a["token_weights"][0, -1], torch.tensor(.875))

    def test_ess_bisection_ties_and_limits(self):
        ranks, w, tau, ess, boundary = ess_weights([2., 5., -2., 2., .5], 2)
        np.testing.assert_allclose(ranks, [.625, 1., 0., .625, .25])
        self.assertTrue((w > 0).all())
        self.assertGreater(tau, 0)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertAlmostEqual(ess, 2)
        self.assertIsNone(boundary)
        self.assertAlmostEqual(ess_weights([1.], 1)[3], 1.)
        _, w, tau, ess, edge = ess_weights([3., 3., 1.], 1)
        self.assertTrue((w.astype(np.float32) > 0).all())
        self.assertGreater(tau, 0)
        self.assertAlmostEqual(ess, 2.)
        self.assertIsNotNone(edge)


if __name__ == "__main__":
    unittest.main()
