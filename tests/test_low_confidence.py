"""Unit checks for the LowConfidence-Matched selector and the frozen v4 NPO path."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from prepare_low_confidence import rank_low_confidence
from trainer.masking import eligible_positions
from trainer.utils import compute_masked_batch_nll, compute_masked_dpo_loss, compute_model_token_logprobs


class ToyPolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)

    def forward(self, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.logits)


class LowConfidenceTests(unittest.TestCase):
    def test_rank_and_ties(self):
        self.assertEqual(rank_low_confidence([-100, -4, -7, -7, -1], [0, 1, 1, 1, 1], 1), [0, 0, 1, 0, 0])
        self.assertEqual(rank_low_confidence([-100, -4, -7, -7, -1], [0, 1, 1, 1, 1], 3), [0, 1, 1, 1, 0])

    def test_exact_counts_and_empty(self):
        for count in range(4):
            mask = rank_low_confidence([None, -8, -3, -5, None], [0, 1, 1, 1, 0], count)
            self.assertEqual(sum(mask), count)
        self.assertEqual(rank_low_confidence([None, None], [0, 0], 0), [0, 0])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            rank_low_confidence([None, -2], [0, 1], 2)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            rank_low_confidence([None, float("nan"), -4], [0, 1, 1], 1)

    def test_shift_ignored_prompt_and_real_eos_equal_to_padding(self):
        ids = torch.tensor([[0, 2, 3, 1, 1]])
        data = {"input_ids": ids, "labels": torch.tensor([[-100, -100, 3, 1, -100]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])}
        logits = torch.randn(1, 5, 5, generator=torch.Generator().manual_seed(17))
        scores = compute_model_token_logprobs(logits, ids)
        self.assertEqual(eligible_positions(data).tolist(), [[False, False, True, True, False]])
        torch.testing.assert_close(scores[0, 2], logits[0, 1].log_softmax(-1)[3])
        torch.testing.assert_close(scores[0, 3], logits[0, 2].log_softmax(-1)[1])
        scores[0, 1] = scores[0, 4] = -1000
        self.assertEqual(rank_low_confidence(scores[0].tolist(), eligible_positions(data)[0].tolist(), 2), [0, 0, 1, 1, 0])

    def test_identical_mask_and_per_sample_normalization_before_nonlinearity(self):
        g = torch.Generator().manual_seed(7)
        current, reference = [ToyPolicy(torch.randn(3, 5, 6, generator=g, dtype=torch.float64)) for _ in range(2)]
        data = {"input_ids": torch.tensor([[0, 2, 3, 1, 1]] * 3),
                "labels": torch.tensor([[-100, -100, 3, 1, -100]] * 3),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]] * 3)}
        supplied = torch.tensor([[1, 0, 1, 1, 1], [0, 0, 0, 1, 0], [0] * 5], dtype=torch.bool)
        valid = eligible_positions(data) & supplied
        beta = .1
        with patch("trainer.utils.compute_masked_batch_nll", wraps=compute_masked_batch_nll) as called:
            result, _ = compute_masked_dpo_loss(current, reference, lose_inputs=data, lose_mask=supplied,
                                                normalization="token_mean", beta=beta)
        self.assertEqual(called.call_count, 2)
        for args in called.call_args_list:
            torch.testing.assert_close(args.kwargs["token_mask"], valid)
        target_logp = compute_model_token_logprobs(current.logits, data["input_ids"])
        reference_logp = compute_model_token_logprobs(reference.logits, data["input_ids"])
        change = [torch.mean(reference_logp[i, valid[i]] - target_logp[i, valid[i]]) for i in (0, 1)]
        expected = (-2 / beta * F.logsigmoid(beta * torch.stack(change))).mean()
        torch.testing.assert_close(result, expected)
        empty, _ = compute_masked_dpo_loss(current, reference, lose_inputs=data, lose_mask=torch.zeros_like(supplied),
                                           normalization="token_mean", beta=beta)
        self.assertEqual(empty.item(), 0)
        self.assertTrue(empty.requires_grad)
        self.assertEqual(torch.count_nonzero(torch.autograd.grad(empty, current.logits)[0]), 0)
        self.assertIsNone(reference.logits.grad)


if __name__ == "__main__":
    unittest.main()
