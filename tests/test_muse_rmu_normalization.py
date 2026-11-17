"""Check corrected RMU against independently calculated masked MSE."""

import unittest

import torch

from trainer.unlearn.rmu import RMU


class MuseRMUNormalizationTests(unittest.TestCase):
    def test_unequal_counts_and_empty_sample(self):
        actual = torch.tensor(
            [[[2., 0.], [5., 5.], [4., 4.]],
             [[1., 3.], [2., 4.], [3., 5.]],
             [[9., 9.], [8., 8.], [7., 7.]]],
            requires_grad=True,
        )
        target = torch.zeros_like(actual)
        mask = torch.tensor([[True, False, False], [True, True, True], [False, False, False]])
        # Per-token MSEs are [2] and [5, 10, 17] in the two valid samples.
        expected = (2 + (5 + 10 + 17) / 3) / 2
        loss = RMU.compute_activation_loss(None, actual, target, mask)
        torch.testing.assert_close(loss, torch.tensor(expected))
        gradients, = torch.autograd.grad(loss, actual)
        expected_gradients = torch.zeros_like(actual)
        expected_gradients[0, 0] = actual.detach()[0, 0] / 2
        expected_gradients[1] = actual.detach()[1] / 6
        torch.testing.assert_close(gradients, expected_gradients)

    def test_all_empty_is_graph_connected_zero(self):
        actual = torch.randn(3, 4, 2, requires_grad=True)
        target = torch.ones_like(actual)
        mask = torch.zeros((3, 4), dtype=torch.bool)
        loss = RMU.compute_activation_loss(None, actual, target, mask)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.item(), 0)
        gradients, = torch.autograd.grad(loss, actual)
        torch.testing.assert_close(gradients, torch.zeros_like(actual))


if __name__ == "__main__":
    unittest.main()
