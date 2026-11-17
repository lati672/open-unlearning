"""Mask counts and tie breaking for the frozen density screening experiment."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from prepare_density_ablation import DENSITIES, build_masks, count_at_density


class DensityTests(unittest.TestCase):
    def test_half_up_and_nonempty_minimum(self):
        self.assertEqual([count_at_density(n, 50) for n in range(8)], [0, 1, 1, 2, 2, 3, 3, 4])
        self.assertEqual(count_at_density(10, 15), 2)
        self.assertEqual(count_at_density(50, 15), 8)
        self.assertEqual(count_at_density(1, 15), 1)
        self.assertEqual(count_at_density(100, 34), 34)

    def test_rank_ties_and_boundaries(self):
        sample = {3: {"eligible": [False, False, True, True, True, False], "delta": [None, -100, -3, -3, -1, -100]}}
        masks = build_masks(sample)
        self.assertEqual(masks["Discrepancy-15"][3], [0, 0, 1, 0, 0, 0])
        self.assertEqual(masks["Discrepancy-50"][3], [0, 0, 1, 1, 0, 0])
        self.assertEqual(masks["BelowMean"][3], [0, 0, 1, 1, 0, 0])
        self.assertEqual(masks["ZeroThreshold"][3], [0, 0, 1, 1, 1, 0])

    def test_random_exact_count_eligibility_and_regeneration(self):
        samples = {i: {"eligible": [False] + [True] * i + [False], "delta": [None] + [-1] * i + [None]}
                   for i in range(1, 25)}
        masks = build_masks(samples)
        self.assertEqual(masks, build_masks(dict(reversed(list(samples.items())))))
        for doc, example in samples.items():
            for density in DENSITIES:
                a = masks[f"Discrepancy-{density}"][doc]
                b = masks[f"Random-{density}"][doc]
                self.assertEqual(sum(a), sum(b))
                self.assertFalse(any(x and not y for x, y in zip(b, example["eligible"])))


if __name__ == "__main__":
    unittest.main()
