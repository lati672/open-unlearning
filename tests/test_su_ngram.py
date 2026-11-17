import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from data.su_ngram import NGramAssistant, canonical_json, paper_mask, score_matched_mask, window_union
from trainer.unlearn.selective_gradient_difference import SelectiveGradientDifference


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.zeros(2, 5, 6))

    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(logits=self.scores[input_ids[:, 0]])


def batch(mask):
    ids = torch.tensor([[0, 2, 3, 4, 4], [1, 2, 3, 4, 4]])
    inputs = dict(input_ids=ids, attention_mask=torch.tensor([[1, 1, 1, 1, 0]] * 2),
                  labels=torch.tensor([[-100, -100, 3, 4, -100]] * 2), index=torch.tensor([12, 3]))
    return {"forget": {**inputs, "reference_mask": torch.tensor(mask)}, "retain": inputs}


def loss(model, inputs, forget=1, retain=0):
    trainer = SimpleNamespace(gamma_forget=forget, alpha_retain=retain)
    return SelectiveGradientDifference.compute_loss(trainer, model, inputs)


def assistant(**kwargs):
    return NGramAssistant(training_split=["toy"], tokenizer_id="toy", tokenizer_revision="r1",
                          vocabulary_size=8, **kwargs)


class SUTests(unittest.TestCase):
    def test_counts_and_probabilities(self):
        sequences = [[0, 1, 2], [0, 1, 2, 5], [0, 1, 3]]
        mle = assistant().fit(sequences)
        self.assertEqual(mle.totals[(0, 1)], 3)
        self.assertEqual(mle.counts[(0, 1)][2], 2)
        self.assertEqual(mle.probability([0], 1), 1)
        self.assertEqual(mle.probability([0, 1], 2), 2 / 3)
        self.assertEqual(mle.probability([0, 1], 4), 0)
        self.assertEqual(mle.probability([7, 1], 4), 0)
        self.assertEqual(mle.metadata["token_count"], 10)
        self.assertEqual(mle.metadata["predicted_token_count"], 7)
        self.assertEqual(mle.scores([0, 1, 2]), [None, 1, 2 / 3])
        self.assertNotIn((2, 0), mle.totals)
        smoothed = assistant(smoothing="add_alpha", alpha=0.5).fit(sequences)
        self.assertEqual(smoothed.probability([0, 1], 2), 2.5 / 7)
        self.assertEqual(smoothed.probability([0, 1], 4), 0.5 / 7)
        self.assertEqual(smoothed.probability([7, 1], 4), 1 / 8)
        self.assertAlmostEqual(sum(smoothed.probability([0, 1], token) for token in range(8)), 1)

    def test_deterministic_serialization_and_digest(self):
        sequences = [[0, 1, 2, 4], [0, 2, 1], [1, 2, 3]]
        with tempfile.TemporaryDirectory() as directory:
            path, other = (Path(directory) / name for name in ["first.json", "second.json"])
            first = assistant().fit(sequences)
            first.save(path)
            assistant().fit(reversed(sequences)).save(other)
            self.assertEqual(path.read_bytes(), other.read_bytes())
            loaded = NGramAssistant.load(path)
            self.assertEqual(first.scores(sequences[0]), loaded.scores(sequences[0]))
            self.assertEqual(first.save(path), loaded.save(other))
            self.assertEqual(path.read_bytes(), other.read_bytes())
            other.write_text(other.read_text().replace('"token_count":10', '"token_count":11'))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                NGramAssistant.load(other)

    def test_strict_threshold_and_final_eligibility(self):
        centers, expanded, final = paper_mask([None, .91, .9, .1, .99], [0, 1, 1, 1, 0],
                                               [0, 0, 1, 1, 0], .9, 2)
        self.assertEqual(centers, [0, 1, 0, 0, 0])
        self.assertEqual(expanded, [1, 1, 1, 1, 0])
        self.assertEqual(final, [0, 0, 1, 1, 0])

    def test_window_boundaries(self):
        self.assertEqual(window_union([1, 0, 0, 0, 0, 0, 1], 2), [1, 1, 1, 0, 1, 1, 1])
        self.assertEqual(window_union([1], 2), [1])

    def test_window_union(self):
        self.assertEqual(window_union([0, 0, 1, 0, 1, 0, 0, 0], 2), [1, 1, 1, 1, 1, 1, 1, 0])

    def test_score_matched_stable_ties(self):
        self.assertEqual(score_matched_mask([None, 1, .7, .7, .9], [0, 0, 1, 1, 1], 2), [0, 0, 1, 0, 1])
        self.assertEqual(score_matched_mask([None, 1, .7], [0, 0, 1], 0), [0, 0, 0])
        with self.assertRaises(ValueError):
            score_matched_mask([None, 1, .7], [0, 0, 1], 2)

    def test_shift_and_forget_only_gradient(self):
        model = ToyLM()
        inputs = batch([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
        def logprob():
            logp = F.log_softmax(model.scores, dim=-1)
            return torch.stack([logp[0, 1, 3], logp[1, 2, 4]]).mean()
        initial = logprob().item()
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        objective = loss(model, inputs)
        torch.testing.assert_close(objective, logprob())
        objective.backward()
        touched = model.scores.grad.abs().sum(dim=-1).nonzero().tolist()
        self.assertEqual(touched, [[0, 1], [1, 2]])
        optimizer.step()
        self.assertLess(logprob().item(), initial)

    def test_empty_mask_and_retain_direction(self):
        model = ToyLM()
        inputs = batch([[0] * 5, [0] * 5])
        objective = loss(model, inputs)
        self.assertTrue(objective.requires_grad)
        self.assertEqual(objective.item(), 0)
        self.assertEqual(torch.count_nonzero(torch.autograd.grad(objective, model.scores)[0]), 0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        before = loss(model, inputs, forget=0, retain=1).item()
        loss(model, inputs, forget=0, retain=1).backward()
        optimizer.step()
        self.assertLess(loss(model, inputs, forget=0, retain=1).item(), before)

    def test_reject_invalid_mask_and_missing_id(self):
        model = ToyLM()
        for position in [0, 1, 4]:
            inputs = batch([[0] * 5, [0] * 5])
            inputs["forget"]["reference_mask"][0, position] = 1
            with self.assertRaisesRegex(ValueError, "eligible"):
                loss(model, inputs)
        inputs = batch([[0] * 5, [0] * 5])
        inputs["forget"].pop("index")
        with self.assertRaisesRegex(KeyError, "stable"):
            loss(model, inputs)

    def test_frozen_mask_reload_hash_padding_and_reordering(self):
        tokenizer = SimpleNamespace(pad_token_id=4)
        samples = [
            dict(input_ids=torch.tensor([0, 2, 3, 4]), labels=torch.tensor([-100, -100, 3, 4]),
                 attention_mask=torch.tensor([1, 1, 1, 1]), index=12),
            dict(input_ids=torch.tensor([1, 4]), labels=torch.tensor([-100, 4]),
                 attention_mask=torch.tensor([1, 1]), index=3),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.jsonl"
            records = [dict(doc_id=item["index"], input_ids=item["input_ids"].tolist(), mask=mask)
                       for item, mask in zip(samples, [[0, 0, 1, 0], [0, 1]])]
            path.write_text("".join(canonical_json(row) + "\n" for row in records))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for _ in range(2):
                collator = FrozenMaskCollator(DataCollatorForSupervisedDataset(tokenizer, index="index"), path)
                result = collator(samples[::-1])
                self.assertEqual(result["index"].tolist(), [3, 12])
                self.assertEqual(result["reference_mask"].tolist(), [[0, 1, 0, 0], [0, 0, 1, 0]])
                self.assertEqual(result["attention_mask"].tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
