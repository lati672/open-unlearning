import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.collators import DataCollatorForSupervisedDataset, DataCollatorWithEntityMask, DataCollatorWithLogProbs
from data.unlearn import ForgetRetainDataset
from trainer.masking import MaskSelector, density_matched_random_mask, eligible_positions, sample_indices, nonempty_mean
from trainer.unlearn.entity_rmu import EntityRMU
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.rmu import RMU
from trainer.unlearn.npo import NPO, RandomNPO
from trainer.unlearn.adaptive_rmu import AdaptiveRMU, RandomRMU
from trainer.utils import build_adaptive_mask, compute_masked_dpo_loss, compute_model_token_logprobs


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids, **kwargs):
        return ":".join(str(i) for i in ids)


class ToyLM(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


def sample(index, ids=(1, 2, 0)):
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor([-100] + list(ids[1:])),
            "attention_mask": torch.ones(len(ids)), "index": index,
            "entity_mask": torch.tensor([0] + [1] * (len(ids) - 1))}


def distributed_reduction_worker(rank, store, directory):
    torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    value = torch.tensor([3.], requires_grad=True)
    loss = nonempty_mean(value, torch.tensor([rank == 0]))
    loss.backward()
    empty_value = torch.tensor([7.], requires_grad=True)
    empty = nonempty_mean(empty_value, torch.tensor([False]))
    empty.backward()
    Path(directory, f"rank{rank}.json").write_text(json.dumps({"loss": loss.item(), "gradient": value.grad.item(),
        "all_empty_loss": empty.item(), "all_empty_gradient": empty_value.grad.item()}))
    torch.distributed.destroy_process_group()


class MaskTests(unittest.TestCase):
    def test_nested_training_args_are_serializable(self):
        from omegaconf import OmegaConf
        from trainer import load_trainer_args
        with tempfile.TemporaryDirectory() as directory:
            args = load_trainer_args(OmegaConf.create({"output_dir": directory, "use_cpu": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False}, "report_to": []}), [])
            self.assertFalse(json.loads(args.to_json_string())["gradient_checkpointing_kwargs"]["use_reentrant"])

    def test_distributed_nonempty_weighting(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(distributed_reduction_worker,
                args=(str(Path(directory) / "store"), directory), nprocs=2,
                join=True, start_method="spawn")
            rows = [json.loads(Path(directory, f"rank{rank}.json").read_text()) for rank in [0, 1]]
            self.assertEqual([row["loss"] for row in rows], [3., 3.])
            self.assertEqual([row["gradient"] for row in rows], [2., 0.])
            self.assertEqual(sum(row["gradient"] for row in rows) / 2, 1.)
            self.assertEqual([row["all_empty_loss"] for row in rows], [0., 0.])
            self.assertEqual([row["all_empty_gradient"] for row in rows], [0., 0.])

    def test_real_eos_and_padding_all_collators(self):
        examples = [sample(12), sample(99, (1, 3, 4, 0))]
        for side in ["left", "right"]:
            for collator in [DataCollatorForSupervisedDataset(Tokenizer(), side),
                             DataCollatorWithEntityMask(Tokenizer(), [], side)]:
                batch = collator(examples)
                self.assertEqual(batch["attention_mask"].sum().item(), 7)
                self.assertEqual(batch["index"].tolist(), [12, 99])
                # Three zero token IDs: two real EOS and one padding position.
                self.assertEqual((batch["input_ids"].eq(0) & batch["attention_mask"]).sum().item(), 2)
                self.assertEqual(eligible_positions(batch).sum().item(), 5)

    def test_explicit_attention_is_preserved(self):
        item = sample(0)
        item["attention_mask"][1] = 0
        batch = DataCollatorForSupervisedDataset(Tokenizer())([item])
        self.assertFalse(batch["attention_mask"][0, 1])
        self.assertFalse(eligible_positions(batch)[0, 2])

    def test_logprob_default_index_and_left_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lp.jsonl"
            path.write_text("\n".join(json.dumps({"doc_id": i, "token_logprobs": [None, -2., -3., -4.]}) for i in [12, 99]))
            collator = DataCollatorWithLogProbs(Tokenizer(), str(path), padding_side="left")
            batch = collator([sample(12), sample(99, (1, 3, 4, 0))])
            self.assertEqual(batch["index"].tolist(), [12, 99])
            self.assertTrue(torch.isnan(batch["base_logprobs"][0, 0]))
            self.assertTrue(torch.isnan(batch["base_logprobs"][0, 1]))
            self.assertEqual(batch["base_logprobs"][0, 2:].tolist(), [-2., -3.])
            missing = sample(0)
            del missing["index"]
            with self.assertRaises(KeyError):
                collator([missing])

    def test_pre_fix_internal_eos_logprobs_are_rejected(self):
        from data.collators import ATTENTION_MASK_POLICY
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logprobs.json"
            path.write_text(json.dumps({"doc_id": 0, "token_logprobs": [None, -1., -2., -3.]}) + "\n")
            item = sample(0, (1, 0, 2, 0))
            with self.assertRaisesRegex(ValueError, "Stale logprobs"):
                DataCollatorWithLogProbs(Tokenizer(), str(path))([item])
            path.with_suffix(".metadata.json").write_text(json.dumps({"attention_mask_policy": ATTENTION_MASK_POLICY}))
            batch = DataCollatorWithLogProbs(Tokenizer(), str(path))([item])
            self.assertEqual(batch["attention_mask"].sum().item(), 4)

    def test_dataloader_stable_indices_under_shuffle(self):
        forget = [sample(i + 10) for i in range(9)]
        retain = [sample(100)]
        dataset = ForgetRetainDataset(forget, retain)
        collator = DataCollatorForSupervisedDataset(Tokenizer())
        for seed in [0, 1]:
            loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator,
                                generator=torch.Generator().manual_seed(seed))
            indices = [i for batch in loader for i in batch["forget"]["index"].tolist()]
            self.assertEqual(sorted(indices), list(range(10, 19)))

    def test_indices_required_and_validated(self):
        batch = DataCollatorForSupervisedDataset(Tokenizer())([sample(1)])
        del batch["index"]
        for kind in ["adaptive", "random"]:
            with self.assertRaises(KeyError):
                MaskSelector(kind)(batch, torch.zeros_like(batch["labels"], dtype=torch.float))
        for bad in [torch.tensor([-1]), torch.tensor([1.5]), torch.tensor([[1]]), torch.tensor([True])]:
            batch["index"] = bad
            with self.assertRaises(ValueError):
                sample_indices(batch)

    def test_cache_survives_order_padding_and_short_batch(self):
        samples = [sample(10), sample(11, (1, 4, 5, 6, 0))]
        selector = MaskSelector("adaptive")
        batch = DataCollatorForSupervisedDataset(Tokenizer())(samples)
        batch["base_logprobs"] = torch.tensor([[0., -3., -1., 0., 0.], [0., -1., -4., -2., 0.]])
        first = selector(batch, torch.zeros_like(batch["base_logprobs"]))
        last = DataCollatorForSupervisedDataset(Tokenizer(), "left")([samples[1], samples[0]])
        last["base_logprobs"] = torch.zeros_like(last["labels"], dtype=torch.float)
        second = selector(last, last["base_logprobs"])
        self.assertEqual(first[0, :3].tolist(), second[1, -3:].tolist())
        self.assertEqual(first[1].tolist(), second[0].tolist())
        short = DataCollatorForSupervisedDataset(Tokenizer())([samples[0]])
        torch.testing.assert_close(selector(short)[0], first[0, :3])
        changed = DataCollatorForSupervisedDataset(Tokenizer())([sample(10, (1, 7, 0))])
        with self.assertRaisesRegex(ValueError, "reused"):
            selector(changed)

    def test_random_exact_density_reproducible_and_independent(self):
        eligible = torch.tensor([[0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        reference = eligible.clone()
        reference[0, 3:] = False
        reference[1, 2:] = False
        expected = density_matched_random_mask(eligible, reference, [4, 8, 9], 1729)
        torch.manual_seed(9841)
        torch.rand(1000)
        global_state = torch.get_rng_state().clone()
        actual = density_matched_random_mask(eligible, reference, [4, 8, 9], 1729)
        torch.testing.assert_close(torch.get_rng_state(), global_state)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(1), reference.sum(1))
        self.assertEqual(actual.sum(), reference.sum())
        self.assertFalse((actual & ~eligible).any())
        reordered = density_matched_random_mask(eligible.flip(0), reference.flip(0), [9, 8, 4], 1729)
        torch.testing.assert_close(reordered.flip(0), expected)
        draws = {tuple(density_matched_random_mask(eligible, reference, [4, 8, 9], seed)[0].tolist()) for seed in range(10)}
        self.assertGreater(len(draws), 1)

    def test_adaptive_threshold_and_fallback(self):
        inputs = {"labels": torch.tensor([[-100, 1, 1, 1, 1]]),
                  "base_logprobs": torch.tensor([[float("nan"), -1., .2, .8, 2.]])}
        actual = build_adaptive_mask(inputs, torch.zeros_like(inputs["base_logprobs"]), mask_negative_diff=False)
        self.assertEqual(actual.tolist(), [[False, True, True, False, False]])
        for value in [0., float("nan")]:
            inputs["base_logprobs"].fill_(value)
            actual = build_adaptive_mask(inputs, torch.zeros_like(inputs["base_logprobs"]), mask_negative_diff=False)
            torch.testing.assert_close(actual, eligible_positions(inputs))

    def test_npo_normalizations_same_mask_and_denominator(self):
        torch.manual_seed(8)
        ids = torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4]])
        inputs = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
        model, ref = [ToyLM(torch.randn(2, 4, 5, dtype=torch.float64)) for _ in range(2)]
        mask = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 0]], dtype=torch.bool)
        delta = compute_model_token_logprobs(model.logits, ids) - compute_model_token_logprobs(ref.logits, ids)
        sums = (delta[:, 1:] * mask[:, 1:]).sum(1)
        for normalization in ["summed", "token_mean"]:
            expected = sums if normalization == "summed" else sums / mask.sum(1)
            expected = (-2 / .3 * F.logsigmoid(-.3 * expected)).mean()
            actual, _ = compute_masked_dpo_loss(model, ref, lose_inputs=inputs, lose_mask=mask, beta=.3, normalization=normalization)
            torch.testing.assert_close(actual, expected)

    def test_npo_mixed_empty_and_all_empty_gradients(self):
        torch.manual_seed(5)
        inputs = {"input_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
                  "labels": torch.tensor([[-100, 1, 2], [-100, 1, 2]])}
        model, ref = [ToyLM(torch.randn(2, 3, 4, dtype=torch.float64)) for _ in range(2)]
        for normalization in ["summed", "token_mean"]:
            mask = torch.tensor([[0, 1, 1], [0, 0, 0]], dtype=torch.bool)
            loss, _ = compute_masked_dpo_loss(model, ref, lose_inputs=inputs, lose_mask=mask, normalization=normalization)
            gradients = torch.autograd.grad(loss, model.logits)[0]
            self.assertEqual(gradients[1].count_nonzero(), 0)
            one = {k: v[:1] for k, v in inputs.items()}
            one_loss, _ = compute_masked_dpo_loss(ToyLM(model.logits[:1]), ToyLM(ref.logits[:1]), lose_inputs=one, lose_mask=mask[:1], normalization=normalization)
            torch.testing.assert_close(loss, one_loss)
            empty, _ = compute_masked_dpo_loss(model, ref, lose_inputs=inputs, lose_mask=torch.zeros_like(mask), normalization=normalization)
            self.assertEqual(empty.item(), 0.)
            self.assertTrue(empty.requires_grad)
            self.assertEqual(torch.autograd.grad(empty, model.logits)[0].count_nonzero(), 0)

    def test_rmu_entity_values_and_gradients_match(self):
        torch.manual_seed(2)
        for mask in [torch.tensor([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.bool), torch.zeros(3, 3, dtype=torch.bool)]:
            activations = torch.randn(3, 3, 7, requires_grad=True)
            target = torch.randn_like(activations)
            a = RMU.compute_activation_loss(None, activations, target, mask)
            b = EntityRMU._safe_activation_loss(activations, target, mask)
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(torch.autograd.grad(a, activations, retain_graph=True)[0], torch.autograd.grad(b, activations)[0])
            if not mask.any():
                self.assertTrue(a.requires_grad)
                self.assertEqual(a.item(), 0.)

    def test_full_rmu_entity_objectives_match_with_explicit_control(self):
        import copy
        from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
        torch.manual_seed(41)
        config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            bos_token_id=1, eos_token_id=0, pad_token_id=0)
        model = LlamaForCausalLM(config)
        batch = DataCollatorWithEntityMask(Tokenizer(), [])([
            {"forget": sample(1), "retain": sample(2)}])
        batch["forget"]["control_vec"] = torch.zeros(1, 1, 16)
        values, gradients = [], []
        with tempfile.TemporaryDirectory() as directory:
            for trainer_cls in [RMU, EntityRMU]:
                current = copy.deepcopy(model)
                args = TrainingArguments(output_dir=directory, use_cpu=True, report_to=[])
                trainer = trainer_cls(model=current, args=args, module_regex=r"model\.layers\.0",
                                      trainable_params_regex=[".*"], retain_loss_type="EMBED_DIFF")
                loss = trainer.compute_loss(current, batch)
                values.append(loss)
                gradients.append(torch.autograd.grad(loss, list(current.parameters()), allow_unused=True))
        torch.testing.assert_close(values[0], values[1])
        for a, b in zip(*gradients):
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                torch.testing.assert_close(a, b)

    def test_all_empty_retain_nll(self):
        model = ToyLM(torch.randn(1, 3, 5, requires_grad=True))
        inputs = {"input_ids": torch.ones(1, 3, dtype=torch.long), "labels": torch.full((1, 3), -100)}
        loss = GradDiff.compute_retain_loss(SimpleNamespace(retain_loss_type="NLL"), model, inputs)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(torch.autograd.grad(loss, model.logits)[0].count_nonzero(), 0)

    def test_frozen_reference_and_random_match(self):
        batch = DataCollatorForSupervisedDataset(Tokenizer())([sample(3, (1, 2, 3, 4, 0))])
        batch["reference_mask"] = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.bool)
        selected = MaskSelector("adaptive")(batch)
        torch.testing.assert_close(selected, batch["reference_mask"])
        random = MaskSelector("random")(batch)
        torch.testing.assert_close(random.sum(1), selected.sum(1))

    def test_frozen_mask_rejects_changed_logprobs(self):
        import hashlib
        from data.collators import FrozenMaskCollator
        with tempfile.TemporaryDirectory() as directory:
            lp = Path(directory) / "logprobs.json"
            lp.write_text("original")
            manifest = Path(directory) / "masks.jsonl"
            manifest.write_text(json.dumps({"doc_id": 0, "input_ids": [1, 2, 0], "mask": [0, 1, 0]}) + "\n")
            metadata_path = manifest.with_suffix(".metadata.json")
            checksum = hashlib.sha256(lp.read_bytes()).hexdigest()
            collator = SimpleNamespace(logprob_path=str(lp))
            for model_key, commit_key in [("original_model", "original_model_commit"),
                                          ("target_model", "target_model_commit")]:
                with self.subTest(model_key=model_key):
                    metadata_path.write_text(json.dumps({"logprob_sha256": checksum,
                        model_key: "tofu_full", commit_key: "example_revision"}))
                    frozen = FrozenMaskCollator(collator, manifest)
                    self.assertEqual(frozen.metadata["target_model"], "tofu_full")
                    self.assertEqual(frozen.metadata["target_model_commit"], "example_revision")
            lp.write_text("changed")
            with self.assertRaisesRegex(ValueError, "logprobs changed"):
                FrozenMaskCollator(collator, str(manifest))

    def test_frozen_random_manifest_replays_token_identity(self):
        from data.collators import FrozenMaskCollator

        examples = [sample(3, (1, 2, 3, 4, 0)), sample(9)]
        ids = [item["input_ids"].tolist() for item in examples]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            random_mask = Path(directory) / "random.jsonl"
            for path, masks in [(source, [[0, 1, 0, 1, 0], [0, 0, 0]]),
                                (random_mask, [[0, 0, 1, 0, 1], [0, 0, 0]])]:
                path.write_text("\n".join(json.dumps({"doc_id": i, "input_ids": tokens, "mask": mask})
                    for i, tokens, mask in zip([3, 9], ids, masks)))
            collator = FrozenMaskCollator(DataCollatorForSupervisedDataset(Tokenizer()), source, random_mask)
            batch = collator(examples)
            for seed in [0, 1, 2]:
                selected = MaskSelector("random", seed=seed)(batch)
                torch.testing.assert_close(selected, batch["frozen_random_mask"])
            self.assertEqual(selected.sum(-1).tolist(), [2, 0])

    def test_real_trainers_one_optimizer_step(self):
        from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
        from data.collators import FrozenMaskCollator
        config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             bos_token_id=1, eos_token_id=0, pad_token_id=0)
        for trainer_cls in [NPO, RandomNPO, RMU, EntityRMU, AdaptiveRMU, RandomRMU]:
            with tempfile.TemporaryDirectory() as directory:
                model = LlamaForCausalLM(config)
                examples = [{"forget": sample(i), "retain": sample(i + 5)} for i in range(2)]
                base_collator = DataCollatorWithEntityMask(Tokenizer(), [], index="index")
                manifest = Path(directory) / "masks.jsonl"
                manifest.write_text("\n".join(json.dumps({"doc_id": i, "input_ids": [1, 2, 0], "mask": [0, 1, 0]}) for i in range(2)))
                collator = FrozenMaskCollator(base_collator, str(manifest))
                args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=1,
                                         per_device_train_batch_size=2, report_to=[],
                                         remove_unused_columns=False, disable_tqdm=True)
                kwargs = {"model": model, "args": args, "train_dataset": examples,
                          "data_collator": collator, "retain_loss_type": "NLL"}
                if issubclass(trainer_cls, RMU):
                    kwargs.update(module_regex=r"model\.layers\.0", trainable_params_regex=[".*"])
                elif trainer_cls is NPO:
                    kwargs.update(mask="adaptive")
                trainer = trainer_cls(**kwargs)
                result = trainer.train()
                self.assertEqual(result.global_step, 1)
                self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))

    def test_real_trainers_all_empty_batch(self):
        from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
        config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            bos_token_id=1, eos_token_id=0, pad_token_id=0)
        for trainer_cls in [NPO, EntityRMU]:
            with tempfile.TemporaryDirectory() as directory:
                example = {"forget": sample(1), "retain": sample(2)}
                for part in example.values():
                    part["labels"].fill_(-100)
                    part["entity_mask"].zero_()
                args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=1,
                    per_device_train_batch_size=1, report_to=[], remove_unused_columns=False, disable_tqdm=True)
                kwargs = {"model": LlamaForCausalLM(config), "args": args, "train_dataset": [example],
                    "data_collator": DataCollatorWithEntityMask(Tokenizer(), [])}
                if trainer_cls is NPO:
                    kwargs.update(mask="entity", normalization="token_mean")
                else:
                    kwargs.update(module_regex=r"model\.layers\.0", trainable_params_regex=[".*"], retain_loss_type="EMBED_DIFF")
                result = trainer_cls(**kwargs).train()
                self.assertEqual(result.training_loss, 0.)


if __name__ == "__main__":
    unittest.main()
