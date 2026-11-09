import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from freeze_mask_protocol import DEFAULT_PROTOCOL, fixed_rmu_vector
from train_mask_experiment import StepCheckpointEvaluation
from trainer.unlearn.rmu import RMU


class ToyModel(torch.nn.Linear):
    def forward(self, inputs):
        return {"loss": super().forward(inputs).square().mean()}


class FrozenProtocolTests(unittest.TestCase):
    def test_legacy_steps_and_checkpoint_cleanup(self):
        training = json.loads(DEFAULT_PROTOCOL.read_text())["training"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer = Trainer(ToyModel(1, 1), TrainingArguments(
                output_dir=directory, use_cpu=True, num_train_epochs=20,
                per_device_train_batch_size=4, gradient_accumulation_steps=4,
                max_steps=120, save_steps=60, save_strategy="no", report_to=[], disable_tqdm=True,
            ), train_dataset=[{"inputs": torch.tensor([1.])}] * 100)

            def evaluate(model, target):
                self.assertTrue(list(target.parent.glob("*.safetensors")))
                target.mkdir(exist_ok=True)
                model.eval()
                random.random()
                np.random.rand()
                torch.rand(1)
                return {"loss": 1.0}

            callback = StepCheckpointEvaluation(trainer, training, evaluate, output)
            trainer.add_callback(callback)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 120)
            self.assertEqual(trainer.state.epoch, 19.2)
            for step, epoch in [(60, 9.6), (120, 19.2)]:
                checkpoint = output / f"checkpoint-{step}"
                result = json.loads((checkpoint / "evaluated.json").read_text())
                self.assertEqual(result["actual_epoch"], epoch)
                self.assertTrue(result["weights_removed_after_eval"])
                self.assertFalse(list(checkpoint.glob("*.safetensors")))
                self.assertTrue((checkpoint / "trainer_state.json").exists())

            torch_state = torch.get_rng_state().clone()
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            callback.save_and_evaluate(trainer.state)
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            self.assertEqual(python_state, random.getstate())
            np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])

    def test_eval_failure_preserves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer = Trainer(ToyModel(1, 1), TrainingArguments(
                output_dir=directory, use_cpu=True, report_to=[], disable_tqdm=True,
            ))
            trainer.state.global_step = 60
            trainer.state.epoch = 9.6

            def fail(*args):
                raise RuntimeError("eval failed")

            callback = StepCheckpointEvaluation(trainer, {}, fail, output)
            with self.assertRaisesRegex(RuntimeError, "eval failed"):
                callback.save_and_evaluate(trainer.state)
            self.assertTrue(list((output / "checkpoint-60").glob("*.safetensors")))
            self.assertFalse((output / "checkpoint-60/evaluated.json").exists())

    def test_frozen_control_vector_preserves_legacy_rng_draw(self):
        protocol = json.loads(DEFAULT_PROTOCOL.read_text())
        for seed in protocol["training"]["seeds"]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                vector = fixed_rmu_vector(protocol)
                state = torch.get_rng_state()
                legacy = SimpleNamespace(control_vec=None, fixed_control_vec=None, steering_coeff=2)
                fixed = SimpleNamespace(control_vec=None, fixed_control_vec=vector, steering_coeff=2)
                RMU.get_control_vector(legacy, 2048)
                legacy_rng = torch.get_rng_state()
                torch.set_rng_state(state)
                self.assertTrue(torch.equal(RMU.get_control_vector(fixed, 2048), vector))
                self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
