import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inspect_run_config import inspect_binary, inspect_text


class RunConfigTests(unittest.TestCase):
    def test_real_training_arguments_in_both_torch_formats(self):
        import torch
        from transformers import TrainingArguments
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, report_to=[],
                                     per_device_train_batch_size=7, gradient_accumulation_steps=3,
                                     num_train_epochs=10, learning_rate=1e-5)
            path = Path(directory) / "training_args.bin"
            for use_zip in [True, False]:
                torch.save(args, path, _use_new_zipfile_serialization=use_zip)
                result = inspect_binary(path)
                self.assertEqual(result["per_device_train_batch_size"], 7)
                self.assertEqual(result["gradient_accumulation_steps"], 3)
                self.assertEqual(result["num_train_epochs"], 10)

    def test_state_does_not_confuse_logged_lr_with_initial_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer_state.json"
            path.write_text(json.dumps({"train_batch_size": 4, "log_history": [{"learning_rate": 1e-7}]}))
            result = inspect_text(path)
            self.assertEqual(result["train_batch_size"], 4)
            self.assertNotIn("learning_rate", result)


if __name__ == "__main__":
    unittest.main()
