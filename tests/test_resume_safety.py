"""Diagnostic snapshots must never be accepted as exact training resumes."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model import ModelConfig
from train import SyntheticBatches, TrainConfig, Trainer, WeatherScaler, make_synthetic_data
from train import run_synthetic


class ResumeSafetyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make_failure_snapshot(self, output):
        args = SimpleNamespace(
            resume=None, device="cpu", amp=False, seed=42,
            microbatch_size=2, accumulation_steps=2, updates=1,
            output=output, scenarios=2, sampling_steps=2, chunk_size=2,
        )
        original_next = SyntheticBatches.__next__
        calls = 0

        def fail_second_microbatch(batches):
            nonlocal calls
            batch = original_next(batches)
            calls += 1
            if calls == 2:
                # One backward pass already happened; the optimizer has not stepped.
                batch["y"].fill_(float("nan"))
            return batch

        with patch.object(SyntheticBatches, "__next__", fail_second_microbatch):
            with self.assertRaisesRegex(ValueError, "complete finite windows"):
                run_synthetic(args, "loop")
        self.assertEqual(calls, 2)
        path = output / "loop" / "failure.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.assertTrue(state["metadata"]["failure"])
        self.assertEqual(state["updates"], 0)
        self.assertEqual(state["data"]["cursor"], 4)
        self.assertFalse((output / "loop" / "checkpoint.pt").exists())
        return path

    def test_partial_accumulation_snapshot_rejected_even_if_renamed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure = self.make_failure_snapshot(root / "failed")
            renamed = failure.with_name("checkpoint.pt")
            failure.rename(renamed)
            rng_before = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, "diagnostic failure snapshot"):
                Trainer.load(renamed)
            # Reject before constructing Trainer, which resets the global RNG.
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            self.assertTrue(renamed.exists())  # Keep the evidence for diagnosis.

    def test_runner_rejects_failure_before_creating_resume_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure = self.make_failure_snapshot(root / "failed")
            args = SimpleNamespace(
                resume=failure, device="cpu", output=root / "resumed",
                updates=1, scenarios=2, sampling_steps=2, chunk_size=2,
            )
            with self.assertRaisesRegex(ValueError, "diagnostic failure snapshot"):
                run_synthetic(args, "loop")
            self.assertFalse(args.output.exists())

    def test_normal_checkpoint_accepted_regardless_of_filename(self):
        model_config = ModelConfig(horizon=6, hidden_dim=16, ffn_dim=32)
        config = TrainConfig(microbatch_size=2, accumulation_steps=2)
        data = make_synthetic_data(model_config, samples=7)
        trainer = Trainer(model_config, config, WeatherScaler.fit(data["nwp"]))
        batches = SyntheticBatches(data, config.microbatch_size)
        trainer.train_update(batches)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.pt"
            # The metadata determines resumability, not the file's name.
            for metadata in ({}, {"failure": False}):
                with self.subTest(metadata=metadata):
                    trainer.save(path, batches, metadata)
                    restored, state, actual_metadata = Trainer.load(path)
                    self.assertEqual(restored.updates, trainer.updates)
                    self.assertEqual(state["cursor"], batches.cursor)
                    self.assertEqual(actual_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
