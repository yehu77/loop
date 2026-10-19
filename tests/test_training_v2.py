"""Protocol boundaries, exact continuation, and paired architecture initialization."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model import ModelConfig
from train import TrainConfig, Trainer, SyntheticBatches, WeatherScaler, make_synthetic_data


class ProtocolV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = ModelConfig(variant="base", horizon=6, hidden_dim=16, ffn_dim=32)
        self.config = TrainConfig(microbatch_size=2, accumulation_steps=2,
                                  training_protocol_version=2, ema_start_update=2)
        self.data = make_synthetic_data(self.model, samples=7)
        self.scaler = WeatherScaler.fit(self.data["nwp"])

    def create(self, variant="base", config=None):
        trainer = Trainer(replace(self.model, variant=variant), config or self.config, self.scaler)
        return trainer, SyntheticBatches(self.data, 2)

    def equal_states(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0, msg=key)

    def test_boundary_copy_then_average_and_copy_buffers(self):
        trainer, batches = self.create()
        initial = {k: v.clone() for k, v in trainer.ema.state_dict().items()}
        trainer.train_update(batches)
        self.assertFalse(trainer.ema_initialized)
        self.equal_states(initial, trainer.ema.state_dict())
        trainer.train_update(batches)
        self.assertTrue(trainer.ema_initialized)
        self.assertEqual(trainer.ema_num_averaged_updates, 0)
        self.equal_states(trainer.model.state_dict(), trainer.ema.state_dict())
        before = {k: v.clone() for k, v in trainer.ema.named_parameters()}
        trainer.model.position_encoding.add_(0.25)
        trainer.train_update(batches)
        self.assertEqual(trainer.ema_num_averaged_updates, 1)
        for name, value in trainer.model.named_parameters():
            expected = before[name].lerp(value, 1 - self.config.ema_decay)
            torch.testing.assert_close(dict(trainer.ema.named_parameters())[name], expected, atol=0, rtol=0)
        self.equal_states(dict(trainer.model.named_buffers()), dict(trainer.ema.named_buffers()))

    def test_skipped_attempt_does_not_start_or_advance_ema(self):
        trainer, batches = self.create()
        for successful_count in (0, 1, 2, 3):
            while trainer.updates < successful_count:
                trainer.train_update(batches)
            before = {k: v.clone() for k, v in trainer.ema.state_dict().items()}
            count, ready = trainer.ema_num_averaged_updates, trainer.ema_initialized
            # Simulate the GradScaler contract: skipped optimizer step and reduced scale.
            with patch.object(trainer.amp_scaler, "step"), patch.object(
                    trainer.amp_scaler, "get_scale", side_effect=[2., 1.]):
                result = trainer.train_update(batches)
            self.assertFalse(result["did_update"])
            self.assertEqual(trainer.updates, successful_count)
            self.assertEqual((trainer.ema_num_averaged_updates, trainer.ema_initialized), (count, ready))
            self.equal_states(before, trainer.ema.state_dict())

    def test_resume_before_at_and_after_start_matches_continuous(self):
        for boundary in (1, 2, 3):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                trainer, batches = self.create()
                for _ in range(boundary):
                    trainer.train_update(batches)
                path = Path(directory) / "checkpoint.pt"
                trainer.save(path, batches)
                expected = [trainer.train_update(batches) for _ in range(3)]
                restored, state, _ = Trainer.load(path, expected_variant="base",
                                                 expected_training_protocol_version=2)
                other_batches = SyntheticBatches(self.data, 2)
                other_batches.load_state_dict(state)
                actual = [restored.train_update(other_batches) for _ in range(3)]
                self.assertEqual(actual, expected)
                self.equal_states(trainer.model.state_dict(), restored.model.state_dict())
                self.equal_states(trainer.ema.state_dict(), restored.ema.state_dict())
                self.assertEqual(trainer.ema_num_averaged_updates, restored.ema_num_averaged_updates)

    def test_legacy_resume_and_mismatch_rejection_before_rng_changes(self):
        trainer, batches = self.create(config=replace(self.config, training_protocol_version=1, ema_start_update=0))
        trainer.train_update(batches)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            trainer.save(path, batches)
            state = torch.load(path, weights_only=True)
            state["format_version"] = 1
            for key in ("training_protocol_version", "ema_start_update"):
                del state["train_config"][key]
            for key in ("ema_initialized", "ema_num_averaged_updates"):
                del state[key]
            torch.save(state, path)
            restored, _, _ = Trainer.load(path)
            self.assertTrue(restored.ema_initialized)
            self.assertEqual(restored.ema_num_averaged_updates, 1)
            self.equal_states(trainer.ema.state_dict(), restored.ema.state_dict())
            rng = torch.get_rng_state().clone()
            for expectations in ({"expected_variant": "loop"}, {"expected_training_protocol_version": 2},
                                 {"expected_ema_start_update": 2}):
                with self.assertRaises(ValueError):
                    Trainer.load(path, **expectations)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_corrupt_ema_state_rejected(self):
        trainer, batches = self.create()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            trainer.save(path, batches)
            state = torch.load(path, weights_only=True)
            state["ema_initialized"] = True
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "EMA state"):
                Trainer.load(path)

    def test_ema_policy_does_not_change_raw_gradients_updates_or_rng(self):
        original, batches1 = self.create(config=replace(self.config, training_protocol_version=1, ema_start_update=0))
        delayed, batches2 = self.create()
        for _ in range(5):
            first = original.train_update(batches1)
            global_rng = torch.get_rng_state().clone()
            second = delayed.train_update(batches2)
            self.assertEqual(first, second)
            self.equal_states(original.model.state_dict(), delayed.model.state_dict())
            self.equal_states({n: p.grad for n, p in original.model.named_parameters()},
                              {n: p.grad for n, p in delayed.model.named_parameters()})
            self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
            self.assertTrue(torch.equal(original.generator.get_state(), delayed.generator.get_state()))
            self.assertTrue(torch.equal(batches1.generator.get_state(), batches2.generator.get_state()))

    def test_matched_initialization_and_independent_extra_core(self):
        base, _ = self.create()
        global_rng = torch.get_rng_state().clone()
        loop, _ = self.create("loop")
        untied, _ = self.create("untied")
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.equal_states(base.model.state_dict(), loop.model.state_dict())
        state = untied.model.state_dict()
        self.equal_states(base.model.state_dict(), {k: state[k] for k in base.model.state_dict()})
        first, extra = untied.model.cores
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first.parameters(), extra.parameters())))
        for a, b in zip(first.parameters(), extra.parameters()):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        for a, b in zip(base.model.parameters(), loop.model.parameters()):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        repeated, _ = self.create("untied")
        self.equal_states(untied.model.state_dict(), repeated.model.state_dict())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA execution environment required")
    def test_actual_amp_overflow_before_and_after_start(self):
        config = replace(self.config, amp=True, ema_start_update=1)
        trainer = Trainer(self.model, config, self.scaler, "cuda")
        batches = SyntheticBatches(self.data, 2)
        for ready in (False, True):
            before = {k: v.clone() for k, v in trainer.ema.state_dict().items()}
            updates, averaged = trainer.updates, trainer.ema_num_averaged_updates
            hook = trainer.model.output_head.projection.weight.register_hook(
                lambda grad: torch.full_like(grad, float("inf")))
            try:
                self.assertFalse(trainer.train_update(batches)["did_update"])
            finally:
                hook.remove()
            self.assertEqual(trainer.ema_initialized, ready)
            self.assertEqual((trainer.updates, trainer.ema_num_averaged_updates), (updates, averaged))
            self.equal_states(before, trainer.ema.state_dict())
            self.assertTrue(trainer.train_update(batches)["did_update"])


if __name__ == "__main__":
    unittest.main()
