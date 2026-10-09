from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from model import ModelConfig
from train import SyntheticBatches, TrainConfig, Trainer, WeatherScaler, make_synthetic_data


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model_config = ModelConfig(horizon=6, hidden_dim=16, ffn_dim=32)
        self.config = TrainConfig(microbatch_size=2, accumulation_steps=2)
        self.data = make_synthetic_data(self.model_config, samples=7)
        self.weather = WeatherScaler.fit(self.data["nwp"])

    def make_trainer(self, device="cpu", config=None):
        trainer = Trainer(self.model_config, config or self.config, self.weather, device)
        batches = SyntheticBatches(self.data, trainer.config.microbatch_size)
        return trainer, batches

    def test_weather_scaler_uses_only_fit_data_and_handles_constant_feature(self):
        data = torch.tensor([[[1., 2.], [3., 2.]], [[5., 2.], [7., 2.]]])
        scaler = WeatherScaler.fit(data)
        transformed = scaler.transform(data)
        torch.testing.assert_close(transformed.mean((0, 1)), torch.zeros(2))
        self.assertTrue(torch.isfinite(transformed).all())
        before = scaler.mean.clone()
        scaler.transform(torch.full_like(data, 1000))
        torch.testing.assert_close(before, scaler.mean)

    def test_all_variants_acquire_upstream_gradients_after_several_updates(self):
        for variant in ("base", "loop", "untied"):
            trainer = Trainer(replace(self.model_config, variant=variant), self.config, self.weather)
            batches = SyntheticBatches(self.data, 2)
            for _ in range(5):
                result = trainer.train_update(batches)
                self.assertTrue(result["did_update"])
            for name, parameter in trainer.model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            self.assertEqual(trainer.updates, 5)

    def test_checkpoint_replays_output_and_next_training_update(self):
        trainer, batches = self.make_trainer()
        for _ in range(2):
            trainer.train_update(batches)
        fixed = (self.data["y"][:2], self.weather.transform(self.data["nwp"][:2]),
                 self.data["calendar"][:2], torch.tensor([5, 600]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save(path, batches, {"data_kind": "synthetic"})
            expected_output = trainer.model(*fixed).detach()
            expected_step = trainer.train_update(batches)
            restored, state, metadata = Trainer.load(path)
            restored_batches = SyntheticBatches(self.data, 2)
            restored_batches.load_state_dict(state)
            torch.testing.assert_close(restored.model(*fixed), expected_output, atol=0, rtol=0)
            actual_step = restored.train_update(restored_batches)
            self.assertEqual(actual_step, expected_step)
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key], atol=0, rtol=0)
            for key, value in trainer.ema.state_dict().items():
                torch.testing.assert_close(value, restored.ema.state_dict()[key], atol=0, rtol=0)
            self.assertEqual(metadata["data_kind"], "synthetic")

    def test_nonfinite_and_invalid_power_windows_rejected(self):
        for value in (float("nan"), -1.0, 1.1):
            trainer, _ = self.make_trainer()
            data = {name: tensor.clone() for name, tensor in self.data.items()}
            data["y"][:] = value
            with self.assertRaises(ValueError):
                trainer.train_update(SyntheticBatches(data, 2))

    def test_biases_excluded_from_weight_decay(self):
        trainer, _ = self.make_trainer()
        for group in trainer.optimizer.param_groups:
            for parameter in group["params"]:
                self.assertEqual(group["weight_decay"], 0 if parameter.ndim < 2 else 1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA execution environment required")
    def test_amp_overflow_skips_optimizer_and_ema(self):
        trainer, batches = self.make_trainer("cuda", replace(self.config, amp=True))
        before_model = {key: value.clone() for key, value in trainer.model.state_dict().items()}
        before_ema = {key: value.clone() for key, value in trainer.ema.state_dict().items()}
        hook = trainer.model.output_head.projection.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf"))
        )
        try:
            result = trainer.train_update(batches)
        finally:
            hook.remove()
        self.assertFalse(result["did_update"])
        self.assertEqual(trainer.skipped, 1)
        for key, value in before_model.items():
            torch.testing.assert_close(value, trainer.model.state_dict()[key], atol=0, rtol=0)
            torch.testing.assert_close(before_ema[key], trainer.ema.state_dict()[key], atol=0, rtol=0)
        self.assertTrue(trainer.train_update(batches)["did_update"])


if __name__ == "__main__":
    unittest.main()
