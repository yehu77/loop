from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from diffusion import LinearNoiseSchedule, sample_scenarios
from model import ModelConfig
from train import Trainer, TrainConfig, WeatherScaler, SyntheticBatches, make_synthetic_data


def make_trainer(variant, group, device="cpu"):
    config = ModelConfig(variant=variant, history_group=group, hidden_dim=16, ffn_dim=32)
    data = make_synthetic_data(config, samples=7)
    data["history_power"] = torch.rand(7, 24, 1, generator=torch.Generator().manual_seed(134))
    trainer = Trainer(config, TrainConfig(training_protocol_version=2, ema_start_update=2,
                                         microbatch_size=2, accumulation_steps=2),
                      WeatherScaler.fit(data["nwp"]), device)
    return trainer, SyntheticBatches(data, 2), data


def activate(model):
    with torch.no_grad():
        model.output_head.projection.weight.fill_(.03)
        # Unequal channels avoid zero LayerNorm-sum readout.
        model.output_head.projection.weight[0, ::2] = -.07
        d = model.config.hidden_dim
        for core in model.cores:
            for block in core.blocks:
                block.modulator[-1].bias[2*d:3*d] = .6
                block.modulator[-1].bias[5*d:6*d] = .4


class HistoryArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_matched_common_parameters_and_independent_extra_core(self):
        by_group = {}
        for group in ("H0", "H24"):
            models = {v: make_trainer(v, group)[0].model for v in ("base", "loop", "untied")}
            base = models["base"].state_dict()
            for variant in ("loop", "untied"):
                actual = models[variant].state_dict()
                for key, value in base.items():
                    torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
            core0, core1 = models["untied"].cores
            for p, q in zip(core0.parameters(), core1.parameters()):
                self.assertNotEqual(p.data_ptr(), q.data_ptr())
            self.assertFalse(torch.equal(core0.blocks[0].attention.qkv.weight, core1.blocks[0].attention.qkv.weight))
            by_group[group] = models
        for variant in ("base", "loop", "untied"):
            for key, value in by_group["H0"][variant].state_dict().items():
                torch.testing.assert_close(value, by_group["H24"][variant].state_dict()[key], atol=0, rtol=0)

    def check_structure_and_isolation(self, device):
        for variant in ("loop", "untied"):
            for group in ("H0", "H24"):
                trainer, _, data = make_trainer(variant, group, device)
                model = trainer.model.eval()
                activate(model)
                y, c, a, h = (data[k][:2].to(device) for k in ("y", "nwp", "calendar", "history_power"))
                t = torch.tensor([50, 900], device=device)
                cores, masks, heads = [], [], []
                hooks = []
                for core in model.cores:
                    hooks.append(core.register_forward_pre_hook(lambda mod, args: cores.append(id(mod))))
                    for block in core.blocks:
                        hooks.append(block.attention.register_forward_pre_hook(lambda mod, args: masks.append(args[1].clone())))
                hooks.append(model.output_head.register_forward_pre_hook(lambda mod, args: heads.append(args[0].shape)))
                out = model(y, c, a, t, history_power=h)
                for hook in hooks:
                    hook.remove()
                self.assertEqual(cores, [id(model.cores[0]), id(model.cores[0 if variant == "loop" else 1])])
                self.assertEqual(len(masks), 4)
                self.assertTrue(all(m.dtype == torch.bool and torch.equal(m, masks[0]) for m in masks))
                self.assertEqual(heads, [torch.Size([2, 24, 16])])
                self.assertEqual(out.shape, (2, 24, 1))
                self.assertGreater(out.abs().max().item(), 1e-4)
                out.square().mean().backward()
                for core in model.cores:
                    self.assertGreater(core.blocks[0].attention.qkv.weight.grad.abs().sum().item(), 0)
                changed = h.clone()
                if group == "H0":
                    changed[:] = float("nan")
                    torch.testing.assert_close(out, model(y, c, a, t, history_power=changed), atol=0, rtol=0)
                else:
                    changed[:, 0] = 1 - changed[:, 0]
                    self.assertGreater((out-model(y, c, a, t, history_power=changed)).abs().max().item(), 1e-5)
                noise = torch.randn(2, 3, 24, 1, generator=torch.Generator().manual_seed(12)).to(device)
                def sample(history):
                    return sample_scenarios(model, LinearNoiseSchedule(10).to(device), c, a,
                        history_power=history, scenarios=3, sampling_steps=4, chunk_size=2, initial_noise=noise)[0]
                original, other = sample(h), sample(changed)
                self.assertEqual(original.shape, (2, 3, 24, 1))
                if group == "H0":
                    torch.testing.assert_close(original, other, atol=0, rtol=0)
                else:
                    self.assertGreater((original-other).abs().max().item(), 1e-6)
                with self.assertRaisesRegex(ValueError, "final-only"):
                    model.denoise_with_cached_condition(y, model.encode_condition(c, a, h), t, return_stages=True)
                with self.assertRaisesRegex(ValueError, "final-only"):
                    model.forward_stages(y, c, a, t)
                with self.assertRaises(ValueError):
                    model(y, c, a, t, rounds=4, history_power=h)

    def test_two_core_execution_masks_and_active_condition_paths(self):
        self.check_structure_and_isolation("cpu")

    def check_resume(self, device):
        for variant in ("loop", "untied"):
            for group in ("H0", "H24"):
                for boundary in (1, 2, 3):
                    trainer, batches, data = make_trainer(variant, group, device)
                    for _ in range(boundary):
                        trainer.train_update(batches)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "checkpoint.pt"
                        trainer.save(path, batches, {"data_fingerprint": "common_history"})
                        for expected in ({"expected_variant": "untied" if variant == "loop" else "loop"},
                                         {"expected_history_group": "H24" if group == "H0" else "H0"}):
                            with self.assertRaisesRegex(ValueError, "mismatch"):
                                Trainer.load(path, device, expected_data_fingerprint="common_history", **expected)
                        with self.assertRaisesRegex(ValueError, "fingerprint"):
                            Trainer.load(path, device, expected_data_fingerprint="other")
                        expected = trainer.train_update(batches)
                        restored, state, _ = Trainer.load(path, device, expected_data_fingerprint="common_history",
                                                         expected_variant=variant, expected_history_group=group)
                        more = SyntheticBatches(data, 2)
                        more.load_state_dict(state)
                        self.assertEqual(restored.train_update(more), expected)
                        for name in ("model", "ema"):
                            for key, value in getattr(trainer, name).state_dict().items():
                                torch.testing.assert_close(value, getattr(restored, name).state_dict()[key], atol=0, rtol=0)
                        self.assertTrue(torch.equal(trainer.generator.get_state(), restored.generator.get_state()))

    def test_four_combinations_resume_and_wrong_identity_rejection(self):
        self.check_resume("cpu")

    def test_history_intermediate_supervision_and_variable_depth_stay_disabled(self):
        for variant in ("loop", "untied"):
            with self.assertRaises(ValueError):
                ModelConfig(variant=variant, history_group="H24", rounds=4)
            trainer, _, _ = make_trainer(variant, "H24")
            with self.assertRaisesRegex(ValueError, "final-only"):
                Trainer(trainer.model.config, replace(trainer.config, supervision_mode="intermediate", loss_weights=(.5,.5)), trainer.weather_scaler)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA execution environment required")
    def test_cuda_four_combinations_structure_isolation_sampling_and_resume(self):
        self.check_structure_and_isolation("cuda")
        self.check_resume("cuda")


if __name__ == "__main__":
    unittest.main()
