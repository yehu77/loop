from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from data import RealBatches
from diffusion import LinearNoiseSchedule, sample_scenarios
from evaluate import align_artifacts
from history_data import prepare_history
from model import ConditionalLoopDiT, ModelConfig
import test_history_data as fixtures
from train import SyntheticBatches, Trainer, TrainConfig, WeatherScaler, make_synthetic_data
from train_real import load_real_checkpoint, run, validation_artifacts


def active_model(group, device="cpu"):
    torch.manual_seed(42)
    model = ConditionalLoopDiT(ModelConfig(variant="base", hidden_dim=16, ffn_dim=32,
                                           history_group=group)).to(device).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.output_head.projection.weight, std=.12)
        for block in model.cores[0].blocks:
            block.modulator[-1].bias[32:48] = .6
            block.modulator[-1].bias[80:96] = .4
    return model


class HistoryInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(14)
        self.x = torch.randn(3, 24, 1, generator=generator) * .2
        self.c = torch.randn(3, 24, 4, generator=generator)
        self.a = torch.randn(3, 24, 4, generator=generator)
        self.h = torch.rand(3, 24, 1, generator=generator)
        self.t = torch.tensor([200, 700, 950])
        self.noise = torch.randn(3, 3, 24, 1, generator=generator)

    def forward(self, model, history):
        return model(self.x, self.c, self.a, self.t, history_power=history)

    def sample(self, model, history, chunk=4, order=None):
        order = torch.arange(3) if order is None else order
        return sample_scenarios(model, LinearNoiseSchedule(10), self.c[order], self.a[order],
                                history_power=history[order], scenarios=3, sampling_steps=4,
                                chunk_size=chunk, initial_noise=self.noise[order])[0]

    def test_hidden_history_isolation_and_visible_positive_controls(self):
        for group in ("H0", "H1", "H24"):
            model = active_model(group)
            original = self.forward(model, self.h)
            self.assertGreater(original.abs().max().item(), 1e-4)
            changed = self.h.clone()
            changed[:, :23] = float("nan")
            if group == "H0":
                changed[:] = float("nan")
            if group != "H24":
                torch.testing.assert_close(original, self.forward(model, changed), atol=0, rtol=0)
                torch.testing.assert_close(self.sample(model, self.h), self.sample(model, changed), atol=0, rtol=0)
            changed = self.h.clone()
            changed[:, -1 if group == "H1" else 0] = 1 - changed[:, -1 if group == "H1" else 0]
            if group != "H0":
                self.assertGreater((self.forward(model, changed) - original).abs().max().item(), 1e-5)
                self.assertGreater((self.sample(model, changed) - self.sample(model, self.h)).abs().max().item(), 1e-6)

    def test_h0_equals_future_only_reference_at_new_positions(self):
        model = active_model("H0")
        reference = ConditionalLoopDiT(replace(model.config, history_group=None)).eval()
        reference.load_state_dict({k: v for k, v in model.state_dict().items()
                                   if not k.startswith("history_")})
        torch.testing.assert_close(self.forward(model, self.h),
                                   reference(self.x, self.c, self.a, self.t), atol=2e-6, rtol=2e-5)
        actual = self.sample(model, self.h)
        expected = sample_scenarios(reference, LinearNoiseSchedule(10), self.c, self.a,
                                    scenarios=3, sampling_steps=4, chunk_size=4,
                                    initial_noise=self.noise)[0]
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)

    def test_cache_mask_dtype_single_transform_and_each_block(self):
        model = active_model("H1")
        cached = model.encode_condition(self.c, self.a, self.h)
        self.assertEqual(cached["key_visibility"].dtype, torch.bool)
        self.assertEqual(cached["key_visibility"].sum(1).tolist(), [25, 25, 25])
        expected = model.history_projection(2 * self.h[:, -1:] - 1) + model.history_position_encoding[:, -1:]
        torch.testing.assert_close(cached["history"][:, -1:], expected, atol=0, rtol=0)
        calls = []
        hooks = [block.attention.register_forward_pre_hook(lambda mod, args: calls.append(args[1].clone()))
                 for block in model.cores[0].blocks]
        try:
            out = model.denoise_with_cached_condition(self.x, cached, self.t)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(torch.equal(mask, cached["key_visibility"]) for mask in calls))
        torch.testing.assert_close(out, self.forward(model, self.h), atol=0, rtol=0)
        self.assertEqual(out.shape, (3, 24, 1))

    def test_sampling_cache_row_order_and_chunk_alignment(self):
        model = active_model("H24")
        cached = model.encode_condition(self.c, self.a, self.h)
        saved = {k: v.clone() for k, v in cached.items()}
        model.denoise_with_cached_condition(self.x, cached, self.t)
        model.denoise_with_cached_condition(-self.x, cached, self.t.flip(0))
        reference = self.sample(model, self.h)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(self.sample(model, self.h, chunk=2, order=order),
                                   reference[order], atol=3e-6, rtol=3e-5)
        for k in cached:
            torch.testing.assert_close(cached[k], saved[k], atol=0, rtol=0)
        # Independent row-wise generation detects histories accidentally shared across days.
        for i in range(3):
            single = sample_scenarios(model, LinearNoiseSchedule(10), self.c[i:i+1], self.a[i:i+1],
                                      history_power=self.h[i:i+1], scenarios=3, sampling_steps=4,
                                      chunk_size=1, initial_noise=self.noise[i:i+1])[0]
            torch.testing.assert_close(single[0], reference[i], atol=3e-6, rtol=3e-5)

    def make_trainer(self, group, device="cpu"):
        cfg = ModelConfig(variant="base", hidden_dim=16, ffn_dim=32, history_group=group)
        data = make_synthetic_data(cfg, samples=6)
        data["history_power"] = torch.rand(6, 24, 1, generator=torch.Generator().manual_seed(51))
        trainer = Trainer(cfg, TrainConfig(training_protocol_version=2, ema_start_update=2,
                                           microbatch_size=2, accumulation_steps=2),
                          WeatherScaler.fit(data["nwp"]), device)
        return trainer, SyntheticBatches(data, 2), data

    def test_matching_initialization_rng_and_future_only_training(self):
        trainers = [self.make_trainer(group) for group in ("H0", "H1", "H24")]
        first = trainers[0][0]
        initial = {k: v.clone() for k, v in first.model.state_dict().items()}
        for trainer, batches, data in trainers:
            for key, value in initial.items():
                torch.testing.assert_close(value, trainer.model.state_dict()[key], atol=0, rtol=0)
            noise_shapes, output_shapes = [], []
            add = trainer.schedule.add_noise
            def observed(x0, t, noise):
                noise_shapes.append((x0.shape, noise.shape))
                return add(x0, t, noise)
            hook = trainer.model.register_forward_hook(lambda mod, args, out: output_shapes.append(out.shape))
            with patch.object(trainer.schedule, "add_noise", side_effect=observed):
                trainer.train_update(batches)
            hook.remove()
            self.assertEqual(noise_shapes, [(torch.Size([2, 24, 1]), torch.Size([2, 24, 1]))] * 2)
            self.assertEqual(output_shapes, [torch.Size([2, 24, 1])] * 2)
        for trainer, batches, _ in trainers[1:]:
            self.assertTrue(torch.equal(first.generator.get_state(), trainer.generator.get_state()))
            self.assertEqual(batches.cursor, trainers[0][1].cursor)
            self.assertTrue(torch.equal(batches.order, trainers[0][1].order))

    def check_resume(self, device):
        for group in ("H0", "H1", "H24"):
            for saved_update in (1, 2, 3):
                trainer, batches, data = self.make_trainer(group, device)
                for _ in range(saved_update):
                    trainer.train_update(batches)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "checkpoint.pt"
                    trainer.save(path, batches)
                    before = torch.get_rng_state().clone()
                    with self.assertRaisesRegex(ValueError, "history group"):
                        Trainer.load(path, device, expected_history_group="H1" if group != "H1" else "H0")
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    expected = trainer.train_update(batches)
                    restored, state, _ = Trainer.load(path, device, expected_history_group=group)
                    more = SyntheticBatches(data, 2)
                    more.load_state_dict(state)
                    self.assertEqual(expected, restored.train_update(more))
                    for name in ("model", "ema"):
                        for k, v in getattr(trainer, name).state_dict().items():
                            torch.testing.assert_close(v, getattr(restored, name).state_dict()[k], atol=0, rtol=0)

    def test_resume_before_at_after_ema_and_cross_group_rejection(self):
        self.check_resume("cpu")

    def test_real_entry_history_identity_validation_rng_and_resume(self):
        fixture = fixtures.HistoryDataTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        parent, protocol = fixture.source()
        package = prepare_history(parent, fixture.root, protocol)
        path = fixture.root / "history.pt"
        torch.save(package, path)
        args = SimpleNamespace(data=path, output=fixture.root / "run", resume=None, history_group="H1",
                               variant="base", attention_scope="full", training_protocol_version=2,
                               ema_start_update=2, supervision_mode="final-only", loss_weights=[0, 1],
                               device="cpu", amp=False, seed=42, microbatch_size=1, accumulation_steps=1,
                               validation_limit=None, scenarios=3, sampling_steps=3, chunk_size=2,
                               validate_every=1, updates=3)
        run(args)
        trainer, batches, metadata = load_real_checkpoint(args.output / "checkpoint_last.pt", package,
                                                         "cpu", expected_history_group="H1")
        before = trainer.generator.get_state().clone()
        pos = batches.cursor
        predicted, target, _ = validation_artifacts(trainer, package, metadata["run_config"])
        self.assertEqual(predicted["values"].shape, (2, 3, 24, 1))
        self.assertTrue(torch.equal(before, trainer.generator.get_state()))
        self.assertEqual(pos, batches.cursor)
        broken = deepcopy(predicted)
        broken["samples"][0]["history_valid_times"][0] = "1900-01-01T00:00:00"
        with self.assertRaisesRegex(ValueError, "History metadata"):
            align_artifacts(broken, target)
        with self.assertRaisesRegex(ValueError, "history group"):
            load_real_checkpoint(args.output / "checkpoint_last.pt", package, "cpu", expected_history_group="H0")
        original = args.output
        args.resume, args.output, args.updates = original / "checkpoint_last.pt", fixture.root / "resume", 1
        run(args)
        state = torch.load(args.output / "checkpoint_last.pt", weights_only=True)
        self.assertEqual(state["updates"], 4)
        self.assertEqual(state["model_config"]["history_group"], "H1")
        self.assertEqual(json.loads((args.output / "config.json").read_text())["evaluation"], metadata["run_config"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA execution environment required")
    def test_cuda_history_training_sampling_and_exact_resume(self):
        self.check_resume("cuda")
        for group in ("H0", "H1", "H24"):
            active = active_model(group, "cuda")
            x, c, a, t, h = (v.cuda() for v in (self.x, self.c, self.a, self.t, self.h))
            original = active(x, c, a, t, history_power=h)
            changed = h.clone()
            if group in ("H0", "H1"):
                changed[:, :24 if group == "H0" else 23] = float("nan")
                torch.testing.assert_close(original, active(x, c, a, t, history_power=changed), atol=0, rtol=0)
            if group == "H0":
                ref = ConditionalLoopDiT(replace(active.config, history_group=None)).cuda().eval()
                ref.load_state_dict({k: v for k, v in active.state_dict().items() if not k.startswith("history_")})
                torch.testing.assert_close(original, ref(x, c, a, t), atol=3e-6, rtol=3e-5)
            else:
                changed = h.clone()
                slot = -1 if group == "H1" else 0
                changed[:, slot] = 1 - changed[:, slot]
                self.assertGreater((original - active(x, c, a, t, history_power=changed)).abs().max().item(), 1e-5)
            trainer, batches, data = self.make_trainer(group, "cuda")
            for _ in range(4):
                trainer.train_update(batches)
            result, _ = sample_scenarios(trainer.ema, trainer.schedule,
                trainer.weather_scaler.transform(data["nwp"][:2].cuda()), data["calendar"][:2].cuda(),
                history_power=data["history_power"][:2].cuda(), scenarios=3, sampling_steps=4,
                initial_noise=torch.randn(2, 3, 24, 1, device="cuda"), chunk_size=2)
            self.assertEqual(result.shape, (2, 3, 24, 1))
            self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
