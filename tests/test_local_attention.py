"""R02 isolation tests open the zero-initialized head and residual gates."""
from argparse import Namespace
from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from data import prepare_dataset
from diffusion import LinearNoiseSchedule, sample_scenarios
from model import ConditionalLoopDiT, ModelConfig, SelfAttention
from test_data import fixture_protocol, write_sources
from train import Trainer, TrainConfig, WeatherScaler, SyntheticBatches, make_synthetic_data
from train_real import run


def active_pair(device="cpu"):
    torch.manual_seed(128)
    config = ModelConfig(variant="base", horizon=6, hidden_dim=16, ffn_dim=32)
    full = ConditionalLoopDiT(config).to(device)
    with torch.no_grad():
        full.output_head.projection.weight.normal_(std=0.1)
        for block in full.cores[0].blocks:
            block.modulator[-1].bias[32:48].fill_(0.7)
            block.modulator[-1].bias[80:96].fill_(0.4)
    local = ConditionalLoopDiT(replace(config, attention_scope="diagonal")).to(device)
    local.load_state_dict(full.state_dict())
    return full.eval(), local.eval()


class LocalAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def check_isolation(self, device):
        full, local = active_pair(device)
        inputs = [torch.randn(2, 6, d, device=device) for d in (1, 4, 4)]
        times = torch.tensor([200, 700], device=device)
        other = [0, 1, 2, 4, 5]
        for field in range(3):
            changed = [x.clone() for x in inputs]
            changed[field][:, 3] += 2.3
            with torch.no_grad():
                delta = local(*changed, times) - local(*inputs, times)
                positive = full(*changed, times) - full(*inputs, times)
            self.assertGreater(delta[:, 3].abs().max().item(), 1e-5)
            torch.testing.assert_close(delta[:, other], torch.zeros_like(delta[:, other]), atol=1e-7, rtol=0)
            self.assertGreater(positive[:, other].abs().max().item(), 1e-5,
                               "Full-attention positive control must detect cross-hour influence")
        # A complete DDIM chain, with identical initial noise and open gates/head.
        noise = torch.randn(2, 3, 6, 1, device=device)
        schedule = LinearNoiseSchedule(steps=20).to(device)
        for field in (1, 2):
            changed = [x.clone() for x in inputs]
            changed[field][:, 3] += 2.3
            kwargs = dict(scenarios=3, sampling_steps=5, chunk_size=4, initial_noise=noise)
            first, _ = sample_scenarios(local, schedule, inputs[1], inputs[2], **kwargs)
            second, _ = sample_scenarios(local, schedule, changed[1], changed[2], **kwargs)
            torch.testing.assert_close(first[:, :, other], second[:, :, other], atol=1e-7, rtol=0)
            self.assertGreater((first[:, :, 3] - second[:, :, 3]).abs().max().item(), 1e-6)
            a, _ = sample_scenarios(full, schedule, inputs[1], inputs[2], **kwargs)
            b, _ = sample_scenarios(full, schedule, changed[1], changed[2], **kwargs)
            self.assertGreater((a[:, :, other] - b[:, :, other]).abs().max().item(), 1e-6)

    def test_nonzero_forward_and_complete_sampling_isolation_cpu(self):
        self.check_isolation("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA execution environment required")
    def test_nonzero_forward_and_complete_sampling_isolation_cuda(self):
        self.check_isolation("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA execution environment required")
    def test_diagonal_gpu_updates_and_resume(self):
        config = ModelConfig(variant="base", horizon=6, hidden_dim=16, ffn_dim=32,
                             attention_scope="diagonal")
        data = make_synthetic_data(config, samples=7)
        trainer = Trainer(config, TrainConfig(microbatch_size=2, accumulation_steps=2,
                          training_protocol_version=2, ema_start_update=2),
                          WeatherScaler.fit(data["nwp"]), "cuda")
        batches = SyntheticBatches(data, 2)
        for _ in range(3):
            self.assertTrue(trainer.train_update(batches)["did_update"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save(path, batches)
            expected = trainer.train_update(batches)
            restored, state, _ = Trainer.load(path, "cuda", expected_attention_scope="diagonal")
            other = SyntheticBatches(data, 2)
            other.load_state_dict(state)
            self.assertEqual(expected, restored.train_update(other))
            for key, value in trainer.ema.state_dict().items():
                self.assertTrue(torch.equal(value, restored.ema.state_dict()[key]))

    def test_full_call_unchanged_and_diagonal_single_key(self):
        torch.manual_seed(51)
        x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
        full = SelfAttention(16, 4).double()
        q, k, v = full.qkv(x).reshape(2, 6, 3, 4, 4).permute(2, 0, 3, 1, 4).unbind(0)
        old = full.projection(F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                              is_causal=False).transpose(1, 2).reshape(2, 6, 16))
        torch.testing.assert_close(full(x), old, atol=0, rtol=0)
        local = SelfAttention(16, 4, "diagonal").double()
        local.load_state_dict(full.state_dict())
        expected = local.projection(local.qkv(x).chunk(3, -1)[2])
        torch.testing.assert_close(local(x), expected, atol=1e-7, rtol=1e-6)
        local(x).square().sum().backward()
        self.assertLess(local.qkv.weight.grad[:32].abs().max().item(), 1e-12)
        self.assertGreater(local.qkv.weight.grad[32:].abs().max().item(), 0)

    def test_matched_initialization_and_rng_for_all_planned_seeds(self):
        for seed in (42, 43, 44):
            config = ModelConfig(variant="base")
            train = TrainConfig(seed=seed, training_protocol_version=2, ema_start_update=1000)
            scaler = WeatherScaler(torch.zeros(4), torch.ones(4))
            full = Trainer(config, train, scaler)
            rng = torch.get_rng_state().clone()
            local = Trainer(replace(config, attention_scope="diagonal"), train, scaler)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(torch.equal(full.generator.get_state(), local.generator.get_state()))
            self.assertEqual(sum(p.numel() for p in local.model.parameters()), 208065)
            for key, value in full.model.state_dict().items():
                self.assertTrue(torch.equal(value, local.model.state_dict()[key]), key)

    def test_scope_validation_and_checkpoint_resume(self):
        with self.assertRaisesRegex(ValueError, "attention_scope"):
            ModelConfig(attention_scope="causal")
        for scope in ("full", "diagonal"):
            config = ModelConfig(variant="base", horizon=6, hidden_dim=16, ffn_dim=32,
                                 attention_scope=scope)
            data = make_synthetic_data(config, samples=7)
            trainer = Trainer(config, TrainConfig(microbatch_size=2, accumulation_steps=1,
                              training_protocol_version=2, ema_start_update=2), WeatherScaler.fit(data["nwp"]))
            batches = SyntheticBatches(data, 2)
            trainer.train_update(batches)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                trainer.save(path, batches)
                if scope == "full":
                    state = torch.load(path, weights_only=True)
                    del state["model_config"]["attention_scope"]  # Real legacy field omission.
                    torch.save(state, path)
                rng = torch.get_rng_state().clone()
                with self.assertRaisesRegex(ValueError, "attention scope mismatch"):
                    Trainer.load(path, expected_attention_scope="full" if scope == "diagonal" else "diagonal")
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                expected = [trainer.train_update(batches) for _ in range(3)]
                restored, saved, _ = Trainer.load(path, expected_attention_scope=scope)
                other = SyntheticBatches(data, 2)
                other.load_state_dict(saved)
                self.assertEqual(expected, [restored.train_update(other) for _ in range(3)])
                for name in ("model", "ema"):
                    for key, value in getattr(trainer, name).state_dict().items():
                        self.assertTrue(torch.equal(value, getattr(restored, name).state_dict()[key]))

    def test_real_entry_legacy_best_and_local_resume_identity(self):
        for scope in ("full", "diagonal"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = fixture_protocol()
                write_sources(root, protocol)
                torch.save(prepare_dataset(root, protocol), root / "data.pt")
                args = Namespace(data=root / "data.pt", output=root / "first", resume=None,
                                 variant="base", attention_scope=scope, training_protocol_version=2,
                                 ema_start_update=1, device="cpu", updates=2, amp=False, seed=42,
                                 microbatch_size=1, accumulation_steps=1, validate_every=1,
                                 validation_limit=None, scenarios=2, sampling_steps=2, chunk_size=4)
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                if scope == "full":
                    for name in ("checkpoint_last.pt", "checkpoint_best.pt"):
                        path = args.output / name
                        state = torch.load(path, weights_only=True)
                        del state["model_config"]["attention_scope"]
                        torch.save(state, path)
                args.resume = args.output / "checkpoint_last.pt"
                args.output = root / "resumed"
                args.updates = 1
                args.attention_scope = "diagonal" if scope == "full" else "full"
                with self.assertRaisesRegex(ValueError, "attention scope mismatch"):
                    run(args)
                self.assertFalse(args.output.exists())
                args.attention_scope = None  # Omitted resume flag inherits, never changes scope.
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                saved = json.loads((args.output / "config.json").read_text())
                self.assertEqual(saved["model"]["attention_scope"], scope)
