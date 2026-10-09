import unittest

import torch

from diffusion import LinearNoiseSchedule, ddim_step, sample_scenarios
from model import ConditionalLoopDiT, ModelConfig


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.schedule = LinearNoiseSchedule()

    def test_forward_formula_and_clean_endpoint(self):
        self.assertEqual(self.schedule.a[0].item(), 1)
        self.assertEqual(self.schedule.sigma[0].item(), 0)
        x0, noise = torch.rand(2, 6, 1) * 2 - 1, torch.randn(2, 6, 1)
        t = torch.tensor([1, 1000])
        noisy = self.schedule.add_noise(x0, t, noise)
        for i in range(2):
            expected = self.schedule.a[t[i]] * x0[i] + self.schedule.sigma[t[i]] * noise[i]
            torch.testing.assert_close(noisy[i], expected)

    def test_perfect_noise_prediction_recovers_clean_target(self):
        x0, noise = torch.rand(2, 6, 1) - 0.5, torch.randn(2, 6, 1)
        noisy = self.schedule.add_noise(x0, torch.tensor([900, 900]), noise)
        recovered, raw = ddim_step(noisy, noise, self.schedule, 900, 0)
        torch.testing.assert_close(recovered, x0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(raw, x0, atol=1e-5, rtol=1e-5)

    def test_clipping_recomputes_direction_without_clipping_noisy_state(self):
        x = torch.full((1, 6, 1), 3.0)
        epsilon = torch.zeros_like(x)
        result, raw = ddim_step(x, epsilon, self.schedule, 1000, 980)
        self.assertTrue((raw > 1).all())
        expected = self.schedule.a[980] + self.schedule.sigma[980] * (
            x - self.schedule.a[1000]
        ) / self.schedule.sigma[1000]
        torch.testing.assert_close(result, expected)
        self.assertTrue((result > 1).all())
        terminal, _ = ddim_step(x, epsilon, self.schedule, 1000, 0)
        torch.testing.assert_close(terminal, torch.ones_like(x))

    def test_grid_has_exact_number_of_calls_and_reaches_zero(self):
        pairs = self.schedule.sampling_grid(50)
        self.assertEqual(len(pairs), 50)
        self.assertEqual(pairs[0][0], 1000)
        self.assertEqual(pairs[-1][1], 0)
        self.assertTrue(all(t > s for t, s in pairs))
        with self.assertRaises(ValueError):
            self.schedule.sampling_grid(1001)
        with self.assertRaises(ValueError):
            self.schedule.add_noise(torch.zeros(1, 6, 1), torch.tensor([0]), torch.zeros(1, 6, 1))

    def test_sampling_reproducible_chunked_and_condition_encoded_once(self):
        model = ConditionalLoopDiT(ModelConfig(horizon=6, hidden_dim=16, ffn_dim=32))
        c, calendar = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
        initial = torch.randn(2, 5, 6, 1)
        calls = []
        handle = model.condition_mlp.register_forward_hook(lambda *args: calls.append(1))
        try:
            first, diagnostics = sample_scenarios(model, self.schedule, c, calendar, 5, 4, 3,
                                                  initial_noise=initial)
        finally:
            handle.remove()
        second, _ = sample_scenarios(model, self.schedule, c, calendar, 5, 4, 10,
                                     initial_noise=initial)
        torch.testing.assert_close(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first.shape, (2, 5, 6, 1))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        self.assertGreater(first.std(dim=1).mean().item(), 0)
        self.assertEqual(len(diagnostics["preclip_out_of_bounds_rate"]), 4)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
