from dataclasses import replace
import unittest

import torch

from model import ConditionalLoopDiT, ModelConfig, SelfAttention


def activate_zero_layers(model):
    # Test meaningful transformations rather than the all-zero initial output.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "modulator" in name or "output_head.projection" in name:
                parameter.normal_(std=0.05)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = ModelConfig(horizon=6, hidden_dim=16, heads=4, ffn_dim=32)
        self.inputs = (torch.randn(2, 6, 1), torch.randn(2, 6, 4),
                       torch.randn(2, 6, 4), torch.tensor([1, 900]))

    def test_reference_parameter_counts_and_unique_blocks(self):
        for variant, expected in (("base", 208065), ("loop", 208065), ("untied", 365697)):
            model = ConditionalLoopDiT(ModelConfig(variant=variant))
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertIsNot(model.cores[0].blocks[0], model.cores[0].blocks[1])
            if variant == "untied":
                self.assertIsNot(model.cores[0].adapter.weight, model.cores[1].adapter.weight)

    def test_initialization_and_shape(self):
        for variant in ("base", "loop", "untied"):
            model = ConditionalLoopDiT(replace(self.config, variant=variant))
            output = model(*self.inputs)
            self.assertEqual(output.shape, (2, 6, 1))
            self.assertEqual(output.count_nonzero().item(), 0)
            for core in model.cores:
                h, evidence = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
                torch.testing.assert_close(core.adapter(torch.cat((h, evidence), -1)), h)
                torch.testing.assert_close(core(h, evidence, torch.randn(2, 16)), h)

    def test_one_round_matches_base_with_nonzero_output(self):
        loop = ConditionalLoopDiT(self.config)
        activate_zero_layers(loop)
        base = ConditionalLoopDiT(replace(self.config, variant="base"))
        base.load_state_dict(loop.state_dict())
        self.assertGreater(loop(*self.inputs, rounds=1).abs().sum().item(), 0)
        torch.testing.assert_close(loop(*self.inputs, rounds=1), base(*self.inputs))

    def test_shared_core_called_twice_and_first_round_receives_gradient(self):
        model = ConditionalLoopDiT(self.config)
        activate_zero_layers(model)
        states, evidence_ids = [], []

        def capture(module, inputs, output):
            output.retain_grad()
            states.append(output)
            evidence_ids.append(id(inputs[1]))

        handle = model.cores[0].register_forward_hook(capture)
        try:
            model(*self.inputs).square().mean().backward()
        finally:
            handle.remove()
        self.assertEqual(len(states), 2)
        self.assertEqual(evidence_ids[0], evidence_ids[1])
        self.assertIsNotNone(states[0].grad)
        self.assertGreater(states[0].grad.abs().sum().item(), 0)

    def test_shared_gradient_equals_sum_of_independent_core_gradients(self):
        shared = ConditionalLoopDiT(self.config)
        activate_zero_layers(shared)
        untied = ConditionalLoopDiT(replace(self.config, variant="untied"))
        source = shared.state_dict()
        untied.load_state_dict({name: source[name.replace("cores.1.", "cores.0.")]
                                for name in untied.state_dict()})
        shared_output, untied_output = shared(*self.inputs), untied(*self.inputs)
        torch.testing.assert_close(shared_output, untied_output)
        shared_output.square().sum().backward()
        untied_output.square().sum().backward()
        other = dict(untied.named_parameters())
        for name, parameter in shared.named_parameters():
            expected = other[name].grad
            if name.startswith("cores.0."):
                expected = expected + other[name.replace("cores.0.", "cores.1.")].grad
            torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-5)

    def test_attention_uses_future_tokens(self):
        attention = SelfAttention(16, 4)
        x = torch.randn(2, 6, 16)
        perturbed = x.clone()
        perturbed[:, -1] += 5
        self.assertGreater((attention(x)[:, 0] - attention(perturbed)[:, 0]).abs().max().item(), 1e-4)

    def test_cached_condition_matches_forward(self):
        model = ConditionalLoopDiT(self.config)
        activate_zero_layers(model)
        x, c, calendar, t = self.inputs
        condition = model.encode_condition(c, calendar)
        torch.testing.assert_close(model(x, c, calendar, t),
                                   model.denoise_with_cached_condition(x, condition, t))

    def test_bad_shapes_and_rounds_rejected(self):
        model = ConditionalLoopDiT(self.config)
        with self.assertRaises(ValueError):
            model(*self.inputs, rounds=0)
        with self.assertRaises(ValueError):
            model.encode_condition(torch.randn(2, 5, 4), torch.randn(2, 5, 4))


if __name__ == "__main__":
    unittest.main()
