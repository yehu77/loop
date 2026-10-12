import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from metrics import (energy_score, ensemble_crps, evaluate_scenarios,
                     interval_statistics, variogram_score)


class MetricTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(9)
        self.samples = torch.rand(3, 7, 4, 1, generator=generator, dtype=torch.float64)
        self.targets = torch.rand(3, 4, 1, generator=generator, dtype=torch.float64)

    def test_perfect_identical_trajectories_have_zero_scores(self):
        samples = self.targets[:, None].expand(-1, 5, -1, -1)
        for score in (ensemble_crps, energy_score, variogram_score):
            torch.testing.assert_close(score(samples, self.targets),
                                       torch.zeros_like(score(samples, self.targets)), atol=1e-14, rtol=0)
        interval = interval_statistics(samples, self.targets)
        self.assertTrue(interval["covered"].all())
        self.assertEqual(interval["width"].sum().item(), 0)

    def test_single_member_reduces_to_absolute_and_euclidean_error(self):
        samples = torch.tensor([[[[0.], [0.]]]], dtype=torch.float64)
        targets = torch.tensor([[[3.], [4.]]], dtype=torch.float64)
        torch.testing.assert_close(ensemble_crps(samples, targets), torch.tensor([[3., 4.]]).double())
        self.assertEqual(energy_score(samples, targets).item(), 5)

    def test_two_member_hand_calculation_and_one_hour_equivalence(self):
        samples = torch.tensor([[[[0.]], [[2.]]]], dtype=torch.float64)
        targets = torch.tensor([[[1.]]], dtype=torch.float64)
        self.assertEqual(ensemble_crps(samples, targets).item(), 0.5)
        self.assertEqual(energy_score(samples, targets).item(), 0.5)

    def test_crps_matches_direct_ordered_pair_formula(self):
        x, y = self.samples[..., 0], self.targets[..., 0]
        reference = (x - y[:, None]).abs().mean(1)
        reference -= 0.5 * (x[:, :, None] - x[:, None, :]).abs().mean((1, 2))
        torch.testing.assert_close(ensemble_crps(self.samples, self.targets), reference)

    def test_energy_chunking_matches_explicit_vector_distances(self):
        x, y = self.samples[..., 0], self.targets[..., 0]
        reference = (x - y[:, None]).square().sum(-1).sqrt().mean(1)
        reference -= 0.5 * (x[:, :, None] - x[:, None, :]).square().sum(-1).sqrt().mean((1, 2))
        for size in (1, 3, 100):
            torch.testing.assert_close(energy_score(self.samples, self.targets, size), reference)

    def test_variogram_matches_explicit_ordered_hour_pairs(self):
        x, y = self.samples[..., 0], self.targets[..., 0]
        for power in (0.5, 1., 2.):
            reference = torch.zeros(3, dtype=torch.float64)
            for i in range(4):
                for j in range(4):
                    observed = (y[:, i] - y[:, j]).abs().pow(power)
                    expected = (x[:, :, i] - x[:, :, j]).abs().pow(power).mean(1)
                    reference += (observed - expected).square()
            torch.testing.assert_close(variogram_score(self.samples, self.targets, power), reference)

    def test_dependence_changes_joint_scores_but_not_marginal_crps(self):
        correlated = torch.tensor([[[[0.], [0.]], [[1.], [1.]]]], dtype=torch.float64)
        anticorrelated = torch.tensor([[[[0.], [1.]], [[1.], [0.]]]], dtype=torch.float64)
        target = torch.zeros(1, 2, 1, dtype=torch.float64)
        torch.testing.assert_close(ensemble_crps(correlated, target), ensemble_crps(anticorrelated, target))
        self.assertAlmostEqual(energy_score(correlated, target).item(), math.sqrt(2) / 4)
        self.assertGreater(energy_score(anticorrelated, target).item(), energy_score(correlated, target).item())
        self.assertEqual(variogram_score(correlated, target).item(), 0)
        self.assertEqual(variogram_score(anticorrelated, target).item(), 2)

    def test_variogram_alone_cannot_detect_common_level_offset(self):
        samples = self.targets[:, None].repeat(1, 3, 1, 1) + 2
        torch.testing.assert_close(variogram_score(samples, self.targets), torch.zeros(3).double(),
                                   atol=1e-14, rtol=0)
        self.assertGreater(energy_score(samples, self.targets).min().item(), 0)

    def test_empirical_scores_invariant_to_member_order_and_replication(self):
        for score in (ensemble_crps, energy_score, variogram_score):
            reference = score(self.samples, self.targets)
            torch.testing.assert_close(score(self.samples.flip(1), self.targets), reference)
            torch.testing.assert_close(score(self.samples.repeat_interleave(2, dim=1), self.targets), reference)

    def test_linear_quantiles_inclusive_endpoints_and_pointwise_coverage(self):
        samples = torch.arange(4, dtype=torch.float64).view(1, 4, 1, 1).repeat(1, 1, 3, 1)
        target = torch.tensor([[[0.75], [2.25], [2.5]]], dtype=torch.float64)
        stats = interval_statistics(samples, target, level=0.5)
        torch.testing.assert_close(stats["lower"], torch.full((1, 3), 0.75).double())
        torch.testing.assert_close(stats["upper"], torch.full((1, 3), 2.25).double())
        self.assertEqual(stats["covered"].tolist(), [[True, True, False]])
        report = evaluate_scenarios(samples, target, levels=(0.5,))
        self.assertAlmostEqual(report["intervals"][0]["coverage"], 2 / 3)
        self.assertEqual(report["intervals"][0]["mean_width"], 1.5)

    def test_summary_preserves_hour_aggregation_and_is_json_serializable(self):
        report = evaluate_scenarios(self.samples, self.targets)
        self.assertAlmostEqual(report["crps"], sum(report["crps_by_hour"]) / 4)
        self.assertEqual((report["windows"], report["members"], report["horizon"]), (3, 7, 4))
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_invalid_shapes_missing_values_and_options_rejected(self):
        for score in (ensemble_crps, energy_score, variogram_score, interval_statistics):
            for samples in (self.samples[:, :0], self.samples[..., 0], self.samples.long(),
                            self.samples * float("nan")):
                with self.assertRaises(ValueError):
                    score(samples, self.targets)
            with self.assertRaises(ValueError):
                score(self.samples, self.targets[:1])
        for level in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                interval_statistics(self.samples, self.targets, level)
        with self.assertRaises(ValueError):
            energy_score(self.samples, self.targets, 0)
        with self.assertRaises(ValueError):
            variogram_score(self.samples, self.targets, -0.5)
        with self.assertRaises(ValueError):
            variogram_score(self.samples[:, :, :1], self.targets[:, :1])
        with self.assertRaises(ValueError):
            evaluate_scenarios(self.samples, self.targets, levels=(0.9, 0.9))

    def test_cli_generates_report_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(self.samples, root / "scenarios.pt")
            torch.save(self.targets, root / "targets.pt")
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "metrics.py"),
                       "--scenarios", str(root / "scenarios.pt"),
                       "--targets", str(root / "targets.pt"), "--output", str(root / "report.json")]
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            original = (root / "report.json").read_bytes()
            self.assertEqual(json.loads(original)["members"], 7)
            result = subprocess.run(command, capture_output=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(original, (root / "report.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
