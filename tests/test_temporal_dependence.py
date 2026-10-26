from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data import file_hash, sample_id
from diagnostics.temporal_dependence import (MARGINALS, METRICS, aligned_validation,
    compare_scores, make_permutations, permute_members, run, score_panel, transforms)


class TemporalDependenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(19)
        self.x = torch.rand(2, 7, 24, 1, generator=generator)
        self.y = torch.rand(2, 24, 1, generator=generator)
        self.x[0, 0, 0] = self.y[0, 0] = 0
        self.x[0, 1, 0] = 1

    def test_hourly_sets_exact_and_days_hours_not_mixed(self):
        # Each day/hour occupies a disjoint numeric range.
        x = (10000 * torch.arange(2)[:, None, None, None]
             + 100 * torch.arange(24)[None, None, :, None]
             + torch.arange(7)[None, :, None, None]).float()
        original = x.clone()
        indices = make_permutations(2, 7, 24, 1101)
        shuffled = permute_members(x, indices)
        self.assertTrue(torch.equal(x.sort(1).values, shuffled.sort(1).values))
        self.assertTrue(torch.equal(x, original))
        self.assertFalse(torch.equal(indices[:, :1].expand_as(indices), indices))
        self.assertFalse(torch.equal(x, shuffled))

    def test_permutations_reproducible_and_global_rng_untouched(self):
        state = torch.get_rng_state().clone()
        a = make_permutations(2, 7, 24, 1101)
        b = make_permutations(2, 7, 24, 1101)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, make_permutations(2, 7, 24, 1102)))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_marginal_scores_invariant_for_all_five_seeds(self):
        original = score_panel(self.x, self.y)
        for seed in range(1101, 1106):
            x = permute_members(self.x, make_permutations(2, 7, 24, seed))
            compare_scores(original, score_panel(x, self.y), MARGINALS, 1e-10, 1e-10)

    def test_common_permutation_preserves_all_scores(self):
        original = score_panel(self.x, self.y)
        for seed in range(1101, 1106):
            p = make_permutations(2, 7, 24, seed, common=True)
            self.assertTrue(torch.equal(p, p[:, :1].expand_as(p)))
            compare_scores(original, score_panel(permute_members(self.x, p), self.y),
                           METRICS, 1e-10, 1e-10)

    def test_synthetic_dependence_changes_joint_scores_without_marginal_change(self):
        x = torch.tensor([-1., -1., 1., 1.])[None, :, None, None].expand(1, 4, 24, 1)
        y = torch.zeros(1, 24, 1)
        a = score_panel(x, y)
        b = score_panel(permute_members(x, make_permutations(1, 4, 24, 1101)), y)
        compare_scores(a, b, MARGINALS, 1e-10, 1e-10)
        self.assertEqual(a['variogram_score'].item(), 0)
        self.assertGreater(b['variogram_score'].item(), 0)
        self.assertGreater(b['increment_1h_crps'].mean().item(), a['increment_1h_crps'].mean().item())
        self.assertFalse(torch.allclose(a['energy_score'], b['energy_score']))
        # Breaking dependence can improve a different diagnostic; no positive-only filter.
        self.assertLess(b['daily_mean_crps'].item(), a['daily_mean_crps'].item())

    def test_signed_increments_and_normalized_daily_mean(self):
        x = torch.linspace(1, 0, 24).reshape(1, 1, 24, 1)
        y = x[:, 0]
        out = transforms(x, y)
        for lag in (1, 3):
            a, b = out[f'increment_{lag}h_crps']
            self.assertEqual(a.shape, (1, 1, 24 - lag, 1))
            self.assertTrue((a < 0).all())
            torch.testing.assert_close(a, x.double()[:, :, lag:] - x.double()[:, :, :-lag])
            self.assertEqual(b.shape, (1, 24 - lag, 1))
        a, b = out['daily_mean_crps']
        self.assertEqual(a.shape, (1, 1, 1, 1))
        self.assertAlmostEqual(a.item(), 0.5, places=7)
        self.assertAlmostEqual(b.item(), 0.5, places=7)

    def test_invalid_member_indices_rejected(self):
        p = make_permutations(2, 7, 24, 1101)
        for bad in (p.float(), p[:, :-1], torch.zeros_like(p), p + 1):
            with self.assertRaises(ValueError):
                permute_members(self.x, bad)

    def artifacts(self):
        metadata = []
        for day in range(2):
            issue = datetime(2013, 7, 1) + timedelta(days=day)
            metadata.append({'sample_id': sample_id(3, issue), 'zone_id': 3,
                'issue_time': issue.isoformat(), 'valid_times': [
                    (issue + timedelta(hours=h)).isoformat() for h in range(1, 25)],
                'split': 'validation'})
        common = {'schema_version': 1, 'data_fingerprint': 'fixture_data',
                  'protocol_fingerprint': 'fixture_protocol', 'power_unit': 'capacity_fraction',
                  'clock': 'source_naive_unspecified_timezone', 'samples': metadata}
        prediction, target = deepcopy(common), deepcopy(common)
        prediction['values'], target['values'] = self.x.clone(), self.y.clone()
        return prediction, target

    def test_panel_alignment_before_applying_shared_permutations(self):
        pred, target = self.artifacts()
        ids = [r['sample_id'] for r in target['samples']]
        pred['values'] = pred['values'].flip(0)
        pred['samples'].reverse()
        target['values'] = target['values'].flip(0)
        target['samples'].reverse()
        x, y, meta = aligned_validation(pred, target, ids, 'fixture_data')
        self.assertTrue(torch.equal(x, self.x))
        self.assertTrue(torch.equal(y, self.y))
        self.assertEqual([r['sample_id'] for r in meta], ids)

    def test_test_split_or_wrong_identity_rejected(self):
        pred, target = self.artifacts()
        ids = [r['sample_id'] for r in target['samples']]
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            aligned_validation(pred, target, ids, 'different_data')
        pred['samples'][0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'only accepts validation'):
            aligned_validation(pred, target, ids, 'fixture_data')

    def fixture(self, root):
        directory = root / 'source'
        directory.mkdir()
        pred, target = self.artifacts()
        ids = [r['sample_id'] for r in target['samples']]
        torch.save(pred, directory / 'scenarios.pt')
        torch.save(target, directory / 'targets.pt')
        (directory / 'checkpoint_last.pt').write_bytes(b'opaque checkpoint: never deserialized')
        scores = score_panel(self.x, self.y)
        metrics = {k: scores[k].mean().item() for k in ('crps', 'energy_score', 'variogram_score')}
        metrics['diagnostics'] = {'successful_updates': 3000, 'weights': 'ema', 'validation_seed': 2026}
        (directory / 'metrics.json').write_text(json.dumps(metrics))
        (directory / 'config.json').write_text(json.dumps({'training': {'seed': 42,
            'training_protocol_version': 2, 'ema_start_update': 1000}, 'model': {'variant': 'base'},
            'validation_sample_ids': ids}))
        config = {'schema_version': 1, 'stage': 'R01', 'endpoint_update': 3000, 'weights': 'ema',
            'shape': [2, 7, 24, 1], 'training_seeds': [42], 'variants': ['base'],
            'permutation_seeds': list(range(1101, 1106)), 'lags': [1, 3],
            'interval_levels': [0.8, 0.9], 'variogram_power': 0.5, 'atol': 1e-10, 'rtol': 1e-10,
            'data_fingerprint': 'fixture_data', 'sample_ids': ids, 'protected_sources': {},
            'runs': [{'training_seed': 42, 'variant': 'base', 'run_directory': str(directory),
                      'source_directories': [str(directory)], 'protected_files': {
                          str(p): file_hash(p) for p in directory.iterdir()}}]}
        return config

    def test_readonly_end_to_end_keeps_hashes_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.fixture(root)
            report = run(config, root / 'report')
            self.assertEqual(report['source_sha256_before'], report['source_sha256_after'])
            self.assertEqual(len(report['controls']), 5)
            self.assertEqual([r['sample_id'] for r in report['sample_metadata']], config['sample_ids'])
            self.assertTrue((root / 'report/per_day.csv').exists())
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                run(config, root / 'report')

    def test_changed_source_or_overlapping_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                run(config, root / 'source/nested')
            (root / 'source/checkpoint_last.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                run(config, root / 'report')
            self.assertFalse((root / 'report').exists())

    def test_source_change_during_diagnostic_aborts_without_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.fixture(root)
            def change_source(x, y):
                (root / 'source/checkpoint_last.pt').write_bytes(b'changed during scoring')
                return score_panel(x, y)
            with patch('diagnostics.temporal_dependence.score_panel', side_effect=change_source):
                with self.assertRaisesRegex(ValueError, 'changed during diagnostic'):
                    run(config, root / 'report')
            self.assertFalse((root / 'report').exists())

    def test_wrong_endpoint_rejected_even_when_file_hash_matches(self):
        for key, value in (('successful_updates', 2999), ('weights', 'raw')):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = self.fixture(root)
                path = root / 'source/metrics.json'
                metrics = json.loads(path.read_text())
                metrics['diagnostics'][key] = value
                path.write_text(json.dumps(metrics))
                config['runs'][0]['protected_files'][str(path)] = file_hash(path)
                with self.assertRaisesRegex(ValueError, 'fixed v2 EMA endpoint'):
                    run(config, root / 'report')
                self.assertFalse((root / 'report').exists())


if __name__ == '__main__':
    unittest.main()
