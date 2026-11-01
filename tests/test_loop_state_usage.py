"""R03 controls use active heads/gates, not zero-initialization false passes."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from diagnostics.loop_state_usage import (aggregate, branches, check_checkpoint,
                                          measure, noise_bank, run, validate_config)
from diagnostics.temporal_dependence import read
from model import ConditionalLoopDiT, ModelConfig


class LoopStateUsageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(53)
        self.x = torch.randn(3, 6, 1)
        self.nwp = torch.randn(3, 6, 4)
        self.calendar = torch.randn(3, 6, 4)
        self.t = torch.tensor([50, 400, 950])

    def model(self, variant):
        model = ConditionalLoopDiT(ModelConfig(variant=variant, horizon=6, hidden_dim=16, ffn_dim=32)).eval()
        with torch.no_grad():
            model.output_head.projection.weight.normal_(std=.2)
            for core in model.cores:
                for block in core.blocks:
                    block.modulator[-1].bias[32:48].fill_(.6)
                    block.modulator[-1].bias[80:96].fill_(.4)
        return model

    def call(self, model):
        return branches(model, self.x, self.nwp, self.calendar, self.t)

    def test_normal_matches_official_for_all_variants_with_active_gates(self):
        for variant in ('base', 'loop', 'untied'):
            model = self.model(variant)
            result = self.call(model)[0]
            torch.testing.assert_close(result[1.], model(self.x, self.nwp, self.calendar, self.t), atol=0, rtol=0)
            self.assertEqual(set(result), {1.} if variant == 'base' else {0., .5, 1.})
            self.assertGreater(result[1.].abs().max().item(), .01)

    def test_loop_reset_equals_single_readout_and_detects_state_influence(self):
        model = self.model('loop')
        p, h0, h1, inputs, states = self.call(model)
        one = model(self.x, self.nwp, self.calendar, self.t, rounds=1)
        torch.testing.assert_close(p[0.], one, atol=0, rtol=0)
        self.assertGreater((p[0.] - p[1.]).abs().max().item(), .001)
        # Helpful normal state; no hard-coded preference for the sign of delta MSE.
        good = measure(p, p[1.], h0, h1, inputs, states)
        bad = measure(p, p[0.], h0, h1, inputs, states)
        self.assertTrue((good[0.]['delta_mse'] > 0).all())
        self.assertTrue((bad[0.]['delta_mse'] < 0).all())
        torch.testing.assert_close(inputs[.5], h0 + .5 * (h1 - h0), atol=0, rtol=0)

    def test_nonzero_hidden_change_can_have_no_output_effect(self):
        model = self.model('loop')
        with torch.no_grad():
            core = model.cores[0]
            core.adapter.weight.zero_()
            core.adapter.weight[:, 16:].copy_(2 * torch.eye(16))
        p, h0, h1, inputs, states = self.call(model)
        measured = measure(p, torch.randn_like(self.x), h0, h1, inputs, states)
        self.assertGreater(measured[0.]['first_update_rms'].min().item(), .01)
        self.assertGreater(measured[0.]['input_state_change_rms'].min().item(), .01)
        self.assertGreater(p[0.].abs().max().item(), .01)
        for lam in (0., .5):
            self.assertEqual(measured[lam]['output_change_rms'].max().item(), 0.)
            self.assertEqual(measured[lam]['delta_mse'].abs().max().item(), 0.)

    def test_untied_uses_second_core_for_every_intervention(self):
        model = self.model('untied')
        with torch.no_grad():
            model.cores[1].adapter.bias.copy_(torch.linspace(-1, 1, 16))
        with patch.object(model.cores[0], 'forward', wraps=model.cores[0].forward) as first, \
             patch.object(model.cores[1], 'forward', wraps=model.cores[1].forward) as second:
            p, h0, h1, inputs, states = self.call(model)
            self.assertEqual(first.call_count, 1)
            self.assertEqual(second.call_count, 3)
        from model import sinusoidal_embedding
        tau = model.time_mlp(sinusoidal_embedding(self.t, 16))
        for lam in (0., .5, 1.):
            expected = model.output_head(model.cores[1](inputs[lam], h0, tau), tau)
            torch.testing.assert_close(p[lam], expected, atol=0, rtol=0)
        self.assertGreater((p[0.] - model.output_head(h1, tau)).abs().max().item(), .01)

    def test_forward_readonly_and_batching(self):
        model = self.model('loop')
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        full = self.call(model)[0]
        pieces = [branches(model, self.x[i:i+1], self.nwp[i:i+1], self.calendar[i:i+1], self.t[i:i+1])[0]
                  for i in range(3)]
        for lam in full:
            torch.testing.assert_close(full[lam], torch.cat([p[lam] for p in pieces]), atol=1e-6, rtol=1e-5)
            self.assertFalse(full[lam].requires_grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        model.train()
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            self.call(model)

    def test_fixed_noise_is_reproducible_independent_and_rng_isolated(self):
        rng = torch.get_rng_state().clone()
        a = noise_bank([3101,3102,3103,3104], (71,24,1))
        b = noise_bank([3101,3102,3103,3104], (71,24,1))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for seed in a:
            self.assertTrue(torch.equal(a[seed], b[seed]))
        self.assertFalse(torch.equal(a[3101], a[3102]))

    def test_aggregation_retains_t_and_seed_and_rejects_missing_duplicates(self):
        from diagnostics.loop_state_usage import METRICS
        config = dict(training_seeds=[42,43,44], variants=['base','loop','untied'],
                      timesteps=[50,950], lambdas=[0.,.5,1.], sample_ids=['a','b'], noise_seeds=[1,2,3,4])
        rows=[]
        for s in config['training_seeds']:
            for v in config['variants']:
                for t in config['timesteps']:
                    for lam in ([1.] if v=='base' else config['lambdas']):
                        for d, sample in enumerate(config['sample_ids']):
                            for n in config['noise_seeds']:
                                rows.append(dict(training_seed=s,variant=v,t=t,lambda_value=lam,sample_id=sample,noise_seed=n)
                                            | {'lambda':lam} | {m: float(s-43+d+n+t) for m in METRICS})
        days,runs,summary=aggregate(rows[::-1],config)
        self.assertEqual(len(days),84)
        for r in runs:
            self.assertAlmostEqual(r['noise_mse'],r['training_seed']-43+3+r['t'])
        for r in summary:
            self.assertEqual(r['n_training_seeds'],3)
            self.assertEqual(r['metrics']['noise_mse']['sample_std'],1)
        for invalid in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):aggregate(invalid,config)

    def test_checkpoint_rejects_failure_wrong_identity_raw_protocol_or_scope(self):
        manifest={'data_fingerprint':'d','protocol_fingerprint':'p','scaler':{'mean':[0.]*4,'std':[1.]*4}}
        state={'model_config':asdict(ModelConfig(variant='loop')), 'format_version':2,
               'metadata':{'data_kind':'real','data_fingerprint':'d','protocol_fingerprint':'p'},
               'updates':3000,'attempts':3000,'skipped':0,'ema_initialized':True,'ema_num_averaged_updates':2000,
               'train_config':{'seed':42,'training_protocol_version':2,'ema_start_update':1000,'ema_decay':.999,
                               'diffusion_steps':1000,'amp':False},
               'weather_scaler':{'mean':torch.zeros(4),'std':torch.ones(4)},'ema':{'p':torch.ones(1)}}
        spec={'variant':'loop','training_seed':42}
        self.assertEqual(check_checkpoint(state,spec,{'manifest':manifest}).variant,'loop')
        mutations=[('updates',2999),('ema_initialized',False),('ema_num_averaged_updates',1999)]
        for k,v in mutations:
            bad=deepcopy(state);bad[k]=v
            with self.assertRaises(ValueError):check_checkpoint(bad,spec,{'manifest':manifest})
        for path,v in [(['metadata','failure'],True),(['metadata','data_fingerprint'],'other'),
                       (['model_config','attention_scope'],'diagonal'),(['train_config','seed'],43),
                       (['train_config','ema_decay'],.99),(['weather_scaler','mean'],torch.ones(4))]:
            bad=deepcopy(state);bad[path[0]][path[1]]=v
            with self.assertRaises(ValueError):check_checkpoint(bad,spec,{'manifest':manifest})

    def test_frozen_grid_source_hash_and_output_protection(self):
        config=read('configs/diagnostics/loop_state_usage_v1.json')
        validate_config(config)
        changed=deepcopy(config);changed['lambdas'].append(.75)
        with self.assertRaises(ValueError):validate_config(changed)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'Output exists'):run(config,root)
            source=root/'protected';source.write_text('source')
            modified=deepcopy(config);modified['protected_files']={str(source):'wrong'}
            with self.assertRaisesRegex(ValueError,'hash mismatch'):run(modified,root/'new')
            self.assertFalse((root/'new').exists())
            with self.assertRaisesRegex(ValueError,'overlaps'):run(config,Path(config['runs'][0]['checkpoint']).parent/'new')
