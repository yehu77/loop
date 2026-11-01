"""R04 loss/gradient checks activate the initially zero head and gates."""
from argparse import Namespace
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data import prepare_dataset
from diffusion import sample_scenarios
from model import ModelConfig, ConditionalLoopDiT
from test_data import fixture_protocol, write_sources
from train import TrainConfig, Trainer, WeatherScaler, SyntheticBatches, make_synthetic_data
from train_real import run


class IntermediateSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.mc=ModelConfig(variant='loop',horizon=6,hidden_dim=16,ffn_dim=32)
        self.tc=TrainConfig(microbatch_size=2,accumulation_steps=2,training_protocol_version=2,
                            ema_start_update=2,supervision_mode='intermediate',loss_weights=(.5,.5))
        self.data=make_synthetic_data(self.mc,samples=7)
        self.scaler=WeatherScaler.fit(self.data['nwp'])

    def active_model(self,variant):
        torch.manual_seed(128)
        model=ConditionalLoopDiT(replace(self.mc,variant=variant))
        with torch.no_grad():
            model.output_head.projection.weight.normal_(std=.15)
            for core in model.cores:
                for block in core.blocks:
                    block.modulator[-1].bias[32:48].fill_(.6)
                    block.modulator[-1].bias[80:96].fill_(.4)
        return model

    def inputs(self):
        return (torch.randn(2,6,1),torch.randn(2,6,4),torch.randn(2,6,4),torch.tensor([200,700]))

    def test_single_pass_shared_head_and_final_forward_equivalence(self):
        for variant in ('loop','untied'):
            model=self.active_model(variant);inputs=self.inputs()
            count=[0]*len(model.cores);head_calls=[]
            def hook(i):
                def record(module,args,out):count[i]+=1
                return record
            handles=[c.register_forward_hook(hook(i)) for i,c in enumerate(model.cores)]
            handles.append(model.output_head.register_forward_hook(lambda m,a,o:head_calls.append(o)))
            stages=model.forward_stages(*inputs)
            for h in handles:h.remove()
            self.assertEqual(count,[2] if variant=='loop' else [1,1])
            self.assertEqual(len(head_calls),2)
            torch.testing.assert_close(stages[1],model(*inputs),atol=0,rtol=0)
            self.assertIs(stages[0],head_calls[0]);self.assertIs(stages[1],head_calls[1])

    def test_gradient_sum_and_full_final_loss_path(self):
        for variant in ('loop','untied'):
            model=self.active_model(variant);inputs=self.inputs();states=[]
            hook=model.cores[0].register_forward_hook(lambda m,a,o:states.append(o))
            p1,p2=model.forward_stages(*inputs);hook.remove()
            noise=torch.randn_like(p1)
            l1=(p1.float()-noise).square().mean();l2=(p2.float()-noise).square().mean()
            params=list(model.parameters());names=[n for n,p in model.named_parameters()]
            g1=torch.autograd.grad(l1,params,retain_graph=True,allow_unused=True)
            g2=torch.autograd.grad(l2,params,retain_graph=True,allow_unused=True)
            through_first=torch.autograd.grad(l2,states[0],retain_graph=True)[0]
            self.assertGreater(through_first.abs().sum().item(),1e-6)
            combined=torch.autograd.grad(.5*l1+.5*l2,params,allow_unused=True)
            for p,a,b,g in zip(params,g1,g2,combined):
                a=torch.zeros_like(p) if a is None else a;b=torch.zeros_like(p) if b is None else b
                torch.testing.assert_close(g,.5*a+.5*b,atol=2e-6,rtol=1e-4)
            if variant=='untied':
                self.assertTrue(all(g is None for n,g in zip(names,g1) if n.startswith('cores.1.')))
                for prefix in ('cores.0.','cores.1.'):
                    self.assertGreater(sum(g.abs().sum().item() for n,g in zip(names,g2) if n.startswith(prefix) and g is not None),1e-6)

    def test_identical_initial_parameters_and_rng_no_new_head_all_seeds(self):
        for seed in (42,43,44):
            for variant in ('loop','untied'):
                mc=ModelConfig(variant=variant)
                a=Trainer(mc,replace(self.tc,seed=seed,supervision_mode='final-only',loss_weights=(0.,1.)),self.scaler)
                rng=torch.get_rng_state().clone()
                b=Trainer(mc,replace(self.tc,seed=seed),self.scaler)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertTrue(torch.equal(a.generator.get_state(),b.generator.get_state()))
                self.assertEqual(set(a.model.state_dict()),set(b.model.state_dict()))
                for n,p in a.model.state_dict().items():self.assertTrue(torch.equal(p,b.model.state_dict()[n]))
                self.assertEqual(sum(p.numel() for p in b.model.parameters()),208065 if variant=='loop' else 365697)

    def test_training_logs_and_final_only_does_not_compute_stage_outputs(self):
        trainer=Trainer(self.mc,self.tc,self.scaler);batches=SyntheticBatches(self.data,2)
        for _ in range(4):
            r=trainer.train_update(batches)
            self.assertAlmostEqual(r['loss_total'],.5*r['loss1']+.5*r['loss2'],places=6)
            self.assertEqual(r['loss'],r['loss_total'])
        plain=Trainer(self.mc,replace(self.tc,supervision_mode='final-only',loss_weights=(0,1)),self.scaler)
        with patch.object(plain.model,'forward_stages',side_effect=AssertionError('unused path')):
            r=plain.train_update(SyntheticBatches(self.data,2))
        self.assertNotIn('loss1',r)

    def test_resume_identity_and_exact_continuation_before_and_after_ema(self):
        for variant in ('loop','untied'):
            for boundary in (1,3):
                trainer=Trainer(replace(self.mc,variant=variant),self.tc,self.scaler)
                batches=SyntheticBatches(self.data,2)
                for _ in range(boundary):trainer.train_update(batches)
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/'checkpoint.pt';trainer.save(path,batches)
                    rng=torch.get_rng_state().clone()
                    for expect in ({'expected_supervision_mode':'final-only'},{'expected_loss_weights':(.25,.75)}):
                        with self.assertRaisesRegex(ValueError,'mismatch'):Trainer.load(path,**expect)
                        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                    expected=[trainer.train_update(batches) for _ in range(2)]
                    restored,state,_=Trainer.load(path,expected_supervision_mode='intermediate',expected_loss_weights=[.5,.5])
                    other=SyntheticBatches(self.data,2);other.load_state_dict(state)
                    self.assertEqual(expected,[restored.train_update(other) for _ in range(2)])
                    for attr in ('model','ema'):
                        for n,p in getattr(trainer,attr).state_dict().items():self.assertTrue(torch.equal(p,getattr(restored,attr).state_dict()[n]))

    def test_real_entry_ds_and_legacy_final_best_resume(self):
        for mode in ('final-only','intermediate'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);protocol=fixture_protocol();write_sources(root,protocol)
                torch.save(prepare_dataset(root,protocol),root/'data.pt')
                args=Namespace(data=root/'data.pt',output=root/'first',resume=None,variant='loop',
                               attention_scope='full',supervision_mode=mode,loss_weights=None,
                               training_protocol_version=2,ema_start_update=1,device='cpu',updates=2,
                               amp=False,seed=42,microbatch_size=1,accumulation_steps=1,validate_every=1,
                               validation_limit=None,scenarios=2,sampling_steps=2,chunk_size=4)
                with contextlib.redirect_stdout(io.StringIO()):run(args)
                if mode=='final-only':
                    for name in ('checkpoint_last.pt','checkpoint_best.pt'):
                        path=args.output/name;state=torch.load(path,weights_only=True)
                        for key in ('supervision_mode','loss_weights'):del state['train_config'][key]
                        torch.save(state,path)
                args.resume=args.output/'checkpoint_last.pt';args.output=root/'resumed';args.updates=1
                args.supervision_mode='intermediate' if mode=='final-only' else 'final-only'
                with self.assertRaisesRegex(ValueError,'supervision mode mismatch'):run(args)
                self.assertFalse(args.output.exists())
                args.supervision_mode=None
                with contextlib.redirect_stdout(io.StringIO()):run(args)
                saved=json.loads((args.output/'config.json').read_text())
                self.assertEqual(saved['training']['supervision_mode'],mode)

    def test_configuration_validates_weights_and_scope(self):
        self.assertEqual(TrainConfig().supervision_mode,'final-only')
        self.assertEqual(TrainConfig(**json.loads(json.dumps(asdict(self.tc)))),self.tc)
        for values in ({'supervision_mode':'bad'},{'loss_weights':(1,1)},
                       {'supervision_mode':'intermediate','loss_weights':(0,1)},
                       {'loss_weights':(float('nan'),1)}):
            with self.assertRaises(ValueError):TrainConfig(**values)
        with self.assertRaises(ValueError):Trainer(replace(self.mc,variant='base'),self.tc,self.scaler)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA execution environment required')
    def test_gpu_updates_resume_and_final_only_sampling(self):
        for variant in ('loop','untied'):
            trainer=Trainer(replace(self.mc,variant=variant),self.tc,self.scaler,'cuda')
            batches=SyntheticBatches(self.data,2)
            for _ in range(4):self.assertTrue(trainer.train_update(batches)['did_update'])
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'c.pt';trainer.save(path,batches)
                expected=trainer.train_update(batches)
                restored,state,_=Trainer.load(path,'cuda',expected_supervision_mode='intermediate')
                other=SyntheticBatches(self.data,2);other.load_state_dict(state)
                self.assertEqual(expected,restored.train_update(other))
                for n,p in trainer.ema.state_dict().items():self.assertTrue(torch.equal(p,restored.ema.state_dict()[n]))
            with patch.object(trainer.ema,'forward_stages',side_effect=AssertionError('sampling must use final only')):
                samples,_=sample_scenarios(trainer.ema,trainer.schedule,
                    trainer.weather_scaler.transform(self.data['nwp'][:2].cuda()),self.data['calendar'][:2].cuda(),
                    scenarios=3,sampling_steps=4,chunk_size=3,generator=torch.Generator(device='cuda').manual_seed(2026))
            self.assertEqual(samples.shape,(2,3,6,1))
