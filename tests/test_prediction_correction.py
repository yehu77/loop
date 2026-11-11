"""Nondegenerate correction, feedback, freezing and exact-resume regressions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from correction import CorrectedDenoiser, Corrector
from diffusion import LinearNoiseSchedule, sample_scenarios
from model import ConditionalLoopDiT, ModelConfig
from train import SyntheticBatches, WeatherScaler, make_synthetic_data
from train_correction import CorrectionTrainer


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def fixture(device='cpu'):
    torch.manual_seed(9)
    mc = ModelConfig(variant='base', history_group='H0', hidden_dim=16, ffn_dim=32)
    base = ConditionalLoopDiT(mc).to(device)
    with torch.no_grad():
        base.output_head.projection.weight.fill_(0.03)
    data = make_synthetic_data(mc, samples=9)
    data['history_power'] = torch.zeros(9, 24, 1)
    scaler = WeatherScaler.fit(data['nwp'])
    cfg = {'corrector': {'variant':'base', 'rounds':1,'hidden_dim':16,'ffn_dim':32},
           'seeds': {'corrector_initialization':30042,'training_noise':43,'data_order':42},
           'training': {'learning_rate':1e-4,'weight_decay':1e-4,'microbatch_size':2,
                        'accumulation_steps':2,'max_grad_norm':1.,'ema_start_update':3,
                        'ema_decay':.999,'correction_steps':1,'precision':'float32'},
           'base':{'sha256':'synthetic'},'data_fingerprint':'synthetic','source_sha256':{'test':'v1'}}
    return base, data, scaler, cfg


class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_inference_loader_identity_and_json_tuple_compatibility(self):
        import json
        from dataclasses import asdict
        from correction import load_base
        from data import file_hash
        from train import TrainConfig
        b,d,scaler,c=fixture()
        mc=asdict(b.config);tc=asdict(TrainConfig())
        manifest={'data_fingerprint':'data','protocol_fingerprint':'protocol',
                  'scaler':{k:v.tolist() for k,v in scaler.state_dict().items()}}
        archive={'model':mc,'training':tc,'data_fingerprint':'data',
                 'evaluation':{'history_training_source_sha256':{'model.py':'archived'}}}
        state={'model_config':mc,'train_config':tc,'updates':3000,'ema_initialized':True,
               'ema_num_averaged_updates':2000,'ema':b.state_dict(),'weather_scaler':scaler.state_dict(),
               'metadata':{'data_kind':'real','data_fingerprint':'data','protocol_fingerprint':'protocol',
                           'run_config':archive['evaluation']}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'base.pt';arch=Path(tmp)/'archive.json'
            torch.save(state,path);arch.write_text(json.dumps(archive))
            cfg={'data_fingerprint':'data','protocol_fingerprint':'protocol',
                 'base':{'checkpoint':str(path),'sha256':file_hash(path),
                         'archive_config':str(arch),'archive_config_sha256':file_hash(arch)}}
            loaded,_=load_base(cfg,{'manifest':manifest},'cpu')
            self.assertTrue(identical(loaded.state_dict(),b.state_dict()))
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            for mutate in ('hash','failure','clock','source','scaler','variant','group','data'):
                bad=deepcopy(state);conf=deepcopy(cfg)
                if mutate=='failure':bad['metadata']['failure']=True
                elif mutate=='clock':bad['ema_num_averaged_updates']=1999
                elif mutate=='source':bad['metadata']['run_config']={}
                elif mutate=='scaler':bad['weather_scaler']['mean']=bad['weather_scaler']['mean']+1
                elif mutate=='variant':bad['model_config']['variant']='loop'
                elif mutate=='group':bad['model_config']['history_group']='H24'
                elif mutate=='data':bad['metadata']['data_fingerprint']='other'
                torch.save(bad,path)
                conf['base']['sha256']=file_hash(path) if mutate!='hash' else 'bad'
                with self.assertRaises(ValueError):load_base(conf,{'manifest':manifest},'cpu')

    def test_zero_correction_and_k0_sampling(self):
        b,d,s,c = fixture()
        trainer = CorrectionTrainer(c,b,s,'cpu')
        x = torch.randn(2,24,1);t=torch.tensor([1,1000])
        n=s.transform(d['nwp'][:2]);a=d['calendar'][:2];h=d['history_power'][:2]
        expected=b(x,n,a,t,history_power=h)
        for k in (0,1,2):
            w=CorrectedDenoiser(b,trainer.corrector,trainer.schedule,k)
            self.assertTrue(torch.equal(w(x,n,a,t,h),expected))
        w=CorrectedDenoiser(b,trainer.corrector,trainer.schedule,0).train()
        noise=torch.randn(2,3,24,1)
        original=sample_scenarios(b,trainer.schedule,n,a,3,4,2,initial_noise=noise,history_power=h)[0]
        actual=sample_scenarios(w,trainer.schedule,n,a,3,4,2,initial_noise=noise,history_power=h)[0]
        self.assertTrue(torch.equal(original,actual));self.assertFalse(b.training)
        self.assertTrue(w.training)
        with self.assertRaisesRegex(ValueError,'t in'):
            w(x,n,a,torch.tensor([0,1]),h)

    def test_feedback_formula_and_no_internal_clipping(self):
        b,d,s,c=fixture()
        trainer=CorrectionTrainer(c,b,s,'cpu');model=trainer.corrector
        proposals=[]
        def delta(proposal, condition, t):
            proposals.append(proposal.detach().clone())
            return .1*proposal+2.
        model.denoise_with_cached_condition=delta
        x=torch.full((2,24,1),3.);t=torch.tensor([1,1000]);n=s.transform(d['nwp'][:2]);a=d['calendar'][:2];h=d['history_power'][:2]
        eps=b(x,n,a,t,history_power=h).detach();aa=trainer.schedule.a[t,None,None];ss=trainer.schedule.sigma[t,None,None]
        p0=(x-ss*eps)/aa
        w=CorrectedDenoiser(b,model,trainer.schedule,2)
        result=w(x,n,a,t,h)
        dp1=.1*p0+2.;dp2=.1*(p0+dp1)+2.
        self.assertTrue(torch.equal(proposals[0],p0))
        self.assertTrue(torch.equal(proposals[1],p0+dp1))
        self.assertGreater(proposals[1].abs().max(),1)
        self.assertTrue(torch.equal(result,eps-(aa/ss)*(dp1+dp2)))

    def test_boundary_timesteps_gradient_and_learning_freeze(self):
        b,d,s,c=fixture();tr=CorrectionTrainer(c,b,s,'cpu')
        original=deepcopy(b.state_dict())
        before=deepcopy(tr.corrector.state_dict())
        opt_ids={id(p) for g in tr.optimizer.param_groups for p in g['params']}
        self.assertEqual(opt_ids,{id(p) for p in tr.corrector.parameters()})
        self.assertFalse(opt_ids & {id(p) for p in b.parameters()})
        for t0 in (1,1000):
            tr.optimizer.zero_grad(set_to_none=True)
            t=torch.full((2,),t0,dtype=torch.long);noise=torch.randn(2,24,1)
            x=tr.schedule.add_noise(2*d['y'][:2]-1,t,noise)
            pred=tr.wrapper(x,s.transform(d['nwp'][:2]),d['calendar'][:2],t,d['history_power'][:2])
            loss=(pred-noise).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in tr.corrector.parameters()))
            self.assertTrue(all(p.grad is None for p in b.parameters()))
        batches=SyntheticBatches(d,2)
        for _ in range(5):tr.update(batches)
        self.assertTrue(identical(original,b.state_dict()))
        self.assertFalse(torch.equal(before['cores.0.blocks.0.attention.qkv.weight'],tr.corrector.state_dict()['cores.0.blocks.0.attention.qkv.weight']))
        self.assertFalse(b.training)

    def test_cache_direct_chunk_and_reorder(self):
        b,d,s,c=fixture();tr=CorrectionTrainer(c,b,s,'cpu')
        for _ in range(4):tr.update(SyntheticBatches(d,2))
        w=CorrectedDenoiser(b,tr.corrector,tr.schedule,2)
        n=s.transform(d['nwp'][:2]);a=d['calendar'][:2];h=d['history_power'][:2]
        x=torch.randn(2,24,1);t=torch.tensor([50,700])
        cache=w.encode_condition(n,a,h)
        self.assertTrue(all(isinstance(v,torch.Tensor) for v in cache.values()))
        self.assertTrue(torch.equal(w(x,n,a,t,h),w.denoise_with_cached_condition(x,cache,t)))
        noise=torch.randn(2,3,24,1)
        out=sample_scenarios(w,tr.schedule,n,a,3,4,2,initial_noise=noise,history_power=h)[0]
        other=sample_scenarios(w,tr.schedule,n.flip(0),a.flip(0),3,4,3,initial_noise=noise.flip(0),history_power=h.flip(0))[0]
        torch.testing.assert_close(out,other.flip(0),rtol=1e-5,atol=1e-6)

    def check_resume(self,device):
        b,d,s,c=fixture(device);initial=deepcopy(b.state_dict())
        for stop in (2,3,6):
            tr=CorrectionTrainer(c,b,s,device);batches=SyntheticBatches(d,2)
            for _ in range(stop):tr.update(batches)
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'c.pt';tr.save(path,batches)
                resumed=CorrectionTrainer(c,b,s,device);rb=SyntheticBatches(d,2);resumed.restore(path,rb)
                for _ in range(8-stop):resumed.update(rb)
                final=resumed.state(rb)
                continuous=CorrectionTrainer(c,b,s,device);cb=SyntheticBatches(d,2)
                for _ in range(8):continuous.update(cb)
                self.assertTrue(identical(final,continuous.state(cb)))
                for field in ('base','data_fingerprint','source_sha256'):
                    bad=deepcopy(c);bad[field]='wrong'
                    resumed.config=bad
                    with self.assertRaises(ValueError):resumed.restore(path,rb)
                resumed.config=c
                tr.at_boundary=False;tr.save(path,batches)
                with self.assertRaisesRegex(ValueError,'failure'):resumed.restore(path,rb)
        self.assertTrue(identical(initial,b.state_dict()))

    def test_resume_and_identity(self):self.check_resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_resume(self):self.check_resume('cuda')


if __name__=='__main__':unittest.main()
