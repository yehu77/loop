"""Real-data C-only acceptance; small replay from full-shape CUDA validation noise."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from correction import CorrectedDenoiser, load_config, require
from data import RealBatches, file_hash, tensor_hash
from diffusion import sample_scenarios
from tests.test_prediction_correction import identical
from train_correction import CorrectionTrainer, load_inputs, write


def run(args):
    torch.set_num_threads(1)
    config=load_config(args.config)
    package,base,scaler=load_inputs(config,args.device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    before={k:tensor_hash(v) for k,v in base.state_dict().items()}
    c=deepcopy(config);c['training']['ema_start_update']=3
    trainer=CorrectionTrainer(c,base,scaler,args.device)
    batches=RealBatches(package,4,42)
    for _ in range(6):trainer.update(batches)
    trainer.save(output/'checkpoint_6.pt',batches)
    resumed=CorrectionTrainer(c,base,scaler,args.device);rb=RealBatches(package,4,42)
    resumed.restore(output/'checkpoint_6.pt',rb)
    for _ in range(2):resumed.update(rb)
    resumed.save(output/'checkpoint_8_resumed.pt',rb)
    continuous=CorrectionTrainer(c,base,scaler,args.device);cb=RealBatches(package,4,42)
    for _ in range(8):continuous.update(cb)
    continuous.save(output/'checkpoint_8_continuous.pt',cb)
    require(identical(resumed.state(rb),continuous.state(cb)), 'Real 6+2 versus 8 mismatch')
    require(before=={k:tensor_hash(v) for k,v in base.state_dict().items()},'D mutated')
    require(any(not torch.equal(v, trainer.corrector.state_dict()[k]) for k,v in continuous.corrector.state_dict().items()),'C did not learn')
    values=package['splits']['validation']
    full_noise=torch.randn(57,100,24,1,device=args.device,generator=torch.Generator(device=args.device).manual_seed(2026))
    # Preserve both condition-encoder and DDIM batch shapes for bitwise
    # replay. Drawing full noise alone does not preserve encoder GEMM shapes.
    count=57
    n=scaler.transform(values['nwp'][:count].to(args.device));a=values['calendar'][:count].to(args.device);h=values['history_power'][:count].to(args.device)
    noise=full_noise[:count]
    original=sample_scenarios(base,trainer.schedule,n,a,100,50,16,initial_noise=noise,history_power=h)[0]
    wrapper=CorrectedDenoiser(base,resumed.ema,trainer.schedule,0)
    replay=sample_scenarios(wrapper,trainer.schedule,n,a,100,50,16,initial_noise=noise,history_power=h)[0]
    require(torch.equal(original,replay),'K0 does not match direct BASE')
    matches_old=None
    if torch.device(args.device).type=='cuda':
        old=torch.load(config['base']['scenarios'],map_location='cpu',weights_only=True)['values'][:count]
        write(output/'archived_replay.json', {'exact':torch.equal(replay.cpu(),old),
              'max_abs_error':(replay.cpu()-old).abs().max().item(),
              'different_values':int((replay.cpu()!=old).sum()),'days':count})
        require(torch.equal(replay.cpu(),old),'K0 CUDA archived scenario replay mismatch')
        matches_old=True
    for k in (1,2):
        wrapper=CorrectedDenoiser(base,resumed.ema,trainer.schedule,k)
        result=sample_scenarios(wrapper,trainer.schedule,n,a,3,4,2,initial_noise=noise[:,:3],history_power=h)[0]
        require(result.shape==(count,3,24,1) and bool(torch.isfinite(result).all()),'Invalid corrected samples')
    require(file_hash(config['base']['checkpoint'])==config['base']['sha256'],'BASE checkpoint changed')
    write(output/'acceptance.json',{'device':args.device,'torch':str(torch.__version__),
          'normal_resume_6_plus_2_equals_8_all_state':True,'base_parameters_buffers_unchanged':True,
          'k0_direct_exact':True,'k0_archived_cuda_exact':matches_old,'replay_days':count,
          'full_initial_noise_sha256':tensor_hash(full_noise),'temporary_ema_start':3,
          'base_sha256':config['base']['sha256'],'formal_training':False,'test_scoring':False})
    print((output/'acceptance.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
