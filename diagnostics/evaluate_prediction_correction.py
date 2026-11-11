"""Fixed C@3000 EMA, K0/K1/K2, common CUDA noise, saved-artifact re-scoring."""
import argparse
from pathlib import Path
import time

import torch

from correction import CorrectedDenoiser, load_config, require
from data import file_hash, tensor_hash
from diffusion import sample_scenarios
from evaluate import identified_artifact, write_evaluation, evaluate_artifacts, align_artifacts
from diagnostics.temporal_dependence import score_panel, daily_mean, METRICS, write_csv
from train import synchronize
from train_correction import CorrectionTrainer, load_inputs, write


def run(args):
    torch.set_num_threads(1)
    config=load_config(args.config)
    require(args.mode=='fixed','Only fixed K0/K1/K2 evaluation is authorized here')
    package,base,scaler=load_inputs(config,args.device)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    require(state['format']=='prediction_correction_v1' and state['config']==config
            and not state['metadata']['failure'] and state['metadata']['base_sha256']==config['base']['sha256']
            and state['metadata']['data_fingerprint']==config['data_fingerprint'], 'C checkpoint identity mismatch')
    require(state['updates']==state['attempts']==3000 and state['ema_initialized']
            and state['ema_num_averaged_updates']==2000,'Require fixed C@3000 delayed EMA')
    checkpoint_hash=file_hash(args.checkpoint)
    trainer=CorrectionTrainer(config,base,scaler,args.device)
    trainer.ema.load_state_dict(state['ema'])
    e=config['evaluation']
    values=package['splits']['validation'];meta=package['metadata']['validation']
    noise=torch.randn(len(meta),e['scenarios'],24,1,device=args.device,
                      generator=torch.Generator(device=args.device).manual_seed(config['seeds']['validation']))
    n=scaler.transform(values['nwp'].to(args.device));a=values['calendar'].to(args.device);h=values['history_power'].to(args.device)
    target=identified_artifact(values['y'],package['manifest'],meta)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    write(output/'config.json',config)
    rows,daily,arrays=[],[],{}
    for k in (0,1,2):
        wrapper=CorrectedDenoiser(base,trainer.ema,trainer.schedule,k).eval()
        synchronize(trainer.device)
        if trainer.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        x,diag=sample_scenarios(wrapper,trainer.schedule,n,a,e['scenarios'],e['sampling_steps'],e['chunk_size'],initial_noise=noise,history_power=h)
        synchronize(trainer.device);elapsed=time.perf_counter()-start
        prediction=identified_artifact(x,package['manifest'],meta)
        if k==0 and trainer.device.type=='cuda':
            old=torch.load(config['base']['scenarios'],map_location='cpu',weights_only=True)
            oldx,oldy=align_artifacts(old,target)
            require(torch.equal(oldx,prediction['values']) and torch.equal(oldy,target['values']),'Full K0 archived replay mismatch')
        diag.update(K=k,weights='C EMA',successful_updates=3000,base_sha256=config['base']['sha256'],
                    corrector_checkpoint_sha256=checkpoint_hash,initial_noise_sha256=tensor_hash(noise),
                    sampling_seconds=elapsed,generated_trajectories=len(meta)*e['scenarios'])
        path=output/f'K{k}'
        report=write_evaluation(path,prediction,target,diag)
        diskp=torch.load(path/'scenarios.pt',map_location='cpu',weights_only=True)
        diskt=torch.load(path/'targets.pt',map_location='cpu',weights_only=True)
        rescored,per=evaluate_artifacts(diskp,diskt)
        require(rescored=={key:val for key,val in report.items() if key!='diagnostics'},'Independent re-score mismatch')
        xx,yy=align_artifacts(diskp,diskt)
        scores=score_panel(xx,yy);arrays[k]={m:daily_mean(scores[m]) for m in METRICS}
        dp=sum(p.numel() for p in base.parameters());cp=sum(p.numel() for p in trainer.ema.parameters())
        rows.append({'K':k,**{m:scores[m].mean().item() for m in METRICS},
                     'sampling_seconds':elapsed,'D_parameters':dp,'C_parameters':cp,'deployment_parameters':dp+(cp if k else 0),
                     'boundary_fraction':((xx==0)|(xx==1)).double().mean().item(),
                     'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if trainer.device.type=='cuda' else None})
        for i,m in enumerate(meta):
            daily.append({'K':k,'sample_id':m['sample_id'],**{name:arr[i].item() for name,arr in arrays[k].items()}})
        print(rows[-1],flush=True)
    pairs=[]
    for left,right in ((1,0),(2,0),(2,1)):
        for i,m in enumerate(meta):
            pairs.append({'comparison':f'K{left}-K{right}','sample_id':m['sample_id'],
                          **{name:(arrays[left][name][i]-arrays[right][name][i]).item() for name in METRICS}})
    gate=all(rows[2][m]<rows[k][m] for m in ('crps','energy_score') for k in (0,1)) and abs(rows[2]['coverage_90']-.9)<=abs(rows[0]['coverage_90']-.9)
    require(file_hash(args.checkpoint)==checkpoint_hash,'C checkpoint changed')
    load_config(args.config)
    require(all(file_hash(p)==digest for p,digest in config['protected_artifacts'].items()),'Protected artifacts changed')
    summary={'rows':rows,'development_gate_passed':gate,'checkpoint_sha256':checkpoint_hash,
             'base_sha256':config['base']['sha256'],'initial_noise_sha256':tensor_hash(noise),
             'independent_rescore_exact':True,'test_scoring':False,'new_training_runs':1,
             'cost_matching_performed':False,'training_seed':42,'validation_days':len(meta),
             'interpretation':'Single-seed development screen; 57 days are not independent trainings; stop after this table'}
    write(output/'summary.json',summary)
    write_csv(output/'scores.csv',rows);write_csv(output/'scores_per_day.csv',daily);write_csv(output/'paired_daily.csv',pairs)
    torch.save({'sample_ids':config['validation_ids'],'scores':arrays},output/'transformed_scores_per_day.pt')
    write(output/'artifact_sha256.json',{str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()})
    print('Fixed evaluation complete; development gate:',gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--mode',choices=['fixed'],default='fixed')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
