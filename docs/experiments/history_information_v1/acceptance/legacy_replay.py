"""Compare current final-only execution to repository source at the history data protocol commit."""
import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from model import ModelConfig
from train import Trainer,TrainConfig,WeatherScaler,SyntheticBatches,make_synthetic_data
ROOT=Path('/home/u86177/work/loop');sys.path.insert(0,str(ROOT))
# Running this file requires PYTHONPATH=. so imports above resolve the repository.
device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as directory:
 modules={}
 for name in ('model','train'):
  path=Path(directory)/(name+'.py')
  path.write_bytes(subprocess.check_output(['git','show','05d8eb3:'+name+'.py'],cwd=ROOT))
  spec=importlib.util.spec_from_file_location('history_legacy_'+name,path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
  with patch.dict(sys.modules,modules):spec.loader.exec_module(module)
  modules[name]=module
 records=[]
 for variant in ('base','loop','untied'):
  kwargs=dict(variant=variant,horizon=6,hidden_dim=16,ffn_dim=32)
  mc=ModelConfig(**kwargs);data=make_synthetic_data(mc,samples=11)
  scaler=WeatherScaler.fit(data['nwp'])
  tc=dict(microbatch_size=2,accumulation_steps=2,training_protocol_version=2,ema_start_update=2)
  old=modules['train'].Trainer(modules['model'].ModelConfig(**kwargs),modules['train'].TrainConfig(**tc),scaler,device)
  new=Trainer(mc,TrainConfig(**tc),scaler,device)
  a=SyntheticBatches(data,2);b=SyntheticBatches(data,2)
  for _ in range(5):
   x=old.train_update(a);y=new.train_update(b);assert x==y,(x,y)
   for attr in ('model','ema'):
    for n,v in getattr(old,attr).state_dict().items():assert torch.equal(v,getattr(new,attr).state_dict()[n]),n
   for (n,p),(nn,q) in zip(old.model.named_parameters(),new.model.named_parameters()):assert n==nn and torch.equal(p.grad,q.grad),n
   assert torch.equal(old.generator.get_state(),new.generator.get_state())
  records.append({'variant':variant,'updates':5,'loss_gradient_raw_ema_noise_rng_bitwise_equal':True})
 print(json.dumps({'reference':'05d8eb3','device':device,'torch':str(torch.__version__),'records':records},indent=2))
