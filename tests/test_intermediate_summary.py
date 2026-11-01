"""Four-cell interaction accounting and JSON/checkpoint loss-weight identity."""
from argparse import Namespace
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from data import prepare_dataset,file_hash
from diagnostics.summarize_intermediate_supervision import LABELS,METRICS,paired_differences
from summarize_runs import audit_run
from test_data import fixture_protocol,write_sources
from train_real import run
import torch


class IntermediateSummaryTests(unittest.TestCase):
    def test_interaction_pairs_within_seed_and_does_not_imply_absolute_win(self):
        rows=[]
        for table in ('primary','best'):
            for seed in (42,43,44):
                # LOOP benefits more, yet remains worse than UNTIED and BASE.
                scores={'base':2.,'loop':10.,'untied':3.,'loop_ds':10.-(seed-40),'untied_ds':2.5}
                for model in LABELS:rows.append({'table':table,'seed':seed,'model':model,**{k:scores[model] for k in METRICS}})
        pairs,summary=paired_differences(rows[::-1],[42,43,44])
        p=next(r for r in pairs if r['seed']==42 and r['table']=='primary' and r['metric']=='crps')
        self.assertEqual((p['delta_loop'],p['delta_untied'],p['interaction']),(-2.,-.5,-1.5))
        for r in summary:
            self.assertEqual(r['effects']['interaction']['mean'],-2.5)
            self.assertEqual(r['effects']['interaction']['sample_std'],1.)
            self.assertEqual(r['n_training_seeds'],3)
        for invalid in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):paired_differences(invalid,[42,43,44])

    def test_real_ds_artifact_audit_handles_json_lists_without_weakening_identity(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol=fixture_protocol();write_sources(root,protocol)
            data=root/'data.pt';torch.save(prepare_dataset(root,protocol),data)
            output=root/'loop_ds'
            args=Namespace(data=data,output=output,resume=None,variant='loop',attention_scope='full',
                           supervision_mode='intermediate',loss_weights=[.5,.5],training_protocol_version=2,
                           ema_start_update=1,device='cpu',updates=3,amp=False,seed=42,microbatch_size=1,
                           accumulation_steps=1,validate_every=1,validation_limit=None,scenarios=2,sampling_steps=2,chunk_size=4)
            with contextlib.redirect_stdout(io.StringIO()):run(args)
            config=json.loads((output/'config.json').read_text())
            lineage={'training_source_sha256':{},'data_sha256':file_hash(data),'exit_code':0,
                     'sources_unchanged':True,'last_checkpoint_sha256':file_hash(output/'checkpoint_last.pt')}
            output.with_suffix('.lineage.json').write_text(json.dumps(lineage))
            spec={'seed':42,'variant':'loop','segments':[str(output)]}
            frozen={'endpoint_update':3,'ema_start_update':1,'validate_every':1,
                    'validation_days':len(config['validation_sample_ids']),'data_fingerprint':config['data_fingerprint'],
                    'data_sha256':lineage['data_sha256'],'training_source_sha256':{},
                    'training':{k:v for k,v in config['training'].items() if k!='seed'},
                    'model':{k:v for k,v in config['model'].items() if k!='variant'},
                    'evaluation':{k:v for k,v in config['evaluation'].items() if k!='rng_streams'}}
            actual=audit_run(spec,frozen)
            self.assertTrue(actual['evidence']['independent_rescoring_passed'])
            state=torch.load(output/'checkpoint_last.pt',weights_only=True)
            state['train_config']['loss_weights']=(.25,.75)
            torch.save(state,output/'checkpoint_last.pt')
            with self.assertRaisesRegex(ValueError,'configuration differs'):audit_run(spec,frozen)
