"""Resume the inspected R04 seed-44 interruption; retain every original artifact."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from data import file_hash
from diagnostics.run_intermediate_supervision import write
from diagnostics.temporal_dependence import read, require
from diagnostics.summarize_intermediate_supervision import verify_ds_pair, summarize
from summarize_runs import audit_run

CONFIG = Path('configs/intermediate_supervision_v1_resumed.json')
STATUS = Path('outputs/intermediate_v1_resume1_queue.json')


def main():
    torch.set_num_threads(1)
    require(torch.cuda.is_available(), 'CUDA required')
    c = read(CONFIG)
    lock = Path('outputs/intermediate_v1_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not STATUS.exists(), 'Continuation queue already exists; inspect first')
    protected = {**c['protected_reference_files'], **c['protocol']['training_source_sha256'],
                 **c['implementation_sha256'], **c['continuation']['protected_files'],
                 c['data']: c['protocol']['data_sha256'], str(CONFIG): file_hash(CONFIG),
                 c['reference_manifest']: c['reference_manifest_sha256']}
    def verify():
        require(all(file_hash(p) == h for p, h in protected.items()), 'Frozen source/artifact changed')
    verify()
    resume = Path(c['continuation']['resume_checkpoint'])
    state = torch.load(resume, map_location='cpu', weights_only=True)
    require(state['updates'] == state['attempts'] == 1999 and state['skipped'] == 0
            and not state['metadata'].get('failure', False), 'Expected normal update-1999 checkpoint')
    require(c['protocol']['endpoint_update'] == 3000, 'Endpoint must remain 3000')
    jobs = [s for s in c['runs'] if s['seed'] == 44]
    require([s['variant'] for s in jobs] == ['loop', 'untied'], 'Only the two remaining runs')
    for spec in jobs:
        output = Path(spec['segments'][-1])
        require(not output.exists() and not output.with_suffix('.lineage.json').exists(), 'Output exists')
    require(not Path(c['report_output']).exists(), 'Report already exists')
    reference = read(c['reference_manifest'])
    status = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
              'plan_sha256': file_hash(CONFIG), 'completed': [], 'active': None,
              'previous_completed': [s['segments'][-1] for s in c['runs'] if s['seed'] != 44]}
    write(STATUS, status)
    try:
        for spec in jobs:
            verify()
            variant = spec['variant']
            output = Path(spec['segments'][-1])
            command = [sys.executable, 'train_real.py', '--data', c['data'], '--device', 'cuda',
                       '--variant', variant, '--attention-scope', 'full',
                       '--training-protocol-version', '2', '--ema-start-update', '1000',
                       '--supervision-mode', 'intermediate', '--loss-weights', '0.5', '0.5',
                       '--output', str(output)]
            if variant == 'loop':
                command += ['--resume', str(resume), '--updates', '1001']
            else:
                command += ['--updates', '3000', '--microbatch-size', '4', '--accumulation-steps', '8',
                            '--validate-every', '250', '--scenarios', '100', '--sampling-steps', '50',
                            '--chunk-size', '16', '--seed', '44']
            record = {'seed': 44, 'variant': variant, 'attention_scope': 'full',
                      'supervision_mode': 'intermediate', 'output': str(output), 'command': command,
                      'commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      'git_status': subprocess.check_output(['git','status','--porcelain'],text=True),
                      'training_source_sha256': c['protocol']['training_source_sha256'],
                      'data_sha256': c['protocol']['data_sha256'], 'plan_sha256': file_hash(CONFIG),
                      'started_unix': time.time(), 'exit_code': None}
            if variant == 'loop':
                record.update(resume_sha256=file_hash(resume), resumed_successful_updates=1999)
            write(output.with_suffix('.lineage.json'),record)
            with output.with_suffix('.log').open('x') as stream:
                child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                status['active']={'seed':44,'variant':variant,'output':str(output),'pid':child.pid}
                write(STATUS,status)
                code=child.wait()
            record.update(exit_code=code,ended_unix=time.time(),
                          sources_unchanged=all(file_hash(p)==h for p,h in protected.items()))
            if code==0:
                record['last_checkpoint_sha256']=file_hash(output/'checkpoint_last.pt')
                anchor = output/'checkpoint_ema_start.pt' if variant=='untied' else resume.parent/'checkpoint_ema_start.pt'
                record['ema_start_checkpoint_sha256']=file_hash(anchor)
            write(output.with_suffix('.lineage.json'),record)
            require(code==0 and record['sources_unchanged'],'Continuation failed; stop for inspection')
            audited=audit_run(spec,c['protocol'])
            old_spec=next(s for s in reference['runs'] if s['seed']==44 and s['variant']==variant)
            verify_ds_pair(audit_run(old_spec,reference['protocol']),audited)
            record['engineering_audit_passed']=True
            write(output.with_suffix('.lineage.json'),record)
            status['completed'].append(str(output));status['active']=None
            write(STATUS,status)
        verify()
        status.update(status='training_complete',training_ended_unix=time.time());write(STATUS,status)
        report=summarize(c,Path(c['report_output']))
        status.update(status='report_complete_pending_review',output=c['report_output'],report_models=len(report['rows'])//2)
        write(STATUS,status)
        os.environ.setdefault('MPLCONFIGDIR','/tmp/loop-pilot-mpl')
        from diagnostics.plot_intermediate_supervision import plot
        plot(Path(c['report_output']),Path(c['report_output'])/'comparison.png')
        status['ended_unix']=time.time();write(STATUS,status)
    except BaseException as error:
        status.update(status='stopped_for_inspection',error=repr(error),ended_unix=time.time());write(STATUS,status)
        raise
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    main()
