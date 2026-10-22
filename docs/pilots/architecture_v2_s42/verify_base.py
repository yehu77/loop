# ruff: noqa: E402
# Repository path precedes local imports.
# Run from the repository root; reads local artifacts and writes only derived reports.
# Historical v1 EMA reconstructs the same raw trajectory; FP32 equality is not required for EMA.
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import torch
from diagnose_ema import reconstruct_ema_state
from model import ConditionalLoopDiT, ModelConfig
torch.set_num_threads(1)
run = Path('outputs/architecture_v2_s42_base')
old = Path('outputs/real_base_pilot_v1_s42')
end = Path('outputs/real_base_pilot_v1_s42_to3000')

def read(p):
    return json.loads(p.read_text())

def load(p):
    return torch.load(p, map_location='cpu', weights_only=True)
assert read(run.with_suffix('.lineage.json'))['exit_code'] == 0
new = load(run / 'checkpoint_last.pt')
anchor = load(run / 'checkpoint_ema_start.pt')
v1 = load(end / 'checkpoint_last.pt')
oldanchor = load(old / 'checkpoint_last.pt')
assert new['updates'] == 3000 and new['ema_num_averaged_updates'] == 2000 and new['ema_initialized']
assert anchor['updates'] == 1000 and anchor['ema_num_averaged_updates'] == 0
raw_diff = max(((new['model'][k] - v1['model'][k]).abs().max().item() for k in new['model']))
anchor_diff = max(((anchor['model'][k] - oldanchor['model'][k]).abs().max().item() for k in anchor['model']))
original_history = read(old / 'training_history.json') + read(end / 'training_history.json')
new_history = read(run / 'training_history.json')
assert len(new_history) == len(original_history) == 3000
loss_diff = max((abs(a['loss'] - b['loss']) for a, b in zip(original_history, new_history)))
grad_diff = max((abs(a['grad_norm'] - b['grad_norm']) for a, b in zip(original_history, new_history)))
for key in ['noise_rng', 'torch_rng']:
    assert torch.equal(new[key], v1[key]), key
for key in ['order', 'rng']:
    assert torch.equal(new['data'][key], v1['data'][key]), key
assert new['data']['cursor'] == v1['data']['cursor']
assert new['skipped'] == 0
model = ConditionalLoopDiT(ModelConfig(**new['model_config']))
reconstructed = reconstruct_ema_state(model, oldanchor['model'], oldanchor['ema'], v1['ema'], 0.999, 2000)
param_names = set(dict(model.named_parameters()))
ema_diff = max(((new['ema'][k] - reconstructed[k]).abs().max().item() for k in param_names))
report = {'raw_endpoint_max_abs_difference_vs_v1': raw_diff, 'raw_anchor_max_abs_difference_vs_v1': anchor_diff, 'training_loss_max_abs_difference_vs_v1': loss_diff, 'gradient_norm_max_abs_difference_vs_v1': grad_diff, 'training_noise_and_batch_rng_equal': True, 'online_vs_reconstructed_ema_parameter_max_abs_difference': ema_diff, 'ema_averaged_updates': new['ema_num_averaged_updates'], 'interpretation': 'FP32 online recurrence and FP64 reconstruction followed by FP32 rounding need not be bitwise equal'}
assert raw_diff == anchor_diff == loss_diff == grad_diff == 0, report
assert ema_diff < 0.0001, report
old_scores = read(Path('outputs/real_base_ema_history_diagnostic_3000/restarted_ema/metrics.json'))
new_scores = read(run / 'metrics.json')
report['online_vs_offline_score_differences'] = {k: new_scores[k] - old_scores[k] for k in ['crps', 'energy_score', 'variogram_score']}
(run / 'implementation_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
