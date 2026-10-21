# ruff: noqa: E402
# Plotting backend and repository path precede dependent imports.
# Run from the repository root; reads local artifacts and writes only derived reports.
# UNTIED has two segments. Preserve the interrupted launch and exclude update 2499 from selection.
"""Rebuild the frozen seed-42 tables from identity-bound local artifacts."""
import os
os.environ['MPLCONFIGDIR'] = '/tmp/loop-pilot-mpl'
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path.cwd()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate import evaluate_artifacts
torch.set_num_threads(1)
variants = ('base', 'loop', 'untied')

def read(p):
    return json.loads(p.read_text())

def load(p):
    return torch.load(p, map_location='cpu', weights_only=True)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
runs = {v: Path('outputs/architecture_v2_s42_' + v) for v in variants}
dest = Path('docs/pilots/architecture_v2_s42')
dest.mkdir(parents=True, exist_ok=True)
configs = {v: read(p / 'config.json') for v, p in runs.items()}
initials = {v: read(p / 'initialization.json') for v, p in runs.items()}
assert initials['base'] == initials['loop']
assert all((initials['untied'][k] == x for k, x in initials['base'].items()))
reference = configs['base']
base_lineage = read(runs['base'].with_suffix('.lineage.json'))
base_checkpoint = load(runs['base'] / 'checkpoint_last.pt')
assert reference['data_fingerprint'] == '403ff1adc1df6b1bd6dd79b24f206f164daf3161f829bf7a232fddf90a9168a4'
rows = []
histories = {}
vals = {}
scores = {}
manifests = {}
checks = {}
panels = {}
for variant, initial_path in runs.items():
    path = Path('outputs/architecture_v2_s42_untied_resumed') if variant == 'untied' else initial_path
    segments = [initial_path, path] if path != initial_path else [path]
    config = configs[variant]
    if len(segments) == 2:
        resume = read(path.with_suffix('.lineage.json'))
        prior = read(initial_path.with_suffix('.lineage.json'))
        assert prior['exit_code'] is None
        assert resume['prior_checkpoint_unchanged']
        assert resume['resume_sha256'] == digest(initial_path / 'checkpoint_last.pt')
        assert resume['prior_best_sha256'] == digest(initial_path / 'checkpoint_best.pt')
        assert resume['prior_lineage_sha256'] == digest(initial_path.with_suffix('.lineage.json'))
        assert resume['resumed_successful_updates'] == 2499 and resume['additional_successful_updates'] == 501
        assert prior['training_source_sha256'] == resume['training_source_sha256']
        for key in ('training', 'evaluation', 'data_fingerprint', 'validation_sample_ids'):
            assert read(path / 'config.json')[key] == config[key]
    for key in ('training', 'evaluation', 'data_fingerprint', 'validation_sample_ids'):
        assert config[key] == reference[key], key
    assert config['training']['training_protocol_version'] == 2
    assert config['training']['ema_start_update'] == 1000
    assert config['evaluation']['validation_limit'] is None
    assert len(config['validation_sample_ids']) == 71
    lineage = read(path.with_suffix('.lineage.json'))
    assert lineage['exit_code'] == 0 and lineage['sources_unchanged'] and (not lineage['git_status'])
    assert lineage['training_source_sha256'] == base_lineage['training_source_sha256']
    assert lineage['data_sha256'] == base_lineage['data_sha256']
    checkpoint = load(path / 'checkpoint_last.pt')
    assert checkpoint['updates'] == 3000 and checkpoint['ema_initialized']
    assert checkpoint['ema_num_averaged_updates'] == 2000 and (not checkpoint['metadata'].get('failure', False))
    assert checkpoint['metadata']['data_fingerprint'] == reference['data_fingerprint']
    assert torch.equal(checkpoint['noise_rng'], base_checkpoint['noise_rng'])
    for field in ('order', 'rng'):
        assert torch.equal(checkpoint['data'][field], base_checkpoint['data'][field])
    assert checkpoint['data']['cursor'] == base_checkpoint['data']['cursor']
    anchor = load(initial_path / 'checkpoint_ema_start.pt')
    assert anchor['updates'] == 1000 and anchor['ema_num_averaged_updates'] == 0
    assert all((torch.equal(anchor['model'][k], anchor['ema'][k]) for k in anchor['model']))
    histories[variant] = [row for segment in segments for row in read(segment / 'training_history.json')]
    vals[variant] = [row for segment in segments for row in read(segment / 'validation_history.json') if row['updates'] % 250 == 0]
    for segment in segments:
        for node in read(segment / 'validation_history.json'):
            panels[variant, node['updates']] = segment / 'validation' / 'update_{:06d}'.format(node['updates'])
    history, validation = (histories[variant], vals[variant])
    assert len(history) == 3000 and all((x['did_update'] for x in history))
    assert [x['updates'] for x in history] == list(range(1, 3001))
    assert history[-1]['skipped'] == 0 and history[-1]['samples_consumed_total'] == 96000
    assert [x['updates'] for x in validation] == list(range(0, 3001, 250))
    eligible = [x for x in validation if 'ema' in x]
    assert [x['updates'] for x in eligible] == list(range(1000, 3001, 250))
    best = min(eligible, key=lambda x: x['ema']['crps'])['updates']
    cost = read(path / 'cost.json')
    assert cost['auxiliary_best_update'] == best
    for table, relative, update in [('primary', Path('.'), 3000), ('best', Path('selected'), best)]:
        folder = path / relative
        report = read(folder / 'metrics.json')
        assert report['diagnostics']['successful_updates'] == update
        assert report['sample_ids'] == reference['validation_sample_ids']
        predicted, target = (load(folder / 'scenarios.pt'), load(folder / 'targets.pt'))
        repeated, per_window = evaluate_artifacts(predicted, target)
        assert {k: v for k, v in report.items() if k != 'diagnostics'} == repeated
        saved = load(folder / 'scores_per_window.pt')
        for key in ('crps', 'energy_score', 'variogram_score'):
            assert torch.equal(saved[key], per_window[key]), key
        for actual, expected in zip(saved['intervals'], per_window['intervals']):
            assert actual['level'] == expected['level']
            assert torch.equal(actual['covered'], expected['covered'])
            assert torch.equal(actual['width'], expected['width'])
        scores[variant, table] = saved
        interval = next((i for i in report['intervals'] if i['nominal_level'] == 0.9))
        row = {'table': table, 'variant': variant, 'update': update, **{k: report[k] for k in ('crps', 'energy_score', 'variogram_score')}, 'coverage_90': interval['coverage'], 'width_90': interval['mean_width'], 'boundary_fraction': report['diagnostics']['zero_power_fraction'] + report['diagnostics']['full_power_fraction'], 'parameters': cost['parameters'], 'independent_cores': cost['independent_cores'], 'core_executions_per_denoise': cost['core_executions_per_denoise'], 'training_seconds': sum((x['training_seconds'] for x in history)), 'sampling_seconds_7100_trajectories': report['diagnostics']['sampling_seconds'], 'sample_exposures': cost['sample_exposures']}
        rows.append(row)
    target_root = dest / variant
    target_root.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'initialization.json'):
        shutil.copyfile(initial_path / name, target_root / name)
    for name in ('cost.json', 'metrics.json', 'selected/metrics.json', 'rand/metrics.json'):
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path / name, target)
    for name, value in [('training_history.json', history), ('validation_history.json', validation)]:
        (target_root / name).write_text(json.dumps(value, indent=2) + '\n')
    for node in validation:
        for weights in ('raw', 'ema'):
            if weights in node:
                relative = Path('validation') / 'update_{:06d}'.format(node['updates']) / weights / 'metrics.json'
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(panels[variant, node['updates']] / weights / 'metrics.json', target)
    for i, segment in enumerate(segments):
        folder = target_root / 'segments' / str(i)
        folder.mkdir(parents=True, exist_ok=True)
        if len(segments) > 1:
            for name in ('config.json', 'training_history.json', 'validation_history.json'):
                shutil.copyfile(segment / name, folder / name)
        shutil.copyfile(segment.with_suffix('.lineage.json'), folder / 'lineage.json')
    manifests[variant] = {str(segment): {str(p.relative_to(segment)): digest(p) for p in segment.rglob('*') if p.is_file()} for segment in segments}
    checks[variant] = {'endpoint_and_best_independently_rescored_on_cpu': True, 'all_retained_updates_successful': True, 'ema_nodes_match_protocol': True, 'segments': [str(x) for x in segments], 'interrupted_uncommitted_work': 'unknown, at most one update; excluded from retained-trajectory costs' if len(segments) > 1 else None, 'loss_first100_mean': float(np.mean([x['loss'] for x in history[:100]])), 'loss_last100_mean': float(np.mean([x['loss'] for x in history[-100:]])), 'training_gradient_norm_max': max((x['grad_norm'] for x in history))}
with (dest / 'comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
comparisons = {}
daily = []
for table in ('primary', 'best'):
    for left, right in [('loop', 'base'), ('untied', 'base'), ('loop', 'untied')]:
        result = {}
        for key in ('crps', 'energy_score', 'variogram_score'):
            a, b = (scores[left, table][key].numpy(), scores[right, table][key].numpy())
            if a.ndim == 2:
                a, b = (a.mean(1), b.mean(1))
            delta = a - b
            result[key] = {'mean_delta': float(delta.mean()), 'median_delta': float(np.median(delta)), 'days_better': int((delta < 0).sum()), 'days_tied': int((delta == 0).sum())}
            for sample, x, y in zip(reference['validation_sample_ids'], a, b):
                daily.append([table, sample, left, right, key, float(x), float(y), float(x - y)])
        comparisons[table + '/' + left + '_vs_' + right] = result
with (dest / 'daily_comparisons.csv').open('w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['table', 'sample_id', 'variant', 'reference', 'metric', 'score', 'reference_score', 'delta'])
    writer.writerows(daily)
summary = {'tables': rows, 'paired_comparisons': comparisons, 'checks': checks, 'scope': 'seed 42; 71 complete validation days; no test scoring; not multi-seed evidence', 'matched_common_initialization_verified': True, 'matching_final_data_order_and_training_noise_rng_verified': True, 'identical_training_sources_and_data_file_verified': True, 'untied_resume': '2499 + 501 retained successful updates; extra 2499 validation excluded from selection and curves; interrupted uncommitted work not measured'}
(dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(dest / 'local_artifact_sha256.json').write_text(json.dumps(manifests, indent=2) + '\n')
shutil.copyfile(runs['base'] / 'implementation_verification.json', dest / 'base_replay_verification.json')
for name in ('v2_tests_cpu.log', 'v2_tests_cuda.log'):
    shutil.copyfile(Path('outputs') / name, dest / name)
fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
colors = {'base': '#1768AC', 'loop': '#D1495B', 'untied': '#32864B'}
for v in variants:
    h = histories[v]
    valid = vals[v]
    c = colors[v]
    loss = np.array([x['loss'] for x in h])
    axes[0, 0].plot(np.arange(100, 3001), np.convolve(loss, np.ones(100) / 100, 'valid'), color=c, label=v)
    for ax, key in [(axes[0, 1], 'crps'), (axes[0, 2], 'energy_score'), (axes[1, 0], 'variogram_score')]:
        ax.plot([x['updates'] for x in valid], [x['raw'][key] for x in valid], color=c, linestyle='--', alpha=0.5)
        ev = [x for x in valid if 'ema' in x]
        ax.plot([x['updates'] for x in ev], [x['ema'][key] for x in ev], color=c, marker='o', label=v)
    ev = [x for x in valid if 'ema' in x]
    axes[1, 1].plot([x['updates'] for x in ev], [x['ema']['boundary_fraction'] for x in ev], color=c, marker='o', label=v)
    levels = []
    for x in ev:
        report = read(panels[v, x['updates']] / 'ema/metrics.json')
        levels.append(next((i['coverage'] for i in report['intervals'] if i['nominal_level'] == 0.9)))
    axes[1, 2].plot([x['updates'] for x in ev], levels, color=c, marker='o', label=v)
for ax, title in zip(axes.flat, ['Noise MSE (100-update mean)', 'CRPS', 'Energy score', 'Variogram score', 'Delayed EMA boundary fraction', 'Delayed EMA 90% point coverage']):
    ax.set_title(title)
    ax.set_xlabel('Successful updates')
    ax.grid(alpha=0.2)
    ax.legend()
axes[1, 2].axhline(0.9, color='black', linestyle=':', linewidth=1)
fig.suptitle('Protocol v2 / seed 42 / validation only — score panels: solid EMA; dashed raw')
fig.savefig(dest / 'learning_curves.png', dpi=160)
plt.close(fig)
print(json.dumps(summary, indent=2))

# Keep flat update logs reviewable: one complete update record per JSON line.
for history_path in dest.rglob("training_history.json"):
    records = read(history_path)
    history_path.write_text("[\n" + ",\n".join(
        "  " + json.dumps(row) for row in records
    ) + "\n]\n")
