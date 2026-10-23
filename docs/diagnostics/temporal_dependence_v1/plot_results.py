"""Plot all five permutation effects per trained model from R01 CSV output."""
import argparse
import csv
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/loop-pilot-mpl')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Figure already exists')
    with (args.results / 'per_permutation.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    metrics = [('energy_score', 'Energy score'), ('variogram_score', 'Variogram score'),
               ('increment_1h_crps', '1-hour increment CRPS'),
               ('increment_3h_crps', '3-hour increment CRPS'),
               ('daily_mean_crps', 'Daily mean power CRPS')]
    colors = {'base': '#1768AC', 'loop': '#D1495B', 'untied': '#32864B'}
    models = [(v, s) for v in colors for s in (42, 43, 44)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for axis, (metric, title) in zip(axes.flat, metrics):
        for position, (variant, seed) in enumerate(models):
            selected = sorted([r for r in rows if r['metric'] == metric and r['variant'] == variant
                               and int(r['training_seed']) == seed], key=lambda r: int(r['permutation_seed']))
            assert [int(r['permutation_seed']) for r in selected] == list(range(1101, 1106))
            values = [float(r['delta']) for r in selected]
            axis.scatter([position + (i-2)*0.055 for i in range(5)], values,
                         color=colors[variant], s=18, alpha=0.65)
            axis.scatter([position], [statistics.mean(values)], color='black', marker='_', s=90)
        axis.axhline(0, color='#444444', linewidth=1)
        axis.set_ylim(bottom=min(0, axis.get_ylim()[0]))
        axis.set_xticks(range(9), [f'{v.upper()}\n{s}' for v, s in models], fontsize=7)
        axis.set_ylabel('Shuffled minus original')
        axis.set_title(title)
        axis.grid(axis='y', alpha=0.2)
    axes[1, 2].axis('off')
    axes[1, 2].text(0.03, 0.85, 'Each colored dot: one fixed permutation\n'
                   'Black mark: mean of five permutations\n\n'
                   'Positive: original temporal pairing scores better\n'
                   'Axes have different metric scales\n\n'
                   '9 fixed EMA endpoints; 71 validation days\n'
                   'No new training, sampling or test scoring\n'
                   'Permutation repeats are not training replications',
                   va='top', fontsize=11, linespacing=1.6)
    fig.suptitle('R01: temporal pairing destruction with exactly preserved hourly sample sets')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
