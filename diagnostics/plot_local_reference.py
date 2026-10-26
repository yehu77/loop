"""Plot saved R02 summaries; no training, sampling or checkpoint selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(results, output):
    if output.exists():
        raise ValueError("Figure exists; choose a new path")
    summary = json.loads((results / "summary.json").read_text())
    primary = [r for r in summary["rows"] if r["table"] == "primary"]
    seeds = summary["seeds"]
    models = ("base", "local", "loop", "untied")
    colors = dict(base="#1874b4", local="#dd7926", loop="#32865e", untied="#8263a5")
    metrics = (("crps", "Hourly CRPS"), ("energy_score", "Energy score"),
               ("variogram_score", "Variogram score"), ("increment_1h_crps", "1h increment CRPS"),
               ("increment_3h_crps", "3h increment CRPS"), ("daily_mean_crps", "Daily mean power CRPS"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (metric, title) in zip(axes.flat, metrics):
        for j, seed in enumerate(seeds):
            values = [next(r[metric] for r in primary if r["seed"] == seed and r["model"] == m) for m in models]
            axis.plot(range(4), values, color="#b5b5b5", linewidth=0.8, alpha=0.7)
            for i, model in enumerate(models):
                axis.scatter(i, values[i], color=colors[model], marker=("o", "s", "^")[j], s=45)
        axis.set_xticks(range(4), [m.upper() for m in models])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("R02: delayed EMA at update 3000 | dots: training seeds 42, 43, 44\n"
                 "Lines connect the same seed; all scores lower is better; validation only")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results, args.output)


if __name__ == "__main__":
    main()
