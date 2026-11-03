"""Plot the completed single-seed history screening without changing scores."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(report, output):
    if output.exists():
        raise FileExistsError("Refusing to overwrite figure")
    summary = json.loads((report / "summary.json").read_text())
    rows = [r for r in summary["lead_groups"] if r["table"] == "primary"]
    leads = ["1-6", "7-12", "13-24"]
    groups = ["H0", "H1", "H24"]
    colors = ["#426d91", "#d97924", "#208477"]
    indexed = {(r["group"], r["lead_group"]): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for group, color in zip(groups, colors):
        axes[0, 0].plot(leads, [indexed[group, lead]["crps"] for lead in leads],
                        marker="o", label=group, color=color)
        axes[1, 0].plot(leads, [100 * indexed[group, lead]["coverage_90"] for lead in leads],
                        marker="o", label=group, color=color)
        axes[1, 1].plot(leads, [indexed[group, lead]["width_90"] for lead in leads],
                        marker="o", label=group, color=color)
    for group, color, offset in (("H1", colors[1], -.17), ("H24", colors[2], .17)):
        values = [100 * (indexed[group, lead]["crps"] / indexed["H0", lead]["crps"] - 1) for lead in leads]
        axes[0, 1].bar([i + offset for i in range(3)], values, width=.32, label=group + " vs H0", color=color)
    axes[0, 1].set_xticks(range(3), leads)
    axes[0, 1].axhline(0, color="black", linewidth=.8)
    axes[1, 0].axhline(90, linestyle="--", color="black", linewidth=1, label="Nominal 90%")
    titles = ["Lead-group CRPS (lower is better)", "CRPS change vs H0 (%)",
              "90% point coverage (%)", "90% interval width (capacity fraction)"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Forecast lead (hours)")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("History information: seed 42, EMA at update 3000\n"
                 "57 common validation days; single-seed development results", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.output)
