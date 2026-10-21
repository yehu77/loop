"""Plot raw/EMA curves from an audited summary directory; no training is performed."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/loop-pilot-mpl")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Figure exists; choose a new destination")
    summary = json.loads((args.summary / "summary.json").read_text())
    seeds = summary["seeds"]
    metrics = (("crps", "CRPS"), ("energy_score", "ES"), ("variogram_score", "VS"))
    colors = {"base": "#1768AC", "loop": "#D1495B", "untied": "#32864B"}
    figure, axes = plt.subplots(len(seeds), 3, figsize=(12, 3 * len(seeds)),
                                squeeze=False, layout="constrained")
    for row, seed in enumerate(seeds):
        for variant, color in colors.items():
            path = args.summary / f"s{seed}_{variant}" / "validation_history.json"
            history = json.loads(path.read_text())
            # Show all post-start common nodes; earlier diagnostics remain in JSON.
            history = [node for node in history if "ema" in node]
            updates = [node["updates"] for node in history]
            for column, (metric, label) in enumerate(metrics):
                axis = axes[row, column]
                axis.plot(updates, [n["ema"][metric] for n in history], color=color,
                          marker="o", markersize=3, label=variant.upper())
                axis.plot(updates, [n["raw"][metric] for n in history], color=color,
                          linestyle="--", alpha=0.45)
                axis.set_title(f"Seed {seed}: {label}")
                axis.set_xlabel("Successful updates")
                axis.grid(alpha=0.2)
        axes[row, 0].legend(fontsize=8)
    figure.suptitle("Frozen v2 / validation only: delayed EMA solid; raw dashed\n"
                    "Same updates and sample exposure, different computation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
