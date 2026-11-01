"""Plot R03 saved per-noise-level denoising results; no model execution."""
import argparse
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(results, output):
    if output.exists():
        raise ValueError("Figure exists; choose a new path")
    rows = list(csv.DictReader((results / "per_run.csv").open()))
    times = [50, 200, 400, 700, 950]
    colors = {0.: "#d36b28", .5: "#7753a6", 1.: "#197b73"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ri, variant in enumerate(("loop", "untied")):
        for ci, (metric, label) in enumerate((("noise_mse", "Final noise MSE"),
                                             ("delta_mse", "MSE minus normal"),
                                             ("output_change_rms", "Output-change RMS"))):
            ax = axes[ri, ci]
            for lam in ([0., .5, 1.] if ci == 0 else [0., .5]):
                group = [r for r in rows if r["variant"] == variant and float(r["lambda"]) == lam]
                means = [statistics.mean(float(r[metric]) for r in group if int(r["t"]) == t) for t in times]
                ax.plot(times, means, color=colors[lam], marker="o", label=f"lambda={lam:g}")
                for seed, marker in zip((42, 43, 44), ("o", "s", "^")):
                    values = [float(next(r for r in group if int(r["training_seed"]) == seed and int(r["t"]) == t)[metric]) for t in times]
                    ax.scatter(times, values, color=colors[lam], marker=marker, s=15, alpha=.45)
            if ci == 0:
                base = [r for r in rows if r["variant"] == "base"]
                ax.plot(times, [statistics.mean(float(r[metric]) for r in base if int(r["t"]) == t) for t in times],
                        color="black", linestyle="--", label="BASE normal")
            # Absolute deltas remain linear: zero/negative values would remain visible.
            if ci == 0:
                ax.set_yscale("log")
            else:
                ax.axhline(0, color="gray", linewidth=.6)
            ax.set_title(variant.upper() + ": " + label)
            ax.set_xticks(times)
            ax.set_xlabel("Diffusion noise level t")
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
    fig.suptitle("R03: fixed-input denoiser diagnostic, validation only\n"
                 "Lines: three-seed means; small markers: seeds 42/43/44; MSE is not a scenario score")
    fig.tight_layout(rect=(0, 0, 1, .92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results, args.output)
