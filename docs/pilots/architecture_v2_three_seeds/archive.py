"""Copy small audited artifacts into a new archive, excluding tensor/checkpoint files."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Archive exists; choose a new destination")
    source_root, output_root = args.summary.resolve(), args.output.resolve()
    if source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Archive and source summary must not overlap")
    evidence = json.loads((args.summary / "evidence.json").read_text())
    # summarize_runs.py produces only JSON/CSV files. Do not copy arbitrary extras.
    for source in args.summary.rglob("*"):
        if source.is_file() and source.suffix not in (".json", ".csv"):
            raise ValueError(f"Unexpected summary artifact: {source}")
    shutil.copytree(args.summary, args.output)
    for run in evidence:
        target = args.output / f"s{run['seed']}_{run['variant']}"
        sources = [Path(path) for path in run["segments"]]
        for relative in ("config.json", "initialization.json"):
            shutil.copyfile(sources[0] / relative, target / relative)
        for relative in ("cost.json", "metrics.json", "selected/metrics.json", "rand/metrics.json"):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[-1] / relative, destination)
        # Later segments win a repeated common-node evaluation, as in the audit.
        for source in sources:
            for metric_path in (source / "validation").glob("update_*/*/metrics.json"):
                update = int(metric_path.parents[1].name.split("_")[1])
                if update % run["config"]["evaluation"]["validate_every"]:
                    continue
                destination = target / metric_path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(metric_path, destination)
    print(args.output)


if __name__ == "__main__":
    main()
