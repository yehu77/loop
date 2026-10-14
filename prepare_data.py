"""Prepare a versioned real-data package, or trace one sample to original CSV rows."""

import argparse
import json
from pathlib import Path
import tempfile

import torch

from data import load_package, prepare_dataset, trace_sample


def write_package(package, source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing data package: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        torch.save(package, staging / "data.pt")
        artifacts = {"data_manifest.json": package["manifest"],
                     "sample_metadata.json": package["metadata"],
                     "dropped_windows.json": package["dropped"]}
        first_id = package["metadata"]["train"][0]["sample_id"]
        artifacts["trace_sample.json"] = trace_sample(package, source, first_id)
        for name, content in artifacts.items():
            (staging / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data_v1.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--package", type=Path, help="Existing data.pt for tracing")
    parser.add_argument("--sample-id", help="Trace a prepared sample to its 24 source records")
    args = parser.parse_args()
    if args.sample_id:
        if args.package is None:
            parser.error("--sample-id requires --package")
        print(json.dumps(trace_sample(load_package(args.package), args.source, args.sample_id), indent=2))
    else:
        if args.output is None:
            parser.error("Preparation requires --output")
        protocol = json.loads(args.config.read_text())
        package = prepare_dataset(args.source, protocol)
        write_package(package, args.source, args.output)
        print(json.dumps({"selected_zone": package["manifest"]["selected_zone"],
                          "counts": package["manifest"]["counts"],
                          "data_fingerprint": package["manifest"]["data_fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
