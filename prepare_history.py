"""Prepare the common H0/H1/H24 data package; does not train or score any model."""

import argparse
import json
from pathlib import Path
import tempfile

import torch

from data import file_hash, load_package
from history_data import prepare_history, trace_history_sample, validate_history_package


def write_history_package(package, parent, source, output):
    validate_history_package(package)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite history data: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "history"
        staging.mkdir()
        torch.save(package, staging / "data.pt")
        artifacts = {"data_manifest": package["manifest"], "sample_metadata": package["metadata"],
                     "dropped_history_windows": package["dropped"],
                     "dropped_future_windows": package["parent_dropped"],
                     "trace_sample": trace_history_sample(package, parent, source,
                                                          package["metadata"]["train"][0]["sample_id"])}
        for name, value in artifacts.items():
            (staging / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/history_information_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = file_hash(args.parent_data)
    parent = load_package(args.parent_data)
    package = prepare_history(parent, args.source, json.loads(args.config.read_text()))
    if before != file_hash(args.parent_data):
        raise ValueError("Parent package changed during history preparation")
    write_history_package(package, parent, args.source, args.output)
    print(json.dumps({"counts": package["manifest"]["counts"],
                      "data_fingerprint": package["manifest"]["data_fingerprint"],
                      "parent_sha256": before}, indent=2))


if __name__ == "__main__":
    main()
