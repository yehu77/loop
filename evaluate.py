"""Identity-checked real-data evaluation around the unchanged metric formulas."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path

import torch

from data import sample_id
from metrics import (energy_score, ensemble_crps, evaluate_scenarios,
                     interval_statistics, variogram_score)


def identified_artifact(values, manifest, samples):
    protocol = manifest["protocol"]
    return {"schema_version": 1, "values": values.detach().cpu(),
            "data_fingerprint": manifest["data_fingerprint"],
            "protocol_fingerprint": manifest["protocol_fingerprint"],
            "power_unit": protocol["power_unit"], "clock": protocol["clock"],
            "samples": deepcopy(samples)}


def _index(artifact):
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
        raise ValueError("Evaluation requires an identified artifact, not a bare tensor")
    if not isinstance(artifact.get("values"), torch.Tensor):
        raise ValueError("Artifact values must be a tensor")
    if artifact["values"].ndim not in (3, 4) or artifact["values"].shape[-2:] != (24, 1):
        raise ValueError("Artifact tensor must represent the declared 24 hourly power values")
    records = artifact["samples"]
    if not records or len(records) != len(artifact["values"]):
        raise ValueError("Sample metadata and tensor row counts differ")
    index = {}
    for row, metadata in enumerate(records):
        key = metadata["sample_id"]
        if key in index:
            raise ValueError("Duplicate sample_id: " + key)
        issue = datetime.fromisoformat(metadata["issue_time"])
        expected_times = [(issue + timedelta(hours=h)).isoformat() for h in range(1, 25)]
        if (issue.tzinfo is not None or issue.time() != datetime.min.time()
                or metadata["valid_times"] != expected_times
                or key != sample_id(metadata["zone_id"], issue)):
            raise ValueError("Invalid sample clock or identity: " + key)
        if metadata["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid sample split")
        index[key] = row
    if len({m["zone_id"] for m in records}) != 1 or len({m["split"] for m in records}) != 1:
        raise ValueError("Evaluation must contain one zone and one split")
    return index


def align_artifacts(prediction, target):
    predicted_index, target_index = _index(prediction), _index(target)
    for key in ("data_fingerprint", "protocol_fingerprint", "power_unit", "clock"):
        if not prediction.get(key) or prediction[key] != target.get(key):
            raise ValueError("Artifact identity mismatch: " + key)
    if target["power_unit"] != "capacity_fraction" or target["clock"] != "source_naive_unspecified_timezone":
        raise ValueError("Unsupported power units or clock")
    if set(predicted_index) != set(target_index):
        raise ValueError("Missing or unexpected sample_ids; evaluation sets must match exactly")
    order = []
    for key, target_row in target_index.items():
        predicted_row = predicted_index[key]
        for field in ("zone_id", "issue_time", "valid_times", "split"):
            if prediction["samples"][predicted_row][field] != target["samples"][target_row][field]:
                raise ValueError("Sample metadata mismatch: " + key + "/" + field)
        order.append(predicted_row)
    return prediction["values"][order].cpu(), target["values"].cpu()


def evaluate_artifacts(prediction, target):
    scenarios, targets = align_artifacts(prediction, target)
    report = evaluate_scenarios(scenarios, targets)
    report["data_fingerprint"] = target["data_fingerprint"]
    report["sample_ids"] = [m["sample_id"] for m in target["samples"]]
    report["split"] = target["samples"][0]["split"]
    report["conventions"]["alignment"] = "sample_id joined; zone, issue/valid times, split, unit and fingerprints checked"
    per_window = {"sample_ids": report["sample_ids"], "data_fingerprint": target["data_fingerprint"],
                  "crps": ensemble_crps(scenarios, targets),
                  "energy_score": energy_score(scenarios, targets),
                  "variogram_score": variogram_score(scenarios, targets), "intervals": []}
    for level in (0.8, 0.9):
        stats = interval_statistics(scenarios, targets, level)
        per_window["intervals"].append({"level": level, "covered": stats["covered"],
                                         "width": stats["width"]})
    return report, per_window


def write_evaluation(output, prediction, target, diagnostics=None):
    report, per_window = evaluate_artifacts(prediction, target)
    report["diagnostics"] = diagnostics or {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(prediction, output / "scenarios.pt")
    torch.save(target, output / "targets.pt")
    torch.save(per_window, output / "scores_per_window.pt")
    metadata = {name: {key: value for key, value in artifact.items() if key != "values"}
                for name, artifact in (("scenarios", prediction), ("targets", target))}
    (output / "sample_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction = torch.load(args.scenarios, map_location="cpu", weights_only=True)
    target = torch.load(args.targets, map_location="cpu", weights_only=True)
    write_evaluation(args.output, prediction, target)
    print("Report: " + str((args.output / "metrics.json").resolve()))


if __name__ == "__main__":
    main()
