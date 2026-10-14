"""Reproducible single-zone CSV preparation, sample identities and source tracing."""

import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

import torch

from train import SyntheticBatches, WeatherScaler


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return fingerprint({"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                        "bytes": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})


def issue_days(bounds):
    first, last = (datetime.fromisoformat(day) for day in bounds)
    if first.tzinfo or last.tzinfo or first.time() != datetime.min.time() or last.time() != datetime.min.time():
        raise ValueError("Issue boundaries must be naive midnight dates")
    if last < first:
        raise ValueError("Reversed split boundary")
    return [first + timedelta(days=day) for day in range((last - first).days + 1)]


def validate_protocol(protocol):
    required = {
        "schema_version": 1, "horizon": 24,
        "clock": "source_naive_unspecified_timezone",
        "issue_rule": "daily_midnight_leads_1_to_24", "power_unit": "capacity_fraction",
        "missing_rule": "drop_entire_window",
        "nwp_fields": ["U10", "V10", "U100", "V100"],
        "calendar_fields": ["hour_sin", "hour_cos", "year_fraction_sin", "year_fraction_cos"],
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError("Unsupported v1 protocol field: " + key)
    if set(protocol["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Exactly train, validation and test splits are required")
    previous_end = None
    for split in ("train", "validation", "test"):
        days = issue_days(protocol["splits"][split])
        if previous_end is not None and days[0] <= previous_end:
            raise ValueError("Splits must be chronological and have disjoint issue dates")
        previous_end = days[-1]
    selection = protocol["zone_selection"]
    zones = selection["candidate_zones"]
    if not zones or len(set(zones)) != len(zones) or any(type(z) is not int or z < 1 for z in zones):
        raise ValueError("Candidate zones must be distinct positive integers")
    if selection["rule"] != "lowest_eligible_zone" or not 0 < selection["minimum_training_complete_fraction"] <= 1:
        raise ValueError("Invalid zone selection rule")


def read_csv_records(path, numeric_fields, expected_zone=None):
    records = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"ZONEID", "TIMESTAMP", *numeric_fields}.issubset(reader.fieldnames or []):
            raise ValueError("Missing columns in " + str(path))
        for row in reader:
            zone = int(row["ZONEID"])
            time = datetime.strptime(row["TIMESTAMP"], "%Y%m%d %H:%M")
            if time.minute or time.second or (expected_zone is not None and zone != expected_zone):
                raise ValueError("Unexpected zone or nonhourly timestamp in " + str(path))
            key = (zone, time)
            if key in records:
                raise ValueError("Duplicate primary key in {}: {}".format(path, key))
            values, invalid = {}, []
            for field in numeric_fields:
                try:
                    value = float(row[field])
                except (ValueError, TypeError):
                    value = float("nan")
                values[field] = value
                if not math.isfinite(value):
                    invalid.append(field + ":missing_or_nonfinite")
                elif field == "TARGETVAR" and not 0 <= value <= 1:
                    invalid.append(field + ":outside_capacity_range")
            records[key] = {"values": values, "invalid": invalid,
                            "source": {"file": Path(path).name, "line": reader.line_num}}
    return records


def sample_id(zone, issue):
    return "zone{:02d}_{}".format(zone, issue.strftime("%Y%m%dT%H%M%S"))


def calendar_features(times):
    result = []
    for time in times:
        year_start, year_end = datetime(time.year, 1, 1), datetime(time.year + 1, 1, 1)
        hour_phase = 2 * math.pi * time.hour / 24
        year_phase = 2 * math.pi * (time - year_start).total_seconds() / (year_end - year_start).total_seconds()
        result.append([math.sin(hour_phase), math.cos(hour_phase), math.sin(year_phase), math.cos(year_phase)])
    return result


def build_windows(zone, days, targets, weather, protocol):
    tensors = {"y": [], "nwp": [], "calendar": []}
    metadata, dropped = [], []
    for issue in days:
        times = [issue + timedelta(hours=hour) for hour in range(1, 25)]
        reasons, provenance, y, nwp = [], [], [], []
        for time in times:
            key = (zone, time)
            target, forecast = targets.get(key), weather.get(key)
            errors = []
            if target is None:
                errors.append("missing_target_row")
            else:
                errors.extend(e for e in target["invalid"] if e.startswith("TARGETVAR:"))
            if forecast is None:
                errors.append("missing_nwp_row")
            else:
                errors.extend(e for e in forecast["invalid"] if not e.startswith("TARGETVAR:"))
            if errors:
                reasons.append({"valid_time": time.isoformat(), "reasons": errors})
            else:
                y.append([target["values"]["TARGETVAR"]])
                nwp.append([forecast["values"][field] for field in protocol["nwp_fields"]])
                provenance.append({"valid_time": time.isoformat(), "target": target["source"],
                                   "nwp": forecast["source"]})
        identity = {"sample_id": sample_id(zone, issue), "zone_id": zone,
                    "issue_time": issue.isoformat(), "valid_times": [t.isoformat() for t in times]}
        if reasons:
            dropped.append({**identity, "reasons": reasons})
            continue
        metadata.append({**identity, "source_rows": provenance})
        tensors["y"].append(y)
        tensors["nwp"].append(nwp)
        tensors["calendar"].append(calendar_features(times))
    arrays = {key: torch.tensor(values, dtype=torch.float32).reshape(-1, 24, 1 if key == "y" else 4)
              for key, values in tensors.items()}
    return arrays, metadata, dropped


def prepare_dataset(source, protocol):
    validate_protocol(protocol)
    source = Path(source)
    fields = ["TARGETVAR", *protocol["nwp_fields"]]
    candidates, loaded, files = [], {}, {}

    def read(name, numeric_fields, zone=None):
        path = source / name
        before = file_hash(path)
        records = read_csv_records(path, numeric_fields, zone)
        if file_hash(path) != before:
            raise ValueError("Source changed during preparation: " + name)
        files[name] = {"sha256": before, "bytes": path.stat().st_size}
        return records

    days = issue_days(protocol["splits"]["train"])
    for zone in sorted(protocol["zone_selection"]["candidate_zones"]):
        records = read("Train_W_Zone{}.csv".format(zone), fields, zone)
        loaded[zone] = records
        _, valid, _ = build_windows(zone, days, records, records, protocol)
        candidates.append({"zone_id": zone, "training_candidate_windows": len(days),
                           "training_complete_windows": len(valid),
                           "complete_fraction": len(valid) / len(days)})
    eligible = [c["zone_id"] for c in candidates
                if c["complete_fraction"] >= protocol["zone_selection"]["minimum_training_complete_fraction"]]
    if not eligible:
        raise ValueError("No zone meets the predefined training completeness threshold")
    zone = min(eligible)
    targets = dict(loaded[zone])
    weather = dict(loaded[zone])
    predicted = read("TestPred_W_Zone{}.csv".format(zone), protocol["nwp_fields"], zone)
    observed = read("TestTar_W.csv", ["TARGETVAR"])
    for destination, addition in ((weather, predicted), (targets, observed)):
        for key, row in addition.items():
            if key[0] != zone:
                continue
            if key in destination:
                raise ValueError("Duplicate primary key across source files: " + str(key))
            destination[key] = row

    splits, metadata, drops, seen = {}, {}, {}, set()
    for split in ("train", "validation", "test"):
        arrays, meta, dropped = build_windows(zone, issue_days(protocol["splits"][split]),
                                               targets, weather, protocol)
        if not meta:
            raise ValueError("No valid windows in split " + split)
        for item in meta:
            item["split"] = split
            keys = {(zone, time) for time in item["valid_times"]}
            if seen & keys:
                raise ValueError("Target-time overlap across windows/splits")
            seen.update(keys)
        splits[split], metadata[split], drops[split] = arrays, meta, dropped
    scaler = WeatherScaler.fit(splits["train"]["nwp"])
    manifest = {
        "schema_version": 1, "protocol": protocol, "protocol_fingerprint": fingerprint(protocol),
        "selected_zone": zone, "zone_selection_audit": candidates, "source_files": files,
        "counts": {name: {"kept": len(metadata[name]), "dropped": len(drops[name]),
                          "candidate": len(issue_days(protocol["splits"][name]))} for name in splits},
        "sample_ids": {name: [m["sample_id"] for m in metadata[name]] for name in splits},
        "metadata_fingerprint": fingerprint(metadata), "dropped_fingerprint": fingerprint(drops),
        "tensor_fingerprints": {name: {key: tensor_hash(value) for key, value in arrays.items()}
                                for name, arrays in splits.items()},
        "scaler": {key: value.tolist() for key, value in scaler.state_dict().items()},
    }
    manifest["data_fingerprint"] = fingerprint(manifest)
    return {"manifest": manifest, "splits": splits, "metadata": metadata, "dropped": drops}


def validate_package(package):
    manifest = package["manifest"]
    validate_protocol(manifest["protocol"])
    if fingerprint(manifest["protocol"]) != manifest["protocol_fingerprint"]:
        raise ValueError("Protocol fingerprint mismatch")
    unsigned = {k: v for k, v in manifest.items() if k != "data_fingerprint"}
    if fingerprint(unsigned) != manifest["data_fingerprint"]:
        raise ValueError("Data manifest fingerprint mismatch")
    if fingerprint(package["metadata"]) != manifest["metadata_fingerprint"]:
        raise ValueError("Sample metadata fingerprint mismatch")
    if fingerprint(package["dropped"]) != manifest["dropped_fingerprint"]:
        raise ValueError("Dropped-window fingerprint mismatch")
    for section in ("splits", "metadata", "dropped"):
        if set(package[section]) != {"train", "validation", "test"}:
            raise ValueError("Prepared package must contain all three splits")
    for split, arrays in package["splits"].items():
        if set(arrays) != {"y", "nwp", "calendar"}:
            raise ValueError("Prepared tensors must include y, nwp and calendar")
        count = len(package["metadata"][split])
        ids = [row["sample_id"] for row in package["metadata"][split]]
        if ids != manifest["sample_ids"][split] or count != manifest["counts"][split]["kept"]:
            raise ValueError("Manifest sample order/count mismatch")
        for key, values in arrays.items():
            if values.shape != (count, 24, 1 if key == "y" else 4):
                raise ValueError("Prepared tensor shape mismatch")
            if tensor_hash(values) != manifest["tensor_fingerprints"][split][key]:
                raise ValueError("Data tensor fingerprint mismatch: " + split + "/" + key)
    return package


def load_package(path):
    return validate_package(torch.load(path, map_location="cpu", weights_only=True))


def trace_sample(package, source, requested_id):
    manifest = package["manifest"]
    matches = [m for records in package["metadata"].values() for m in records
               if m["sample_id"] == requested_id]
    if len(matches) != 1:
        raise ValueError("Sample identity must match exactly one prepared window")
    sample = matches[0]
    names = {ref[k]["file"] for ref in sample["source_rows"] for k in ("target", "nwp")}
    contents = {}
    for name in names:
        path = Path(source) / name
        if file_hash(path) != manifest["source_files"][name]["sha256"]:
            raise ValueError("Original source fingerprint changed: " + name)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            contents[name] = {reader.line_num: row for row in reader}
    rows = []
    for lead, references in enumerate(sample["source_rows"], 1):
        raw = {kind: contents[references[kind]["file"]][references[kind]["line"]]
               for kind in ("target", "nwp")}
        rows.append({"lead_hour": lead, "valid_time": references["valid_time"],
                     "source_rows": references, "raw_records": raw})
    return {"sample": sample, "data_fingerprint": manifest["data_fingerprint"], "rows": rows}


class RealBatches(SyntheticBatches):
    """Reuse the existing shuffled iterator while binding it to the data identity."""
    def __init__(self, package, batch_size, seed=42):
        super().__init__(package["splits"]["train"], batch_size, seed)
        self.data_fingerprint = package["manifest"]["data_fingerprint"]

    def state_dict(self):
        return {**super().state_dict(), "data_fingerprint": self.data_fingerprint}

    def load_state_dict(self, state):
        if state.get("data_fingerprint") != self.data_fingerprint:
            raise ValueError("Batch iterator data fingerprint mismatch")
        super().load_state_dict(state)
