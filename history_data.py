"""Identity-bound rolling power history, kept separate from frozen v1 packages."""

from datetime import datetime, timedelta
from pathlib import Path

import torch

from data import (file_hash, fingerprint, read_csv_records, tensor_hash,
                  trace_sample, validate_package)
from train import WeatherScaler


def validate_history_protocol(protocol):
    expected = {
        "schema_version": 1, "history_hours": 24, "observation_delay_hours": 1,
        "availability_status": "explicit_benchmark_assumption_not_verified_operational_latency",
        "history_fields": ["TARGETVAR"], "history_order": "oldest_to_latest",
        "common_window_rule": "parent_valid_future_and_all_24_history_hours_valid",
        "rolling_policy": "observed_power_before_each_issue_including_earlier_validation_days",
        "history_missing_rule": "drop_common_window_no_imputation",
        "power_transform": "2*y-1_using_capacity_fraction",
        "weather_scaler": "refit_on_common_training_windows_only",
        "groups": {"H0": [], "H1": [23], "H24": "all_24"},
        "test_policy": "prepare_and_audit_only_no_scoring",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError("Unsupported history protocol: " + key)
    if type(protocol.get("zone_id")) is not int or protocol["zone_id"] < 1:
        raise ValueError("Invalid history zone")
    if not isinstance(protocol.get("parent_data_fingerprint"), str):
        raise ValueError("Missing parent data fingerprint")


def history_times(issue):
    """One-hour delay: previous midnight through previous 23:00, in source clock."""
    return [issue - timedelta(hours=24 - i) for i in range(24)]


def visible_history(history, group):
    """Mask before encoding; an invisible raw value must never reach the model."""
    if group not in ("H0", "H1", "H24"):
        raise ValueError("History group must be H0, H1 or H24")
    if history.ndim != 3 or history.shape[1:] != (24, 1):
        raise ValueError("history must be [B, 24, 1]")
    mask = torch.zeros(history.shape[:2], dtype=torch.bool, device=history.device)
    if group == "H1":
        mask[:, -1] = True
    elif group == "H24":
        mask[:] = True
    # torch.where, not multiplication: hidden NaN/Inf must not contaminate values.
    values = torch.where(mask.unsqueeze(-1), 2 * history - 1, torch.zeros_like(history))
    return values, mask


def read_history_targets(source, parent):
    """Only source power columns; never require historical NWP completeness."""
    zone = parent["manifest"]["selected_zone"]
    names = ["Train_W_Zone{}.csv".format(zone), "TestTar_W.csv"]
    records = {}
    for name in names:
        path = Path(source) / name
        expected = parent["manifest"]["source_files"][name]["sha256"]
        if file_hash(path) != expected:
            raise ValueError("History source fingerprint changed: " + name)
        rows = read_csv_records(path, ["TARGETVAR"], zone if name.startswith("Train") else None)
        if file_hash(path) != expected:
            raise ValueError("History source changed during read: " + name)
        for key, row in rows.items():
            if key[0] != zone:
                continue
            if key in records:
                raise ValueError("Duplicate history primary key across files")
            records[key] = row
    return records


def prepare_history(parent, source, protocol):
    validate_package(parent)
    validate_history_protocol(protocol)
    original = parent["manifest"]
    if (protocol["parent_data_fingerprint"] != original["data_fingerprint"]
            or protocol["zone_id"] != original["selected_zone"]):
        raise ValueError("History protocol and parent dataset identity differ")
    records = read_history_targets(source, parent)
    splits, metadata, dropped, indices, counts = {}, {}, {}, {}, {}
    for split in ("train", "validation", "test"):
        kept, meta, rejected, histories = [], [], [], []
        for i, sample in enumerate(parent["metadata"][split]):
            times = history_times(datetime.fromisoformat(sample["issue_time"]))
            errors, values, provenance = [], [], []
            for time in times:
                row = records.get((protocol["zone_id"], time))
                reasons = ["missing_history_power_row"] if row is None else row["invalid"]
                if reasons:
                    errors.append({"valid_time": time.isoformat(), "reasons": reasons})
                else:
                    values.append([row["values"]["TARGETVAR"]])
                    provenance.append({"valid_time": time.isoformat(), "target": row["source"]})
            identity = {k: sample[k] for k in ("sample_id", "zone_id", "issue_time", "split")}
            if errors:
                rejected.append({**identity, "history_reasons": errors})
            else:
                kept.append(i)
                histories.append(values)
                meta.append({**sample, "history_valid_times": [t.isoformat() for t in times],
                             "history_source_rows": provenance,
                             "history_available_by": sample["issue_time"]})
        if not kept:
            raise ValueError("No common windows in " + split)
        arrays = {key: tensor[kept].clone() for key, tensor in parent["splits"][split].items()}
        arrays["history_power"] = torch.tensor(histories, dtype=torch.float32)
        splits[split], metadata[split], dropped[split], indices[split] = arrays, meta, rejected, kept
        counts[split] = {"candidate": original["counts"][split]["candidate"],
                         "parent_future_kept": len(parent["metadata"][split]),
                         "parent_future_dropped": len(parent["dropped"][split]),
                         "additional_history_dropped": len(rejected), "common_kept": len(kept)}
    scaler = WeatherScaler.fit(splits["train"]["nwp"])
    manifest = {
        "schema": "rolling_history_v1", "protocol": protocol,
        "protocol_fingerprint": fingerprint(protocol),
        "parent_data_fingerprint": original["data_fingerprint"],
        "parent_protocol": original["protocol"], "source_files": original["source_files"],
        "counts": counts, "parent_indices": indices,
        "sample_ids": {s: [r["sample_id"] for r in metadata[s]] for s in splits},
        "metadata_fingerprint": fingerprint(metadata), "dropped_fingerprint": fingerprint(dropped),
        "parent_dropped_fingerprint": fingerprint(parent["dropped"]),
        "tensor_fingerprints": {s: {k: tensor_hash(v) for k, v in arrays.items()}
                                for s, arrays in splits.items()},
        "scaler": {k: v.tolist() for k, v in scaler.state_dict().items()},
    }
    manifest["data_fingerprint"] = fingerprint(manifest)
    return {"manifest": manifest, "splits": splits, "metadata": metadata,
            "dropped": dropped, "parent_dropped": parent["dropped"]}


def validate_history_package(package):
    m = package["manifest"]
    if m.get("schema") != "rolling_history_v1":
        raise ValueError("Unsupported history package")
    validate_history_protocol(m["protocol"])
    if (fingerprint({k: v for k, v in m.items() if k != "data_fingerprint"}) != m["data_fingerprint"]
            or fingerprint(m["protocol"]) != m["protocol_fingerprint"]):
        raise ValueError("History manifest fingerprint mismatch")
    for section in ("metadata", "dropped", "parent_dropped"):
        if fingerprint(package[section]) != m[section + "_fingerprint"]:
            raise ValueError("History " + section + " fingerprint mismatch")
    if set(package["splits"]) != {"train", "validation", "test"}:
        raise ValueError("History package requires all three splits")
    for split, arrays in package["splits"].items():
        meta = package["metadata"][split]
        if (len(meta) != m["counts"][split]["common_kept"]
                or [r["sample_id"] for r in meta] != m["sample_ids"][split]):
            raise ValueError("History sample identity mismatch")
        if set(arrays) != {"y", "nwp", "calendar", "history_power"}:
            raise ValueError("History tensor fields mismatch")
        for key, values in arrays.items():
            shape = (len(meta), 24, 1 if key in ("y", "history_power") else 4)
            if (values.shape != shape or not torch.isfinite(values).all()
                    or tensor_hash(values) != m["tensor_fingerprints"][split][key]):
                raise ValueError("History tensor fingerprint/shape mismatch: " + key)
        for r in meta:
            expected = [t.isoformat() for t in history_times(datetime.fromisoformat(r["issue_time"]))]
            if r["history_valid_times"] != expected or r["history_available_by"] != r["issue_time"]:
                raise ValueError("History temporal alignment mismatch")
    return package


def load_history_package(path):
    return validate_history_package(torch.load(path, map_location="cpu", weights_only=True))


def trace_history_sample(package, parent, source, sample_id):
    validate_history_package(package)
    validate_package(parent)
    if parent["manifest"]["data_fingerprint"] != package["manifest"]["parent_data_fingerprint"]:
        raise ValueError("Trace parent identity mismatch")
    samples = [r for rows in package["metadata"].values() for r in rows if r["sample_id"] == sample_id]
    if len(samples) != 1:
        raise ValueError("Trace ID must be a retained common window")
    sample = samples[0]
    future = trace_sample(parent, source, sample_id)
    records = read_history_targets(source, parent)
    history = []
    for i, reference in enumerate(sample["history_source_rows"]):
        row = records[(sample["zone_id"], datetime.fromisoformat(reference["valid_time"]))]
        if row["source"] != reference["target"]:
            raise ValueError("History trace row mismatch")
        history.append({"slot": i, "valid_time": reference["valid_time"],
                        "source": row["source"], "power": row["values"]["TARGETVAR"]})
    return {"sample_id": sample_id, "issue_time": sample["issue_time"],
            "data_fingerprint": package["manifest"]["data_fingerprint"],
            "history_rows": history, "future_rows": future["rows"]}
