"""Build joint windows and fingerprint splits from the fixed three-year calendar."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .collect import GRID, save
from .multiyear import ARCHIVE, calendar, daily_batch
from .prepare import MissingWeather, assemble, cached, point_values


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    save(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def freeze(root, pilot):
    months = json.loads((ARCHIVE / "coverage.json").read_bytes())
    baseline = json.loads((ARCHIVE / "acceptance.json").read_bytes())["days"]
    splits = {s: [] for s in ("train", "validation", "test")}
    ledger = {}
    month_id = None
    for day, rule in calendar().items():
        key = str(day)
        try:
            if not rule["target_complete"] or not rule["run_listed"]:
                raise ValueError(rule["reason"] or "coverage unavailable")
            if key in baseline:
                folder = pilot / key
                entry = baseline[key]
                for source in entry["sources"]:
                    if digest(folder / source["file"]) != source["sha256"]:
                        raise ValueError("Pilot source changed")
            else:
                folder, results, weather = daily_batch(root, day, False)
                if key[:7] != month_id:
                    expected = months[key[:7]]["sources"][1]
                    meter_folder = pilot / "months" / key[:7]
                    meter, receipt = cached(meter_folder, "meter", expected["url"])
                    if receipt != expected:
                        raise ValueError("Monthly receipt differs from frozen inventory")
                    month_id = key[:7]
                wind = []
                links = []
                for result, point in zip(results, GRID, strict=True):
                    values, (version, snapshot), link = point_values(result, day, point)
                    wind.append(values)
                    links.append(link)
                meter_source = dict(file=os.path.relpath(meter_folder / "meter.json", folder),
                                    sha256=receipt["sha256"], bytes=receipt["bytes"], url=receipt["url"])
                sources = [dict(file="weather.json", sha256=weather["sha256"], bytes=weather["bytes"],
                                canonical=links)]
                entry = assemble(folder, day, wind, version, snapshot, sources, meter, meter_source)
            joint = folder / "joint.json"
            if digest(joint) != entry["sample_sha256"]:
                raise ValueError("Joint sample fingerprint mismatch")
            row = dict(path=os.path.relpath(joint, root), sha256=entry["sample_sha256"],
                       day=key, snapshot=entry["snapshot"], version=entry["version"])
            ledger[key] = dict(**row, status="accepted", split=rule["split"],
                               boundary_holdout=rule["boundary_holdout"], sources=entry["sources"])
            if not rule["boundary_holdout"]:
                splits[rule["split"]].append(row)
        except MissingWeather as exc:
            ledger[key] = dict(status="excluded_missing_weather", reason=str(exc), split=rule["split"],
                               weather_sha256=weather["sha256"])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            ledger[key] = dict(status="pending" if isinstance(exc, FileNotFoundError) else "failed",
                               reason=str(exc), split=rule["split"])
    receipts = []
    errors = []
    transport_failures = 0
    for key, entry in ledger.items():
        attempts = []
        for folder in sorted((root / key).glob("attempt-*")):
            weather_path = folder / "weather.receipt.json"
            day_errors = folder / "errors.jsonl"
            failures = [json.loads(line) for line in day_errors.read_text().splitlines()] if day_errors.exists() else []
            errors.extend(failures)
            attempt = dict(directory=folder.name, http_errors=[dict(status=r["status"], bytes=r["bytes"],
                           utc=r["observed_utc"], retry_after=r["retry_after"]) for r in failures])
            transport = folder / "transport.jsonl"
            if transport.exists():
                events = [json.loads(line) for line in transport.read_text().splitlines()]
                attempt["transport_failures"] = events
                transport_failures += len(events)
            if weather_path.exists():
                receipt = json.loads(weather_path.read_bytes())
                receipts.append(receipt)
                attempt.update(utc=receipt["observed_utc"], bytes=receipt["bytes"], sha256=receipt["sha256"],
                               receipt_sha256=digest(weather_path))
            attempts.append(attempt)
        entry["weather_attempts"] = attempts
    write(root / "ledger.json", ledger)
    files = {}
    for name, rows in splits.items():
        path = root / f"{name}.json"
        write(path, rows)
        files[name] = dict(days=len(rows), sha256=digest(path))
    summary = dict(splits=files, ledger_sha256=digest(root / "ledger.json"),
                   dataset_train_ready=all(r["status"] in ("accepted", "excluded_missing_weather") for r in ledger.values()) and all(splits.values()),
                   joint_accepted=sum(r["status"] == "accepted" for r in ledger.values()),
                   cached_pilot_days=len(baseline), cached_months=len(months),
                   weather_http_responses=len(receipts)+len(errors), weather_http_errors=len(errors),
                   recorded_transport_failures=transport_failures,
                   recorded_request_attempts=len(receipts)+len(errors)+transport_failures,
                   weather_work_units=sum(int(r["work_units"] or 0) for r in receipts),
                   weather_response_bytes=sum(r["bytes"] for r in receipts+errors))
    write(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("pilot", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.root, args.pilot), indent=2))
