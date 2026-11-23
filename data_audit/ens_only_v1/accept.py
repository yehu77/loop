"""Rebuild the ENS-only offline acceptance and coverage inventory."""
import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path

from data_audit.audit import DAY, FARM, HERE, UTC, digest, hourly
from .probe import DAYS

ROOT = Path(__file__).parent
HOUR = timedelta(hours=1)
LEADS = (24, 30, 36, 42, 48)
ASSUMPTION = "operational_ens_delivered_by_noon_utc"


def verified_json(path, sha):
    body = path.read_bytes()
    if digest(body) != sha:
        raise ValueError(f"Source fingerprint mismatch: {path.name}")
    return json.loads(body)


def evaluate(day, protocol, ens, meter):
    origin = datetime.combine(date.fromisoformat(day), time(), tzinfo=UTC)
    issue = origin + 12 * HOUR
    start = origin + DAY
    errors = {}
    declared = (protocol.get("name") == "ens_only_offline_v1" and
                protocol.get("farm") == FARM and protocol.get("access_assumption") == ASSUMPTION and
                protocol.get("cycle_hour_utc") == 0 and protocol.get("issue_hour_utc") == 12 and
                protocol.get("grid_use") == "nearby_native_points" and
                protocol.get("reference_site") == {"repd_id": 4261, "latitude": 57.446611,
                                                   "longitude": -3.475393})
    if not declared:
        errors["protocol"] = "Missing or incompatible offline declaration"
    target = []
    try:
        target = hourly(meter, start)
        for n, hour in enumerate(target):
            hour.update(start_lead_from_initialization_h=24 + n, start_lead_from_issue_h=12 + n)
            if hour["quality"] != "complete":
                raise ValueError("Incomplete target hour")
            if any(not meter[i].get("settlementRunType") for i in hour["response_indices"]):
                raise ValueError("Settlement revision missing")
    except (ValueError, TypeError, KeyError) as error:
        errors["target"] = str(error)

    ens = ens or {}
    rows = ens.get("rows") or []
    expected = {(m, lat, lon, lead) for m in range(1, 51) for lat in (57, 57.5)
                for lon in (-4, -3.5) for lead in LEADS}
    try:
        keys = {(r["member"], r["lat"], r["lon"], r["lead_hours"]) for r in rows}
        if (ens.get("product") != "ENS" or ens.get("wind_units") != "m s**-1" or
                keys != expected or len(rows) != len(expected) or
                not all(math.isfinite(r[k]) for r in rows for k in ("u10", "v10"))):
            raise ValueError("Missing or invalid native ensemble panel")
    except (KeyError, TypeError, ValueError) as error:
        errors["ens"] = str(error)

    try:
        if protocol.get("issue_hour_utc") != 12 or protocol.get("cycle_hour_utc") != 0:
            raise ValueError("Missing or incompatible UTC issue/cycle schedule")
        if ens.get("origin") is None or not rows or datetime.fromisoformat(ens["origin"]) != origin:
            raise ValueError("Missing or mismatched cycle")
        if ens.get("raw_time_units") != "hours since 1900-01-01 00:00:00.0":
            raise ValueError("Missing native ENS time reference")
        epoch = datetime(1900, 1, 1, tzinfo=UTC)
        for row in rows:
            valid = datetime.fromisoformat(row["valid_time"])
            if valid != epoch + row["raw_time"] * HOUR:
                raise ValueError("Raw and decoded weather times disagree")
            if valid != origin + row["lead_hours"] * HOUR or not start <= valid <= start + DAY:
                raise ValueError("Inconsistent weather times")
        if ens.get("available_time") is not None:
            raise ValueError("This protocol audits a source with unknown actual delivery")
    except (ValueError, TypeError, KeyError) as error:
        errors["time"] = str(error)
    return dict(protocol=protocol.get("name"), farm=FARM, initialization=origin.isoformat(),
                issue_time=issue.isoformat(), target_start=start.isoformat(),
                target_end=(start + DAY).isoformat(), actual_available_time=None,
                access_assumption=protocol.get("access_assumption"), operational_availability_verified=False,
                weather_knots=[dict(valid_time=(origin + h * HOUR).isoformat(),
                                    initialization_lead_h=h, issue_lead_h=h - 12) for h in LEADS],
                power_target_complete="target" not in errors, ens_panel_complete="ens" not in errors,
                time_relations_pass="time" not in errors, offline_sample_pass=not errors,
                errors=errors, target_MW=target)


def inventory():
    protocol = json.loads((ROOT / "protocol.json").read_bytes())
    frozen = json.loads((HERE / "sample.json").read_bytes())
    ens = verified_json(HERE / "ens.json", frozen["ens_sha256"])
    manifest = verified_json(HERE / "b1610/requests.json", frozen["meter_manifest_sha256"])
    meter_hash = manifest["2019-08-25"]["sha256"]
    sample = evaluate("2019-08-24", protocol, ens,
                      verified_json(HERE / frozen["target_source"], meter_hash))
    sample.update(meter_source="../evidence/" + frozen["target_source"], meter_sha256=meter_hash,
                  ens_source="../evidence/ens.json", ens_sha256=frozen["ens_sha256"])
    runs = [sample]
    probes = ROOT / "probes"
    ledger = json.loads((probes / "ledger.json").read_bytes())
    for day in DAYS:
        entries = {e["kind"]: e for e in ledger if e["day"] == day}
        weather, meter = entries["ens"], entries["meter"]
        response = verified_json(probes / weather["body"], weather["sha256"])
        if weather["http_status"] != 401 or response.get("detail") != "authentication required":
            raise ValueError("Reinspect changed ECDS response before classifying coverage")
        if meter["http_status"] != 200:
            raise ValueError("Missing official meter response")
        result = evaluate(day, protocol, None, verified_json(probes / meter["body"], meter["sha256"]))
        result.update(ens_blocker=response["detail"], meter_source="probes/" + meter["body"],
                      meter_sha256=meter["sha256"])
        runs.append(result)
    count = sum(r["offline_sample_pass"] for r in runs)
    accepted = {r["initialization"] for r in runs if r["offline_sample_pass"]}
    splits = protocol.get("split_manifest") or {}
    partitions = [splits.get(name, []) for name in ("train", "validation", "test")]
    assigned = [cycle for partition in partitions for cycle in partition]
    ready = all(partitions) and len(set(assigned)) == len(assigned) and set(assigned) == accepted
    return dict(protocol_sha256=digest((ROOT / "protocol.json").read_bytes()),
                samples=runs, verified_complete_windows=count, dataset_train_ready=ready,
                split_counts=None if not ready else {k: len(v) for k, v in splits.items()},
                split_status="unassigned" if not splits else "declared")


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2, allow_nan=False))
