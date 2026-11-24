"""Rebuild identified Berry Burn joint samples from the saved public point responses."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_audit.audit import FARM, hourly
from .collect import DAYS, GRID, LEADS, PRODUCT, VARS, WX, query, save, stamp


def cached(folder, name, url, request=None):
    receipt = json.loads((folder / f"{name}.receipt.json").read_bytes())
    body = (folder / f"{name}.json").read_bytes()
    if (receipt["url"] != url or receipt["request"] != request or receipt["status"] != 200
            or receipt["bytes"] != len(body) or receipt["sha256"] != hashlib.sha256(body).hexdigest()):
        raise ValueError(f"{name} provenance mismatch")
    return json.loads(body), receipt


def point_values(result, day, point):
    product = result["dataProduct"]
    snapshot = result["snapshotId"]
    loc = result["point"]
    if (product["id"] != PRODUCT or not product["version"] or not snapshot or loc["method"] != "nearest"
            or any((loc[k]["latitude"], loc[k]["longitude"]) != point for k in ("selected", "requested"))):
        raise ValueError("Product or native grid point mismatch")
    if result["variables"] != {v: {"unit": "m s-1"} for v in VARS}:
        raise ValueError("Variable identity or units mismatch")
    f, = result["forecasts"]
    times = [stamp(day + timedelta(days=h//24), h % 24) for h in LEADS]
    if f["initTime"] != stamp(day) or f["leadTimeHours"] != LEADS or f["validTimes"] != times:
        raise ValueError("Initialization, valid time or native lead mismatch")
    members = {m["member"]: m["data"] for m in f["ensembleMembers"]}
    if len(f["ensembleMembers"]) != 51 or set(members) != set(range(51)):
        raise ValueError("Missing or duplicate ensemble identity")
    values = []
    for m in range(1, 51):
        data = members[m]
        if set(data) != set(VARS) or any(len(data[v]) != len(LEADS) for v in VARS):
            raise ValueError(f"Incomplete series for member {m}")
        sequence = [[data[v][i] for v in VARS] for i in range(len(LEADS))]
        if any(type(x) not in (int, float) or not math.isfinite(x) for row in sequence for x in row):
            raise ValueError(f"Missing wind value for member {m}")
        values.append(sequence)
    return values, (product["version"], snapshot), f["links"]["canonical"]


def prepare_day(folder, day):
    target = day + timedelta(days=1)
    samples = []
    sources = []
    identities = set()
    for i, pt in enumerate(GRID):
        payload, meta = cached(folder, f"point{i}", WX, {"queries": [query(day, pt)]})
        result, = payload["results"]
        values, identity, canonical = point_values(result, day, pt)
        samples.append(values)
        sources.append(dict(file=f"point{i}.json", bytes=meta["bytes"], sha256=meta["sha256"],
                            work_units=meta["work_units"], canonical=canonical))
        identities.add(identity)
    if len(identities) != 1:
        raise ValueError("Mixed snapshots within one sample")
    version, snapshot = identities.pop()
    url = ("https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610/stream"
           f"?settlementDateFrom={target}&settlementDateTo={target + timedelta(days=1)}&bmUnit={FARM}")
    meter, meta = cached(folder, "meter", url)
    hours = hourly(meter, datetime.combine(target, datetime.min.time(), timezone.utc))
    if any(h["quality"] != "complete" for h in hours):
        raise ValueError("Incomplete next-day UTC power window")
    for h in hours:
        h["settlement_runs"] = [meter[j]["settlementRunType"] for j in h["response_indices"]]
        if not all(h["settlement_runs"]):
            raise ValueError("Missing meter revision")
    sources.append(dict(file="meter.json", bytes=meta["bytes"], sha256=meta["sha256"], url=url))
    sample = dict(protocol="berry_burn_dynamical_ens_v1", farm=FARM, origin=stamp(day),
                  issue=stamp(day, 12), actual_available_time=None, operational_availability_verified=False,
                  availability_assumption="operational_ens_delivered_by_noon_utc",
                  product=PRODUCT, version=version, snapshot=snapshot, grid=GRID, members=list(range(1, 51)),
                  lead_hours=LEADS, issue_lead_hours=[h-12 for h in LEADS], variables=VARS,
                  valid_times=[stamp(day + timedelta(days=h//24), h%24) for h in LEADS],
                  wind_unit="m s-1", wind_axes=["grid", "member", "lead_hours", "variables"],
                  wind=samples, target_MW=hours, sources=sources)
    encoded = json.dumps(sample, allow_nan=False, separators=(",", ":")).encode()
    save(folder / "joint.json", encoded)
    return dict(accepted=True, version=version, snapshot=snapshot, sources=sources,
                sample_sha256=hashlib.sha256(encoded).hexdigest())


def inventory(root):
    runs = {}
    for day in DAYS:
        try:
            runs[str(day)] = prepare_day(root / str(day), day)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            runs[str(day)] = dict(accepted=False, reason=str(exc))
    return dict(protocol="berry_burn_dynamical_ens_v1", accepted=sum(r["accepted"] for r in runs.values()),
                days=runs, dataset_train_ready=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), indent=2))
