"""Offline acceptance of the frozen Berry Burn evidence. Print reproducible JSON."""
import hashlib
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .fetch import FARM

HERE = Path(__file__).parent / "evidence"
UTC = timezone.utc
GB = ZoneInfo("Europe/London")
ORIGIN = datetime(2019, 8, 24, tzinfo=UTC)
DAY = timedelta(days=1)
HALF = timedelta(minutes=30)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def interval(row):
    day = date.fromisoformat(row["settlementDate"])
    period = row["settlementPeriod"]
    midnight = datetime.combine(day, time(), GB).astimezone(UTC)
    tomorrow = datetime.combine(day + DAY, time(), GB).astimezone(UTC)
    if not 1 <= period <= (tomorrow - midnight) // HALF:
        raise ValueError("Invalid period for settlement day")
    end = midnight + period * HALF
    raw_end = datetime.fromisoformat(row["halfHourEndTime"])
    if raw_end.tzinfo is None:
        raw_end = raw_end.replace(tzinfo=UTC)
    if end != raw_end or row["bmUnit"] != FARM or row["dataset"] != "B1610":
        raise ValueError("Settlement key, timestamp or source mismatch")
    return end - HALF, end


def hourly(rows, start):
    values = {}
    for i, row in enumerate(rows):
        stamp, _ = interval(row)
        if start <= stamp < start + DAY:
            if stamp in values:
                raise ValueError("Duplicate half hour")
            values[stamp] = (i, row["quantity"])
    result = []
    for h in range(24):
        stamp = start + 2 * h * HALF
        pairs = [values.get(stamp + k * HALF, (None, None)) for k in (0, 1)]
        valid = all(q is not None and math.isfinite(q) for _, q in pairs)
        result.append(dict(start=stamp.isoformat(), end=(stamp + 2 * HALF).isoformat(),
                           MW=sum(q for _, q in pairs) if valid else None,
                           quality="complete" if valid else "missing_meter",
                           response_indices=[i for i, _ in pairs]))
    return result


def weather_identity(dataset, product, origin):
    if product not in ("ENS", "HRES") or dataset["product"] != product:
        raise ValueError("Only operational forecasts may condition the sample")
    if datetime.fromisoformat(dataset["origin"]) != origin:
        raise ValueError("Mixed forecast cycles")
    for row in dataset["rows"]:
        valid = datetime.fromisoformat(row["valid_time"])
        if row["lead_hours"] < 0 or valid != origin + timedelta(hours=row["lead_hours"]):
            raise ValueError("Inconsistent forecast valid time")


def available(dataset, issue):
    published = dataset.get("available_time")
    return published is not None and datetime.fromisoformat(dataset["origin"]) <= datetime.fromisoformat(published) <= issue


def check_ensemble(dataset):
    weather_identity(dataset, "ENS", ORIGIN)
    per_trajectory = {}
    for row in dataset["rows"]:
        key = (row["member"], row["lat"], row["lon"])
        per_trajectory.setdefault(key, []).append(row["lead_hours"])
        if not all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) for k in ("u10", "v10")):
            raise ValueError("Missing ensemble wind")
    expected = {(m, lat, lon) for m in range(1, 51) for lat in (57, 57.5) for lon in (-4, -3.5)}
    if set(per_trajectory) != expected or any(v != [24, 30, 36, 42, 48] for v in per_trajectory.values()):
        raise ValueError("Incomplete or unordered ensemble trajectories")


def report(root=HERE):
    requests = json.loads((root / "b1610/requests.json").read_bytes())
    target = None
    counts = {}
    for day, metadata in requests.items():
        body = (root / f"b1610/{day}.json").read_bytes()
        if digest(body) != metadata["sha256"]:
            raise ValueError("Meter response fingerprint mismatch")
        rows = json.loads(body)
        for row in rows:
            interval(row)
        counts[day] = len(rows)
        if day == "2019-08-25":
            target = hourly(rows, ORIGIN + DAY)
    ens = (root / "ens.json").read_bytes()
    check_ensemble(json.loads(ens))
    return dict(train_ready=False, farm=FARM, forecast_origin=ORIGIN.isoformat(),
                issue_time=None, weather_publication_time=None, hres=None,
                ens_file="ens.json", ens_sha256=digest(ens), response_counts=counts,
                target_source="b1610/2019-08-25.json", target=target,
                meter_manifest_sha256=digest((root / "b1610/requests.json").read_bytes()),
                blockers=["same-cycle HRES slice absent", "issue and product availability unverified",
                          "author processed time index and BAV flags not reconciled"],
                complete_windows_by_split=dict(train=None, validation=None, test=None))


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
