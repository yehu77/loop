"""Inventory 2021–2023 run identities and next-day B1610 targets, without scoring."""
import argparse
import calendar
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from data_audit.audit import FARM, hourly, interval
from .collect import PRODUCT, fetch, stamp
from .prepare import cached

DAY = timedelta(days=1)
YEARS = {2021: "train", 2022: "validation", 2023: "test"}


def month(root, year, number, online):
    start = date(year, number, 1)
    n = calendar.monthrange(year, number)[1]
    stop = start + n * DAY
    # Endpoint bounds are exclusive. Adjacent midnight cycles lie outside.
    url = "https://api.dynamical.org/v1/forecasts/runs?" + urlencode(dict(
        dataProductId=PRODUCT, initTimesAfter=stamp(start - DAY), initTimesBefore=stamp(stop), limit=100))
    meter_url = ("https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610/stream"
                 f"?settlementDateFrom={start+DAY}&settlementDateTo={stop+DAY}&bmUnit={FARM}")
    folder = root / start.strftime("%Y-%m")
    folder.mkdir(exist_ok=True, parents=True)
    if online:
        fetch(folder, "runs", url)
        fetch(folder, "meter", meter_url)
    catalog, run_meta = cached(folder, "runs", url)
    meter, meter_meta = cached(folder, "meter", meter_url)
    found = [r["initTime"] for r in catalog["runs"]]
    if catalog["dataProductId"] != PRODUCT or not catalog["snapshotId"]:
        raise ValueError("Run listing source mismatch")
    expected = {stamp(start + i * DAY) for i in range(n)}
    if set(found) - expected or len(found) != len(set(found)) or len(found) >= 100:
        raise ValueError("Duplicate or truncated run listing")
    buckets = {}
    for item in meter:
        utc, _ = interval(item)
        buckets.setdefault(utc.date(), []).append(item)
    days = {}
    for i in range(n):
        day = start + i * DAY
        target = day + DAY
        rows = buckets.get(target, [])
        try:
            hours = hourly(rows, datetime.combine(target, time(), timezone.utc))
            ok = all(h["quality"] == "complete" for h in hours) and all(r.get("settlementRunType") for r in rows)
            reason = None if ok else "missing quantity, interval or settlement revision"
        except ValueError as exc:
            ok, reason = False, str(exc)
        days[str(day)] = dict(run_listed=stamp(day) in found, target_complete=ok, meter_rows=len(rows),
                              reason=reason, split=YEARS[year],
                              boundary_holdout=(year < 2023 and day == date(year, 12, 31)))
    return dict(days=days, snapshot=catalog["snapshotId"], sources=[run_meta, meter_meta])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--first", action="store_true")
    args = parser.parse_args()
    result = {}
    for year in YEARS:
        for number in range(1, 13):
            result[f"{year}-{number:02}"] = month(args.root, year, number, args.fetch)
            if args.first:
                break
        if args.first:
            break
    print(json.dumps(result, indent=2))
