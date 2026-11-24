"""Summarize HTTP payloads and distinguish candidate coverage from decoded samples."""
import argparse
import json
from pathlib import Path

from data_audit.audit import interval
from .collect import PRODUCT
from .coverage import YEARS


def report(root):
    trial = json.loads((root / "acceptance.json").read_bytes())
    months = json.loads((root / "coverage.json").read_bytes())
    receipts = [json.loads(p.read_bytes()) for p in root.glob("2021-08-*/*.receipt.json")]
    errors = [json.loads(line) for p in root.glob("2021-08-*/errors.jsonl") for line in p.read_text().splitlines()]
    attempts = receipts + errors
    weather = [r for r in attempts if r["url"].startswith("https://api.dynamical.org/")]
    trial_bytes = sum(r["bytes"] for r in attempts)
    stats = {}
    for year, split in YEARS.items():
        dates = {day: row for m, listing in months.items() if m.startswith(str(year))
                 for day, row in listing["days"].items()}
        accepted = [day for day in dates if trial["days"].get(day, {}).get("accepted")]
        stats[str(year)] = dict(split=split, candidates=len(dates),
            run_listed=sum(r["run_listed"] for r in dates.values()),
            meter_complete=sum(r["target_complete"] for r in dates.values()),
            listed_with_complete_target=sum(r["run_listed"] and r["target_complete"] for r in dates.values()),
            split_eligible=sum(r["run_listed"] and r["target_complete"] and not r["boundary_holdout"]
                               for r in dates.values()),
            boundary_holdout=[d for d,r in dates.items() if r["boundary_holdout"]],
            joint_verified=len(accepted),
            excluded={d: (r["reason"] or "run not listed") for d,r in dates.items()
                      if not r["target_complete"] or not r["run_listed"]})
    month_receipts = [r for m in months.values() for r in m["sources"]]
    # Both routes queried the same unit; compare the 48 source records by physical start.
    month_rows = json.loads((root / "months/2021-08/meter.json").read_bytes())
    by_start = {interval(r)[0].isoformat(): r for r in month_rows}
    matched_halves = 0
    for day, entry in trial["days"].items():
        if not entry["accepted"]:
            continue
        daily = json.loads((root / day / "meter.json").read_bytes())
        joint = json.loads((root / day / "joint.json").read_bytes())
        for h in joint["target_MW"]:
            for i in h["response_indices"]:
                row = daily[i]
                if by_start.get(interval(row)[0].isoformat()) != row:
                    raise ValueError(f"Meter query revisions or values disagree: {day}")
                matched_halves += 1
    return dict(product=PRODUCT, matched_meter_half_hours=matched_halves, pilot=dict(days=30, joint_verified=trial["accepted"],
        http_responses=len(attempts), http_error_responses=len(errors), weather_requests=len(weather),
        weather_body_bytes=sum(r["bytes"] for r in weather), all_response_body_bytes=trial_bytes,
        post_body_bytes=sum(r["request_body_bytes"] for r in attempts),
        work_units=sum(int(r["work_units"] or 0) for r in weather),
        estimated_1095_day_response_body_bytes=round(trial_bytes * 1095/30),
        remaining_weather_requests_with_cached_meter=4 * (1095 - trial["accepted"]),
        estimated_remaining_weather_bytes_with_cached_meter=round(
            sum(r["bytes"] for r in weather) * (1095 - trial["accepted"]) / 30)),
        coverage_http_responses=len(month_receipts), coverage_response_bytes=sum(r["bytes"] for r in month_receipts),
        by_year=stats, dataset_train_ready=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(report(parser.parse_args().root), indent=2))
