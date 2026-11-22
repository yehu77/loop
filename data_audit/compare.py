"""Compare settlement records to author-clock predictions and optional table export."""
import argparse
import calendar
import csv
import json
import sys
from datetime import datetime, timedelta

from .audit import HALF, HERE, interval
from .fetch import FARM


def predicted_label(day, period):
    # Mirrors the authors' year-ordered raw-file clock before duplicate removal.
    naive = datetime.fromisoformat(day) + (period - 1) * HALF
    boundaries = []
    for month in (3, 10):
        last = datetime(naive.year, month, calendar.monthrange(naive.year, month)[1])
        boundaries.append(last - timedelta(days=(last.weekday() - 6) % 7))
    if boundaries[0] + timedelta(days=1) <= naive <= boundaries[1]:
        naive += timedelta(hours=1)
    return naive


def comparison(export=None):
    table = {}
    if export is not None:
        with export.open(newline="") as stream:
            for item in csv.DictReader(stream):
                stamp = datetime.fromisoformat(item["Time"])
                if stamp.tzinfo is not None or stamp in table:
                    raise ValueError("Preserve the original, unique naive author index")
                table[stamp] = item[FARM]
    for path in sorted((HERE / "b1610").glob("2*.json")):
        for i, row in enumerate(json.loads(path.read_bytes())):
            start, _ = interval(row)
            label = predicted_label(row["settlementDate"], row["settlementPeriod"])
            yield dict(file=path.name, row=i, utc_start=start.isoformat(),
                       expected_author_label=label.isoformat(),
                       label_minus_utc_start_hours=(label - start.replace(tzinfo=None)).total_seconds() / 3600,
                       official_MW=2 * row["quantity"] if row["quantity"] is not None else None,
                       author_cell=table.get(label), lookup=("no_export" if export is None
                           else "found" if label in table else "missing_label"))


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-csv", type=Path)
    records = list(comparison(parser.parse_args().author_csv))
    writer = csv.DictWriter(sys.stdout, fieldnames=records[0])
    writer.writeheader()
    writer.writerows(records)
