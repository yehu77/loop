"""Collect four points per daily batch, preserving responses and superseded attempts."""
import argparse
import json
from datetime import date
from pathlib import Path

from .collect import GRID, WX, fetch, query
from .prepare import cached

ARCHIVE = Path("docs/data/dynamical_ens_2021")


def calendar():
    months = json.loads((ARCHIVE / "coverage.json").read_bytes())
    return {date.fromisoformat(day): row for month in months.values() for day, row in month["days"].items()}


def daily_batch(root, day, online):
    payload = {"queries": [query(day, point) for point in GRID]}
    for number in range(1, 4):
        folder = root / str(day) / f"attempt-{number}"
        if online:
            folder.mkdir(parents=True, exist_ok=True)
            fetch(folder, "weather", WX, payload)
        response, receipt = cached(folder, "weather", WX, payload)
        points = response["results"]
        if len(points) != len(GRID):
            raise ValueError("Incomplete weather batch")
        if len({(p["snapshotId"], p["dataProduct"]["version"]) for p in points}) == 1:
            return folder, points, receipt
    raise ValueError("Mixed daily snapshots in three archived attempts")


def collect(root):
    pilot = json.loads((ARCHIVE / "acceptance.json").read_bytes())["days"]
    for day in calendar():
        if str(day) not in pilot:
            daily_batch(root, day, True)
            print(day, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    collect(parser.parse_args().root)
