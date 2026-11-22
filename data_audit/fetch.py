"""Replay bounded per-farm B1610 requests into a fresh directory."""
import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

DAYS = ("2019-01-15", "2019-03-31", "2019-08-25", "2019-10-27",
        "2020-01-15", "2020-03-29", "2020-10-25", "2023-07-01")
FARM = "E_BRYBW-1"


def collect(destination):
    destination.mkdir(exist_ok=False, parents=True)
    manifest = {}
    for day in DAYS:
        tomorrow = date.fromisoformat(day) + timedelta(days=1)
        url = ("https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610/stream"
               f"?settlementDateFrom={day}&settlementDateTo={tomorrow}&bmUnit={FARM}")
        with urlopen(url, timeout=40) as response:
            body = response.read(500001)
        if len(body) > 500000:
            raise ValueError("Response exceeds audit budget")
        rows = json.loads(body)
        if not isinstance(rows, list) or any(r["bmUnit"] != FARM for r in rows):
            raise ValueError("Unexpected response schema or unit")
        (destination / f"{day}.json").write_bytes(body)
        manifest[day] = dict(url=url, bytes=len(body), records=len(rows),
                             sha256=hashlib.sha256(body).hexdigest())
    (destination / "requests.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    collect(parser.parse_args().destination)
