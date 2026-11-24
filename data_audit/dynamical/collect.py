"""Fetch the August 2021 trial sequentially and retain exact HTTP bodies and receipts."""
import argparse
import hashlib
import json
import random
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from data_audit.audit import FARM

PRODUCT = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
WX = "https://api.dynamical.org/v1/forecasts"
GRID = [(lat, lon) for lat in (57.25, 57.50) for lon in (-3.50, -3.25)]
VARS = ["wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"]
DAYS = [date(2021, 8, i) for i in range(1, 31)]
LEADS = list(range(24, 49, 3))


def stamp(day, hour=0):
    return f"{day}T{hour:02}:00:00Z"


def query(day, point):
    target = day + timedelta(days=1)
    return dict(dataProductId=PRODUCT, initTime=stamp(day), validTimeStart=stamp(target),
                validTimeEnd=stamp(target + timedelta(days=1), 1), maxLeadTimeHours=48,
                location=dict(zip(("latitude", "longitude"), point)), variables=VARS)


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def fetch(folder, name, url, payload=None):
    path = folder / f"{name}.json"
    receipt = folder / f"{name}.receipt.json"
    body = json.dumps(payload, separators=(",", ":")).encode() if payload else None
    if receipt.exists():
        meta = json.loads(receipt.read_bytes())
        if meta["url"] != url or meta["request"] != payload or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Cache identity mismatch: {path}")
        return
    req = Request(url, data=body, headers={"Content-Type": "application/json", "Accept": "application/json",
                                           "Accept-Encoding": "identity"})
    def transport_failure(exc, stage):
        event = dict(error=type(exc).__name__, stage=stage, url=url, request=payload,
                     observed_utc=datetime.now(timezone.utc).isoformat(), partial_body_bytes=None)
        with (folder / "transport.jsonl").open("a") as stream:
            print(json.dumps(event), file=stream)

    for attempt in range(5):
        try:
            response = urlopen(req, timeout=180)
        except HTTPError as exc:
            response = exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            transport_failure(exc, "open")
            if attempt == 4:
                raise
            time.sleep(2**attempt + random.random())
            continue
        with response:
            try:
                length = response.headers.get("Content-Length")
                if length and int(length) > 2_000_000:
                    raise ValueError("Unexpectedly large response")
                raw = response.read(2_000_001)
                if length and len(raw) < int(length):
                    raise IncompleteRead(raw, int(length) - len(raw))
            except (IncompleteRead, TimeoutError, ConnectionError) as exc:
                transport_failure(exc, "body")
                if attempt == 4:
                    raise
                time.sleep(2**attempt + random.random())
                continue
            if len(raw) > 2_000_000:
                raise ValueError("Unexpectedly large response")
            meta = dict(url=url, request=payload, request_body_bytes=len(body or b""), status=response.status,
                        bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                        observed_utc=datetime.now(timezone.utc).isoformat(),
                        work_units=response.headers.get("X-Dynamical-Work-Units"))
            if response.status == 200:
                save(path, raw)
                save(receipt, json.dumps(meta, indent=2).encode())
                return
            retry = response.headers.get("Retry-After")
            with (folder / "errors.jsonl").open("a") as stream:
                print(json.dumps(dict(meta, detail=raw.decode(errors="replace")[:2000], retry_after=retry)), file=stream)
            if response.status not in (429, 500, 502, 503, 504) or attempt == 4:
                raise ValueError(f"HTTP {response.status} {raw[:400]!r}")
            wait = 2**attempt
            if retry:
                wait = int(retry) if retry.isdecimal() else (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
            time.sleep(max(0, wait) + random.random())


def collect(root, days):
    for day in days:
        folder = root / str(day)
        folder.mkdir(parents=True, exist_ok=True)
        target = day + timedelta(days=1)
        fetch(folder, "meter", "https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610/stream"
              f"?settlementDateFrom={target}&settlementDateTo={target + timedelta(days=1)}&bmUnit={FARM}")
        for i, point in enumerate(GRID):
            fetch(folder, f"point{i}", WX, dict(queries=[query(day, point)]))
        print(day, "responses saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--first", action="store_true")
    args = parser.parse_args()
    collect(args.root, DAYS[:1] if args.first else DAYS)
