"""Bounded, anonymous probes for the two preselected consecutive cycles."""
import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from data_audit.audit import FARM

DAYS = ("2019-08-25", "2019-08-26")
ENDPOINT = "https://ecds.ecmwf.int/api/retrieve/v1/processes/tigge-forecasts/execution"


def request_spec(day):
    return dict(date=day, expver="prod", origin="ecmf", levtype="sfc",
                param="165/166", step="24/30/36/42/48", time="00:00:00",
                type="pf", number="1/to/50", area="57.5/-4/57/-3.5",
                grid="0.5/0.5", **{"class": "ti"})


def probe(dest):
    dest.mkdir(exist_ok=False, parents=True)
    ledger = []
    for day in DAYS:
        start = date.fromisoformat(day) + timedelta(days=1)
        end = start + timedelta(days=1)
        payload = json.dumps({"inputs": request_spec(day)}).encode()
        meter = ("https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610/stream"
                 f"?settlementDateFrom={start}&settlementDateTo={end}&bmUnit={FARM}")
        for kind, req in (("ens", Request(ENDPOINT, data=payload, headers={"Content-Type": "application/json"})),
                          ("meter", Request(meter))):
            try:
                response = urlopen(req, timeout=40)
            except HTTPError as error:
                response = error
            with response:
                body = response.read(500001)
                if len(body) > 500000:
                    raise ValueError("Probe response exceeds limit")
                name = f"{day}-{kind}.json"
                (dest / name).write_bytes(body)
                ledger.append(dict(day=day, kind=kind, method=req.get_method(), url=req.full_url,
                                   request=json.loads(req.data) if req.data else None,
                                   request_body_bytes=len(req.data or b""), http_status=response.status,
                                   body=name, response_bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
    (dest / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    probe(parser.parse_args().destination)
