"""Authenticated, date-ordered ECDS acquisition with a fixed first-cycle gate."""
import argparse
import fcntl
from functools import lru_cache
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from data_audit.audit import FARM, HERE, digest
from data_audit.ens_only_v1.accept import ROOT, evaluate, verified_json
from data_audit.ens_only_v1.probe import request_spec
from .decode import PanelError, compare, decode

API = "https://ecds.ecmwf.int/api"
DATASET = "tigge-forecasts"
LIMIT = 20_000_000
DAYS = [(date(2019, 8, 24) + timedelta(days=i)).isoformat() for i in range(30)]


class Stop(Exception):
    """Only fixed, credential-free status codes may be persisted."""


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def client():
    conf = Path(os.environ.get("CDSAPI_RC", "~/.cdsapirc")).expanduser()
    if not os.environ.get("CDSAPI_KEY") and not conf.is_file():
        raise Stop("credentials_not_configured")
    import cdsapi
    from cdsapi.api import get_url_key_verify

    try:
        _, key, verify = get_url_key_verify(API, None, None)
    except Exception:
        raise Stop("credentials_not_configured") from None
    if not key or ":" in key or "<" in key:
        raise Stop("personal_token_required")
    return cdsapi.Client(url=API, key=key, verify=verify, quiet=True, debug=False, progress=False)


def download(url, path, limit, status, expected=None):
    if expected is not None and not 0 < expected <= limit:
        raise Stop("delivery_size_outside_limit")
    with urlopen(url, timeout=120) as response:
        size = response.headers.get("Content-Length")
        if size and int(size) > limit:
            raise Stop("delivery_size_outside_limit")
        body = bytearray()
        while len(body) <= limit:
            chunk = response.read(min(65536, limit + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
            status["downloaded_body_bytes_this_run"] += len(chunk)
    if len(body) > limit or (expected is not None and len(body) != expected):
        raise Stop("delivery_size_mismatch")
    tmp = path.with_suffix(".partial")
    tmp.write_bytes(body)
    tmp.replace(path)
    return dict(bytes=len(body), sha256=digest(body))


def artifact(path, record, name, acquire):
    if name in record:
        if not path.is_file() or digest(path.read_bytes()) != record[name]["sha256"]:
            raise Stop("cached_artifact_hash_mismatch")
        return
    if path.exists():
        raise Stop("unreceipted_artifact_requires_review")
    record[name] = acquire()
    write(path.parent / "receipt.json", record)


def classify(error):
    if isinstance(error, PanelError):
        return "content_rejected: " + str(error)
    if isinstance(error, Stop):
        return str(error)
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(error, HTTPError):
        code = error.code
    if code == 401:
        return "authentication_rejected"
    if code == 403:
        return "licence_rejected" if any(t in str(error).lower() for t in ("licence", "license", "terms")) else "permission_denied"
    if code in (400, 422):
        return "request_parameters_rejected"
    if code:
        return f"http_{code}"
    if isinstance(error, ImportError):
        return "client_or_decoder_not_installed"
    if isinstance(error, ValueError):
        return "content_or_identity_mismatch"
    return "execution_error_" + type(error).__name__


def process(day, directory, get_service, old, status):
    directory.mkdir(exist_ok=True)
    req = request_spec(day)
    receipt = directory / "receipt.json"
    record = json.loads(receipt.read_bytes()) if receipt.exists() else dict(dataset=DATASET, request=req)
    if (record["dataset"], record["request"]) != (DATASET, req):
        raise Stop("cached_request_mismatch")
    record["request_sha256"] = digest(json.dumps(req, sort_keys=True, separators=(",", ":")).encode())
    write(receipt, record)
    grib = directory / "ensemble.grib"

    def retrieve():
        service = get_service()
        import eccodes  # Ensure the decoder is installed before submitting the request.
        status["request_submitted_this_run"] = True
        result = service.retrieve(DATASET, req)
        return download(result.location, grib, LIMIT, status, int(result.content_length))

    artifact(grib, record, "grib", retrieve)
    status["grib_bytes"] = record["grib"]["bytes"]
    panel, packing = decode(grib, day)
    if day == DAYS[0]:
        ref = verified_json(HERE / "ens.json", old["samples"][0]["ens_sha256"])
        match = compare(ref, panel, packing)
        write(directory / "comparison.json", match)
        status["reference_comparison"] = "passed"
        if any(v["outside"] for v in match.values()):
            status["reference_comparison"] = "failed"
            raise Stop("first_cycle_tolerance_exceeded")
    if day in DAYS[:3]:
        saved = old["samples"][DAYS.index(day)]
        path = ROOT / saved["meter_source"]
        rows = verified_json(path, saved["meter_sha256"])
        meter_source = dict(path=saved["meter_source"], relative_to="data_audit/ens_only_v1",
                            sha256=saved["meter_sha256"])
    else:
        start = date.fromisoformat(day) + timedelta(days=1)
        end = start + timedelta(days=1)
        url = ("https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610/stream"
               f"?settlementDateFrom={start}&settlementDateTo={end}&bmUnit={FARM}")
        path = directory / "meter.json"
        artifact(path, record, "meter", lambda: dict(url=url, **download(url, path, 500000, status)))
        rows = json.loads(path.read_bytes())
        meter_source = record["meter"]
    accepted = evaluate(day, json.loads((ROOT / "protocol.json").read_bytes()), panel, rows)
    write(directory / "weather.json", panel)
    accepted.update(weather_source="weather.json", grib_sha256=record["grib"]["sha256"], meter_source=meter_source)
    write(directory / "acceptance.json", accepted)
    if not accepted["offline_sample_pass"]:
        raise Stop("window_rejected")
    status["status"] = "accepted"


def run(root):
    old = json.loads((ROOT / "acceptance.json").read_bytes())
    statuses = [dict(day=day, status="not_attempted", grib_bytes=0, downloaded_body_bytes_this_run=0,
                     request_submitted_this_run=False)
                for day in DAYS]
    statuses[0]["reference_comparison"] = "not_run"
    try:
        get_service = lru_cache(maxsize=1)(client)
        for status in statuses:
            process(status["day"], root / status["day"], get_service, old, status)
    except Exception as error:
        reason = classify(error)
        status = next((s for s in statuses if s["status"] != "accepted"), statuses[-1])
        status.update(status="blocked", reason=reason)
    ledger = dict(dataset=DATASET, api=API, dataset_train_ready=False, days=statuses,
                  actual_available_time=None, operational_availability_verified=False,
                  downloaded_body_bytes_this_run=sum(s["downloaded_body_bytes_this_run"] for s in statuses),
                  accepted_windows=sum(s["status"] == "accepted" for s in statuses))
    write(root / "coverage.json", ledger)
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.CRITICAL)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(root)
    print("Accepted authenticated windows:", result["accepted_windows"])
    if result["accepted_windows"] != len(DAYS):
        raise SystemExit(next(s["reason"] for s in result["days"] if "reason" in s))
