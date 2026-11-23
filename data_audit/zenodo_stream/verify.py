"""Verify complete local CDF-2 members and align their winds to frozen B1610 targets."""
import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_audit.audit import HERE, digest
from data_audit.ens import Header
from data_audit.ens_only_v1.accept import ROOT, LEADS, evaluate, verified_json
from .scan import ANCHOR, TARGETS


def panel(path, evidence, origin):
    body = path.read_bytes()
    if len(body) != evidence["bytes"] or digest(body) != evidence["sha256"]:
        raise ValueError("Member length or fingerprint mismatch")
    h = Header(body)
    if dict(h.dims) != dict(time=0, latitude=20, longitude=24, number=50) or h.records != 61:
        raise ValueError("Unexpected dimensions or native time count")
    for dims, _, _, size, begin in h.variables.values():
        last = h.stride * (h.records - 1) if dims[0] == "time" else 0
        if begin + last + size > len(body):
            raise ValueError("Truncated NetCDF variable")
    units = h.variables["time"][1]["units"]
    if units != "hours since 1900-01-01 00:00:00.0":
        raise ValueError("Unrecognized time reference")
    ids = [h.value("number", i) for i in range(50)]
    if set(ids) != set(range(1, 51)):
        raise ValueError("Incomplete native member axis")
    grid = [(i, j, h.value("latitude", i), h.value("longitude", j))
            for i in range(20) for j in range(24)
            if h.value("latitude", i) in (57, 57.5) and h.value("longitude", j) in (-4, -3.5)]
    if len({p[2:] for p in grid}) != 4:
        raise ValueError("Incomplete nearby grid")
    for var in ("u10", "v10"):
        dims, attrs, *_ = h.variables[var]
        if dims != ["time", "number", "latitude", "longitude"] or attrs["units"] != "m s**-1":
            raise ValueError("Unexpected wind schema")
    rows = []
    epoch = datetime(1900, 1, 1, tzinfo=timezone.utc)
    for step in range(61):
        raw_time = h.value("time", 0, step)
        valid = epoch + timedelta(hours=raw_time)
        lead = step * 6
        if valid != origin + timedelta(hours=lead):
            raise ValueError("Incorrect or missing native valid time")
        for m, member in enumerate(ids):
            for i, j, lat, lon in grid:
                index = (m * 20 + i) * 24 + j
                winds = {k: h.value(k, index, step) for k in ("u10", "v10")}
                if any(v is None or not math.isfinite(v) for v in winds.values()):
                    raise ValueError("Missing native wind on protocol grid")
                if lead in LEADS:
                    rows.append(dict(member=member, lat=lat, lon=lon, raw_time=raw_time,
                                     valid_time=valid.isoformat(), lead_hours=lead, **winds))
    return dict(origin=origin.isoformat(), product="ENS", raw_time_units=units,
                wind_units="m s**-1", available_time=None, rows=rows)


def verify(directory):
    scan = json.loads((directory / "scan.json").read_bytes())
    frozen = json.loads((ROOT / "acceptance.json").read_bytes())
    protocol = json.loads((ROOT / "protocol.json").read_bytes())
    report = {}
    for saved, name in zip(frozen["samples"], sorted(TARGETS | {ANCHOR}), strict=True):
        origin = datetime.fromisoformat(saved["initialization"])
        day = origin.date().isoformat()
        entry = scan["complete"].get(name)
        if entry is None:
            report[day] = dict(accepted=False, reason="complete_member_not_obtained")
            continue
        try:
            ens = panel(directory / Path(name).name, entry, origin)
            if name == ANCHOR and ens["rows"] != verified_json(HERE / "ens.json", saved["ens_sha256"])["rows"]:
                raise ValueError("Anchor disagrees with previously frozen wind slice")
            meter = verified_json(ROOT / saved["meter_source"], saved["meter_sha256"])
            acceptance = evaluate(day, protocol, ens, meter)
            (directory / f"{day}-acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
            (directory / f"{day}-ens.json").write_text(json.dumps(ens, sort_keys=True) + "\n")
            report[day] = dict(accepted=acceptance["offline_sample_pass"], errors=acceptance["errors"],
                               member=name, **entry, native_steps=61, members=50, grid_points=4,
                               target_complete=acceptance["power_target_complete"],
                               meter_source=saved["meter_source"], meter_sha256=saved["meter_sha256"])
        except (ValueError, KeyError, OSError) as error:
            report[day] = dict(accepted=False, reason=str(error))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
