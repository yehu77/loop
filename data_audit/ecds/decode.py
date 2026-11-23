"""Validate native TIGGE GRIB identities before adapting them to the frozen evaluator."""
import math
from datetime import datetime, timedelta, timezone

from data_audit.ens_only_v1.accept import LEADS

UTC = timezone.utc
EPOCH = datetime(1900, 1, 1, tzinfo=UTC)
# Verified CDF-2 attributes from the prefix pinned in data_audit.ens.
NC_SCALE = {"u10": 0.0005914715818466534, "v10": 0.0006574601748562554}
GRID = {(lat, lon) for lat in (57, 57.5) for lon in (-4, -3.5)}


class PanelError(ValueError):
    """Credential-free reason a decoded physical field cannot be accepted."""


def decode(path, day):
    import eccodes as ec

    origin = datetime.fromisoformat(day).replace(tzinfo=UTC)
    rows, errors = {}, {}
    with path.open("rb") as stream:
        while (handle := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                def get(key):
                    return ec.codes_get(handle, key)

                lead = get("endStep")
                member = get("perturbationNumber")
                param = get("paramId")
                if (get("centre") != "ecmf" or ec.codes_get_long(handle, "typeOfProcessedData") != 4 or
                    get("dataDate") != int(origin.strftime("%Y%m%d")) or get("dataTime") != 0 or
                    ec.codes_get_long(handle, "stepUnits") != 1 or lead not in LEADS or get("stepType") != "instant" or
                    member not in range(1, 51) or get("typeOfLevel") != "heightAboveGround" or
                    get("level") != 10 or param not in (165, 166) or
                    get("units") not in ("m s**-1", "m s-1") or get("numberOfMissing") != 0):
                    raise PanelError("GRIB identity, units, level, missing values or step mismatch")
                valid = origin + timedelta(hours=lead)
                if (get("validityDate"), get("validityTime")) != (int(valid.strftime("%Y%m%d")), valid.hour * 100):
                    raise PanelError("GRIB validity time disagrees with initialization and lead")
                var = "u10" if param == 165 else "v10"
                packing = get("packingError")
                if not math.isfinite(packing) or packing < 0:
                    raise PanelError("Invalid GRIB packing error bound")
                points = set()
                for lat, east, value in zip(*(ec.codes_get_array(handle, key)
                                             for key in ("latitudes", "longitudes", "values")), strict=True):
                    lon = (float(east) + 180) % 360 - 180
                    key = (member, lead, float(lat), lon)
                    if (lat, lon) not in GRID or (key, var) in errors or not math.isfinite(value):
                        raise PanelError("Unexpected grid, duplicate or invalid wind")
                    points.add((lat, lon))
                    row = rows.setdefault(key, dict(member=member, lead_hours=lead, lat=float(lat), lon=lon,
                        valid_time=valid.isoformat(), raw_time=int((valid - EPOCH).total_seconds() / 3600)))
                    row[var] = float(value)
                    errors[key, var] = packing
                if points != GRID:
                    raise PanelError("Incomplete message grid")
            finally:
                ec.codes_release(handle)
    if len(errors) != 2000:
        raise PanelError("Incomplete native panel")
    return dict(rows=list(rows.values()), origin=origin.isoformat(), product="ENS", available_time=None,
                wind_units="m s**-1", raw_time_units="hours since 1900-01-01 00:00:00.0"), errors


def compare(reference, candidate, packing):
    def index(panel):
        return {(r["member"], r["lead_hours"], r["lat"], r["lon"]): r for r in panel["rows"]}

    old, new = index(reference), index(candidate)
    if (reference["origin"] != candidate["origin"] or reference["wind_units"] != candidate["wind_units"]
        or old.keys() != new.keys() or len(old) != 1000 or len(candidate["rows"]) != 1000):
        raise PanelError("Cross-source identities differ")
    result = {}
    for var, scale in NC_SCALE.items():
        diffs, limits, mismatches = [], [], []
        for k, a in old.items():
            b = new[k]
            if a["valid_time"] != b["valid_time"] or not all(math.isfinite(v) for v in (a[var], b[var])):
                raise PanelError("Cross-source validity or missingness differs")
            diffs.append(abs(a[var] - b[var]))
            limits.append(scale / 2 + packing[k, var] + 1e-12)
            if diffs[-1] > limits[-1]:
                mismatches.append(dict(member=k[0], lead_hours=k[1], lat=k[2], lon=k[3],
                                       reference=a[var], decoded=b[var], allowed=limits[-1]))
        result[var] = dict(count=len(diffs), max_abs_error=max(diffs), max_allowed=max(limits),
                           outside=len(mismatches), mismatches=mismatches)
    return result
