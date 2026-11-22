"""Read the verified leading CDF-2 member without treating missing bytes as fill."""
import argparse
import hashlib
import io
import json
import struct
import tarfile
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

MEMBER = "2019/tigge-GB_2019082400.nc"
PREFIX_SHA256 = "72fa56c0acf4de8a61695da8f6f49a4760b94a848061f39cdcfb0a944482991a"
TYPES = {1: "b", 2: "c", 3: "h", 4: "i", 5: "f", 6: "d"}


class Header:
    """XDR header of this classic NetCDF file (big endian, 4-byte alignment)."""

    def __init__(self, data):
        self.stream = io.BytesIO(data)
        if self.stream.read(4) != b"CDF\x02":
            raise ValueError("Expected CDF-2")
        self.records = self.uint()
        self.dims = [(self.name(), self.uint()) for _ in self.items(10)]
        self.attrs()
        self.variables = {}
        for _ in self.items(11):
            name = self.name()
            dims = [self.dims[self.uint()][0] for _ in range(self.uint())]
            attrs = self.attrs()
            kind, size = self.uint(), self.uint()
            begin, = struct.unpack(">Q", self.stream.read(8))
            self.variables[name] = (dims, attrs, TYPES[kind], size, begin)
        self.stride = sum(size for dims, _, _, size, _ in self.variables.values()
                          if dims[0] == "time")

    def uint(self):
        return struct.unpack(">I", self.stream.read(4))[0]

    def items(self, tag):
        actual, count = self.uint(), self.uint()
        if (actual, count) != (0, 0) and actual != tag:
            raise ValueError("Unexpected NetCDF header tag")
        return range(count)

    def padded(self, size):
        data = self.stream.read(size)
        self.stream.read(-size % 4)
        return data

    def name(self):
        return self.padded(self.uint()).decode()

    def attrs(self):
        attrs = {}
        for _ in self.items(12):
            name, fmt, count = self.name(), TYPES[self.uint()], self.uint()
            raw = self.padded(struct.calcsize(fmt) * count)
            attrs[name] = raw.decode() if fmt == "c" else struct.unpack(f">{count}{fmt}", raw)[0]
        return attrs

    def value(self, name, index, record=0):
        _, attrs, fmt, _, begin = self.variables[name]
        self.stream.seek(begin + record * self.stride + index * struct.calcsize(fmt))
        value, = struct.unpack(">" + fmt, self.stream.read(struct.calcsize(fmt)))
        if value == attrs.get("_FillValue") or value == attrs.get("missing_value"):
            return None
        return value * attrs.get("scale_factor", 1) + attrs.get("add_offset", 0)


def extract(path):
    compressed = path.read_bytes()
    fingerprint = hashlib.sha256(compressed).hexdigest()
    if fingerprint != PREFIX_SHA256:
        raise ValueError("ENS compressed prefix differs from audited source")
    raw = zlib.decompressobj(31).decompress(compressed)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for entry in archive:
            if entry.name == MEMBER:
                break
        else:
            raise ValueError("Expected ENS cycle missing")
    header = Header(raw[entry.offset_data:entry.offset_data + entry.size])
    dims = dict(header.dims)
    if dims != dict(longitude=24, latitude=20, number=50, time=0):
        raise ValueError(f"Unexpected grid {dims}")
    origin = datetime(2019, 8, 24, tzinfo=timezone.utc)
    epoch = datetime(1900, 1, 1, tzinfo=timezone.utc)
    if header.variables["time"][1]["units"] != "hours since 1900-01-01 00:00:00.0":
        raise ValueError("Unexpected time units")
    ids = [header.value("number", i) for i in range(50)]
    if len(set(ids)) != 50:
        raise ValueError("Duplicate ENS identity")
    grid = [(i, j) for i in range(20) for j in range(24)
            if header.value("latitude", i) in (57, 57.5)
            and header.value("longitude", j) in (-4, -3.5)]
    if len(grid) != 4:
        raise ValueError("Unexpected grid coordinates")
    rows = []
    for step in range(4, 9):
        stamp = header.value("time", 0, step)
        valid = epoch + timedelta(hours=stamp)
        if valid != origin + timedelta(hours=6 * step):
            raise ValueError("Cycle and valid time disagree")
        for m, member in enumerate(ids):
            for i, j in grid:
                row = dict(member=member, lead_hours=step * 6, valid_time=valid.isoformat(),
                           raw_time=stamp, lat=header.value("latitude", i),
                           lon=header.value("longitude", j))
                for var in ("u10", "v10"):
                    axes, attrs, *_ = header.variables[var]
                    if axes != ["time", "number", "latitude", "longitude"] or attrs["units"] != "m s**-1":
                        raise ValueError(f"Unexpected schema for {var}")
                    row[var] = header.value(var, (m * 20 + i) * 24 + j, step)
                rows.append(row)
    return dict(member_file=MEMBER, prefix_sha256=fingerprint,
                origin=origin.isoformat(), product="ENS", available_time=None,
                raw_time_units=header.variables["time"][1]["units"], wind_units="m s**-1",
                variables={name: attrs.get("units") for name, (_, attrs, *_) in header.variables.items()},
                rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    print(json.dumps(extract(parser.parse_args().prefix), sort_keys=True))
