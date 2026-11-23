"""One bounded HTTP continuation of the verified local Zenodo prefix."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

from data_audit.ens import PREFIX_SHA256

LIMIT = 256 * 1024**2
TOTAL = 36034983151
URL = "https://zenodo.org/records/13255991/files/ecmwf_ens_2019-2020.tar.gz?scan=256m"
TARGETS = {f"2019/tigge-GB_201908{day}00.nc" for day in (25, 26)}
ANCHOR = "2019/tigge-GB_2019082400.nc"


class Stream:
    def __init__(self, prefix, remote, suffix, report):
        self.prefix = prefix
        self.remote = remote
        self.suffix = suffix
        self.report = report
        self.hash = hashlib.sha256()

    def read(self, size):
        part = self.prefix.read(size)
        if part:
            return part
        remaining = LIMIT - self.report["prefix_bytes"] - self.report["download_bytes"]
        if not remaining:
            return b""
        data = self.remote.read(min(size, remaining))
        self.report["download_bytes"] += len(data)
        self.suffix.write(data)
        self.hash.update(data)
        return data


def scan(prefix, out):
    fingerprint = hashlib.sha256(prefix.read_bytes()).hexdigest()
    if fingerprint != PREFIX_SHA256:
        raise ValueError("Local prefix fingerprint mismatch")
    start = prefix.stat().st_size
    out.mkdir(parents=True, exist_ok=False)
    requested = f"bytes={start}-{LIMIT - 1}"
    ledger = dict(url=URL, requested_range=requested, archive_bytes=TOTAL,
                  prefix_bytes=start, prefix_sha256=fingerprint, limit_bytes=LIMIT,
                  http_requests=1, download_bytes=0, headers_seen=[], complete={})
    stream = None
    try:
        request = Request(URL, headers={"Range": requested, "Accept-Encoding": "identity"})
        with urlopen(request, timeout=240) as remote:
            ledger.update(status=remote.status, content_range=remote.headers.get("Content-Range"))
            if (remote.status != 206 or ledger["content_range"] != f"bytes {start}-{LIMIT-1}/{TOTAL}"
                    or remote.headers.get("Content-Encoding", "identity") != "identity"):
                raise ValueError("Server did not confirm exact archive continuation")
            with prefix.open("rb") as local, (out / "continuation.bin").open("xb") as suffix:
                stream = Stream(local, remote, suffix, ledger)
                with tarfile.open(fileobj=stream, mode="r|gz", bufsize=131072) as archive:
                    for entry in archive:
                        ledger["headers_seen"].append(dict(path=entry.name, size=entry.size))
                        if entry.name not in TARGETS | {ANCHOR} or entry.name in ledger["complete"]:
                            continue
                        if not entry.isfile():
                            raise ValueError("Expected a regular file")
                        name = Path(entry.name).name
                        partial = out / (name + ".partial")
                        digest = hashlib.sha256()
                        count = 0
                        with partial.open("xb") as dst, archive.extractfile(entry) as src:
                            while block := src.read(262144):
                                count += dst.write(block)
                                digest.update(block)
                        if count != entry.size:
                            raise ValueError("Incomplete tar member")
                        partial.rename(out / name)
                        ledger["complete"][entry.name] = dict(bytes=count, sha256=digest.hexdigest())
                        if TARGETS <= ledger["complete"].keys():
                            break
        if TARGETS <= ledger["complete"].keys():
            ledger["stop"] = "both_targets_extracted"
        elif start + ledger["download_bytes"] == LIMIT:
            ledger["stop"] = "compressed_byte_limit"
        else:
            ledger["stop"] = "end_of_archive"
    except Exception as error:
        ledger["stop"] = ("compressed_byte_limit" if start + ledger["download_bytes"] == LIMIT
                          else "read_failed")
        ledger["error"] = f"{type(error).__name__}: {error}"
    finally:
        if stream:
            ledger["continuation_sha256"] = stream.hash.hexdigest()
        (out / "scan.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(ledger["stop"], ledger["download_bytes"], sorted(ledger["complete"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scan(args.prefix, args.output)
