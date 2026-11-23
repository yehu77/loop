# Zenodo ENS bounded archive scan

The first 256 MiB of the compressed archive contained neither requested header for 2019-08-25 00 UTC or 2019-08-26 00 UTC. The date inventory beyond that boundary remains unknown. The scan saw 11 NetCDF member headers in nonchronological order. It stopped inside the June 9 12 UTC member when the approved byte range ended. No further request was made. [Source record](https://zenodo.org/records/13255991) · [Ranges, encountered names and SHA256 digests](../data_audit/zenodo_stream/scan_result.json)

The existing verified 16,000,000-byte prefix supplied the beginning of the gzip stream. One HTTP 206 response supplied the next 252,435,456 bytes. The local continuation and extracted raw file remain in `outputs/zenodo_ens_scan_v1/`. The gzip trailer and whole-archive checksum are outside this scan.

The August 24 anchor was recovered as a complete 26,356,124-byte tar member. Its CDF-2 dimensions, variable extents, 50 member IDs, and 61 native six-hour times (0…360 h) passed. Both 10 m wind components were finite at every member/time on the four protocol grid points. The five-knot panel matched the earlier frozen rows exactly and passed the original ENS-only evaluation with its 24 official hourly targets. The sample count remains one. [Acceptance by origin](../data_audit/zenodo_stream/acceptance.json)

Offline replay using the retained raw member requires no additional archive reads over the network:

```bash
python -m data_audit.zenodo_stream.verify outputs/zenodo_ens_scan_v1 > /tmp/zenodo-ens-acceptance.json
cmp /tmp/zenodo-ens-acceptance.json data_audit/zenodo_stream/acceptance.json
```

The previous 15 audit tests passed. A local negative control removed just the last byte of the anchor and supplied matching truncated size and SHA256; the independent NetCDF variable-extent check rejected it.
