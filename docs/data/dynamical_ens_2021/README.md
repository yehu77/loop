# Berry Burn public ECMWF ENS pilot

The sample ID is the Berry Burn BM unit and the UTC initialization date. This data version uses the dynamical.org materialized ECMWF IFS ENS 15-day 0.25° product. The provider backfilled runs to 2020-10-15. Catalog coverage motivates the date search; local acceptance requires actual member values and next-day meter readings. [Data dictionary and licence](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) · [Backfill announcement](https://dynamical.org/updates/2026-11-20/)

## Joint sample

- Initialize at 00 UTC and issue at 12 UTC under the research assumption that operational ENS access delivered the necessary fields by noon. Historical delivery time is unknown. API retrieval timestamps identify this audit only.
- Use latitudes 57.25 and 57.50 and longitudes −3.50 and −3.25. These bracket the REPD 4261 representative location (57.446611, −3.475393). Verify the selected native coordinate of every point response and retain four separate points.
- Retain perturbed members 1–50 in numerical ID order. The source response also preserves control member 0. Save Cartesian u/v wind in m/s at 10 m and 100 m. These heights describe model fields.
- The local wind array axes are grid × member × lead × variable (4 × 50 × 9 × 4). The three-hourly weather knots have initialization leads +24 through +48 and issue leads +12 through +36. The last knot anchors the right boundary of the last target interval.
- Use Elexon B1610 for E_BRYBW-1 and combine two complete half hours per UTC hour into mean MW using the existing settlement-period conversion. Keep signed values, settlement revisions and raw source row indices. There are 24 hourly targets, starting 12 through 35 hours after issue. [Official metered-volume definition](https://www.elexon.co.uk/bsc/article/bsc-insight-how-to-get-more-detailed-metered-volume-data-from-our-insights-solution/)

## Evidence and replay

The 30 pilot cycles are August 1–30, 2021. Each date directory contains four original point responses and one meter response with request receipts. Each receipt carries the URL, parameters, response SHA256, status and body size. The assembled joint.json gives the variable order, times, member IDs, grid points, observed targets, original response references, immutable weather links, product version and Icechunk snapshot. Coverage and payload statistics are derived from these files. The synthetic regression checks exercise member reordering, missing values, invalid units and the end anchor.

From the repository root:

```bash
root=outputs/dynamical_ens_2021_v1
python -m data_audit.dynamical.collect "$root"
python -m data_audit.dynamical.prepare "$root" > "$root/acceptance.json"
python -m data_audit.dynamical.coverage "$root/months" --fetch > "$root/coverage.json"
python -m data_audit.dynamical.report "$root"
python -m unittest discover -s tests -p test_dynamical.py -v
```

Cached responses are hash checked. For offline coverage replay, omit `--fetch`. Live reads are sequential. Rate-limit and server-busy retries honor Retry-After plus jitter; response-body byte totals exclude transport overhead. Actual server read cost can exceed the size transferred to the client. [API bounds and replay links](https://dynamical.org/api/)

## Multi-year scope

Use initialization year for the candidate split: 2021 training, 2022 validation, 2023 test. Count every December 31 cycle in its initialization year's coverage. Hold out 2021-12-31 and 2022-12-31 at the two split boundaries: their final target intervals end after the next split's first noon issue. This protects event-time ordering; reproducing an operational fitting date would also require the availability time of the settlement revision used for fitting. The final 2023-12-31 cycle targets 2024-01-01. Enumerate actual run IDs and check each day's 48 half-hour meter intervals and settlement revisions. Only the 30 pilot days have all four weather fields decoded across the four native points. Full per-day candidate membership, target completeness, split and monthly response fingerprints are recorded in coverage.json. This is data preparation; no model selection or test scoring takes place.

ECMWF IFS ENS data provided by ECMWF and distributed by dynamical.org under CC BY 4.0 and ECMWF Terms of Use. Processing selects four native grid points and excludes the control member from the assembled array. Contains BMRS data © Elexon Limited copyright and database right 2026. [ECMWF dataset terms](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) · [BMRS terms](https://www.elexon.co.uk/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/)

## Results

All 30 consecutive pilot cycles passed, with 216,000 accepted wind values and 720 hourly mean power labels. All 1,440 underlying half-hour readings matched the separate monthly retrieval. All four-point weather panels used product version 0.1.0 and snapshot WQ1MP2C2VR20KK8EAW80.

| Initialization year | Runs listed | Complete target days | Split-eligible candidates | Verified joint windows |
|---|---:|---:|---:|---:|
| 2021 | 365 | 365 | 364 | 30 |
| 2022 | 365 | 365 | 364 | 0 |
| 2023 | 365 | 365 | 365 | 0 |

The pilot returned 150 successful responses and no HTTP errors. The response bodies totaled 2,890,909 bytes, including 2,523,285 weather bytes. Weather queries consumed 220,320 reported work units. The separate three-year coverage audit used 72 responses totaling 13,201,689 bytes. Reusing the monthly meter data, the remaining 1,065 four-point panels would require 4,260 successful point requests and approximately 89.6 MB of response bodies at the pilot rate. This estimate excludes retries and transport overhead. Before comparing ensemble mean against mean plus spread, decode the remaining daily panels, verify their identities and fields, and freeze the resulting split manifests.
