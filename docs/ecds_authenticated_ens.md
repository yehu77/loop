# Berry Burn authenticated ENS retrieval

Branch base `d71298e`. The [offline protocol](ens_only_offline_v1.md) controls the farm, grid, meter target and issue time. This intake covers the 30 consecutive 00 UTC cycles from 2019-08-24 through 2019-09-22. Actual availability stays unknown. Operational availability and dataset training readiness stay false.

## Local acceptance record

The preflight found neither a configured CDSAPI_KEY nor a client configuration file. No authenticated request was submitted. The first cycle stopped locally, the other 29 cycles remain unattempted, and the body byte count for ECDS and B1610 retrievals was zero. Account and licence status are unknown. No HTTP response was attributed to this run. The existing Zenodo window is an independently accepted offline reference. There is no real first-day GRIB comparison or new complete authenticated window yet. [Daily inventory](../data_audit/ecds/coverage.json)

## Setup and replay

Log in to the ECMWF Data Store, open the TIGGE download form and accept the dataset terms with the account that will submit the request. In the same WSL user account create the personal configuration shown on the official setup page in `~/.cdsapirc` (mode 600). Use `url: https://ecds.ecmwf.int/api` and the personal token supplied by ECDS. Keep the file outside Git. [Official API setup](https://ecds.ecmwf.int/how-to-api) · [TIGGE dataset and licence](https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download)

From the repository root, use the prepared local Python environment. To recreate it on another machine, use Python 3.10+ and install `cdsapi>=0.7.7` and `eccodes` in a separate environment.

```bash
outputs/ecds-client-env/bin/python -m data_audit.ecds.run --output outputs/ecds_ens_v1
outputs/ecds-client-env/bin/python -m unittest discover -s tests -p test_ecds_acquisition.py -v
```

The runner uses the official client for request submission and the result's delivery URL for a bounded binary read. A reported GRIB size over 20,000,000 bytes stops before the file transfer. No signed URL or HTTP authentication headers enter the receipts. Each date stores the exact request, its SHA256 and local file hashes. Completed artifacts are verified before reuse. The first three targets reuse the frozen B1610 responses. Later target responses remain beside their GRIBs. The ledger distinguishes missing configuration, HTTP 401, permission/licence rejection, request-parameter rejection, delivery failure and content rejection. No subsequent date runs past a failed gate. `downloaded_body_bytes_this_run` counts data payload bytes read from delivery and meter endpoints, excluding metadata, installation traffic and replay of local files.

## Frozen first-cycle comparison

Join by ECMWF initialization, member, valid time, coordinates and wind component. Decode GRIB message identity, height, units, missing-value count, coordinates and instantaneous lead before adapting to the existing evaluator. Keep the original GRIB for native metadata. The numeric hours-since-1900 coordinate in the adapter is derived from GRIB validity time for compatibility. Preserve signs and compare all 2,000 wind values in m/s. Normalize longitude to −180…180. [TIGGE GRIB encoding](https://confluence.ecmwf.int/spaces/TIGGE/pages/51721961/Rules+for+data+encoding+and+exchange)

The SHA256-verified source prefix for the old packed NetCDF is preserved locally at `outputs/ecds_ens_v1/reference/zenodo-13255991-prefix.gz`. Its verified SHA256 and reconstruction command are in the [original evidence record](data_feasibility_v1.md). The GRIB download never uses this archive as a fallback.

Before receiving any new wind values, the per-value absolute tolerance was fixed to half the reference NetCDF scale factor plus the message-specific ecCodes `packingError` plus 1e-12 m/s for floating arithmetic. The CDF-2 header of `2019/tigge-GB_2019082400.nc` yields scales 0.0005914715818466534 (u10) and 0.0006574601748562554 (v10). This assumes nearest-integer NetCDF packing. Different grids, heights, missingness or source revisions require inspection. The median, correlation or average error cannot override an individual mismatch. The comparison records maximum errors and out-of-bound keys and values for both components.

Once this and the first day's complete power window pass, 08-25 and 08-26 run in order. Only three accepted windows unlock the remaining 27. All outputs stay under the new acquisition directory. Thirty accepted days would establish retrieval and window construction only.

## Verification

Local Python 3.10.20, cdsapi 0.7.7, eccodes Python 2.50.1 and eccodeslib 2.48.2.27. Four acquisition tests passed, including a generated 500-message GRIB2 file in reverse field order, 0…360 longitudes, a deliberate value mismatch, a wrong cycle, cache replay and a no-credential stop. These are controlled decoder and workflow checks. They do not measure ECDS numerical agreement. The two previous data audit suites also passed (7 + 8). The final preflight reproduced the committed coverage ledger.
