# Berry Burn day-ahead data contract v1

The audit unit is BM unit `E_BRYBW-1`. The nominated weather initialization is 2019-08-24 00:00 UTC. The outcome covers [2019-08-25 00:00, 2019-08-26 00:00) UTC. Initialization identifies the model run. An operational forecast also needs an issue time before delivery and evidence that each input was released by that time. Both weather release times and the operational issue time are currently unknown.

## Meter and target

Use the official B1610 BM unit, local settlement date and settlement period as the raw key. Convert midnight in `Europe/London` to UTC **before** adding elapsed half hours. Period s covers [midnight + (s−1)/2 hours, midnight + s/2 hours). Local days contain 46, 48 or 50 periods. Compare the computed right endpoint with the API's literal `halfHourEndTime`. The suffix-free field agrees with UTC on the archived rows. Keep both representations and the settlement run type for revision tracing. [Elexon settlement period](https://www.elexon.co.uk/bsc/glossary/settlement-period/) · [Elexon B1610 units](https://www.elexon.co.uk/bsc/article/bsc-insight-how-to-get-more-detailed-metered-volume-data-from-our-insights-solution/)

The outcome is delivered metered generation. Each B1610 quantity in MWh divided by 0.5 h gives interval-average MW. Sum the two energies in a UTC hour and divide by 1 h for the hourly target. Keep zeros and signed values. Flag absent or nonfinite halves, reject duplicate intervals, retain both source-row indices. BAV is ancillary quality/curtailment information, currently unknown per row. Conditioning or filtering on BAV changes the population. The authors' MW table masks observations and their final table divides by estimated available capacity. Neither substitution is part of this target. Meter readings are verification outcomes only. Historical power needs separate as-of evidence for the relevant revision. [Author data dictionary](https://zenodo.org/records/13256014)

## Forecast fields

| Field | HRES | ENS |
|---|---|---|
| Identity | Operational deterministic GRIB, decoded initialization and step | TIGGE operational member, dated cycle filename and NetCDF valid-time coordinate |
| Wind | u/v at 100 m and 10 m, m/s. Confirm parameter IDs, height and units on the actual messages | u10/v10, m/s, raw scaled shorts with fill checked before scale and offset |
| Time axis | Keep native steps and instantaneous/accumulation metadata | Keep leads +24,+30,+36,+42,+48 h, including next midnight as a boundary knot |
| Space | Record native grid, point coordinates and scan direction from GRIB | Four grid points, lat 57/57.5 N × lon −4/−3.5 E; retain both axes |
| Other fields | Archive lists 2 m temperature, downward solar radiation, three cloud layers and precipitation; candidate contract selects wind only | Source units for all variables retained in ens.json; only wind values extracted |

The ENS patch brackets the published research location 57.48712 N, 3.54822 W. Reconcile with the authors' farm metadata before fixing a spatial interpolation rule. A member ID remains the same along its entire trajectory. Require a complete, unique grid × member × lead panel. Preserve the lead and physical height when deriving features; a downstream hourly interpolation would use the original five weather knots. [Wind farm location, Table A.1](https://api.repository.cam.ac.uk/server/api/core/bitstreams/46bff039-dcbc-408d-aa14-0342e41a868d/content) · [ENS](https://zenodo.org/records/13255991) · [HRES](https://zenodo.org/records/13254469)

## Sample acceptance

Require both products from the exact initialization, validated target coverage and recorded source hashes. Publication must precede the declared issue time; unknown publication stays unknown. No zero-lag operational availability assumption is adopted here. Exclude future ERA5, realized power, future BAV and retrospective capacity estimates from the conditions. JSON null denotes unavailable evidence. A blocked sample may retain its measured components. Its readiness flag remains false and split counts remain null until a full inventory and chronological split exist.
