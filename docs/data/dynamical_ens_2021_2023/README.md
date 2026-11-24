# Berry Burn ensemble information experiment

## Calendar and acquisition

Use the daily calendar, target checks and year boundaries frozen in the 30-day pilot at commit 33b3e7a. Reuse its 30 raw daily meter and four-point weather responses after comparing their hashes with the archived receipts. For every other date send one four-query batch and store its raw HTTP body, request, work units and SHA256. A day must have a common product version and snapshot across the four coordinates. Each fresh attempt after a mixed daily snapshot has its own directory. Replays verify the full request and cached body before decoding.

Use the archived monthly meter receipt for the initialization month. Every half-hour index in the new joint file refers to that original monthly response. Retain signed metered MW, UTC intervals, settlement revisions and the offline noon-delivery assumption. The two boundary days go through joint acceptance and stay outside the experiment splits.

## Fixed comparison

Fit a 64-nearest-neighbour empirical distribution of hourly power to the 2021 training hours. Each neighbour contributes equal probability. Evaluate on 2022 validation days. Reserve the 2023 split. Use training-only column means and population standard deviations, squared Euclidean distance, and chronological tie breaking. The neighbourhood size and all feature choices are fixed before validation scoring. This comparison tests marginal distributions.

At each native weather knot, grid point and height, calculate member mean u, mean v and mean horizontal speed. For group B also calculate the population standard deviation of member speed. Interpolate each statistic linearly to the midpoint of the hourly meter interval. Both groups include UTC hour and annual sine/cosine terms and the issue-relative midpoint lead. This yields 29 features for A and 37 for B. Both use the same empirical CRPS in MW, 90% central point coverage and interval width. Report the paired difference B minus A per day and its average.

## Replay

Run from the repository root with the verified pilot and monthly responses present. The information experiment requires NumPy.

```bash
root=outputs/dynamical_ens_multiyear
python -m data_audit.dynamical.multiyear "$root"
python -m data_audit.dynamical.freeze "$root" outputs/dynamical_ens_2021_v1
OPENBLAS_NUM_THREADS=1 python -m data_audit.dynamical.spread "$root"
```

## Frozen data and validation results

All 1,095 cycles passed. The two year-boundary holdouts leave 364 training, 364 validation and 365 test windows. The daily ledger binds each sample to its source bodies and product snapshot. Each split manifest binds its dates to joint file hashes. The summary records the split and ledger fingerprints. The original 30 days retain snapshot WQ1MP2C2VR20KK8EAW80; the remaining 1,065 use 7WQ9WED3MTEE5W9TFMAG. The product version in both is 0.1.0.

Expansion used 1,065 successful batch requests plus a 503 response (January 24, 2022; Retry-After 5 seconds) and a body-read timeout (December 24, 2022). The restarted collector reused the preceding 692 complete batch receipts. Complete response bodies including the 503 account for 84,665,770 bytes; the timeout's partial transfer is unknown. All 36 monthly power responses were reused. Raw response bodies and joints remain in the local output trees.

| 2022 validation, 364 days | CRPS (MW) | 90% coverage | Mean 90% width (MW) |
|---|---:|---:|---:|
| A, central wind information | 6.198885 | 85.22% | 31.4452 |
| B, central wind plus speed spread | 6.258133 | 85.05% | 31.6358 |

The paired mean difference B minus A was +0.059248 MW. B improved CRPS on 167 of 364 days. Adding spread widened the mean interval slightly while lowering coverage. Under the frozen neighbour model and this validation year, speed spread gave no aggregate improvement. This measures the value of eight additional features in the chosen distance rule; the result does not establish that ensembles lack useful uncertainty information. The test split remains reserved.
