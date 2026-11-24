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
