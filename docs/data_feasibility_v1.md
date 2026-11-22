# Berry Burn feasibility audit

**Decision.** Keep the requested origin. The 24-hour delivered-power target and native 50-member ENS patch are reproducible. Same-cycle HRES and operational release evidence are missing. The [partial sample](../data_audit/evidence/sample.json) records `train_ready=false`. Zero joint samples have passed this audit. Full training, validation and test counts are unknown.

Source revision `be0d9023a1dcb4351a2ce81193d9d03d2aaaf685`, branch `codex/berry-burn-data-audit`. Only the new data audit was exercised.

## What the meter evidence establishes

Eight single-unit API requests produced 343 rows. The [request ledger](../data_audit/evidence/b1610/requests.json) records exact URLs, byte counts, row counts and SHA256 of the unmodified responses. All 343 settlement keys reproduce the returned half-hour end. Each nonempty query includes one extra half hour at the upper bound. For the August target, 48 rows form 24 complete hours; the first two energies are 2.694 and 2.554 MWh, giving **5.248 MW** for 00:00–01:00 UTC. The returned DF records have null publication times. These revised readings establish delivered power retrospectively. Recent historical power has no verified operational availability here.

| Query UTC day | Rows | Author code's predicted naive label minus official UTC interval **start** |
|---|---:|---|
| 2019-01-15 | 0 | Empty response retained |
| 2020-01-15 | 49 | 0 h throughout |
| 2019-08-25 and 2023-07-01 | 49 each | +2 h throughout |
| 2019-03-31 and 2020-03-29 | 49 each | 46 at 0 h, then 3 at +2 h |
| 2019-10-27 and 2020-10-25 | 49 each | 48 at +1 h, then 1 at 0 h |

**Code evidence, separate from table evidence.** In `read_bmra/bmra_wind.py`, `_readRawFiles` forms naive settlement date + (period−1) × 30 minutes. For year-ordered inputs, `_identify_TZchanges` adds an hour beginning the Monday after the March transition through the October transition's nominal midnight. The summer rule explains how a two-hour discrepancy can arise relative to UTC start. The comparison script reproduces this stage only. The subsequent pipeline removes all duplicate labels, reindexes half-hourly, and masks wherever its BAV table is non-null. The documentation describes the intended filter as nonzero BAV; explicit zeros in the raw BAV encoding still need inspection. The author's pre-2021 raw-file multiplier is 2. Current Insights returns MWh; use the official quantity definition independently of legacy API vintages. [Preprocessing code](https://zenodo.org/records/13309890) · [BMRA archive](https://zenodo.org/records/13256014)

The archive prefix identifies a 53,426,661-byte MW Parquet member. The available prefix lacks its complete time index and the annual RAW CSVs. No row-level *observed* author-table offsets or BAV decisions are asserted. The 2023 anecdote warrants a keyed author export; the clock evidence rules out a universal two-hour repair. The optional comparison input is a CSV with the original naive `Time` index and `E_BRYBW-1` MW column. Preserve missing values.

## What the weather evidence establishes

The ENS archive starts with `2019/tigge-GB_2019082400.nc`. A 16,000,000-byte compressed prefix contains the necessary CDF-2 coordinate arrays and wind records. The [extracted panel](../data_audit/evidence/ens.json) contains 1,000 rows (50 IDs × five native times × four cells), with two finite components per row. At lead +24, member 1, 57.5 N/4 W, u/v are −0.273278/+0.048044 m/s. Raw time 1048800 hours since 1900-01-01 gives 2019-08-25 00:00 UTC. File conversion history dates describe retrospective processing. [ENS archive](https://zenodo.org/records/13255991)

The HRES prefix starts with `2019/hres-UKNS_20191225.grib`. Its first GRIB1 message encodes 2019-12-25 00:00, table 128 parameter 165, step 0 and a 116 × 96 regular grid at 0.1°. This verifies a different cycle only. The requested August day may exist deeper in the 47.9 GB gzip stream. A December substitution would also need a matched ENS slice. No alternative cycle was accepted. [HRES archive](https://zenodo.org/records/13254469)

## Smallest next acquisition

Request the August 24 00 UTC operational HRES surface wind subset from MARS or the authors, with original GRIB metadata and surrounding grid, and a one-column BMRA slice with raw settlement keys, original labels and BAVs for the eight dates. Verify issue-time delivery or explicitly freeze a retrospective release assumption. These small extracts are worth acquiring. Defer full weather downloads until one joint sample passes. Ensemble uncertainty research then needs an inventory of matched cycles, missing members, revisions and chronological split coverage, followed by an agreed treatment of native 6-hour leads and wind height. One complete day cannot estimate ensemble calibration.

## Replay and evidence hashes

From the repository root, offline:

```bash
python -m data_audit.audit > /tmp/berry-sample.json
cmp /tmp/berry-sample.json data_audit/evidence/sample.json
python -m data_audit.compare > /tmp/berry-clock.csv
python -m unittest discover -s tests -p 'test_berry_burn_audit.py' -v
```

For a new snapshot use a fresh directory. The API can revise responses. To reproduce the ENS extraction, require HTTP 206 with the requested byte range and compare the prefix digest below.

```bash
python -m data_audit.fetch /tmp/berry-b1610-new
curl -fLsS --range 0-15999999 --max-filesize 16000000 'https://zenodo.org/records/13255991/files/ecmwf_ens_2019-2020.tar.gz?prefix=16m' -o /tmp/berry-ens-prefix
python -m data_audit.ens /tmp/berry-ens-prefix > /tmp/berry-panel.json
cmp /tmp/berry-panel.json data_audit/evidence/ens.json
```

| Source file or byte range | Observed SHA256 |
|---|---|
| ENS compressed bytes 0–15999999 | `72fa56c0acf4de8a61695da8f6f49a4760b94a848061f39cdcfb0a944482991a` |
| [HRES](https://zenodo.org/records/13254469/files/ecmwf_hres_2019-2021.tar.gz), bytes 0–1999999 | `cab2c92f118471accc1cc3283acd664d4a71fe377feaea600a8737fae5abb8ff` |
| [BMRA](https://zenodo.org/records/13256014/files/BMRA_Data.tar.gz), bytes 0–1999999 | `dc53dfa4d8ebeacceae8f06b8d0bf55bbfddba6f185f275aa9226f6e2f009db3` |
| [Author software](https://zenodo.org/records/13309890/files/read_bmra.tar.gz), complete 239259 bytes | `3a2502d004994055bc74825a1ff4e1bb1cccb234a11bc5041bf9a66dec2b7863` |

Use the same curl command with bytes 0–1999999 and a 2000000-byte limit for the other prefixes. Inspect leading names with `tar tvzf`; end-of-input is expected for a prefix. Run `tar xOzf read_bmra.tar.gz read_bmra/read_bmra/bmra_wind.py` for the preprocessing source. The checked-in evidence and tests replay without network access.

Local audit tests passed (8 passed, 0 skipped). The checks cover sample identity, seasonal clocks, measured units, missing halves, native ENS trajectories, mixed origins, as-of checks and byte-for-byte replay. Archived evidence retains original source attribution. Contains BMRS data © Elexon Limited copyright and database right 2026. [BMRS terms](https://www.elexon.co.uk/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/) · [ECMWF ENS dataset and licence](https://zenodo.org/records/13255991)
