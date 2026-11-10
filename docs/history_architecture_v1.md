# History × architecture interaction v1

## Question and frozen scope

Reviewed base: `699980c`; implementation branch:
`codex/history-architecture-interaction`. This single-seed development experiment
asks whether full history benefits shared two-round computation more than BASE
or ordinary untied depth. It does not require LOOP to win.

Reuse the saved BASE-H0/H24 results from the previous screening, without loading
those checkpoints for training under changed sources. Add exactly four runs:
LOOP-H0 → UNTIED-H0 → LOOP-H24 → UNTIED-H24. Every new run starts from random
initialization. No H1, new seeds, intermediate supervision, or test scoring.

The machine-readable freeze is [configs/history_architecture_v1.json](../configs/history_architecture_v1.json).
All six cells use the same 545 training windows, 57 validation days, training-only
statistics and one-hour observation-delay benchmark. The unresolved timestamp
and availability assumptions in [the data protocol](history_information_v1.md)
remain in force. Prepared test windows are not scored.

## Interface and initialization

The 48-token interface retains 24 historical and 24 future positions. All blocks
receive the same Boolean key visibility mask; H0 hides every historical key and
H24 exposes all 24. Only the future positions are noised, predicted and sampled.
BASE executes Core0 once; LOOP executes Core0 twice; UNTIED executes independent
Core0 and Core1. Intermediate readout remains rejected in history mode.

Canonical BASE initialization uses seed 42. Common parameters and Core0 match
across architectures; UNTIED Core1 keeps its independent initialization from
seed 10042. The history projection uses isolated seed 20042. Full initial state
hashes match between H0/H24 within each architecture; only corresponding common
keys are compared across architectures. No learned BASE parameters are reused.

Training stays FP32, final-only noise MSE, AdamW with the existing learning rate
and clipping, microbatch 4 × accumulation 8, and 3,000 successful updates.
EMA copies raw at update 1,000 and averages from 1,001 with decay 0.999. Validation
uses all 57 days, 100 scenarios, DDIM50, chunk16, and every 250 updates. Primary
results use final EMA at 3,000; auxiliary best uses minimum final-readout EMA CRPS
at the nine common eligible nodes, with earliest tie. Data, training-noise and
validation random streams retain the previous rules.

## Acceptance before launch

Local environment: Python 3.10.20, PyTorch 2.10.0+cu128, RTX 2060.
[Archived evidence](experiments/history_architecture_v1/acceptance/review.json):

- Full CPU suite: 150 tests, **143 passed / 7 skipped**.
- CUDA architecture suite: **5 passed / 0 skipped**. Active output heads and
  attention gates check hidden-history isolation, visible-history influence,
  four masked blocks, core sharing/independence and future-only sampling.
- Full-width real-data CUDA smoke runs: all four cells, **6+2 resumed updates
  exactly match 8 continuous updates**, including raw, EMA and RNG states.
  Wrong-source checkpoints are rejected; variant/group/data guards are covered.
- BASE-H0/H24 replay against `699980c`: five updates and fixed-noise sampling
  match exactly on CPU and CUDA, including initialization, gradients, raw/EMA
  and training noise state.
- Existing BASE final/best artifacts independently re-score identically.
  All 2,629 protected artifacts passed hash checks before launch.

Commands (acceptance outputs already exist; use a fresh output for reruns):

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
python -m unittest tests.test_history_architecture -v
python -m diagnostics.check_history_architecture --mode replay --device cuda
python -m diagnostics.check_history_architecture --mode real --device cuda \
  --data outputs/history_information_v1_20261103/data.pt --output NEW_DIRECTORY
```

These short checks certify the paths exercised, not formal forecasting results.
The temporary smoke EMA start is 3; formal EMA start remains 1,000.

## Bounded execution and recovery

```bash
python -m diagnostics.run_history_architecture
```

The queue refuses existing output directories, records source/data/initialization
identity, and audits each completed cell before advancing. Status is in
`outputs/history_architecture_v1_queue.json`; runs use separate
`outputs/history_architecture_v1_s42_{variant}_{group}` directories. Any exception
stops for inspection, preserving artifacts. Normal source-bound recovery uses
`--updates 3000-COMPLETED` as an additional update count, with a new continuation
directory and audited segment lineage. Never relax checkpoint source guards or
clear a failure marker. Frozen source files must remain unchanged while running.

New cost records include actual CUDA allocator peaks over the training segment
plus raw/EMA validation, before best reload. Existing BASE formal peak memory was
not recorded and remains missing; short smoke peaks are not substitutes.
Training-update time excludes validation and checkpoint IO; sampling latency is
reported separately. Equal update counts mean equal sample exposures, not equal
compute.

## Predeclared report and interpretation

The read-only six-cell summarizer keys every cell by `(variant, history_group)`.
It checks source lineage, checkpoint clocks, initial common parameters, training
streams, target/sample identity and fixed validation noise, then independently
re-scores saved scenarios. It preserves per-day and 1–6/7–12/13–24 lead results.

Three required tables are: six fixed-endpoint cells; same-input differences;
and within-architecture H24−H0 effects. For every score, also calculate:

```
I_LOOP-BASE   = (LOOP_H24 - LOOP_H0) - (BASE_H24 - BASE_H0)
I_LOOP-UNTIED = (LOOP_H24 - LOOP_H0) - (UNTIED_H24 - UNTIED_H0)
```

Negative interactions indicate a larger LOOP improvement for lower-is-better
scores; they do not establish better absolute quality. Coverage and width must
be interpreted jointly, rather than assigning them the same score direction.
CRPS remains primary; ES, VS, signed 1h/3h increment CRPS, daily-mean CRPS,
coverage, width and measured costs remain visible. Auxiliary best is separate.

One training seed and 57 repeatedly used development days cannot establish
stable generalization. Similar LOOP/UNTIED improvement supports depth, not a
shared-computation advantage. Small, contradictory or only local gains do not
justify automatic expansion. All four runs must finish before deciding whether
independent seeds are warranted; no additional training is queued.

## Run status

Interface acceptance is complete. Formal queue launch and completion evidence
will be appended without modifying the frozen sources or previous reports.

The four-job queue was launched from implementation freeze `1bb6837` on the
local RTX 2060. The first LOOP-H0 config matches every frozen training/evaluation
field, all 57 validation IDs and saved BASE initial tensors.
[Launch snapshot](experiments/history_architecture_v1/launch.json) records the
process identity. This snapshot is not a completion claim; live status remains
local and the final six-cell report is pending the four runs.

## Completed experiment and review (2026-11-10)

All four new jobs finished 3,000 successful updates (3,000 attempts, no skipped updates), in the frozen order. Each passed the per-run engineering audit. The automatic queue stopped after reporting; no extra training was launched. All six auxiliary best checkpoints are also update 3,000, so best does not change the ranking.

This review independently re-ran source/identity/checkpoint audits and re-scored the saved validation artifacts. The full summary reproduced exactly; all 3,169 protected artifacts and frozen implementation hashes match. No GPU training or original acceptance tests were rerun during this review, and the test split remains unscored.

### Six-cell fixed-endpoint table

All scores below are lower-is-better. Coverage and width are reported separately. Seed 42; 57 common validation days; final delayed EMA at update 3,000.

| Model | Input | CRPS | ES | VS | 1h increment | 3h increment | Daily mean |
|---|---|---:|---:|---:|---:|---:|---:|
| BASE | H0 | 0.074919 | 0.462751 | 14.516202 | 0.050975 | 0.086097 | 0.036394 |
| BASE | H24 | 0.074377 | 0.455977 | 13.974425 | 0.051606 | 0.085051 | 0.036270 |
| LOOP | H0 | 0.076796 | 0.472289 | 14.515052 | 0.050711 | 0.085801 | 0.039211 |
| LOOP | H24 | 0.076153 | 0.464278 | 13.989749 | 0.050971 | 0.085539 | 0.038349 |
| UNTIED | H0 | 0.074619 | 0.463169 | 14.301930 | 0.050785 | 0.085503 | 0.036548 |
| UNTIED | H24 | 0.075147 | 0.460370 | 14.079697 | 0.050888 | 0.085023 | 0.036416 |

| Model | Input | 90% point coverage | Width | Parameters | Train-update seconds | Sampling seconds | CUDA allocated / reserved MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| BASE | H0 | 91.08% | 0.446282 | 208,193 | 289.97 | 57.97 | not recorded |
| BASE | H24 | 87.28% | 0.422035 | 208,193 | 287.27 | 56.76 | not recorded |
| LOOP | H0 | 91.96% | 0.458327 | 208,193 | 446.56 | 95.99 | 25.88 / 32.00 |
| LOOP | H24 | 88.96% | 0.442202 | 208,193 | 428.58 | 94.40 | 25.88 / 32.00 |
| UNTIED | H0 | 91.45% | 0.436544 | 365,825 | 470.95 | 95.14 | 28.90 / 34.00 |
| UNTIED | H24 | 86.92% | 0.422274 | 365,825 | 455.20 | 93.91 | 28.90 / 34.00 |

Sampling seconds cover the same 57 × 100 trajectories, DDIM50/chunk16. Training-update seconds exclude validation and IO. CUDA allocator peaks are not total device/driver memory. Historical BASE formal peak memory is unavailable. Timing comes from these local runs, not repeated latency trials.

### Same-input differences (H24)

Positive score differences favor the comparison model, not LOOP.

| Comparison | ΔCRPS | ΔES | ΔVS | Δ1h increment | Δ3h increment | Δdaily mean |
|---|---:|---:|---:|---:|---:|---:|
| loop-base | +0.001776 | +0.008301 | +0.015324 | -0.000636 | +0.000488 | +0.002079 |
| untied-base | +0.000769 | +0.004393 | +0.105272 | -0.000718 | -0.000028 | +0.000146 |
| loop-untied | +0.001007 | +0.003908 | -0.089948 | +0.000082 | +0.000516 | +0.001933 |

### History effects and interactions

| H24−H0 effect / interaction | ΔCRPS | ΔES | ΔVS | Δ1h increment | Δ3h increment | Δdaily mean |
|---|---:|---:|---:|---:|---:|---:|
| base | -0.000542 | -0.006775 | -0.541777 | +0.000631 | -0.001046 | -0.000124 |
| loop | -0.000642 | -0.008011 | -0.525303 | +0.000259 | -0.000262 | -0.000861 |
| untied | +0.000527 | -0.002799 | -0.222233 | +0.000103 | -0.000480 | -0.000132 |
| I: loop-base | -0.000100 | -0.001237 | +0.016474 | -0.000372 | +0.000784 | -0.000737 |
| I: loop-untied | -0.001169 | -0.005212 | -0.303069 | +0.000156 | +0.000218 | -0.000729 |

### Lead-group CRPS effects

| Model | H24−H0, hours 1–6 | Hours 7–12 | Hours 13–24 |
|---|---:|---:|---:|
| BASE | -0.007220 | -0.001271 | +0.003162 |
| LOOP | -0.004362 | -0.000452 | +0.001123 |
| UNTIED | -0.005115 | -0.001545 | +0.004384 |

### Decision

The current fixed-R2 history interface does **not** establish a useful shared-loop quality–cost advantage. Under H24, LOOP CRPS is 2.39% worse than BASE and 1.34% worse than UNTIED; ES is also worse than both. LOOP improves 1h increment CRPS over BASE, and its 90% point coverage (88.96%) is closer to nominal than BASE (87.28%) and UNTIED (86.92%), but its intervals are wider. These mixed benefits do not reverse the primary result.

The CRPS interaction versus BASE is only −0.000100: the LOOP–BASE gap shrinks from +0.001877 to +0.001776, while LOOP remains worse. The interaction versus UNTIED is −0.001169, partly because UNTIED CRPS worsens after history is added; it is not evidence that LOOP-H24 wins. LOOP-H24 has lower daily CRPS on 25/57 days versus BASE and 28/57 versus UNTIED. Those are paired weather-day descriptions, not independent training repetitions.

All three architectures improve short-lead CRPS after adding history but worsen at hours 13–24. BASE and LOOP gain only slightly overall; UNTIED worsens. All three narrow their intervals and reduce coverage. Thus history effects are heterogeneous, and there is no uniform full-trajectory gain. LOOP-H24 sampling takes 94.40 seconds versus BASE 56.76 (1.66×); training-update time is 1.49×. LOOP has 43.1% fewer parameters than UNTIED but does not improve its CRPS or sampling latency in this run.

Following the predeclared stop rule, archive this experiment and do not automatically add seeds, recurrence depth, training length or tuning to the current R2 recipe. This is one-seed development evidence on repeatedly used validation days; it does not rule out all recurrent architectures. Any new direction needs a separately specified hypothesis. Existing data, historical results and the sealed test split remain unchanged.

[Machine-readable results and paired daily/lead differences](experiments/history_architecture_v1/results/summary.json) and [independent review evidence](experiments/history_architecture_v1/results/review.json) are archived. Large scenarios/checkpoints remain local.
