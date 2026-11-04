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
