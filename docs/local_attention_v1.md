# R02: strictly hourly conditional reference (LOCAL-2)

## Frozen question and scope

Based on `ef50c1b`, R02 removes cross-hour information exchange from BASE-2 and
trains from scratch. R01's shuffled outputs are not a trained local reference.
Existing nine v2 runs and R01 are retained unchanged. Only LOCAL-2 seeds 42, 43,
44 are authorized; R03 and subsequent stages remain unstarted.

## Model and checkpoint identity

LOCAL-2 is `variant="base", attention_scope="diagonal"`. Boolean SDPA masks allow
only the diagonal (True means allowed). All encoders, hourly positions, adapters,
AdaLN, FFN, output head and initialization remain matched. Each denoising call
executes one two-block Core. No history, convolution, extra loss or recurrence
is introduced. There are 208,065 parameter elements, matching BASE. A single
allowed key makes Q/K irrelevant to attention weights in exact arithmetic;
floating-point kernels can leave tiny gradient residuals. This is an interaction
ablation, not equal effective capacity or equal FLOPs. The dense masked operator
is retained; measured timings describe this implementation, not an optimized MLP.

Absent `attention_scope` defaults to `full`, with the historical SDPA call intact.
The field is saved in model config and checked when explicitly supplied on resume.
An omitted resume flag inherits the saved scope. Old best-checkpoint comparisons
normalize missing defaults. Tensor state dictionaries alone do not identify the
scope; use the saved model configuration when loading weights.

## Frozen experiment

`configs/local_attention_v1.json` pins source/data hashes, original artifact hashes,
all 71 validation IDs, ordered runs, and the unchanged v2 recipe: Zone 3,
547/71/90 windows; FP32; microbatch 4 × accumulation 8; 3,000 successful updates;
EMA copies raw at 1,000 and averages with 0.999 thereafter. Validate every 250
updates with 100 scenarios, 50 DDIM steps and chunk size 16. Initialization, data
order and training-noise streams match the same-seed historical BASE; validation
seed 2026 and RAND seed 2027 stay fixed. The test split is not scored.

Primary comparison: delayed EMA at update 3,000, LOCAL minus same-seed original
BASE. Auxiliary best: minimum EMA validation CRPS at the nine common nodes
1,000–3,000, earlier ties. Increment scores do not select checkpoints. Report
CRPS, ES, VS, 80%/90% point coverage and width, signed 1h/3h increment CRPS and
daily-mean-power CRPS, together with per-day differences and measured costs.
Average days within each trained model, then report the three paired seed
differences, mean and sample SD. Days are not independent training repetitions.
LOOP and UNTIED remain contextual references. No scoring-driven recipe changes.

## Verification and execution

Tests explicitly open the zero output head and attention residual gates. For
noisy power, NWP and calendar perturbations, LOCAL must isolate other hours and
the matched full-attention positive control must detect cross-hour changes.
Complete DDIM chains use fixed initial noise; both condition types are checked.
Tests also cover historical full execution, matched initialization/RNG, legacy
best resume, scope mismatch rejection before RNG mutation, and exact continuation.

```bash
python -m unittest discover -s tests -p test_local_attention.py -v
python -m diagnostics.run_local_reference
```

The queue refuses existing runs, checks frozen hashes before each launch, and
stops on failure. Seed 42 must pass endpoint, initialization, RNG, EMA and scoring
audits before 43 and 44 start; its numerical ranking is not a gate. Logs and full
checkpoints remain under `outputs/local_v2_s{seed}`. Interruptions require inspection
and a new resume segment, never deletion or replacement of old artifacts.

## Status

Training implementation and the predeclared run configuration were frozen in
`46360ec`. CPU regression: **104 tests, 100 passed, 4 CUDA-only skipped**
(Python 3.10.20, PyTorch 2.10.0+cu128; CUDA unavailable inside the sandbox).
The **7 LOCAL-specific tests all passed on the actual RTX 2060**, including
FP32 optimizer updates and exact resumed continuation. CPU logs are
`outputs/r02_tests_cpu.log`; GPU-specific logs are `outputs/r02_tests_cuda.log`.
The two report tests check seed-level pairing and reject scope, initialization,
protocol, sample identity, RNG and data-position drift. Syntax and whitespace
checks passed. These are engineering results, not forecasting outcomes.

The queue is running, starting with seed 42. No formal R02 result is claimed yet.
Once all three runs pass engineering audits, generate the independent report:

```bash
python -m diagnostics.summarize_local_reference \
  --output outputs/local_attention_v1_results
MPLCONFIGDIR=/tmp/loop-pilot-mpl python -m diagnostics.plot_local_reference \
  --results outputs/local_attention_v1_results \
  --output outputs/local_attention_v1_results/comparison.png
```

The summary re-scores primary and auxiliary-best saved panels for all 12 models
(nine historical plus three LOCAL), preserving per-lead arrays locally and
per-day CSVs. It verifies original input hashes before and after scoring. LOCAL
initial tensors and final data/noise RNG positions must match same-seed BASE.
Training costs exclude validation and checkpoint I/O; observed latency is not
a theoretical FLOP comparison. The old BASE primary/best panels for seed 42 have
already been independently re-scored successfully with the unchanged evaluator.

