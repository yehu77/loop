# R04: intermediate supervision with a shared output head

## Frozen question and four-cell comparison

Based on `0e0c685`, R04 tests a new training constraint, not a repair for a
confirmed defect. Existing LOOP/UNTIED final-only seeds 42/43/44 supply two
cells; six new runs supply LOOP-DS/UNTIED-DS at those same seeds. Historical BASE
remains the shallow reference. R01–R03 and LOCAL results remain unchanged.

One pass produces H1 then H2. The same existing output head reads both states;
LOOP reuses Core0 and UNTIED uses Core1 at its second stage. The loss is
`0.5*MSE(eps1.float(), noise.float()) + 0.5*MSE(eps2.float(), noise.float())`.
Both losses use identical noisy input, timestep, conditions and noise target.
There is no detached state, extra head, embedding, loss, recurrence depth or
input branch. Weights sum to one to avoid an arithmetic doubling of loss;
this does not guarantee matched gradient magnitude, direction or optimization.
Parameter counts remain 208,065 (LOOP) and 365,697 (UNTIED).

Final-only retains the original model call and loss branch and does not compute
intermediate predictions/losses. Sampling and validation use the final readout
only, never averaging the two predictions. The optional `forward_stages` path
calls each stage once, calling the same head twice within that pass.

## Configuration and checkpoint identity

`TrainConfig.supervision_mode` defaults to `final-only`; missing legacy
`loss_weights` default to `[0,1]`. DS uses `intermediate` with `[0.5,0.5]`.
Both fields are stored in checkpoints and run configs. Explicit resume flags
must match saved values before trainer construction/RNG changes; omitted flags
inherit the saved mode. Old v1/v2 checkpoints keep final-only behavior. JSON
lists and checkpoint tuples are compared through normalized TrainConfig values,
including old best checkpoints. Mode/weight mismatches are rejected.

All six formal runs start from matched random initialization, not trained
checkpoints. The existing v2 recipe is frozen in
`configs/intermediate_supervision_v1.json`: Zone 3, original fingerprint and
547/71/90 split; full attention, two Core executions; FP32, effective batch 32,
AdamW and clipping unchanged; 3,000 successful updates; EMA copied at 1,000 then
averaged at 0.999. Use all 71 validation days, 100 scenarios, 50 DDIM steps,
chunk size 16, validation every 250. Init/data/noise RNG streams match the
same-seed corresponding original architecture; validation seed 2026 and RAND
seed 2027 remain fixed. Test data is not scored.

Order: seed 42 LOOP-DS then UNTIED-DS, followed by the same order for 43 and 44.
No recipe change depends on early scores. Each completion must pass artifact,
EMA, initial-state, final RNG/data-position and scoring audits before advancing.
On any failure the queue stops for inspection and never overwrites old outputs.
There is no automatic extension, extra seed, weight search or R05/R06 run.

## Evaluation and interaction

Primary: final-readout delayed EMA at update 3,000. Auxiliary best: final-readout
EMA validation CRPS at the nine common nodes 1,000–3,000, earlier ties. Neither
intermediate MSE nor transformed scores select checkpoints. DS logs retain
loss1, loss2 and loss_total (`loss` remains an alias for total); these training
losses are not directly equivalent to original final-only training losses.

Score CRPS, ES, VS, 1h/3h increment CRPS, daily-mean-power CRPS, point coverage and
width at 80%/90%, plus parameter count, synchronized update cost and sampling
latency. Reuse the existing identity-aligned evaluators and R01 transformations.
Per-lead arrays stay local; per-day and per-seed values are retained. Report
absolute model scores alongside paired supervision effects within each seed:

- delta_loop = LOOP-DS minus LOOP-final;
- delta_untied = UNTIED-DS minus UNTIED-final;
- interaction = delta_loop minus delta_untied.

For lower-is-better scores a negative interaction means larger improvement for
LOOP, not necessarily better absolute quality than UNTIED or BASE. Coverage must
be judged against its nominal level and width jointly; smaller coverage is not
an improvement. Average days within each trained model, then report all three
seed differences, their mean and sample SD. The 71 days are not independent
trainings; SD is descriptive, not a confidence interval. No result is chosen
based on which table favors LOOP. All conclusions remain development evidence.

## Checks and execution

Nonzero output heads and active attention/FFN gates are used in gradient tests.
Tests check single-pass Core/head counts, exact final-output equivalence, the
weighted gradient identity, final-loss gradient through H1, and UNTIED's
first-loss exclusion of Core1 versus final-loss access to both Cores. They also
check matched parameters/RNG, exact continuation before/after EMA start,
legacy best restore, mode/weight rejection, final-only sampling, and complete
four-cell interaction accounting. A small real-entry fixture tests loss-weight
list/tuple identity through independent saved-artifact re-scoring.

```bash
python -m unittest discover -s tests -v
python -m diagnostics.run_intermediate_supervision
```

The queue checks frozen training/diagnostic hashes and original artifact hashes,
then writes independent `outputs/intermediate_v1_s{seed}_{variant}` directories.
After six successful audited runs it automatically writes the read-only report
to `outputs/intermediate_supervision_v1_results`, including primary/best tables,
paired daily effects, per-seed interactions, learning histories and a figure.
No additional R03-style readout diagnostic is required or launched in this v1.

## Status

Engineering acceptance is complete. Training and reporting sources were frozen
in `cd57515`; the six-run queue has started with seed-42 LOOP-DS. Startup checks
confirmed exact historical LOOP initialization and the unchanged 71-day evaluation
panel. Formal results are still pending. CPU regression: **123 total, 118 passed, 5 CUDA-only skipped**.
The **8 R04 supervision tests all passed in the actual RTX 2060 environment**,
including CUDA FP32 updates/resume and final-only sampling (other tests in that
suite use CPU). Environment: Python 3.10.20, PyTorch 2.10.0+cu128. Gradient tests
use nonzero head/gates and test `0.5*g1+0.5*g2` at `atol=2e-6, rtol=1e-4`.

A direct replay against model/trainer source from `0e0c685` also passed on **both
CPU and CUDA**: BASE, LOOP and UNTIED each performed five synthetic updates, with
bitwise-equal loss records, gradients, raw/EMA tensors and training-noise RNG
states at every update. This short replay certifies the tested calculations;
it is not a new scientific training run or a claim of long-run equivalence on
all hardware. Seed-42 historical LOOP/UNTIED primary and best artifacts were
independently re-scored under the normalized checkpoint auditor successfully.
Syntax and whitespace checks passed; all 129 protected reference files matched.

[Acceptance logs and replay records](diagnostics/intermediate_supervision_v1/acceptance/)
are archived. The replay script was invoked from the repository root with
`PYTHONPATH=. python outputs/check_r04_legacy_replay.py cpu` and `cuda`; archived
source identifies its exact historical comparison. No R04 forecasting outcome
has been observed or claimed at this point.
