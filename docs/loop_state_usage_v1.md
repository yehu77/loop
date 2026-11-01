# R03: use of the first Core's state

## Frozen question and scope

Based on `0323751`, this read-only diagnostic asks whether retaining the first
Core's state improves the final noise estimate. It uses the nine original v2
BASE/LOOP/UNTIED delayed-EMA checkpoints at successful update 3,000. LOCAL is not
retrained or included. There is no new training, DDIM trajectory sampling, test
scoring, extra depth, intervention tuning or R04 execution. Formal model,
training, diffusion and evaluation implementations are unchanged.

## Panel and intervention

`configs/diagnostics/loop_state_usage_v1.json` freezes the 71 validation IDs,
source/data/checkpoint hashes, times 50/200/400/700/950 and noise seeds
3101–3104. Each local CPU RNG generates one FP32 `[71,24,1]` Gaussian tensor;
these same four tensors are reused at every time and for every model/seed/branch.
Thus every model sees 1,420 fixed day–noise–time inputs. True validation power
is used only to construct `x_t = a_t*(2*y-1) + sigma_t*epsilon` and evaluate
injected-noise MSE; it is never supplied as an extra network condition.

Inference uses CPU FP32, `eval()`, disabled gradients and batches of 16; MSE and
RMS reductions use FP64. CPU diagnostics do not certify GPU numerical identity
or latency. The saved training-only scaler is checked against the data manifest.

LOOP computes H0=E and H1=Core0(H0,E,t). Its second call receives H0, the midpoint
H0+0.5*(H1-H0), or H1, corresponding to lambda 0, 0.5 or 1. UNTIED receives the
same interventions but executes Core1 on the second call. BASE executes its
single Core once and has only the normal branch. E, t and input noise are fixed
throughout; the output head is called after the final Core.

Endpoint lambdas use H0/H1 directly to avoid interpolation cancellation error.
Every normal branch is checked against official `forward`; LOOP reset is checked
against `forward(..., rounds=1)` at FP32 `atol=1e-6, rtol=1e-5`. This single readout
is from the **two-round-trained** model and is not a separately trained BASE.

## Measurements and aggregation

For each fixed input and branch, retain noise MSE and its difference from normal,
output-change RMS, first-update RMS (H1-H0), second-input-change RMS (H_lambda-H1)
and final-state-change RMS relative to normal. RMS averages all nonbatch axes.
BASE has zero intervention-change values; first-update RMS is still reported.
Hidden norms are descriptive and not directly comparable measures of usefulness
across different learned representations. Finite values are required.

Average the four noises **within each day**, then average the 71 days within
each trained model. Preserve separate rows for every t, training seed and lambda;
there is no pooled headline across noise levels. Three-seed means and sample SDs
use the training seed as their unit, not days or noise repeats. Positive delta
MSE means normal state retention had lower error; negative values are retained.
There is no significance threshold or search for the best intervention.

## Verification and reproduction

Tests activate nonzero output heads and residual gates. They cover official
forward equivalence, the LOOP reset identity, use of UNTIED's distinct second
Core, helpful and harmful state interventions, and nonzero hidden changes with
no output influence. Additional checks cover batch partitioning, fixed-noise RNG
isolation, checkpoint identity, frozen-grid and output protections, and correct
nested aggregation with missing/duplicate observations rejected.

```bash
python -m unittest discover -s tests -p test_loop_state_usage.py -v
python -m diagnostics.loop_state_usage --output outputs/loop_state_usage_v1
```

The script rejects existing outputs and overlaps with source runs, verifies
protected file hashes before and after execution, and compares all model
parameters/buffers before and after inference. Noise hashes, sample metadata,
checkpoint hashes, equivalence errors and execution provenance are saved with
per-input, per-day and per-model/time CSVs. Noise/input tensors stay local.

## Interpretation boundary

State usage, lower error when using state, and a reusable iterative correction
rule are separate claims. R03 addresses the first two. A degraded reset result
may reflect output-head adaptation to the final representation or intervention
outside training states. It does not establish stepwise correction, depth
extrapolation, recurrence superiority or better final CRPS/ES. Near-zero changes
can have several causes, including weak state use, redundant updates or a nearby
fixed point. These supervised denoiser scores do not replace forecasting results.

## Status

Implementation and protocol prepared; test and diagnostic results pending.
