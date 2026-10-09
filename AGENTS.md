# Repository Guidelines

## Scope and Evidence

This is an independent, from-scratch single-farm conditional wind-power scenario
project. Do not import a user's previous projects, data splits, checkpoints, or
experimental conclusions. Public papers and baseline implementations may be used
as explicitly attributed references.

The v1 implementation exists in `model.py`, `diffusion.py`, and `train.py`.
It currently uses synthetic data for execution checks, not real wind-power
performance experiments. `README.md` describes the architecture and commands;
`docs/acceptance.md` records the original local RTX 2060 acceptance results.
Do not rewrite historical results or present another machine's measurements as
new local evidence. The GPU reports and checkpoints named there are local-only.

## Model and Experimental Boundaries

A sample is one farm's entire future trajectory. The reference configuration is
24 hourly tokens, four NWP features, and four deterministic calendar features.
Do not silently add multi-farm spatial modeling, history inputs, new losses,
random recurrence depths, or truncated gradients.

`base` executes one Core once; `loop` reuses one Core; `untied` executes distinct
Cores. Every Core has an input-reinjection adapter and two AdaLN blocks.
Keep condition access and all non-Core components matched when comparing them.
Parameter count, total computation, and measured latency are different budgets.
Synthetic execution success is not evidence that recurrence improves forecasting.

## Files and Commands

- `model.py`: model configurations, components, weight sharing, initialization.
- `diffusion.py`: forward noising, DDIM sampling, clipping diagnostics.
- `train.py`: synthetic data, training-only scaler, updates, EMA, checkpoints, CLI.
- `check_wind_env.py`: standalone environment check; Python 3.8+.
- `tests/`: standard-library unittest regression tests.
- `outputs/`: local generated artifacts; excluded from version control.

Model code targets Python 3.10+ and PyTorch 2.10. Use an existing validated
environment when available; do not reinstall it merely to run a review.

```bash
python -m unittest discover -s tests -v
python -m py_compile model.py diffusion.py train.py check_wind_env.py
python train.py --variant all --device cpu --updates 3 --scenarios 4 --sampling-steps 4
```

GPU checks, only where CUDA is actually available:

```bash
python train.py --variant all --device cuda
python train.py --variant loop --device cuda --amp
```

Report passed and skipped tests separately. CPU review cannot certify CUDA AMP,
GPU memory use, or RTX 2060 throughput. Optional Ruff settings are in
`pyproject.toml`; keep four-space indentation and existing file conventions.

## Checkpoint and Artifact Safety

Normal checkpoints save the optimizer, EMA, scaler, data position, and RNG state
at a completed update boundary. A snapshot marked `metadata.failure` may have
consumed microbatches without completing an optimizer update. It is diagnostic
only and `Trainer.load` rejects it, even if renamed. Fix the original error and
resume from an earlier normal checkpoint, or start a new run. Do not clear the
failure flag to disguise a partial update as exact continuation.

Keep `.aws/`, `.codex/`, `.agents/`, credentials, datasets, checkpoints, and large
generated artifacts out of commits. Do not access or modify environment metadata
as part of model development.

## Changes and Pull Requests

Keep changes small and explain the failure or hypothesis each change addresses.
Add regression tests for bug fixes. Use a separate branch and a reviewable PR;
do not force-push, auto-merge, or rewrite the main branch as part of a review.
Include the reviewed base commit, changed behavior, validation commands, test
environment, and untested boundaries. Preserve the user's existing work.
