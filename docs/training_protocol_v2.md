# Training protocol v2: matched architecture comparison

Frozen before the seed-42 architecture runs. This supersedes the EMA maintenance
and comparison rules for **new runs**, without changing historical v1 results.
The model, loss, diffusion schedule, sampler, optimizer and data protocol stay fixed.

## Data and budget

Use `outputs/data_v1_20261014/data.pt`, fingerprint
`403ff1adc1df6b1bd6dd79b24f206f164daf3161f829bf7a232fddf90a9168a4`:
Zone 3, 547 training / 71 complete validation / 90 test windows. Test is not scored.
Each variant trains from scratch for 3,000 successful updates in FP32, seed 42,
microbatch 4, accumulation 8, AdamW lr 1e-4, decay 1e-4, gradient clip 1.
All variants therefore expose 96,000 training windows if no steps are skipped;
this is **not** equal computation. No seed search or additional EMA tuning.

## EMA state and continuation

Before successful update 1,000, EMA is unavailable for evaluation or selection.
Immediately after update 1,000, copy the complete raw state into EMA, mark ready,
and set the averaging counter to zero. After updates 1,001 onward, average
parameters with decay 0.999 and copy buffers. Skipped attempts do neither.

Checkpoint format 2 stores readiness and the averaging count, alongside the
version/start in `train_config`. Restore validates these against successful
updates. Legacy format-1 checkpoints retain v1 semantics (EMA from initialization).
Explicit variant, protocol or start mismatches are rejected. An absent best is
valid before EMA eligibility. `checkpoint_ema_start.pt` preserves the copy boundary.

## Initialization and random streams

Canonical initialization is the original BASE constructor with seed 42. BASE and
LOOP receive identical tensors. UNTIED receives matching shared components and
core 0; its additional core retains initialization from an isolated CPU stream
seed 10042. All cores have independent storage where specified by the architecture.
The extra construction leaves the canonical global RNG unchanged.

Independent generators: shuffled data seed 42; training timesteps/noise seed 43;
validation initial noise seed 2026; RAND training-trajectory draws seed 2027.
Their state/seed is saved. No trained weights are transferred between variants.

## Evaluation and costs

Validate raw every 250 updates including 0; evaluate EMA only when ready. Use all
71 validation days, 100 scenarios each, DDIM 50 steps, chunk size 16. The primary
table is delayed EMA at update **3,000**. The auxiliary best table chooses minimum
EMA CRPS among 1000,1250,...,3000; strict improvement preserves the earlier tie.
The run root stores endpoint scores; `selected/` stores auxiliary best scores.
Per-window scores and sample identities remain available for paired analysis.

Report CRPS, ES, VS, 90% point coverage/width, boundary fraction, parameters,
independent/executed cores, sample exposure and measured costs. Training seconds
cover synchronized optimizer updates, excluding validation and checkpoint IO.
Sampling seconds cover the same 7,100-trajectory panel and include sampler work,
excluding scoring. These single-device observations are not FLOP counts or
controlled hardware benchmarks. Keep raw/EMA learning curves as diagnostics.

## Execution

```bash
python train_real.py --data outputs/data_v1_20261014/data.pt \
  --variant base --training-protocol-version 2 --device cuda \
  --updates 3000 --microbatch-size 4 --accumulation-steps 8 \
  --validate-every 250 --scenarios 100 --sampling-steps 50 --chunk-size 16 \
  --seed 42 --output outputs/architecture_v2_s42_base
```

Run BASE first, verify raw replay and delayed EMA, then repeat with `loop` and
`untied` and distinct output directories. Resume uses saved evaluation settings;
`--updates` is additional successful updates. Small EMA start values exist only
for interface tests, not alternate scientific protocols.
