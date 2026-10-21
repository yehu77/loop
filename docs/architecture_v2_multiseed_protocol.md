# Frozen v2 robustness check: seeds 42, 43, 44

This extends the completed seed-42 panel with exactly six new runs: seeds 43 and
44, each in the order BASE, LOOP, UNTIED. It does not change the training recipe
in [training_protocol_v2.md](training_protocol_v2.md), search seeds, or score test
data. Stop after the six runs and summarize all three seeds before deciding on
further research. Training source reference: `72c97f8`.

## Fixed settings

Each new run starts from scratch: 3,000 successful updates, FP32, microbatch 4
and accumulation 8, EMA copied from raw at update 1,000 and averaged with decay
0.999 thereafter. Use the frozen Zone-3 package (547/71/90 windows), all 71
complete validation days, 100 scenarios, 50 DDIM steps and chunk size 16. Validate
every 250 updates. The data fingerprint and exact configurations are recorded in
[`configs/architecture_v2_runs.json`](../configs/architecture_v2_runs.json).

For training seed `s`, initialization and data order use `s`, training noise uses
`s+1`, and independent extra-Core initialization uses `s+10000`. Validation seed
2026 and RAND seed 2027 remain fixed. Match common initial tensors and training
random streams within each seed; do not transfer trained parameters.

Each launch records the command, commit, training-source hashes and data hash in
its adjacent `.lineage.json`. Preserve original outputs on interruption. Inspect
the normal checkpoint, then resume only the remaining successful-update budget
into a new directory and record the prior checkpoint hash. Declare all segments
in chronological order in the manifest. Do not overwrite or silently restart.

## Read-only summary

From the repository root, after the declared runs finish:

```bash
python summarize_runs.py --manifest configs/architecture_v2_runs.json \
  --output outputs/architecture_v2_three_seed_summary
```

Use `--seed 42` (repeatable) to audit a completed subset. The output must be a
new directory outside all source runs. The archived seed-42 script remains
unchanged; the new entry handles arbitrary explicit continuation segments.

The audit checks source/data identity, frozen settings, successful-update and EMA
counts, checkpoint lineage, matched initialization and terminal training RNG
states. It independently rescores endpoint and selected-best scenarios by sample
identity, checks stored per-window arrays, and rejects incomplete model panels.
It merges histories only through each saved segment boundary and excludes
off-grid resume evaluations from best selection.

## Reporting and decision

The primary table compares delayed EMA at update 3,000. The auxiliary table uses
minimum EMA validation CRPS at 1000,1250,...,3000, with earlier ties preferred.
Keep both tables, including CRPS, ES, VS, 90% point coverage/width, boundary
fraction, parameters and measured costs. Full-run training time excludes
validation/checkpoint IO and unrecorded work before an interruption; it is not a
FLOP budget. Sampling time covers the same 7,100-trajectory panel.

For LOOP−BASE, LOOP−UNTIED and UNTIED−BASE, report each seed's paired differences,
their mean and sample standard deviation (denominator `n-1`). The three training
seeds are the replication units; 71 validation days are not 71 independent
trainings. Preserve per-day scores for diagnostics without pooling them into
213 independent model experiments. Standard deviation is not a confidence
interval. Lower CRPS/ES/VS is better; assess coverage jointly with width.

If LOOP offers no quality benefit at higher cost, retire this fixed two-round,
fully shared configuration as the default method. Small changing rankings imply
unstable benefit. Any favorable result remains limited to this common recipe,
three seeds and the complete-day validation panel; no claim of convergence,
optimal hyperparameters or test generalization follows.

Report-layer checks: `python -m unittest discover -s tests -p test_run_summary.py -v`.
