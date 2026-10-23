"""R01: permute ensemble members independently by day/hour, without resampling."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from data import file_hash, fingerprint, tensor_hash
from evaluate import align_artifacts
from metrics import energy_score, ensemble_crps, interval_statistics, variogram_score

MARGINALS = ("crps", "coverage_80", "width_80", "coverage_90", "width_90")
STRUCTURAL = ("energy_score", "variogram_score", "increment_1h_crps",
              "increment_3h_crps", "daily_mean_crps")
METRICS = MARGINALS + STRUCTURAL


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def make_permutations(days, members, hours, seed, common=False):
    """CPU-local RNG; [day, hour, output_member] contains input member indices."""
    require(all(type(n) is int and n > 0 for n in (days, members, hours)), "Invalid dimensions")
    require(type(seed) is int and seed >= 0, "Invalid permutation seed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.empty(days, hours, members, dtype=torch.long)
    for day in range(days):
        if common:
            indices[day] = torch.randperm(members, generator=generator)[None, :]
        else:
            for hour in range(hours):
                indices[day, hour] = torch.randperm(members, generator=generator)
    return indices


def permute_members(scenarios, indices):
    require(scenarios.ndim == 4 and scenarios.shape[-1] == 1, "Expected [D,M,H,1]")
    days, members, hours, _ = scenarios.shape
    require(indices.dtype == torch.long and indices.shape == (days, hours, members),
            "Invalid permutation shape or dtype")
    require(indices.device == scenarios.device, "Permutation device differs")
    expected = torch.arange(members, device=indices.device).expand_as(indices)
    require(torch.equal(indices.sort(dim=-1).values, expected), "Indices must be permutations")
    return scenarios.permute(0, 2, 1, 3).gather(2, indices[..., None]).permute(0, 2, 1, 3)


def transforms(scenarios, targets):
    """Transform in FP64 before subtraction/averaging; never clip signed increments."""
    x, y = scenarios.double(), targets.double()
    require(x.shape[2] > 3, "Need at least four hours for increment diagnostics")
    result = {f"increment_{lag}h_crps": (x[:, :, lag:] - x[:, :, :-lag],
                                        y[:, lag:] - y[:, :-lag]) for lag in (1, 3)}
    result["daily_mean_crps"] = (x.mean(dim=2, keepdim=True), y.mean(dim=1, keepdim=True))
    return result


def score_panel(scenarios, targets):
    result = {"crps": ensemble_crps(scenarios, targets),
              "energy_score": energy_score(scenarios, targets),
              "variogram_score": variogram_score(scenarios, targets, power=0.5)}
    for name, (x, y) in transforms(scenarios, targets).items():
        result[name] = ensemble_crps(x, y)
    for level in (0.8, 0.9):
        interval = interval_statistics(scenarios, targets, level)
        suffix = str(round(level * 100))
        result["coverage_" + suffix] = interval["covered"].double()
        result["width_" + suffix] = interval["width"]
    return result


def compare_scores(original, other, keys, atol, rtol):
    errors = {}
    for key in keys:
        require(torch.allclose(original[key], other[key], atol=atol, rtol=rtol),
                "Invariance failed: " + key)
        errors[key] = (original[key] - other[key]).abs().max().item()
    return errors


def aligned_validation(prediction, target, ids, fingerprint):
    # Reject other splits before calling any scoring routine.
    require(all(m["split"] == "validation" for a in (prediction, target) for m in a["samples"]),
            "R01 only accepts validation artifacts")
    require(prediction["data_fingerprint"] == target["data_fingerprint"] == fingerprint,
            "Unexpected data fingerprint")
    x, y = align_artifacts(prediction, target)
    target_ids = [m["sample_id"] for m in target["samples"]]
    require(len(ids) == len(set(ids)) and set(ids) == set(target_ids), "Wrong validation panel")
    order = [target_ids.index(key) for key in ids]
    return x[order], y[order], [target["samples"][i] for i in order]


def daily_mean(values):
    return values.reshape(values.shape[0], -1).mean(dim=1)


def stats(values):
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config):
    require(config["schema_version"] == 1 and config["stage"] == "R01", "Unsupported diagnostic")
    require(config["permutation_seeds"] == [1101, 1102, 1103, 1104, 1105], "Permutation seeds are frozen")
    require(config["lags"] == [1, 3] and config["interval_levels"] == [0.8, 0.9]
            and config["variogram_power"] == 0.5, "Diagnostic definitions are frozen")
    require(config["atol"] == config["rtol"] == 1e-10, "Invariance tolerances are frozen")
    require(config["endpoint_update"] == 3000 and config["weights"] == "ema", "Wrong endpoint")
    identities = [(r["training_seed"], r["variant"]) for r in config["runs"]]
    require(len(identities) == len(set(identities)) and set(identities) == {
        (s, v) for s in config["training_seeds"] for v in config["variants"]}, "Incomplete model panel")
    require(config["shape"][0] == len(config["sample_ids"]) and config["shape"][2:] == [24, 1],
            "Wrong hourly panel shape")


def run(config, output):
    output = Path(output).resolve()
    require(not output.exists(), "Output exists; choose a new directory")
    validate_config(config)
    protected = dict(config["protected_sources"])
    for spec in config["runs"]:
        for directory in spec["source_directories"]:
            source = Path(directory).resolve()
            require(output != source and output not in source.parents and source not in output.parents,
                    "Output overlaps a source run")
        for name, digest in spec["protected_files"].items():
            require(name not in protected or protected[name] == digest, "Conflicting source hashes")
            protected[name] = digest
    before = {name: file_hash(name) for name in protected}
    require(before == protected, "Frozen source artifact hash mismatch")
    days, members, hours, _ = config["shape"]
    ids = config["sample_ids"]
    permutations = {s: make_permutations(days, members, hours, s) for s in config["permutation_seeds"]}
    common = {s: make_permutations(days, members, hours, s, common=True) for s in permutations}
    details, aggregates, controls, saved = [], [], [], {}
    reference_targets, reference_metadata = None, None
    start = time.perf_counter()
    for spec in config["runs"]:
        directory = Path(spec["run_directory"])
        identity = {k: spec[k] for k in ("training_seed", "variant")}
        cfg = read(directory / "config.json")
        prior_metrics = read(directory / "metrics.json")
        diagnostics = prior_metrics["diagnostics"]
        require(diagnostics["successful_updates"] == 3000 and diagnostics["weights"] == "ema"
                and diagnostics["validation_seed"] == 2026, "Not the fixed v2 EMA endpoint")
        require(cfg["training"]["seed"] == spec["training_seed"]
                and cfg["model"]["variant"] == spec["variant"]
                and cfg["training"]["training_protocol_version"] == 2
                and cfg["training"]["ema_start_update"] == 1000, "Training identity differs")
        require(cfg["validation_sample_ids"] == ids, "Configured validation IDs differ")
        prediction = torch.load(directory / "scenarios.pt", map_location="cpu", weights_only=True)
        target = torch.load(directory / "targets.pt", map_location="cpu", weights_only=True)
        x, y, metadata = aligned_validation(prediction, target, ids, config["data_fingerprint"])
        require(list(x.shape) == config["shape"], "Unexpected ensemble shape")
        if reference_targets is None:
            reference_targets, reference_metadata = y.clone(), metadata
        else:
            require(torch.equal(y, reference_targets) and metadata == reference_metadata,
                    "Cross-model target values or metadata differ")
        original = score_panel(x, y)
        for metric in ("crps", "energy_score", "variogram_score"):
            require(abs(original[metric].mean().item() - prior_metrics[metric]) <= config["atol"],
                    "Original score differs from saved endpoint: " + metric)
        run_key = f"s{spec['training_seed']}_{spec['variant']}"
        saved[run_key] = {"original": original, "shuffled": {}}
        for seed, indices in permutations.items():
            shuffled = permute_members(x, indices)
            require(torch.equal(x.sort(dim=1).values, shuffled.sort(dim=1).values),
                    "Per-day/hour marginal sample values changed")
            scored = score_panel(shuffled, y)
            marginal_error = compare_scores(original, scored, MARGINALS, config["atol"], config["rtol"])
            common_scores = score_panel(permute_members(x, common[seed]), y)
            common_error = compare_scores(original, common_scores, METRICS, config["atol"], config["rtol"])
            controls.append({**identity, "permutation_seed": seed, "sorted_marginals_exact": True,
                             "marginal_max_abs_errors": marginal_error, "common_permutation_max_abs_errors": common_error})
            saved[run_key]["shuffled"][seed] = scored
            for metric in METRICS:
                a, b = daily_mean(original[metric]), daily_mean(scored[metric])
                delta = b - a
                aggregates.append({**identity, "permutation_seed": seed, "metric": metric,
                                   "original": a.mean().item(), "shuffled": b.mean().item(),
                                   "delta": delta.mean().item(), "days_positive": int((delta > 0).sum()),
                                   "days_negative": int((delta < 0).sum())})
                for i, sample in enumerate(ids):
                    details.append({**identity, "permutation_seed": seed, "sample_id": sample,
                                    "metric": metric, "original": a[i].item(),
                                    "shuffled": b[i].item(), "delta": delta[i].item()})
        print(json.dumps({**identity, "permutations_completed": len(permutations)}), flush=True)
    per_run = []
    for spec in config["runs"]:
        for metric in METRICS:
            rows = [r for r in aggregates if r["training_seed"] == spec["training_seed"]
                    and r["variant"] == spec["variant"] and r["metric"] == metric]
            delta = stats([r["delta"] for r in rows])
            per_run.append({"training_seed": spec["training_seed"], "variant": spec["variant"],
                            "metric": metric, "original": rows[0]["original"],
                            "shuffled_mean": statistics.mean(r["shuffled"] for r in rows),
                            **{"delta_" + k: v for k, v in delta.items()}})
    per_model = []
    for variant in config["variants"]:
        for metric in METRICS:
            rows = [r for r in per_run if r["variant"] == variant and r["metric"] == metric]
            per_model.append({"variant": variant, "metric": metric,
                              "original_mean": statistics.mean(r["original"] for r in rows),
                              "shuffled_mean": statistics.mean(r["shuffled_mean"] for r in rows),
                              "delta_across_training_seeds": stats([r["delta_mean"] for r in rows]),
                              "mean_within_run_permutation_sd": statistics.mean(r["delta_sample_std"] for r in rows)})
    after = {name: file_hash(name) for name in protected}
    require(after == before, "Source files changed during diagnostic")
    axes = {f"increment_{lag}h_crps": {
        "lead_pairs": [[h, h + lag] for h in range(1, hours - lag + 1)],
        "unit": "signed capacity_fraction difference", "within_day_reduction": "mean of per-pair CRPS"}
        for lag in (1, 3)}
    axes["daily_mean_crps"] = {"leads": list(range(1, hours + 1)), "unit": "capacity_fraction",
                               "within_day_reduction": "CRPS of the scenario-wise hourly mean"}
    report = {"stage": "R01", "config": config, "models": per_model,
              "diagnostic_config_fingerprint": fingerprint(config), "transformed_axes": axes,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
              "source_sha256_before": before, "source_sha256_after": after,
              "source_files_unchanged": True, "controls": controls,
              "permutation_sha256": {str(s): tensor_hash(p) for s, p in permutations.items()},
              "common_permutation_sha256": {str(s): tensor_hash(p) for s, p in common.items()},
              "sample_metadata": reference_metadata, "python": sys.version.split()[0], "torch": torch.__version__,
              "implementation_sha256": file_hash(__file__), "elapsed_seconds": time.perf_counter() - start,
              "conventions": {"delta": "shuffled minus original; positive score delta favors original pairing",
                  "increments": "signed capacity_fraction differences, per lead CRPS then within-day mean; no clipping",
                  "daily_mean": "24-hour mean capacity_fraction, not MWh",
                  "repeats": "five permutations are Monte Carlo controls, not new trainings or weather data",
                  "aggregation": "average five permutations within each trained model first; then describe three training seeds",
                  "tolerance": "engineering invariance checks only, not statistical significance",
                  "permutation": "CPU Generator, randperm(M) in canonical day/hour order; indices reused across all nine panels",
                  "test_scoring": False, "new_training": False, "new_sampling": False}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_csv(output / "per_permutation.csv", aggregates)
    write_csv(output / "per_day.csv", details)
    write_csv(output / "per_run.csv", per_run)
    torch.save({"sample_ids": ids, "data_fingerprint": config["data_fingerprint"],
                "permutations": permutations, "common_permutations": common}, output / "permutations.pt")
    torch.save({"sample_ids": ids, "data_fingerprint": config["data_fingerprint"],
                "scores": saved, "conventions": report["conventions"], "transformed_axes": axes},
               output / "scores_per_lead.pt")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(read(args.config), args.output)
    print("R01 complete: " + str(args.output.resolve()))


if __name__ == "__main__":
    main()
