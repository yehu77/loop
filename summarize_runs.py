"""Read-only v2 run audit and paired-seed summaries; never updates source runs."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import torch

from data import file_hash
from evaluate import evaluate_artifacts

VARIANTS = ("base", "loop", "untied")
SCORES = ("crps", "energy_score", "variogram_score", "coverage_90", "width_90",
          "boundary_fraction")
COSTS = ("training_seconds", "sampling_seconds")
PAIRS = (("loop", "base"), ("loop", "untied"), ("untied", "base"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def stats(values):
    require(bool(values) and all(math.isfinite(x) for x in values), "Invalid observations")
    return {"n_seeds": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def paired_summary(rows):
    """Each observation is one training seed, never one validation day."""
    indexed = {}
    for row in rows:
        key = (row["table"], row["seed"], row["variant"])
        require(key not in indexed, "Duplicate seed/model/table")
        indexed[key] = row
    seeds = sorted({row["seed"] for row in rows})
    tables = sorted({row["table"] for row in rows})
    require(bool(seeds), "No seeds selected")
    pairs, models = [], []
    for table in tables:
        for seed in seeds:
            for variant in VARIANTS:
                require((table, seed, variant) in indexed, "Incomplete paired seed panel")
        for variant in VARIANTS:
            models.append({"table": table, "variant": variant, "metrics": {
                metric: stats([indexed[table, seed, variant][metric] for seed in seeds])
                for metric in SCORES + COSTS}})
        for left, right in PAIRS:
            for metric in SCORES:
                values = [indexed[table, seed, left][metric] - indexed[table, seed, right][metric]
                          for seed in seeds]
                pairs.append({"table": table, "left": left, "right": right, "metric": metric,
                              "by_seed": [{"seed": seed, "delta": value}
                                          for seed, value in zip(seeds, values)], **stats(values)})
    return {"seeds": seeds, "models": models, "paired_differences": pairs,
            "uncertainty_unit": "training seed; sample standard deviation uses n-1; days are not independent trainings"}


def stitch(histories, endpoints):
    """Use only the saved boundary of each segment and reject gaps/duplicate updates."""
    require(len(histories) == len(endpoints) and bool(histories), "Invalid segment list")
    result, previous = [], 0
    for history, endpoint in zip(histories, endpoints):
        require(endpoint > previous, "Segment endpoints must increase")
        retained = [row for row in history if row["updates"] <= endpoint]
        require([row["updates"] for row in retained] == list(range(previous + 1, endpoint + 1)),
                "Missing or duplicate retained updates")
        require(all(row["did_update"] and row["skipped"] == 0 for row in retained),
                "This frozen FP32 panel requires successful updates without skips")
        result.extend(retained)
        previous = endpoint
    return result


def select_nodes(histories, endpoints, start, interval, endpoint):
    nodes = {}
    previous = 0
    for history, boundary in zip(histories, endpoints):
        for row in history:
            update = row["updates"]
            if previous <= update <= boundary and update % interval == 0:
                if update in nodes:
                    # A resume re-evaluation may differ in timing, not scores or EMA state.
                    for weights in ("raw", "ema"):
                        require((weights in nodes[update]) == (weights in row), "Resume EMA readiness differs")
                        if weights in row:
                            for metric in ("crps", "energy_score", "variogram_score", "boundary_fraction"):
                                require(nodes[update][weights][metric] == row[weights][metric],
                                        "Resume validation differs")
                nodes[update] = row
        previous = boundary
    require(sorted(nodes) == list(range(0, endpoint + 1, interval)), "Missing common validation nodes")
    for update, row in nodes.items():
        require(row["ema_initialized"] == (update >= start)
                and row["ema_num_averaged_updates"] == max(0, update - start), "Validation EMA state differs")
    eligible = [nodes[u] for u in sorted(nodes) if u >= start]
    require(all("ema" in row and row["ema_initialized"] for row in eligible), "EMA not ready at candidate")
    require(all("ema" not in row for u, row in nodes.items() if u < start), "Premature EMA evaluation")
    best = min(eligible, key=lambda row: (row["ema"]["crps"], row["updates"]))["updates"]
    return [nodes[u] for u in sorted(nodes)], best


def audit_run(spec, protocol):
    seed, variant = spec["seed"], spec["variant"]
    directories = [Path(p).resolve() for p in spec["segments"]]
    require(bool(directories) and len(set(directories)) == len(directories), "Invalid run directories")
    configs = [read(p / "config.json") for p in directories]
    lineages = [read(p.with_suffix(".lineage.json")) for p in directories]
    checkpoints = [load(p / "checkpoint_last.pt") for p in directories]
    config = configs[0]
    training, evaluation = config["training"], config["evaluation"]
    require(config["resume_from"] is None, "First segment must start from scratch")
    require(config["model"]["variant"] == variant and training["seed"] == seed, "Run identity mismatch")
    require(training["training_protocol_version"] == 2, "Expected v2 training")
    require({k: v for k, v in training.items() if k != "seed"} == protocol["training"],
            "Training differs from the frozen protocol")
    require({k: v for k, v in evaluation.items() if k != "rng_streams"} == protocol["evaluation"],
            "Evaluation differs from the frozen protocol")
    require({k: v for k, v in config["model"].items() if k != "variant"} == protocol["model"],
            "Dimensions differ from the frozen protocol")
    require(config["data_fingerprint"] == protocol["data_fingerprint"], "Unexpected data identity")
    require(len(config["validation_sample_ids"]) == protocol["validation_days"], "Wrong validation panel size")
    require(evaluation["validation_limit"] is None and evaluation["test_evaluation"] == "not performed",
            "Expected full validation and no test evaluation")
    expected_streams = {"canonical_initialization": seed, "extra_cores": seed + 10000,
                        "data_order": seed, "training_noise": seed + 1, "validation": 2026, "rand": 2027}
    require(evaluation["rng_streams"] == expected_streams, "Unexpected RNG streams")
    require(training["ema_start_update"] == protocol["ema_start_update"], "EMA start mismatch")
    require(evaluation["validate_every"] == protocol["validate_every"], "Validation interval mismatch")
    for i, (directory, cfg, lineage, checkpoint) in enumerate(zip(directories, configs, lineages, checkpoints)):
        for key in ("training", "model", "evaluation", "data_fingerprint", "validation_sample_ids"):
            require(cfg[key] == config[key], "Segment configuration drift: " + key)
        require(lineage["training_source_sha256"] == protocol["training_source_sha256"], "Training source drift")
        require(lineage["data_sha256"] == protocol["data_sha256"], "Prepared file changed")
        require(not checkpoint["metadata"].get("failure", False), "Diagnostic failure snapshot is not resumable")
        require(checkpoint["metadata"]["data_fingerprint"] == config["data_fingerprint"], "Checkpoint data differs")
        require(checkpoint["train_config"] == training and checkpoint["model_config"] == config["model"],
                "Checkpoint configuration differs")
        require(checkpoint["metadata"]["run_config"] == evaluation, "Checkpoint evaluation differs")
        update = checkpoint["updates"]
        require(checkpoint["ema_initialized"] == (update >= training["ema_start_update"]), "Wrong EMA readiness")
        require(checkpoint["ema_num_averaged_updates"] == max(0, update - training["ema_start_update"]),
                "Wrong EMA averaging count")
        if i:
            prior = directories[i - 1] / "checkpoint_last.pt"
            require(Path(cfg["resume_from"]).resolve() == prior, "Resume path does not link prior segment")
            require(lineage.get("resume_sha256") == file_hash(prior), "Resume checkpoint hash mismatch")
            require(lineage.get("resumed_successful_updates") == checkpoints[i - 1]["updates"],
                    "Resume update boundary mismatch")
        if i == len(directories) - 1:
            require(lineage["exit_code"] == 0 and lineage["sources_unchanged"], "Run not completed successfully")
            require(lineage["last_checkpoint_sha256"] == file_hash(directory / "checkpoint_last.pt"),
                    "Terminal checkpoint changed")
    endpoints = [checkpoint["updates"] for checkpoint in checkpoints]
    require(endpoints[-1] == protocol["endpoint_update"], "Incomplete endpoint budget")
    anchors = [p / "checkpoint_ema_start.pt" for p in directories
               if (p / "checkpoint_ema_start.pt").exists()]
    require(bool(anchors), "Missing EMA copy-boundary checkpoint")
    for path in anchors:
        anchor = load(path)
        require(anchor["updates"] == protocol["ema_start_update"]
                and anchor["ema_initialized"] and anchor["ema_num_averaged_updates"] == 0,
                "Invalid EMA copy boundary")
        require(anchor["model_config"] == config["model"] and anchor["train_config"] == training
                and anchor["metadata"]["data_fingerprint"] == config["data_fingerprint"],
                "EMA anchor identity differs")
        require(anchor["model"].keys() == anchor["ema"].keys()
                and all(torch.equal(value, anchor["ema"][key])
                        for key, value in anchor["model"].items()), "EMA anchor is not an exact raw copy")
    history = stitch([read(p / "training_history.json") for p in directories], endpoints)
    validation, best = select_nodes([read(p / "validation_history.json") for p in directories], endpoints,
                                    protocol["ema_start_update"], protocol["validate_every"], endpoints[-1])
    terminal = directories[-1]
    best_checkpoint = load(terminal / "checkpoint_best.pt")
    require(not best_checkpoint["metadata"].get("failure", False), "Invalid best snapshot")
    require(best_checkpoint["updates"] == best, "Best is not minimum EMA CRPS on common nodes")
    require(best_checkpoint["metadata"]["data_fingerprint"] == config["data_fingerprint"], "Best data differs")
    require(best_checkpoint["model_config"] == config["model"] and best_checkpoint["train_config"] == training,
            "Best configuration differs")
    cost = read(terminal / "cost.json")
    require(cost["auxiliary_best_update"] == best and cost["primary_update"] == endpoints[-1], "Cost panel differs")
    rows, daily = [], []
    for table, directory, update in (("primary", terminal, endpoints[-1]), ("best", terminal / "selected", best)):
        saved = read(directory / "metrics.json")
        require(saved["diagnostics"]["successful_updates"] == update and saved["diagnostics"]["weights"] == "ema",
                "Wrong selected weight or update")
        predicted, target = load(directory / "scenarios.pt"), load(directory / "targets.pt")
        report, per_window = evaluate_artifacts(predicted, target)
        require({k: v for k, v in saved.items() if k != "diagnostics"} == report, "Independent re-scoring differs")
        require(report["sample_ids"] == config["validation_sample_ids"] and report["split"] == "validation",
                "Scoring panel identity mismatch")
        scores = load(directory / "scores_per_window.pt")
        require(len(scores["intervals"]) == len(per_window["intervals"]), "Interval count differs")
        for metric in ("crps", "energy_score", "variogram_score"):
            require(torch.equal(scores[metric], per_window[metric]), "Per-window score mismatch")
        for actual, expected in zip(scores["intervals"], per_window["intervals"]):
            require(actual["level"] == expected["level"] and torch.equal(actual["covered"], expected["covered"])
                    and torch.equal(actual["width"], expected["width"]), "Interval arrays differ")
        interval = next(item for item in report["intervals"] if item["nominal_level"] == 0.9)
        row = {"table": table, "seed": seed, "variant": variant, "update": update,
               **{metric: report[metric] for metric in ("crps", "energy_score", "variogram_score")},
               "coverage_90": interval["coverage"], "width_90": interval["mean_width"],
               "boundary_fraction": saved["diagnostics"]["zero_power_fraction"] + saved["diagnostics"]["full_power_fraction"],
               "parameters": cost["parameters"], "training_seconds": sum(r["training_seconds"] for r in history),
               "sampling_seconds": saved["diagnostics"]["sampling_seconds"],
               "retained_sample_exposures": history[-1]["samples_consumed_total"]}
        rows.append(row)
        intervals = next(item for item in per_window["intervals"] if item["level"] == 0.9)
        for index, sample in enumerate(report["sample_ids"]):
            daily.append({"table": table, "seed": seed, "variant": variant, "sample_id": sample,
                          "crps": per_window["crps"][index].mean().item(),
                          "energy_score": per_window["energy_score"][index].item(),
                          "variogram_score": per_window["variogram_score"][index].item(),
                          "coverage_90": intervals["covered"][index].double().mean().item(),
                          "width_90": intervals["width"][index].mean().item()})
    evidence = {"seed": seed, "variant": variant, "segments": [str(p) for p in directories],
                "lineages": lineages, "config": config,
                "interrupted_segment_present": any(record["exit_code"] != 0 for record in lineages),
                "cost_scope": "saved retained updates; unrecorded interrupted work and validation/IO excluded",
                "independent_rescoring_passed": True,
                "artifact_sha256": {str(p): file_hash(p)
                                    for directory in directories for p in directory.glob("*.pt")}}
    return {"rows": rows, "daily": daily, "evidence": evidence, "config": config,
            "initialization": read(directories[0] / "initialization.json"),
            "checkpoint": checkpoints[-1], "history": history, "validation": validation}


def verify_panel(runs):
    reference = runs[0]["config"]
    for run in runs:
        config = run["config"]
        for key in ("data_fingerprint", "validation_sample_ids"):
            require(config[key] == reference[key], "Cross-run panel differs")
        require({k: v for k, v in config["training"].items() if k != "seed"}
                == {k: v for k, v in reference["training"].items() if k != "seed"}, "Training recipe differs")
        require({k: v for k, v in config["model"].items() if k != "variant"}
                == {k: v for k, v in reference["model"].items() if k != "variant"}, "Model dimensions differ")
        require({k: v for k, v in config["evaluation"].items() if k != "rng_streams"}
                == {k: v for k, v in reference["evaluation"].items() if k != "rng_streams"}, "Evaluation recipe differs")
    for seed in sorted({r["config"]["training"]["seed"] for r in runs}):
        group = {r["config"]["model"]["variant"]: r for r in runs if r["config"]["training"]["seed"] == seed}
        require(set(group) == set(VARIANTS), "Missing architecture in seed panel")
        base, loop, untied = (group[v] for v in VARIANTS)
        require(base["initialization"] == loop["initialization"], "BASE/LOOP initialization differs")
        require(all(untied["initialization"].get(k) == v for k, v in base["initialization"].items()),
                "UNTIED common initialization differs")
        for run in (loop, untied):
            left, right = base["checkpoint"], run["checkpoint"]
            require(torch.equal(left["noise_rng"], right["noise_rng"]), "Training-noise RNG differs")
            for key in ("order", "rng"):
                require(torch.equal(left["data"][key], right["data"][key]), "Batch RNG/order differs")
            require(left["data"]["cursor"] == right["data"]["cursor"], "Batch cursor differs")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(manifest, output, seeds=None):
    require(not output.exists(), "Report destination already exists; choose a new directory")
    specifications = manifest["runs"]
    selected = set(seeds or manifest["seeds"])
    require(selected <= set(manifest["seeds"]), "Undeclared seed requested")
    specifications = [item for item in specifications if item["seed"] in selected]
    identities = [(item["seed"], item["variant"]) for item in specifications]
    require(len(identities) == len(set(identities)), "Duplicate manifest run")
    require(set(identities) == {(seed, variant) for seed in selected for variant in VARIANTS},
            "Manifest must include each model for every requested seed")
    # The output cannot contain or be contained by any input directory.
    for item in specifications:
        for segment in item["segments"]:
            source = Path(segment).resolve()
            require(output.resolve() != source and output.resolve() not in source.parents
                    and source not in output.resolve().parents, "Output overlaps a source run")
    runs = [audit_run(spec, manifest["protocol"]) for spec in specifications]
    verify_panel(runs)
    rows = [row for run in runs for row in run["rows"]]
    summary = paired_summary(rows)
    summary["scope"] = "validation only; fixed endpoint is primary, minimum common-node EMA CRPS is auxiliary"
    summary["rows"] = rows
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "evidence.json").write_text(json.dumps([r["evidence"] for r in runs], indent=2) + "\n")
    write_csv(output / "scores.csv", rows)
    write_csv(output / "scores_per_day.csv", [row for run in runs for row in run["daily"]])
    differences = []
    for item in summary["paired_differences"]:
        for row in item["by_seed"]:
            differences.append({k: item[k] for k in ("table", "left", "right", "metric")} | row)
    write_csv(output / "paired_seed_differences.csv", differences)
    for run in runs:
        name = "s{}_{}".format(run["config"]["training"]["seed"], run["config"]["model"]["variant"])
        destination = output / name
        destination.mkdir()
        (destination / "training_history.json").write_text("[\n" + ",\n".join(
            json.dumps(row) for row in run["history"]) + "\n]\n")
        (destination / "validation_history.json").write_text(json.dumps(run["validation"], indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Seeds, variants, ordered run directories and frozen protocol")
    parser.add_argument("--seed", type=int, action="append", help="Optional declared seed subset; repeat to select several")
    parser.add_argument("--output", type=Path, required=True, help="New report directory outside source runs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = summarize(read(args.manifest), args.output, args.seed)
    print(json.dumps({"output": str(args.output), "seeds": summary["seeds"], "runs": len(summary["rows"]) // 2}))


if __name__ == "__main__":
    main()
