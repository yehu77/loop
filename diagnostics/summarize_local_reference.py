"""Read-only R02 paired report, reusing v2 audits and R01 transformed scores."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch

from data import file_hash
from diagnostics.temporal_dependence import (METRICS, aligned_validation, daily_mean,
                                             read, require, score_panel, write_csv)
from model import ModelConfig
from summarize_runs import audit_run


LABELS = ("base", "loop", "untied", "local")


def verify_local_pair(base, local):
    """Compare saved identities and RNG positions, not just the seed integer."""
    a, b = base["config"], local["config"]
    for key in ("training", "evaluation", "data_fingerprint", "validation_sample_ids"):
        require(a[key] == b[key], "Paired configuration mismatch: " + key)
    ma, mb = ModelConfig(**a["model"]), ModelConfig(**b["model"])
    require(ma.variant == mb.variant == "base" and ma.attention_scope == "full"
            and mb.attention_scope == "diagonal", "Expected full BASE and diagonal LOCAL")
    require({k: v for k, v in asdict(ma).items() if k != "attention_scope"}
            == {k: v for k, v in asdict(mb).items() if k != "attention_scope"},
            "Non-attention architecture changed")
    require(base["initialization"] == local["initialization"], "Paired initialization differs")
    ca, cb = base["checkpoint"], local["checkpoint"]
    require(ca["updates"] == cb["updates"] == 3000, "Incomplete paired endpoints")
    require(torch.equal(ca["noise_rng"], cb["noise_rng"]), "Paired training noise differs")
    for key in ("order", "rng"):
        require(torch.equal(ca["data"][key], cb["data"][key]), "Paired data order differs")
    require(ca["data"]["cursor"] == cb["data"]["cursor"], "Paired data position differs")


def paired_differences(rows, seeds):
    index = {(r["table"], r["seed"], r["model"]): r for r in rows}
    require(len(index) == len(rows), "Duplicate run summary")
    require(set(index) == {(t, s, m) for t in ("primary", "best") for s in seeds for m in LABELS},
            "Incomplete paired panel")
    pairs, summaries = [], []
    for table in ("primary", "best"):
        for metric in METRICS:
            values = []
            for seed in seeds:
                delta = index[table, seed, "local"][metric] - index[table, seed, "base"][metric]
                pairs.append({"table": table, "seed": seed, "metric": metric, "local_minus_base": delta})
                values.append(delta)
            summaries.append({"table": table, "metric": metric, "mean_local_minus_base": statistics.mean(values),
                              "sample_std": statistics.stdev(values), "n_training_seeds": len(seeds)})
    return pairs, summaries


def summarize(config, output):
    output = Path(output).resolve()
    require(not output.exists(), "Report exists; choose a new directory")
    require(config["seeds"] == [42, 43, 44], "R02 seeds are frozen")
    manifest = Path(config["reference_manifest"])
    require(file_hash(manifest) == config["reference_manifest_sha256"], "Reference manifest changed")
    reference = read(manifest)
    specs = [(s, reference["protocol"], s["variant"]) for s in reference["runs"]]
    specs += [(s, config["protocol"], "local") for s in config["runs"]]
    identities = [(s["seed"], label) for s, _, label in specs]
    require(len(identities) == len(set(identities)) and set(identities)
            == {(s, label) for s in config["seeds"] for label in LABELS}, "Wrong run manifest")
    protected = dict(config["protected_reference_files"])
    for spec, _, _ in specs:
        for directory in map(Path, spec["segments"]):
            source = directory.resolve()
            require(output != source and output not in source.parents and source not in output.parents,
                    "Output overlaps source")
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix in (".pt", ".json"):
                    digest = file_hash(path)
                    require(str(path) not in protected or protected[str(path)] == digest,
                            "Historical artifact changed")
                    protected[str(path)] = digest
            lineage = directory.with_suffix(".lineage.json")
            protected[str(lineage)] = file_hash(lineage)
    require(all(file_hash(p) == h for p, h in protected.items()), "Frozen artifact changed")
    audited = {(spec["seed"], label): audit_run(spec, protocol) for spec, protocol, label in specs}
    for seed in config["seeds"]:
        verify_local_pair(audited[seed, "base"], audited[seed, "local"])
    rows, daily, tensors, evidence = [], [], {}, []
    reference_targets, reference_metadata = None, None
    for (seed, label), run in audited.items():
        spec = next(s for s, _, name in specs if s["seed"] == seed and name == label)
        terminal = Path(spec["segments"][-1])
        evidence.append({"model": label, **run["evidence"]})
        for table, directory in (("primary", terminal), ("best", terminal / "selected")):
            prediction = torch.load(directory / "scenarios.pt", map_location="cpu", weights_only=True)
            target = torch.load(directory / "targets.pt", map_location="cpu", weights_only=True)
            x, y, metadata = aligned_validation(prediction, target, config["sample_ids"],
                                                config["protocol"]["data_fingerprint"])
            require(x.shape == (71, 100, 24, 1), "Wrong panel size")
            if reference_targets is None:
                reference_targets, reference_metadata = y, metadata
            require(torch.equal(reference_targets, y) and reference_metadata == metadata,
                    "Targets differ across panels")
            scores = score_panel(x, y)
            baseline = next(r for r in run["rows"] if r["table"] == table)
            row = {"table": table, "seed": seed, "model": label, "update": baseline["update"],
                   **{key: value.mean().item() for key, value in scores.items()},
                   **{key: baseline[key] for key in ("parameters", "training_seconds", "sampling_seconds",
                                                     "retained_sample_exposures", "boundary_fraction")}}
            for metric in ("crps", "energy_score", "variogram_score"):
                require(abs(row[metric] - baseline[metric]) < 1e-10, "Re-scoring mismatch")
            rows.append(row)
            per_day = {key: daily_mean(value) for key, value in scores.items()}
            for i, sample_id in enumerate(config["sample_ids"]):
                daily.append({"table": table, "seed": seed, "model": label, "sample_id": sample_id,
                              **{key: value[i].item() for key, value in per_day.items()}})
            tensors[f"{table}_s{seed}_{label}"] = scores
    pairs, pair_summary = paired_differences(rows, config["seeds"])
    indexed = {(r["table"], r["seed"], r["model"], r["sample_id"]): r for r in daily}
    daily_pairs = []
    for table in ("primary", "best"):
        for seed in config["seeds"]:
            for sample_id in config["sample_ids"]:
                a, b = (indexed[table, seed, label, sample_id] for label in ("local", "base"))
                daily_pairs.append({"table": table, "seed": seed, "sample_id": sample_id,
                                    **{key: a[key] - b[key] for key in METRICS}})
    summary = {"seeds": config["seeds"], "rows": rows, "paired_summary": pair_summary,
               "score_direction": "lower is better except coverage (compare with nominal level); width needs coverage",
               "uncertainty_unit": "3 training seeds, sample SD with n-1; not 213 independent trainings",
               "data_fingerprint": config["protocol"]["data_fingerprint"], "sample_ids": config["sample_ids"],
               "test_evaluation": "not performed", "protected_file_count": len(protected),
               "source_files_unchanged": all(file_hash(p) == h for p, h in protected.items())}
    require(summary["source_files_unchanged"], "Source changed during scoring")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary", summary), ("evidence", evidence), ("config", config),
                        ("protected_sha256", protected)):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    write_csv(output / "scores.csv", rows)
    write_csv(output / "scores_per_day.csv", daily)
    write_csv(output / "paired_seed_differences.csv", pairs)
    write_csv(output / "paired_daily_differences.csv", daily_pairs)
    torch.save({"sample_ids": config["sample_ids"], "sample_metadata": reference_metadata,
                "data_fingerprint": summary["data_fingerprint"], "scores": tensors}, output / "scores_per_lead.pt")
    for seed in config["seeds"]:
        run = audited[seed, "local"]
        directory = output / f"s{seed}_local"
        directory.mkdir()
        for name, value in (("training_history", run["history"]), ("validation_history", run["validation"]),
                            ("config", run["config"])):
            (directory / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local_attention_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = summarize(read(args.config), args.output)
    print(json.dumps({"output": str(args.output), "runs": len(result["rows"]) // 2,
                      "source_files_unchanged": result["source_files_unchanged"]}))


if __name__ == "__main__":
    main()
