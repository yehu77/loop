"""Six-cell history x architecture report from saved validation artifacts only."""
import argparse
import json
from pathlib import Path

import torch

from data import file_hash
from diagnostics.summarize_history_information import LEADS, audit_history_run
from diagnostics.temporal_dependence import (MARGINALS, METRICS, aligned_validation,
                                             daily_mean, read, require, score_panel, write_csv)
from evaluate import evaluate_artifacts
from summarize_runs import load


VARIANTS = ("base", "loop", "untied")
GROUPS = ("H0", "H24")


def all_specs(config):
    reference = read(config["reference_config"])
    require(file_hash(config["reference_config"]) == config["reference_config_sha256"], "Reference plan changed")
    specs = [{**s, "variant": "base", "reused": True} for s in reference["runs"] if s["group"] in GROUPS]
    specs += [{**s, "reused": False} for s in config["new_runs"]]
    keys = [(s["variant"], s["group"]) for s in specs]
    require(len(keys) == len(set(keys)) == 6 and set(keys) == {(v, g) for v in VARIANTS for g in GROUPS}, "Incomplete six-cell panel")
    return specs


def audit_cell(spec, config):
    if spec["variant"] == "base":
        protocol = read(config["reference_config"])
    else:
        protocol = {**config, "model_common": {**config["model_common"], "variant": spec["variant"]}}
    require(protocol["data_fingerprint"] == config["data_fingerprint"]
            and protocol["sample_ids"] == config["sample_ids"]
            and protocol["training"] == config["training"], "Reference/new recipe differs")
    evaluation = {k: v for k, v in protocol["evaluation_common"].items() if k != "history_training_source_sha256"}
    expected = {k: v for k, v in config["evaluation_common"].items() if k != "history_training_source_sha256"}
    require(evaluation == expected, "Reference/new evaluation differs")
    item = audit_history_run(spec, protocol)
    item["variant"] = spec["variant"]
    return item


def verify_common_initialization_and_streams(reference, candidate):
    # Only canonical BASE keys are required to match across architectures.
    require(all(candidate["initialization"].get(k) == v for k, v in reference["initialization"].items()),
            "Canonical common initialization differs")
    a, b = reference["state"], candidate["state"]
    require(torch.equal(a["noise_rng"], b["noise_rng"]), "Training noise RNG differs")
    for key in ("order", "rng"):
        require(torch.equal(a["data"][key], b["data"][key]), "Training data order differs")
    require(a["data"]["cursor"] == b["data"]["cursor"], "Training data cursor differs")
    for key in ("mean", "std"):
        require(torch.equal(a["weather_scaler"][key], b["weather_scaler"][key]), "Training standardization differs")


def contrasts(rows, keys, metrics):
    """Same-input differences, within-variant history effects, and two interactions."""
    panels = {}
    for row in rows:
        panel = tuple(row[k] for k in keys)
        cell = (row["variant"], row["history_group"])
        require(cell not in panels.setdefault(panel, {}), "Duplicate architecture/history cell")
        panels[panel][cell] = row
    result = []
    for panel, cells in panels.items():
        require(set(cells) == {(v, g) for v in VARIANTS for g in GROUPS}, "Incomplete architecture/history panel")
        context = dict(zip(keys, panel))
        history = {v: {m: cells[v, "H24"][m] - cells[v, "H0"][m] for m in metrics} for v in VARIANTS}
        for group in GROUPS:
            for left, right in (("loop", "base"), ("untied", "base"), ("loop", "untied")):
                result.append({**context, "kind": "same_input", "comparison": left + "-" + right,
                               "condition": group, **{m: cells[left, group][m] - cells[right, group][m] for m in metrics}})
        for variant in VARIANTS:
            result.append({**context, "kind": "history_effect", "comparison": variant,
                           "condition": "H24-H0", **history[variant]})
        for right in ("base", "untied"):
            result.append({**context, "kind": "interaction", "comparison": "loop-" + right,
                           "condition": "history_effect_difference",
                           **{m: history["loop"][m] - history[right][m] for m in metrics}})
    return result


def summarize(config, output):
    output = Path(output)
    require(not output.exists(), "Output exists; refuse overwrite")
    specs = all_specs(config)
    protected = dict(config["protected_artifacts"])
    for spec in specs:
        for directory in map(Path, spec["segments"]):
            a, b = output.resolve(), directory.resolve()
            require(a != b and a not in b.parents and b not in a.parents, "Report overlaps source")
            for p in directory.rglob("*"):
                if p.is_file():
                    digest = file_hash(p)
                    require(str(p) not in protected or protected[str(p)] == digest, "Protected source changed")
                    protected[str(p)] = digest
            p = directory.with_suffix(".lineage.json")
            digest = file_hash(p)
            require(str(p) not in protected or protected[str(p)] == digest, "Source lineage changed")
            protected[str(p)] = digest
    require(all(file_hash(p) == h for p, h in protected.items()), "Protected artifact changed")
    audited = {(s["variant"], s["group"]): audit_cell(s, config) for s in specs}
    reference = audited["base", "H0"]
    for item in audited.values():
        verify_common_initialization_and_streams(reference, item)
    for variant in VARIANTS:
        require(audited[variant, "H0"]["initialization"] == audited[variant, "H24"]["initialization"],
                "Same-architecture H0/H24 initialization differs")
    rows, daily, leads, lead_daily, evidence, tensors = [], [], [], [], [], {}
    reference_y = reference_meta = reference_noise = reference_rand = None
    for spec in specs:
        variant, group, terminal = spec["variant"], spec["group"], Path(spec["segments"][-1])
        run = audited[variant, group]
        evidence.append({"reused": spec["reused"], **{k: v for k, v in run.items() if k not in ("state", "history", "validation")}})
        rand = load(terminal / "rand/scenarios.pt")
        rand_target = load(terminal / "rand/targets.pt")
        rand_x, rand_y, rand_meta = aligned_validation(rand, rand_target, config["sample_ids"], config["data_fingerprint"])
        if reference_rand is None:
            reference_rand = rand_x
        require(torch.equal(rand_x, reference_rand), "RAND panels differ")
        for table, directory in (("primary", terminal), ("best", terminal / "selected")):
            prediction, target = load(directory / "scenarios.pt"), load(directory / "targets.pt")
            x, y, meta = aligned_validation(prediction, target, config["sample_ids"], config["data_fingerprint"])
            require(x.shape == (57, 100, 24, 1), "Wrong validation scene shape")
            if reference_y is None:
                reference_y, reference_meta = y, meta
            require(torch.equal(y, reference_y) and meta == reference_meta
                    and torch.equal(rand_y, y) and rand_meta == meta, "Target/history identities differ")
            saved = read(directory / "metrics.json")
            report, per_window = evaluate_artifacts(prediction, target)
            require({k: v for k, v in saved.items() if k != "diagnostics"} == report, "Saved re-score differs")
            diag = saved["diagnostics"]
            update = 3000 if table == "primary" else run["best"]
            require(diag["history_group"] == group and diag["weights"] == "ema"
                    and diag["successful_updates"] == update, "Wrong group/weights/endpoint")
            if reference_noise is None:
                reference_noise = diag["initial_noise_sha256"]
            require(diag["initial_noise_sha256"] == reference_noise, "Initial sampling noise differs")
            saved_scores = load(directory / "scores_per_window.pt")
            require(saved_scores["sample_ids"] == config["sample_ids"]
                    and saved_scores["data_fingerprint"] == config["data_fingerprint"], "Score identity differs")
            for k in ("crps", "energy_score", "variogram_score"):
                require(torch.equal(saved_scores[k], per_window[k]), "Per-window score mismatch")
            require(len(saved_scores["intervals"]) == len(per_window["intervals"]), "Interval count differs")
            for a, b in zip(saved_scores["intervals"], per_window["intervals"]):
                require(a["level"] == b["level"] and torch.equal(a["covered"], b["covered"])
                        and torch.equal(a["width"], b["width"]), "Interval score mismatch")
            scores = score_panel(x, y)
            context = {"table": table, "variant": variant, "history_group": group}
            rows.append({**context, "seed": 42, "update": update,
                         **{k: v.mean().item() for k, v in scores.items()},
                         "parameters": run["cost"]["parameters"],
                         "training_seconds": sum(h["training_seconds"] for h in run["history"]),
                         "sampling_seconds": diag["sampling_seconds"],
                         "cuda_peak_allocated_bytes": run["cost"].get("cuda_peak_allocated_bytes"),
                         "cuda_peak_reserved_bytes": run["cost"].get("cuda_peak_reserved_bytes"),
                         "boundary_fraction": ((x == 0) | (x == 1)).double().mean().item()})
            by_day = {k: daily_mean(v) for k, v in scores.items()}
            for i, sample_id in enumerate(config["sample_ids"]):
                daily.append({**context, "sample_id": sample_id, **{k: v[i].item() for k, v in by_day.items()}})
            for name, (start, end) in LEADS.items():
                subset = {k: scores[k][:, start:end] for k in MARGINALS}
                leads.append({**context, "lead_group": name, **{k: v.mean().item() for k, v in subset.items()}})
                for i, sample_id in enumerate(config["sample_ids"]):
                    lead_daily.append({**context, "lead_group": name, "sample_id": sample_id,
                                       **{k: v[i].mean().item() for k, v in subset.items()}})
            tensors[table + "_" + variant + "_" + group] = scores
    effects = contrasts(rows, ("table",), METRICS)
    daily_effects = contrasts(daily, ("table", "sample_id"), METRICS)
    lead_effects = contrasts(leads, ("table", "lead_group"), MARGINALS)
    lead_daily_effects = contrasts(lead_daily, ("table", "lead_group", "sample_id"), MARGINALS)
    summary = {"rows": rows, "contrasts": effects, "lead_rows": leads, "lead_contrasts": lead_effects,
               "seed": 42, "data_fingerprint": config["data_fingerprint"], "sample_ids": config["sample_ids"],
               "initial_noise_sha256": reference_noise,
               "uncertainty_unit": "one training seed; 57 days are not independent model trainings",
               "score_direction": "lower score is better; negative interaction means greater LOOP history improvement",
               "cost_caveat": "train_update time excludes validation/IO; old BASE formal peak memory was not recorded",
               "test_scoring": False, "protected_artifact_count": len(protected), "sources_unchanged": True}
    require(all(file_hash(p) == h for p, h in protected.items()), "Artifacts changed during re-scoring")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary", summary), ("config", config), ("evidence", evidence), ("protected_sha256", protected)):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    for name, value in (("scores", rows), ("scores_per_day", daily), ("lead_scores", leads),
                        ("lead_scores_per_day", lead_daily), ("contrasts", effects),
                        ("daily_contrasts", daily_effects), ("lead_contrasts", lead_effects),
                        ("lead_daily_contrasts", lead_daily_effects)):
        write_csv(output / (name + ".csv"), value)
    torch.save({"sample_ids": config["sample_ids"], "scores": tensors}, output / "scores_per_lead.pt")
    for (variant, group), item in audited.items():
        p = output / (variant + "_" + group)
        p.mkdir()
        for name in ("history", "validation", "config"):
            (p / (name + ".json")).write_text(json.dumps(item[name], indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/history_architecture_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summarize(read(args.config), args.output)
