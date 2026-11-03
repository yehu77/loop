"""Read-only matched H0/H1/H24 audit and fixed-endpoint/auxiliary-best report."""
import argparse
import json
from pathlib import Path

import torch

from data import file_hash
from diagnostics.temporal_dependence import (MARGINALS, METRICS, aligned_validation,
                                             daily_mean, read, require, score_panel, write_csv)
from evaluate import evaluate_artifacts
from summarize_runs import load, select_nodes, stitch
from train import TrainConfig


GROUPS = ("H0", "H1", "H24")
PAIRS = (("H1", "H0"), ("H24", "H0"), ("H24", "H1"))
LEADS = {"1-6": (0, 6), "7-12": (6, 12), "13-24": (12, 24)}


def audit_history_run(spec, frozen):
    group = spec["group"]
    require(group in GROUPS, "Unknown history group")
    directories = [Path(p) for p in spec["segments"]]
    require(len(directories) == len(set(directories)) > 0, "Duplicate/empty segments")
    configs = [read(p / "config.json") for p in directories]
    states = [load(p / "checkpoint_last.pt") for p in directories]
    lineages = [read(p.with_suffix(".lineage.json")) for p in directories]
    cfg = configs[0]
    expected_model = {**frozen["model_common"], "history_group": group}
    expected_evaluation = {**frozen["evaluation_common"], "history_group": group}
    require(cfg["resume_from"] is None, "First segment must start from scratch")
    for i, (path, config, state, lineage) in enumerate(zip(directories, configs, states, lineages)):
        require(config["model"] == expected_model, "Model/group differs from frozen interface")
        require(TrainConfig(**config["training"]) == TrainConfig(**frozen["training"]), "Training differs")
        require(config["evaluation"] == expected_evaluation, "Evaluation differs")
        require(config["validation_sample_ids"] == frozen["sample_ids"], "Validation dates differ")
        require(config["data_fingerprint"] == frozen["data_fingerprint"], "Data fingerprint differs")
        require(lineage["training_source_sha256"] == frozen["training_source_sha256"]
                and lineage["data_sha256"] == frozen["data_sha256"], "Source/data version differs")
        require(state["model_config"] == expected_model
                and TrainConfig(**state["train_config"]) == TrainConfig(**frozen["training"]), "Checkpoint config differs")
        meta = state["metadata"]
        require(not meta.get("failure", False) and meta["data_kind"] == "real", "Invalid checkpoint")
        require(meta["data_fingerprint"] == frozen["data_fingerprint"]
                and meta["protocol_fingerprint"] == frozen["history_protocol_fingerprint"]
                and meta["run_config"] == expected_evaluation, "Checkpoint identity differs")
        require(state["attempts"] == state["updates"] and state["skipped"] == 0, "Unexpected skipped update")
        require(state["ema_initialized"] == (state["updates"] >= 1000)
                and state["ema_num_averaged_updates"] == max(0, state["updates"] - 1000), "EMA clock differs")
        if i:
            prior = directories[i-1] / "checkpoint_last.pt"
            require(Path(config["resume_from"]).resolve() == prior.resolve()
                    and lineage["resume_sha256"] == file_hash(prior)
                    and lineage["resumed_successful_updates"] == states[i-1]["updates"], "Broken continuation chain")
        if i == len(directories)-1:
            require(lineage["exit_code"] == 0 and lineage["sources_unchanged"], "Incomplete terminal run")
            require(lineage["last_checkpoint_sha256"] == file_hash(path / "checkpoint_last.pt"), "Checkpoint changed")
    endpoints = [s["updates"] for s in states]
    require(endpoints[-1] == 3000, "Endpoint must be 3000")
    history = stitch([read(p / "training_history.json") for p in directories], endpoints)
    validation, best = select_nodes([read(p / "validation_history.json") for p in directories],
                                   endpoints, 1000, 250, 3000)
    anchors = [p / "checkpoint_ema_start.pt" for p in directories if (p / "checkpoint_ema_start.pt").exists()]
    require(len(anchors) == 1, "Expected one EMA copy boundary")
    anchor = load(anchors[0])
    require(anchor["updates"] == 1000 and anchor["ema_num_averaged_updates"] == 0
            and anchor["model_config"] == expected_model
            and anchor["metadata"]["data_fingerprint"] == frozen["data_fingerprint"]
            and all(torch.equal(v, anchor["ema"][k]) for k, v in anchor["model"].items()), "Invalid EMA copy")
    selected = load(directories[-1] / "checkpoint_best.pt")
    require(selected["updates"] == best and selected["model_config"] == expected_model
            and TrainConfig(**selected["train_config"]) == TrainConfig(**frozen["training"])
            and selected["metadata"]["run_config"] == expected_evaluation
            and selected["metadata"]["data_fingerprint"] == frozen["data_fingerprint"]
            and not selected["metadata"].get("failure", False), "Best checkpoint mismatch")
    cost = read(directories[-1] / "cost.json")
    require(cost["primary_update"] == 3000 and cost["auxiliary_best_update"] == best, "Cost identity differs")
    return {"group": group, "state": states[-1], "best": best, "history": history,
            "validation": validation, "initialization": read(directories[0] / "initialization.json"),
            "cost": cost, "config": cfg, "lineages": lineages}


def paired_rows(rows, key_fields, metrics):
    """No seed SD: this experiment has only one training seed."""
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        require(row["group"] not in indexed.setdefault(key, {}), "Duplicate history panel")
        indexed[key][row["group"]] = row
    result = []
    for key, groups in indexed.items():
        require(set(groups) == set(GROUPS), "Incomplete history comparison")
        for left, right in PAIRS:
            result.append({**dict(zip(key_fields, key)), "comparison": left + "-" + right,
                           **{m: groups[left][m] - groups[right][m] for m in metrics}})
    return result


def summarize(frozen, output):
    output = Path(output)
    require(not output.exists(), "Report exists; refuse overwrite")
    require([s["group"] for s in frozen["runs"]] == list(GROUPS), "Wrong group panel")
    protected = dict(frozen["protected_artifacts"])
    for spec in frozen["runs"]:
        for directory in map(Path, spec["segments"]):
            require(output.resolve() != directory.resolve() and output.resolve() not in directory.resolve().parents
                    and directory.resolve() not in output.resolve().parents, "Output overlaps source")
            protected.update({str(p): file_hash(p) for p in directory.rglob("*") if p.is_file()})
            p = directory.with_suffix(".lineage.json")
            protected[str(p)] = file_hash(p)
    require(all(file_hash(p) == h for p, h in protected.items()), "Protected artifact changed")
    audited = {s["group"]: audit_history_run(s, frozen) for s in frozen["runs"]}
    reference = audited["H0"]
    for item in audited.values():
        require(item["initialization"] == reference["initialization"], "Initial tensors differ")
        require(torch.equal(item["state"]["noise_rng"], reference["state"]["noise_rng"]), "Training noise differs")
        for key in ("order", "rng"):
            require(torch.equal(item["state"]["data"][key], reference["state"]["data"][key]), "Data order differs")
        require(item["state"]["data"]["cursor"] == reference["state"]["data"]["cursor"], "Data cursor differs")
    rows, daily, leads, lead_daily, tensors, evidence = [], [], [], [], {}, []
    common_targets = common_metadata = common_noise = rand_values = None
    for spec in frozen["runs"]:
        group, terminal = spec["group"], Path(spec["segments"][-1])
        run = audited[group]
        evidence.append({k: v for k, v in run.items() if k not in ("state", "history", "validation")})
        for table, directory in (("primary", terminal), ("best", terminal / "selected"), ("rand", terminal / "rand")):
            prediction, target = load(directory / "scenarios.pt"), load(directory / "targets.pt")
            x, y, meta = aligned_validation(prediction, target, frozen["sample_ids"], frozen["data_fingerprint"])
            require(x.shape == (57, 100, 24, 1), "Wrong scene panel")
            if common_targets is None:
                common_targets, common_metadata = y, meta
            require(torch.equal(y, common_targets) and meta == common_metadata, "Different target/history identities")
            report, per_window = evaluate_artifacts(prediction, target)
            saved = read(directory / "metrics.json")
            require({k: v for k, v in saved.items() if k != "diagnostics"} == report, "Independent re-score mismatch")
            saved_scores = load(directory / "scores_per_window.pt")
            for key in ("crps", "energy_score", "variogram_score"):
                require(torch.equal(saved_scores[key], per_window[key]), "Per-day metric mismatch")
            require(saved_scores["sample_ids"] == frozen["sample_ids"]
                    and saved_scores["data_fingerprint"] == frozen["data_fingerprint"], "Score identity mismatch")
            for a, b in zip(saved_scores["intervals"], per_window["intervals"]):
                require(a["level"] == b["level"] and torch.equal(a["covered"], b["covered"])
                        and torch.equal(a["width"], b["width"]), "Interval scores differ")
            diag = saved["diagnostics"]
            update = None if table == "rand" else 3000 if table == "primary" else run["best"]
            if table != "rand":
                require(diag["history_group"] == group and diag["weights"] == "ema"
                        and diag["successful_updates"] == update, "Prediction group/weight/endpoint mismatch")
                if common_noise is None:
                    common_noise = diag["initial_noise_sha256"]
                require(diag["initial_noise_sha256"] == common_noise, "Validation initial noise differs")
            else:
                indices = torch.tensor(diag["training_indices"])
                require(indices.min() >= 0 and indices.max() < 545, "RAND index outside common training data")
                if rand_values is None:
                    rand_values = x
                require(torch.equal(x, rand_values), "RAND panels differ")
            scores = score_panel(x, y)
            rows.append({"table": table, "group": group, "seed": 42, "update": update,
                         **{k: v.mean().item() for k, v in scores.items()},
                         "parameters": run["cost"]["parameters"] if table != "rand" else None,
                         "training_seconds": sum(r["training_seconds"] for r in run["history"]) if table != "rand" else None,
                         "sampling_seconds": diag.get("sampling_seconds"),
                         "boundary_fraction": ((x == 0) | (x == 1)).double().mean().item()})
            for i, sample_id in enumerate(frozen["sample_ids"]):
                daily.append({"table": table, "group": group, "sample_id": sample_id,
                              **{k: daily_mean(v)[i].item() for k, v in scores.items()}})
            for name, (start, end) in LEADS.items():
                subset = {k: scores[k][:, start:end] for k in MARGINALS}
                leads.append({"table": table, "group": group, "lead_group": name,
                              **{k: v.mean().item() for k, v in subset.items()}})
                for i, sample_id in enumerate(frozen["sample_ids"]):
                    lead_daily.append({"table": table, "group": group, "lead_group": name,
                                       "sample_id": sample_id, **{k: v[i].mean().item() for k, v in subset.items()}})
            tensors[table + "_" + group] = scores
    pairs = paired_rows(rows, ("table",), METRICS)
    paired_daily = paired_rows(daily, ("table", "sample_id"), METRICS)
    paired_leads = paired_rows(leads, ("table", "lead_group"), MARGINALS)
    paired_lead_daily = paired_rows(lead_daily, ("table", "lead_group", "sample_id"), MARGINALS)
    require(all(file_hash(p) == h for p, h in protected.items()), "Source artifact changed during scoring")
    output.mkdir(parents=True, exist_ok=False)
    summary = {"rows": rows, "paired_effects": pairs, "lead_groups": leads,
               "paired_lead_effects": paired_leads, "training_seed": 42,
               "interpretation": "single-seed development screening on 57 complete rolling-history days; no seed uncertainty estimate",
               "data_fingerprint": frozen["data_fingerprint"], "sample_ids": frozen["sample_ids"],
               "initial_noise_sha256": common_noise, "test_scoring": False,
               "protected_artifact_count": len(protected), "sources_unchanged": True}
    for name, value in (("summary", summary), ("evidence", evidence), ("config", frozen), ("protected_sha256", protected)):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    for name, value in (("scores", rows), ("scores_per_day", daily), ("lead_scores", leads),
                        ("lead_scores_per_day", lead_daily), ("paired_effects", pairs),
                        ("paired_daily_effects", paired_daily), ("paired_lead_effects", paired_leads),
                        ("paired_lead_daily_effects", paired_lead_daily)):
        write_csv(output / (name + ".csv"), value)
    torch.save({"sample_ids": frozen["sample_ids"], "scores": tensors}, output / "scores_per_lead.pt")
    for group, run in audited.items():
        p = output / group
        p.mkdir()
        for name in ("history", "validation", "config"):
            (p / (name + ".json")).write_text(json.dumps(run[name], indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/history_information_v1/execution.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summarize(read(args.config), args.output)
