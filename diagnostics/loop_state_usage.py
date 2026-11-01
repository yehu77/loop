"""R03: read-only supervised denoiser interventions, never trajectory sampling."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from data import file_hash, load_package, tensor_hash
from diagnostics.temporal_dependence import aligned_validation, read, require, write_csv
from diffusion import LinearNoiseSchedule
from evaluate import identified_artifact
from model import ConditionalLoopDiT, ModelConfig, sinusoidal_embedding
from train import WeatherScaler

METRICS = ("noise_mse", "delta_mse", "output_change_rms", "first_update_rms",
           "input_state_change_rms", "final_state_change_rms")


def rms(x):
    return x.double().flatten(1).square().mean(1).sqrt()


@torch.no_grad()
def branches(model, x, nwp, calendar, t):
    """Return final predictions; endpoint lambda values avoid cancellation rounding."""
    require(not any(m.training for m in model.modules()), "Diagnostic requires eval mode")
    require(model.config.attention_scope == "full", "R03 requires full attention")
    require(model.config.variant == "base" or model.config.rounds == 2, "Only two-Core models")
    condition = model.encode_condition(nwp, calendar)
    h0 = model.input_projection(torch.cat((x, condition), dim=-1)) + model.position_encoding
    tau = model.time_mlp(sinusoidal_embedding(t, model.config.hidden_dim))
    h1 = model.cores[0](h0, h0, tau)
    if model.config.variant == "base":
        return {1.0: model.output_head(h1, tau)}, h0, h1, {1.0: h1}, {1.0: h1}
    second = model.cores[0] if model.config.variant == "loop" else model.cores[1]
    inputs = {0.0: h0, 0.5: h0 + 0.5 * (h1 - h0), 1.0: h1}
    states = {lam: second(value, h0, tau) for lam, value in inputs.items()}
    return {lam: model.output_head(value, tau) for lam, value in states.items()}, h0, h1, inputs, states


def measure(predictions, noise, h0, h1, inputs, states):
    normal = predictions[1.0]
    baseline = (normal.double() - noise.double()).flatten(1).square().mean(1)
    result = {}
    for lam, prediction in predictions.items():
        loss = (prediction.double() - noise.double()).flatten(1).square().mean(1)
        result[lam] = {"noise_mse": loss, "delta_mse": loss - baseline,
                       "output_change_rms": rms(prediction - normal),
                       "first_update_rms": rms(h1 - h0),
                       "input_state_change_rms": rms(inputs[lam] - h1),
                       "final_state_change_rms": rms(states[lam] - states[1.0])}
        require(all(torch.isfinite(v).all() for v in result[lam].values()), "Non-finite diagnostic")
    return result


def noise_bank(seeds, shape):
    return {seed: torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
            for seed in seeds}


def validate_config(config):
    require(config["training_seeds"] == [42, 43, 44] and config["variants"] == ["base", "loop", "untied"],
            "Frozen training panel changed")
    require(config["timesteps"] == [50, 200, 400, 700, 950]
            and config["noise_seeds"] == [3101, 3102, 3103, 3104]
            and config["lambdas"] == [0.0, 0.5, 1.0], "Frozen intervention grid changed")
    require(config["device"] == "cpu" and config["dtype"] == "float32" and config["batch_size"] == 16,
            "Frozen execution settings changed")
    require(config["shape"] == [71, 24, 1] and len(set(config["sample_ids"])) == 71,
            "Unexpected validation panel")
    identities = [(s["training_seed"], s["variant"]) for s in config["runs"]]
    require(len(identities) == 9 and set(identities) == {(s, v) for s in config["training_seeds"]
                                                      for v in config["variants"]}, "Missing or duplicate model")
    require(config["atol"] == 1e-6 and config["rtol"] == 1e-5, "Frozen FP32 equivalence tolerance changed")


def check_checkpoint(state, spec, package):
    model_config = ModelConfig(**state["model_config"])
    expected = ModelConfig(variant=spec["variant"])
    manifest = package["manifest"]
    require(model_config == expected, "Checkpoint model differs")
    require(not state["metadata"].get("failure", False) and state["metadata"].get("data_kind") == "real"
            and state["format_version"] == 2 and state["updates"] == 3000
            and state["attempts"] == 3000 and state["skipped"] == 0,
            "Requires normal fixed-endpoint v2 checkpoint")
    require(state["ema_initialized"] and state["ema_num_averaged_updates"] == 2000,
            "Wrong delayed EMA state")
    train = state["train_config"]
    require(train["seed"] == spec["training_seed"] and train["training_protocol_version"] == 2
            and train["ema_start_update"] == 1000 and train["ema_decay"] == 0.999
            and train["diffusion_steps"] == 1000 and not train["amp"], "Training recipe differs")
    for key in ("data_fingerprint", "protocol_fingerprint"):
        require(state["metadata"][key] == manifest[key], "Checkpoint data identity differs")
    for key in ("mean", "std"):
        require(torch.equal(state["weather_scaler"][key], torch.tensor(manifest["scaler"][key])),
                "Checkpoint training scaler differs")
    require(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state["ema"].values()),
            "Invalid EMA parameter or buffer")
    return model_config


def aggregate(rows, config):
    """Noise repeats -> days -> per-trained-model means -> three-seed summaries."""
    groups = {}
    identities = set()
    for row in rows:
        key = tuple(row[k] for k in ("training_seed", "variant", "t", "lambda", "sample_id"))
        identity = key + (row["noise_seed"],)
        require(identity not in identities, "Duplicate diagnostic input")
        identities.add(identity)
        groups.setdefault(key, []).append(row)
    expected = {(s, v, t, lam, sample) for s in config["training_seeds"] for v in config["variants"]
                for t in config["timesteps"] for lam in ([1.0] if v == "base" else config["lambdas"])
                for sample in config["sample_ids"]}
    require(set(groups) == expected, "Incomplete diagnostic panel")
    daily = []
    for key, group in sorted(groups.items()):
        require(sorted(r["noise_seed"] for r in group) == config["noise_seeds"], "Missing noise repeat")
        daily.append(dict(zip(("training_seed", "variant", "t", "lambda", "sample_id"), key))
                     | {m: statistics.mean(r[m] for r in group) for m in METRICS})
    runs = []
    run_groups = {}
    for row in daily:
        key = tuple(row[k] for k in ("training_seed", "variant", "t", "lambda"))
        run_groups.setdefault(key, []).append(row)
    for key, group in sorted(run_groups.items()):
        runs.append(dict(zip(("training_seed", "variant", "t", "lambda"), key))
                    | {m: statistics.mean(r[m] for r in group) for m in METRICS}
                    | {"days_positive_delta_mse": sum(r["delta_mse"] > 0 for r in group),
                       "days_negative_delta_mse": sum(r["delta_mse"] < 0 for r in group)})
    summaries = []
    for variant in config["variants"]:
        for t in config["timesteps"]:
            for lam in ([1.0] if variant == "base" else config["lambdas"]):
                group = [r for r in runs if (r["variant"], r["t"], r["lambda"]) == (variant, t, lam)]
                summaries.append({"variant": variant, "t": t, "lambda": lam, "n_training_seeds": len(group),
                                  "metrics": {m: {"mean": statistics.mean(r[m] for r in group),
                                                   "sample_std": statistics.stdev(r[m] for r in group)}
                                              for m in METRICS}})
    return daily, runs, summaries


def run(config, output):
    output = Path(output).resolve()
    require(not output.exists(), "Output exists; refuse overwrite")
    validate_config(config)
    protected = config["protected_files"]
    for name in protected:
        path = Path(name).resolve()
        require(path != output and output not in path.parents, "Output overlaps a protected input")
    for spec in config["runs"]:
        directory = Path(spec["checkpoint"]).resolve().parent
        require(output != directory and directory not in output.parents, "Output overlaps source run")
    require(all(file_hash(p) == h for p, h in protected.items()), "Frozen source hash mismatch")
    package = load_package(config["data"])
    manifest = package["manifest"]
    require(manifest["data_fingerprint"] == config["data_fingerprint"], "Prepared data differs")
    metadata = package["metadata"]["validation"]
    require([m["sample_id"] for m in metadata] == config["sample_ids"]
            and all(m["split"] == "validation" for m in metadata), "Validation identities differ")
    values = package["splits"]["validation"]
    y, calendar = values["y"], values["calendar"]
    require(list(y.shape) == config["shape"], "Validation tensor dimensions differ")
    noises = noise_bank(config["noise_seeds"], tuple(y.shape))
    schedule = LinearNoiseSchedule(1000)
    noisy = {(t, seed): schedule.add_noise(2 * y.float() - 1,
             torch.full((len(y),), t, dtype=torch.long), noise) for t in config["timesteps"]
             for seed, noise in noises.items()}
    rows, evidence = [], []
    rng_before = torch.get_rng_state().clone()
    started = time.perf_counter()
    for spec in config["runs"]:
        state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
        mc = check_checkpoint(state, spec, package)
        target = torch.load(Path(spec["checkpoint"]).parent / "targets.pt", map_location="cpu", weights_only=True)
        dummy = identified_artifact(y[:, None], manifest, metadata)
        _, original_y, original_meta = aligned_validation(dummy, target, config["sample_ids"], config["data_fingerprint"])
        require(torch.equal(original_y, y) and original_meta == metadata, "Historical validation target differs")
        with torch.random.fork_rng(devices=[]):
            model = ConditionalLoopDiT(mc).eval().requires_grad_(False)
        model.load_state_dict(state["ema"], strict=True)
        before = {k: tensor_hash(v) for k, v in model.state_dict().items()}
        nwp = WeatherScaler(**state["weather_scaler"]).transform(values["nwp"])
        controls = {"normal_forward_max_abs": 0.0, "loop_reset_single_readout_max_abs": None}
        for t in config["timesteps"]:
            for seed, noise in noises.items():
                for start in range(0, len(y), config["batch_size"]):
                    end = min(len(y), start + config["batch_size"])
                    times = torch.full((end - start,), t, dtype=torch.long)
                    x = noisy[t, seed][start:end]
                    with torch.no_grad():
                        predictions, h0, h1, inputs, states = branches(model, x, nwp[start:end], calendar[start:end], times)
                        official = model(x, nwp[start:end], calendar[start:end], times)
                        torch.testing.assert_close(predictions[1.0], official, atol=config["atol"], rtol=config["rtol"])
                        controls["normal_forward_max_abs"] = max(controls["normal_forward_max_abs"],
                                                                  (predictions[1.0] - official).abs().max().item())
                        if mc.variant == "loop":
                            one = model(x, nwp[start:end], calendar[start:end], times, rounds=1)
                            torch.testing.assert_close(predictions[0.0], one, atol=config["atol"], rtol=config["rtol"])
                            controls["loop_reset_single_readout_max_abs"] = max(
                                controls["loop_reset_single_readout_max_abs"] or 0,
                                (predictions[0.0] - one).abs().max().item())
                        measured = measure(predictions, noise[start:end], h0, h1, inputs, states)
                    for lam, metrics in measured.items():
                        for i in range(end - start):
                            rows.append({"training_seed": spec["training_seed"], "variant": mc.variant,
                                         "t": t, "noise_seed": seed, "lambda": lam,
                                         "sample_id": metadata[start + i]["sample_id"],
                                         **{m: v[i].item() for m, v in metrics.items()}})
        after = {k: tensor_hash(v) for k, v in model.state_dict().items()}
        require(before == after, "Parameter or buffer changed during diagnostic")
        evidence.append({**spec, "model_config": asdict(mc), "controls": controls,
                         "state_before_sha256": before, "state_after_sha256": after,
                         "eval_mode": not any(m.training for m in model.modules())})
        print(f"Completed s{spec['training_seed']} {mc.variant}: 1420 fixed inputs", flush=True)
    daily, runs, summaries = aggregate(rows, config)
    require(torch.equal(rng_before, torch.get_rng_state()), "Global RNG changed")
    require(all(file_hash(p) == h for p, h in protected.items()), "Source changed during diagnostic")
    report = {"stage": "R03", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
              "python": sys.version, "torch": str(torch.__version__), "device": "cpu", "threads": torch.get_num_threads(),
              "elapsed_seconds": time.perf_counter() - started, "config": config,
              "summary_by_noise_level": summaries, "evidence": evidence,
              "protected_hashes_unchanged": True, "global_rng_unchanged": True,
              "noise_tensor_hashes": {str(s): tensor_hash(v) for s, v in noises.items()},
              "noise_by_sample_hashes": {str(s): {m["sample_id"]: tensor_hash(v[i]) for i, m in enumerate(metadata)} for s, v in noises.items()},
              "noisy_input_tensor_hashes": {f"t{t}_noise{s}": tensor_hash(v) for (t, s), v in noisy.items()},
              "sample_metadata": metadata, "row_counts": {"per_input": len(rows), "per_day": len(daily), "per_run": len(runs)},
              "interpretation": "Supervised denoiser MSE, not forecasting scores; mean noises within day then days within model; SD across three training seeds."}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_csv(output / "per_input.csv", rows)
    write_csv(output / "per_day.csv", daily)
    write_csv(output / "per_run.csv", runs)
    torch.save({"sample_ids": config["sample_ids"], "noises": noises, "noisy_inputs": noisy}, output / "fixed_inputs.pt")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diagnostics/loop_state_usage_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(read(args.config), args.output)
    print(json.dumps({"output": str(args.output), "rows": result["row_counts"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
