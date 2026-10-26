"""Identity-bound real-data runner with explicit v1/v2 training protocols."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

from data import RealBatches, load_package, tensor_hash
from diffusion import sample_scenarios
from evaluate import identified_artifact, write_evaluation
from model import ModelConfig
from train import TrainConfig, Trainer, WeatherScaler, synchronize


def validation_panel(package, limit):
    values, metadata = package["splits"]["validation"], package["metadata"]["validation"]
    if limit is not None and not 1 <= limit <= len(metadata):
        raise ValueError("validation_limit must be within the available validation windows")
    return {key: tensor[:limit] for key, tensor in values.items()}, metadata[:limit]


@torch.no_grad()
def validation_artifacts(trainer, package, run_config, weights="ema"):
    values, metadata = validation_panel(package, run_config["validation_limit"])
    if weights not in ("raw", "ema"):
        raise ValueError("weights must be raw or ema")
    if weights == "ema" and not getattr(trainer, "ema_initialized", True):
        raise ValueError("EMA is not initialized; use raw diagnostics before its start update")
    model = trainer.ema if weights == "ema" else trainer.model
    generator = torch.Generator(device=trainer.device).manual_seed(run_config["validation_seed"])
    nwp = trainer.weather_scaler.transform(values["nwp"].to(trainer.device))
    synchronize(trainer.device)
    started = time.perf_counter()
    trajectories, diagnostics = sample_scenarios(
        model, trainer.schedule, nwp, values["calendar"].to(trainer.device),
        scenarios=run_config["scenarios"], sampling_steps=run_config["sampling_steps"],
        chunk_size=run_config["chunk_size"], generator=generator,
    )
    synchronize(trainer.device)
    if run_config.get("record_sampling_cost", False):
        diagnostics["sampling_seconds"] = time.perf_counter() - started
        diagnostics["generated_trajectories"] = len(metadata) * run_config["scenarios"]
    diagnostics.update({"weights": weights, "successful_updates": trainer.updates,
                        "validation_seed": run_config["validation_seed"]})
    manifest = package["manifest"]
    return (identified_artifact(trajectories, manifest, metadata),
            identified_artifact(values["y"], manifest, metadata), diagnostics)


def rand_artifacts(package, run_config):
    values, metadata = validation_panel(package, run_config["validation_limit"])
    generator = torch.Generator().manual_seed(run_config["rand_seed"])
    training = package["splits"]["train"]["y"]
    indices = torch.randint(len(training), (len(metadata), run_config["scenarios"]), generator=generator)
    manifest = package["manifest"]
    return (identified_artifact(training[indices], manifest, metadata),
            identified_artifact(values["y"], manifest, metadata),
            {"baseline": "RAND: complete training trajectories sampled with replacement",
             "training_indices": indices.tolist(), "rand_seed": run_config["rand_seed"]})


def load_real_checkpoint(path, package, device, **expectations):
    trainer, batch_state, metadata = Trainer.load(
        path, device, expected_data_fingerprint=package["manifest"]["data_fingerprint"], **expectations
    )
    if metadata.get("data_kind") != "real":
        raise ValueError("This runner resumes real-data checkpoints only")
    batches = RealBatches(package, trainer.config.microbatch_size, trainer.config.seed)
    batches.load_state_dict(batch_state)
    return trainer, batches, metadata


def run(args):
    package = load_package(args.data)
    manifest = package["manifest"]
    if args.resume:
        trainer, batches, checkpoint_metadata = load_real_checkpoint(
            args.resume, package, args.device, expected_variant=args.variant,
            expected_training_protocol_version=args.training_protocol_version,
            expected_ema_start_update=args.ema_start_update,
            expected_attention_scope=getattr(args, "attention_scope", None))
        run_config = checkpoint_metadata["run_config"]
        best_score = checkpoint_metadata["best_validation_crps"]
        best_update = checkpoint_metadata["best_update"]
        previous_best = args.resume.parent / "checkpoint_best.pt"
        if best_score is not None and not previous_best.is_file():
            raise ValueError("Resume requires checkpoint_best.pt beside the supplied checkpoint")
        best_state = torch.load(previous_best, map_location="cpu", weights_only=True) if best_score is not None else None
        if best_state is not None and (best_state["metadata"].get("data_fingerprint") != manifest["data_fingerprint"]
                or best_state["metadata"].get("run_config") != run_config
                or best_state["metadata"].get("best_validation_crps") != best_score
                or best_state["metadata"].get("best_update") != best_update
                or ModelConfig(**best_state["model_config"]) != trainer.model.config
                or TrainConfig(**best_state["train_config"]) != trainer.config
                or best_state["updates"] != best_update
                or best_state["metadata"].get("failure", False)):
            raise ValueError("Best checkpoint does not match the resumed run")
    else:
        run_config = {
            "validation_limit": args.validation_limit, "validation_seed": 2026, "rand_seed": 2027,
            "scenarios": args.scenarios, "sampling_steps": args.sampling_steps,
            "chunk_size": args.chunk_size, "validate_every": args.validate_every,
            "selection": "lowest validation mean CRPS using EMA; strict improvement; earliest tie",
            "weights_reported": ["raw", "ema"], "test_evaluation": "not performed",
        }
        scaler = WeatherScaler(*(torch.tensor(manifest["scaler"][k]) for k in ("mean", "std")))
        version = args.training_protocol_version or 1
        start = args.ema_start_update if args.ema_start_update is not None else (1000 if version == 2 else 0)
        config = TrainConfig(amp=args.amp, seed=args.seed, microbatch_size=args.microbatch_size,
                             accumulation_steps=args.accumulation_steps,
                             training_protocol_version=version, ema_start_update=start)
        trainer = Trainer(ModelConfig(variant=args.variant or "base",
                                      attention_scope=getattr(args, "attention_scope", None) or "full"),
                          config, scaler, args.device)
        batches = RealBatches(package, config.microbatch_size, config.seed)
        best_score, best_update = None, None
        if version == 2:
            run_config.update({
                "selection": "auxiliary best: ready EMA on scheduled validation nodes; minimum CRPS; earliest tie",
                "primary": "fixed endpoint delayed EMA (formal comparison: update 3000)",
                "record_sampling_cost": True,
                "rng_streams": {"canonical_initialization": config.seed, "extra_cores": config.seed + 10000,
                                "data_order": config.seed, "training_noise": config.seed + 1,
                                "validation": 2026, "rand": 2027},
                "initialization": "BASE canonical; common components and core 0 copied; independent extra cores",
            })
    _, panel_metadata = validation_panel(package, run_config["validation_limit"])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    if not args.resume:
        (output / "initialization.json").write_text(json.dumps({
            key: tensor_hash(value) for key, value in trainer.model.state_dict().items()
        }, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({
        "model": asdict(trainer.model.config), "training": asdict(trainer.config),
        "evaluation": run_config, "additional_successful_update_budget": args.updates,
        "data_fingerprint": manifest["data_fingerprint"],
        "validation_sample_ids": [m["sample_id"] for m in panel_metadata],
        "torch_version": str(torch.__version__), "device": str(trainer.device),
        "resume_from": str(args.resume) if args.resume else None,
    }, indent=2) + "\n")
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.resume and best_score is not None:
        shutil.copyfile(previous_best, output / "checkpoint_best.pt")
    rand_report = write_evaluation(output / "rand", *rand_artifacts(package, run_config))
    validation_history, training_history = [], []

    def metadata():
        return {"data_kind": "real", "data_fingerprint": manifest["data_fingerprint"],
                "protocol_fingerprint": manifest["protocol_fingerprint"], "run_config": run_config,
                "best_validation_crps": best_score, "best_update": best_update}

    def validate():
        nonlocal best_score, best_update
        entry = {"updates": trainer.updates, "ema_initialized": trainer.ema_initialized,
                 "ema_num_averaged_updates": trainer.ema_num_averaged_updates}
        for weights in (("raw", "ema") if trainer.ema_initialized else ("raw",)):
            prediction, target, diagnostics = validation_artifacts(trainer, package, run_config, weights)
            directory = output / "validation" / ("update_{:06d}".format(trainer.updates)) / weights
            report = write_evaluation(directory, prediction, target, diagnostics)
            entry[weights] = {key: report[key] for key in ("crps", "energy_score", "variogram_score")}
            entry[weights]["boundary_fraction"] = diagnostics["zero_power_fraction"] + diagnostics["full_power_fraction"]
            if "sampling_seconds" in diagnostics:
                entry[weights]["sampling_seconds"] = diagnostics["sampling_seconds"]
        validation_history.append(entry)
        score = entry.get("ema", {}).get("crps")
        eligible = (trainer.config.training_protocol_version == 1
                    or trainer.updates % run_config["validate_every"] == 0)
        if score is not None and eligible and (best_score is None or score < best_score):
            best_score, best_update = score, trainer.updates
            trainer.save(output / "checkpoint_best.pt", batches, metadata())
        (output / "validation_history.json").write_text(json.dumps(validation_history, indent=2) + "\n")
        print("validation update={} raw_CRPS={:.6f} EMA_CRPS={} RAND_CRPS={:.6f}".format(
            trainer.updates, entry["raw"]["crps"], score, rand_report["crps"]), flush=True)

    validate()  # Includes the untrained model, or the resumed state, as a diagnostic.
    trainer.save(output / "checkpoint_last.pt", batches, metadata())
    target_updates = trainer.updates + args.updates
    starting_attempts = trainer.attempts
    max_attempts = max(args.updates * 5, args.updates + 10)
    try:
        while trainer.updates < target_updates:
            if trainer.attempts - starting_attempts >= max_attempts:
                raise FloatingPointError("Too many skipped optimizer steps for this short run")
            synchronize(trainer.device)
            started = time.perf_counter()
            entry = trainer.train_update(batches)
            synchronize(trainer.device)
            entry["training_seconds"] = time.perf_counter() - started
            entry["attempts"] = trainer.attempts
            entry["samples_consumed_total"] = trainer.attempts * trainer.config.microbatch_size * trainer.config.accumulation_steps
            training_history.append(entry)
            if (entry["did_update"] and trainer.config.training_protocol_version == 2
                    and trainer.updates == trainer.config.ema_start_update):
                trainer.save(output / "checkpoint_ema_start.pt", batches, metadata())
            if entry["did_update"] and (trainer.updates % run_config["validate_every"] == 0
                                         or trainer.updates == target_updates):
                validate()
            trainer.save(output / "checkpoint_last.pt", batches, metadata())
            (output / "training_history.json").write_text(json.dumps(training_history, indent=2) + "\n")
            print("{} update={} loss={:.6f}".format(trainer.model.config.variant, trainer.updates, entry["loss"]), flush=True)
    except (FloatingPointError, ValueError, RuntimeError, KeyboardInterrupt):
        trainer.save(output / "failure.pt", batches, {**metadata(), "failure": True})
        raise
    if trainer.config.training_protocol_version == 2:
        # Reuse the saved endpoint evaluation; keep best in a separate directory.
        endpoint = output / "validation" / ("update_{:06d}".format(trainer.updates)) / "ema"
        if trainer.ema_initialized:
            for path in endpoint.iterdir():
                shutil.copyfile(path, output / path.name)
        (output / "cost.json").write_text(json.dumps({
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "core_executions_per_denoise": 1 if trainer.model.config.variant == "base" else trainer.model.config.rounds,
            "independent_cores": len(trainer.model.cores),
            "successful_updates": trainer.updates, "attempts": trainer.attempts,
            "sample_exposures": trainer.attempts * trainer.config.microbatch_size * trainer.config.accumulation_steps,
            "training_seconds_this_segment": sum(row["training_seconds"] for row in training_history),
            "timing_scope": "synchronized train_update only; validation, checkpoint IO excluded",
            "primary_update": trainer.updates if trainer.ema_initialized else None,
            "auxiliary_best_update": best_update,
        }, indent=2) + "\n")
    if best_score is None:
        print("Run saved; EMA not yet eligible for best selection", flush=True)
        return
    best, _, _ = load_real_checkpoint(output / "checkpoint_best.pt", package, args.device)
    prediction, target, diagnostics = validation_artifacts(best, package, run_config, "ema")
    selected_dir = output / "selected"
    write_evaluation(selected_dir, prediction, target, diagnostics)
    if trainer.config.training_protocol_version == 1:
        for path in selected_dir.iterdir():
            path.rename(output / path.name)
        selected_dir.rmdir()
    print("Run: {} ; selected update={}".format(output.resolve(), best_update), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared data.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--variant", choices=("base", "loop", "untied"),
                        help="New run defaults to base; resume must match when specified")
    parser.add_argument("--attention-scope", choices=("full", "diagonal"),
                        help="New run defaults to full; resume inherits saved scope unless specified")
    parser.add_argument("--training-protocol-version", type=int, choices=(1, 2),
                        help="New run defaults to v1; v2 enables matched initialization and delayed EMA")
    parser.add_argument("--ema-start-update", type=int,
                        help="Default: v1=0, v2=1000; small values for interface tests only")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--updates", type=int, default=20, help="Additional successful optimizer updates")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--validation-limit", type=int, help="First N chronological validation days for interface checks")
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    if min(args.updates, args.validate_every, args.scenarios, args.chunk_size) < 1 or not 1 <= args.sampling_steps <= 1000:
        parser.error("Invalid update, validation or sampling budget")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this interpreter/execution environment")
    if args.device == "cpu":
        torch.set_num_threads(1)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output = Path("outputs/real") / stamp
    run(args)


if __name__ == "__main__":
    main()
