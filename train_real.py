"""BASE-2 real-data runner using the existing Trainer, EMA and DDIM sampler."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

from data import RealBatches, load_package
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
    model = trainer.ema if weights == "ema" else trainer.model
    generator = torch.Generator(device=trainer.device).manual_seed(run_config["validation_seed"])
    nwp = trainer.weather_scaler.transform(values["nwp"].to(trainer.device))
    trajectories, diagnostics = sample_scenarios(
        model, trainer.schedule, nwp, values["calendar"].to(trainer.device),
        scenarios=run_config["scenarios"], sampling_steps=run_config["sampling_steps"],
        chunk_size=run_config["chunk_size"], generator=generator,
    )
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


def load_real_checkpoint(path, package, device):
    trainer, batch_state, metadata = Trainer.load(
        path, device, expected_data_fingerprint=package["manifest"]["data_fingerprint"]
    )
    if metadata.get("data_kind") != "real" or trainer.model.config.variant != "base":
        raise ValueError("This runner resumes real-data BASE-2 checkpoints only")
    batches = RealBatches(package, trainer.config.microbatch_size, trainer.config.seed)
    batches.load_state_dict(batch_state)
    return trainer, batches, metadata


def run(args):
    package = load_package(args.data)
    manifest = package["manifest"]
    if args.resume:
        trainer, batches, checkpoint_metadata = load_real_checkpoint(args.resume, package, args.device)
        run_config = checkpoint_metadata["run_config"]
        best_score = checkpoint_metadata["best_validation_crps"]
        best_update = checkpoint_metadata["best_update"]
        previous_best = args.resume.parent / "checkpoint_best.pt"
        if not previous_best.is_file():
            raise ValueError("Resume requires checkpoint_best.pt beside the supplied checkpoint")
        best_state = torch.load(previous_best, map_location="cpu", weights_only=True)
        if (best_state["metadata"].get("data_fingerprint") != manifest["data_fingerprint"]
                or best_state["metadata"].get("run_config") != run_config
                or best_state["metadata"].get("best_validation_crps") != best_score
                or best_state["metadata"].get("best_update") != best_update
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
        config = TrainConfig(amp=args.amp, seed=args.seed, microbatch_size=args.microbatch_size,
                             accumulation_steps=args.accumulation_steps)
        trainer = Trainer(ModelConfig(variant="base"), config, scaler, args.device)
        batches = RealBatches(package, config.microbatch_size, config.seed)
        best_score, best_update = None, None
    _, panel_metadata = validation_panel(package, run_config["validation_limit"])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({
        "model": asdict(trainer.model.config), "training": asdict(trainer.config),
        "evaluation": run_config, "additional_successful_update_budget": args.updates,
        "data_fingerprint": manifest["data_fingerprint"],
        "validation_sample_ids": [m["sample_id"] for m in panel_metadata],
        "torch_version": str(torch.__version__), "device": str(trainer.device),
        "resume_from": str(args.resume) if args.resume else None,
    }, indent=2) + "\n")
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.resume:
        shutil.copyfile(previous_best, output / "checkpoint_best.pt")
    rand_report = write_evaluation(output / "rand", *rand_artifacts(package, run_config))
    validation_history, training_history = [], []

    def metadata():
        return {"data_kind": "real", "data_fingerprint": manifest["data_fingerprint"],
                "protocol_fingerprint": manifest["protocol_fingerprint"], "run_config": run_config,
                "best_validation_crps": best_score, "best_update": best_update}

    def validate():
        nonlocal best_score, best_update
        entry = {"updates": trainer.updates}
        for weights in ("raw", "ema"):
            prediction, target, diagnostics = validation_artifacts(trainer, package, run_config, weights)
            directory = output / "validation" / ("update_{:06d}".format(trainer.updates)) / weights
            report = write_evaluation(directory, prediction, target, diagnostics)
            entry[weights] = {key: report[key] for key in ("crps", "energy_score", "variogram_score")}
            entry[weights]["boundary_fraction"] = diagnostics["zero_power_fraction"] + diagnostics["full_power_fraction"]
        validation_history.append(entry)
        score = entry["ema"]["crps"]
        if best_score is None or score < best_score:
            best_score, best_update = score, trainer.updates
            trainer.save(output / "checkpoint_best.pt", batches, metadata())
        (output / "validation_history.json").write_text(json.dumps(validation_history, indent=2) + "\n")
        print("validation update={} raw_CRPS={:.6f} EMA_CRPS={:.6f} RAND_CRPS={:.6f}".format(
            trainer.updates, entry["raw"]["crps"], entry["ema"]["crps"], rand_report["crps"]), flush=True)

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
            if entry["did_update"] and (trainer.updates % run_config["validate_every"] == 0
                                         or trainer.updates == target_updates):
                validate()
            trainer.save(output / "checkpoint_last.pt", batches, metadata())
            (output / "training_history.json").write_text(json.dumps(training_history, indent=2) + "\n")
            print("BASE-2 update={} loss={:.6f}".format(trainer.updates, entry["loss"]), flush=True)
    except (FloatingPointError, ValueError, RuntimeError, KeyboardInterrupt):
        trainer.save(output / "failure.pt", batches, {**metadata(), "failure": True})
        raise
    # Top-level score artifacts always refer to the selected EMA checkpoint.
    best, _, _ = load_real_checkpoint(output / "checkpoint_best.pt", package, args.device)
    prediction, target, diagnostics = validation_artifacts(best, package, run_config, "ema")
    selected_dir = output / "selected"
    write_evaluation(selected_dir, prediction, target, diagnostics)
    for path in selected_dir.iterdir():
        path.rename(output / path.name)
    selected_dir.rmdir()
    print("Run: {} ; selected update={}".format(output.resolve(), best_update), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared data.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
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
