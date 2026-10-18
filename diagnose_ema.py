"""Read-only, fixed 1000-to-3000 EMA-history diagnostic; never updates best."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data import file_hash, load_package
from diffusion import LinearNoiseSchedule
from evaluate import write_evaluation
from model import ConditionalLoopDiT, ModelConfig
from train import WeatherScaler
from train_real import validation_artifacts

TRAINING_SOURCES = ("model.py", "diffusion.py", "train.py", "data.py",
                    "train_real.py", "evaluate.py", "metrics.py")


def reconstruct_ema_state(model, anchor_raw, anchor_ema, endpoint_ema,
                          decay, successful_updates):
    """Apply the exact-arithmetic identity in FP64, then cast to checkpoint dtype.

    FP32 checkpoints contain rounded recurrences, so this is not a bitwise replay
    of a hypothetical FP32 restarted EMA. Buffers are copied from the endpoint,
    matching Trainer's copy-on-successful-update rule.
    """
    if not 0 < decay < 1 or type(successful_updates) is not int or successful_updates < 0:
        raise ValueError("Invalid EMA decay or successful-update interval")
    expected = model.state_dict()
    for state in (anchor_raw, anchor_ema, endpoint_ema):
        if state.keys() != expected.keys():
            raise ValueError("Model state keys differ")
        for name, reference in expected.items():
            value = state[name]
            if (value.shape != reference.shape or value.dtype != reference.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError("Invalid parameter or buffer: " + name)
    result = {name: value.detach().clone() for name, value in endpoint_ema.items()}
    coefficient = decay ** successful_updates
    for name, _ in model.named_parameters():
        value = endpoint_ema[name].double() + coefficient * (
            anchor_raw[name].double() - anchor_ema[name].double())
        result[name] = value.to(dtype=endpoint_ema[name].dtype)
        if not torch.isfinite(result[name]).all():
            raise ValueError("Non-finite reconstructed parameter: " + name)
    return result


def validate_checkpoint_pair(start, end):
    """Validate compatible endpoints; lineage needs the separate run evidence."""
    for state, update in ((start, 1000), (end, 3000)):
        meta = state["metadata"]
        if (state["format_version"] != 1 or meta.get("failure", False)
                or meta.get("data_kind") != "real" or state["updates"] != update
                or state["attempts"] != state["updates"] + state["skipped"]
                or state["skipped"] < 0 or state["model_config"]["variant"] != "base"
                or state["train_config"]["ema_decay"] != 0.999):
            raise ValueError("Requires normal BASE-2 checkpoints at 1000/3000 and decay 0.999")
    for field in ("model_config", "train_config", "rng_device_type"):
        if start[field] != end[field]:
            raise ValueError("Checkpoint configuration mismatch: " + field)
    for field in ("data_fingerprint", "protocol_fingerprint", "run_config"):
        if not start["metadata"].get(field) or start["metadata"][field] != end["metadata"].get(field):
            raise ValueError("Checkpoint metadata mismatch: " + field)
    for field in ("mean", "std"):
        if not torch.equal(start["weather_scaler"][field], end["weather_scaler"][field]):
            raise ValueError("Training scaler mismatch")
    config = end["metadata"]["run_config"]
    if (config["validation_limit"] is not None or config["scenarios"] != 100
            or config["sampling_steps"] != 50 or config["validate_every"] != 250):
        raise ValueError("Diagnostic requires the frozen full-validation sampling protocol")


def validate_lineage(record, start_path, end_path, data_path, start, end):
    """Check launch-time hashes, resume config/history, and replayed anchor scores.

    This audits a locally recorded continuation. Equal configs alone do not
    establish ancestry; the record must be captured before launching the run and
    sealed after successful completion. It is not a cryptographic proof of every
    optimizer transition and must not be fabricated for unrelated checkpoints.
    """
    if (record.get("schema_version") != 1 or record.get("run_exit_code") != 0
            or record.get("additional_successful_updates") != 2000):
        raise ValueError("Missing successful, sealed continuation record")
    for label, path in (("start", start_path), ("end", end_path)):
        if (Path(record[label + "_checkpoint"]).resolve() != path.resolve()
                or record[label + "_sha256"] != file_hash(path)):
            raise ValueError("Continuation checkpoint hash/path mismatch: " + label)
    if record["data_package_sha256"] != file_hash(data_path):
        raise ValueError("Prepared package differs from the launch record")
    if (record["start_best_sha256"] != file_hash(start_path.parent / "checkpoint_best.pt")
            or record["start_config_sha256"] != file_hash(start_path.parent / "config.json")):
        raise ValueError("Original run changed after launch")
    if set(record["training_source_sha256"]) != set(TRAINING_SOURCES):
        raise ValueError("Incomplete launch-time training source hashes")
    for name, digest in record["training_source_sha256"].items():
        if file_hash(Path(__file__).resolve().parent / name) != digest:
            raise ValueError("Training source changed: " + name)
    config = json.loads((end_path.parent / "config.json").read_text())
    if (not config.get("resume_from")
            or Path(config["resume_from"]).resolve() != start_path.resolve()
            or config["additional_successful_update_budget"] != 2000
            or config["evaluation"] != start["metadata"]["run_config"]
            or config["training"] != start["train_config"]
            or config["model"] != start["model_config"]
            or config["data_fingerprint"] != start["metadata"]["data_fingerprint"]):
        raise ValueError("Resume configuration does not match the recorded anchor")
    history = json.loads((end_path.parent / "training_history.json").read_text())
    successful = [row["updates"] for row in history if row["did_update"]]
    if (successful != list(range(1001, 3001))
            or len(history) != end["attempts"] - start["attempts"]
            or history[-1]["skipped"] != end["skipped"]):
        raise ValueError("Continuation successful-update history is incomplete")
    for weights in ("raw", "ema"):
        relative = Path("validation/update_001000") / weights / "metrics.json"
        if file_hash(start_path.parent / relative) != file_hash(end_path.parent / relative):
            raise ValueError("Resumed anchor evaluation differs: " + weights)


def run(args):
    if args.output.exists():
        raise ValueError("Diagnostic output already exists; refusing to overwrite")
    package = load_package(args.data)
    start = torch.load(args.start, map_location="cpu", weights_only=True)
    end = torch.load(args.end, map_location="cpu", weights_only=True)
    validate_checkpoint_pair(start, end)
    record = json.loads(args.lineage.read_text())
    validate_lineage(record, args.start, args.end, args.data, start, end)
    if (end["metadata"]["data_fingerprint"] != package["manifest"]["data_fingerprint"]
            or end["metadata"]["protocol_fingerprint"] != package["manifest"]["protocol_fingerprint"]):
        raise ValueError("Checkpoint and prepared data identities differ")
    if len(package["metadata"]["validation"]) != 71:
        raise ValueError("Expected the frozen 71-day validation panel")
    if torch.device(args.device).type != end["rng_device_type"]:
        raise ValueError("Use the original device type to preserve validation noise generation")

    # No Trainer or optimizer is constructed; only copies used for inference.
    model = ConditionalLoopDiT(ModelConfig(**end["model_config"]))
    restarted = reconstruct_ema_state(model, start["model"], start["ema"],
                                      end["ema"], 0.999, end["updates"] - start["updates"])
    protected = {str(path): file_hash(path) for path in (
        args.start, args.end, args.start.parent / "checkpoint_best.pt",
        args.end.parent / "checkpoint_best.pt")}
    args.output.mkdir(parents=True, exist_ok=False)
    context = SimpleNamespace(
        device=torch.device(args.device), updates=end["updates"],
        schedule=LinearNoiseSchedule(end["train_config"]["diffusion_steps"]).to(args.device),
        weather_scaler=WeatherScaler(**end["weather_scaler"]), model=None, ema=None)
    reports = {}
    for label, state in (("raw", end["model"]), ("original_ema", end["ema"]),
                         ("restarted_ema", restarted)):
        model.load_state_dict(state)
        model.to(args.device).eval()
        context.model = context.ema = model
        prediction, target, diagnostics = validation_artifacts(
            context, package, end["metadata"]["run_config"], "raw" if label == "raw" else "ema")
        diagnostics.update(weights=label, diagnostic_only=True, diagnostic_anchor_update=1000)
        reports[label] = write_evaluation(args.output / label, prediction, target, diagnostics)
        print("{} CRPS={:.8f} ES={:.8f} VS={:.8f}".format(
            label, *(reports[label][k] for k in ("crps", "energy_score", "variogram_score"))), flush=True)
    for path, digest in protected.items():
        if file_hash(path) != digest:
            raise RuntimeError("A protected checkpoint changed during diagnostic evaluation")
    summary = {
        "diagnostic_only": True, "selected_checkpoint_changed": False,
        "diagnostic_script_sha256": file_hash(Path(__file__)),
        "start_update": 1000, "end_update": 3000, "decay": 0.999,
        "correction_coefficient": 0.999 ** 2000,
        "formula": "ema_3000 + beta**2000 * (raw_1000 - ema_1000)",
        "arithmetic": "FP64 reconstruction, cast to FP32; not bitwise hypothetical FP32 replay",
        "buffers": "copied unchanged from endpoint EMA, matching Trainer's buffer rule",
        "data_fingerprint": package["manifest"]["data_fingerprint"],
        "evaluation_config": end["metadata"]["run_config"],
        "protected_checkpoint_sha256": protected, "lineage": record,
        "metrics": {label: {k: report[k] for k in ("crps", "energy_score", "variogram_score", "intervals")}
                    for label, report in reports.items()},
    }
    (args.output / "diagnostic.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--end", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use the original validated GPU environment")
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
