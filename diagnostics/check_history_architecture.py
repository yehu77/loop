"""R2 history acceptance: old BASE replay and four real-data resume checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from data import file_hash
from diffusion import sample_scenarios
from history_data import load_history_package
from model import ModelConfig
from train import Trainer, TrainConfig, WeatherScaler, SyntheticBatches, make_synthetic_data
from train_real import load_real_checkpoint, run


def replay_base(device):
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        modules = {}
        for name in ("model", "train"):
            path = Path(temporary) / (name + ".py")
            path.write_bytes(subprocess.check_output(["git", "show", "699980c:" + name + ".py"], cwd=root))
            spec = importlib.util.spec_from_file_location("history_arch_reference_" + name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            with patch.dict(sys.modules, modules):
                spec.loader.exec_module(module)
            modules[name] = module
        records = []
        for group in ("H0", "H24"):
            kwargs = dict(variant="base", history_group=group)
            mc = ModelConfig(**kwargs)
            data = make_synthetic_data(mc, samples=11)
            data["history_power"] = torch.rand(11, 24, 1, generator=torch.Generator().manual_seed(82))
            scaler = WeatherScaler.fit(data["nwp"])
            tc = dict(training_protocol_version=2, ema_start_update=2, microbatch_size=2, accumulation_steps=2)
            old = modules["train"].Trainer(modules["model"].ModelConfig(**kwargs), modules["train"].TrainConfig(**tc), scaler, device)
            new = Trainer(mc, TrainConfig(**tc), scaler, device)
            assert all(torch.equal(v, new.model.state_dict()[k]) for k, v in old.model.state_dict().items())
            a, b = SyntheticBatches(data, 2), SyntheticBatches(data, 2)
            for _ in range(5):
                assert old.train_update(a) == new.train_update(b)
                for name in ("model", "ema"):
                    assert all(torch.equal(v, getattr(new, name).state_dict()[k])
                               for k, v in getattr(old, name).state_dict().items())
                for p, q in zip(old.model.parameters(), new.model.parameters()):
                    assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)
                assert torch.equal(old.generator.get_state(), new.generator.get_state())
            c, cal, h = (data[k][:2].to(device) for k in ("nwp", "calendar", "history_power"))
            noise = torch.randn(2, 3, 24, 1, generator=torch.Generator().manual_seed(12)).to(device)
            samples = [sample_scenarios(t.ema, t.schedule, scaler.transform(c), cal, history_power=h,
                                       scenarios=3, sampling_steps=4, chunk_size=2, initial_noise=noise)[0]
                       for t in (old, new)]
            assert torch.equal(*samples)
            records.append({"group": group, "updates": 5, "initial_raw_ema_gradients_noise_rng_exact": True,
                            "fixed_noise_sampling_exact": True})
    return {"reference": "699980c", "device": device, "torch": str(torch.__version__), "records": records}


def check_real(data, output, device):
    output.mkdir(parents=True, exist_ok=False)
    package = load_history_package(data)
    records, states, inits = [], {}, {}
    for variant, group in (("loop", "H0"), ("untied", "H0"), ("loop", "H24"), ("untied", "H24")):
        key = variant + "_" + group
        args = SimpleNamespace(data=data, output=output / (key + "_first"), resume=None,
            history_group=group, variant=variant, attention_scope="full", training_protocol_version=2,
            ema_start_update=3, supervision_mode="final-only", loss_weights=[0., 1.], device=device,
            amp=False, seed=42, microbatch_size=4, accumulation_steps=8, validation_limit=2,
            scenarios=4, sampling_steps=4, chunk_size=3, validate_every=3, updates=6)
        run(args)
        inits[variant, group] = json.loads((args.output / "initialization.json").read_text())
        first = args.output
        args.resume, args.output, args.updates = first / "checkpoint_last.pt", output / (key + "_resume"), 2
        run(args)
        resumed = torch.load(args.output / "checkpoint_last.pt", map_location="cpu", weights_only=True)
        args.resume, args.output, args.updates = None, output / (key + "_continuous"), 8
        run(args)
        continuous = torch.load(args.output / "checkpoint_last.pt", map_location="cpu", weights_only=True)
        for field in ("model", "ema"):
            assert all(torch.equal(v, continuous[field][k]) for k, v in resumed[field].items())
        assert resumed["updates"] == continuous["updates"] == 8
        assert torch.equal(resumed["noise_rng"], continuous["noise_rng"])
        assert resumed["data"]["cursor"] == continuous["data"]["cursor"]
        for field in ("order", "rng"):
            assert torch.equal(resumed["data"][field], continuous["data"][field])
        values = torch.load(args.output / "scenarios.pt", weights_only=True)["values"]
        assert values.shape == (2, 4, 24, 1) and torch.isfinite(values).all()
        # Negative source-identity check uses a new copy, not an altered original snapshot.
        continuous["metadata"]["run_config"]["history_training_source_sha256"]["model.py"] = "wrong"
        bad = output / (key + "_wrong_source.pt")
        torch.save(continuous, bad)
        try:
            load_real_checkpoint(bad, package, device, expected_variant=variant, expected_history_group=group)
        except ValueError as error:
            assert "source mismatch" in str(error)
        else:
            raise AssertionError("Wrong source checkpoint accepted")
        states[variant, group] = resumed
        records.append({"variant": variant, "history_group": group, "continuous_updates": 8,
                        "segments": [6, 2], "raw_ema_and_rng_exact": True,
                        "wrong_source_rejected": True, "scenario_shape": list(values.shape)})
    for variant in ("loop", "untied"):
        assert inits[variant, "H0"] == inits[variant, "H24"]
    for group in ("H0", "H24"):
        base = json.loads(Path(f"outputs/history_information_v1_s42_{group.lower()}/initialization.json").read_text())
        for variant in ("loop", "untied"):
            assert all(inits[variant, group][k] == v for k, v in base.items())
    ref = states["loop", "H0"]
    for state in states.values():
        assert torch.equal(ref["noise_rng"], state["noise_rng"])
        assert ref["data"]["cursor"] == state["data"]["cursor"]
        for field in ("order", "rng"):
            assert torch.equal(ref["data"][field], state["data"][field])
    report = {"device": device, "torch": str(torch.__version__), "data_sha256": file_hash(data),
              "records": records, "shared_initialization_matches_saved_base": True,
              "matched_data_and_noise": True, "formal_experiment": False}
    (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "real"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "real" and (args.data is None or args.output is None):
        parser.error("Real acceptance requires --data and --output")
    result = replay_base(args.device) if args.mode == "replay" else check_real(args.data, args.output, args.device)
    print(json.dumps(result, indent=2))
