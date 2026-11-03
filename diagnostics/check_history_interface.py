"""Small real-package CUDA/CPU acceptance: H0/H1/H24, continuous versus resumed."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data import file_hash
from train_real import run


def check(data, output, device):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    records, initialization, initial_state = [], None, None
    for group in ("H0", "H1", "H24"):
        args = SimpleNamespace(data=data, output=output / (group + "_first"), resume=None,
            history_group=group, variant="base", attention_scope="full", training_protocol_version=2,
            ema_start_update=3, supervision_mode="final-only", loss_weights=[0., 1.], device=device,
            amp=False, seed=42, microbatch_size=4, accumulation_steps=8, validation_limit=2,
            scenarios=4, sampling_steps=4, chunk_size=3, validate_every=3, updates=6)
        run(args)
        init = json.loads((args.output / "initialization.json").read_text())
        if initialization is None:
            initialization = init
        assert init == initialization, "Group initialization differs"
        first = args.output
        args.resume, args.output, args.updates = first / "checkpoint_last.pt", output / (group + "_resume"), 2
        run(args)
        resumed = torch.load(args.output / "checkpoint_last.pt", map_location="cpu", weights_only=True)
        args.resume, args.output, args.updates = None, output / (group + "_continuous"), 8
        run(args)
        continuous = torch.load(args.output / "checkpoint_last.pt", map_location="cpu", weights_only=True)
        for key in ("model", "ema"):
            assert all(torch.equal(v, continuous[key][k]) for k, v in resumed[key].items())
        assert resumed["updates"] == continuous["updates"] == 8
        assert torch.equal(resumed["noise_rng"], continuous["noise_rng"])
        for key in ("order", "rng"):
            assert torch.equal(resumed["data"][key], continuous["data"][key])
        assert resumed["data"]["cursor"] == continuous["data"]["cursor"]
        if initial_state is None:
            initial_state = continuous
        else:
            assert torch.equal(initial_state["noise_rng"], continuous["noise_rng"])
            assert torch.equal(initial_state["data"]["order"], continuous["data"]["order"])
            assert initial_state["data"]["cursor"] == continuous["data"]["cursor"]
        values = torch.load(args.output / "scenarios.pt", weights_only=True)["values"]
        assert values.shape == (2, 4, 24, 1) and torch.isfinite(values).all()
        records.append({"group": group, "continuous_updates": 8, "resumed_segments": [6, 2],
                        "raw_ema_and_rng_exact": True, "parameters": sum(v.numel() for k, v in
                         continuous["model"].items() if "position_encoding" not in k),
                        "scenario_shape": list(values.shape)})
    report = {"device": device, "torch": str(torch.__version__), "data_sha256": file_hash(data),
              "records": records, "matched_initializations": True,
              "matched_data_and_noise_rng": True, "formal_experiment": False,
              "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
    (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.data, args.output, args.device), indent=2))
