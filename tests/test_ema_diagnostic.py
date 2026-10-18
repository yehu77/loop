from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from data import file_hash
from diagnose_ema import (TRAINING_SOURCES, reconstruct_ema_state,
                          validate_checkpoint_pair, validate_lineage)


class EMADiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def sequence(self, dtype):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            model = nn.Linear(4, 3).to(dtype)
        model.register_buffer("counter", torch.tensor(3000))
        model.register_buffer("position", torch.ones(3, dtype=dtype))
        generator = torch.Generator().manual_seed(17)
        initial = deepcopy(model.state_dict())
        original = deepcopy(initial)
        anchor = restart = None
        for update in range(1, 3001):
            raw = {name: torch.randn(p.shape, generator=generator, dtype=dtype)
                   for name, p in model.named_parameters()}
            for name in raw:
                original[name].lerp_(raw[name], 0.001)
                if restart is not None:
                    restart[name].lerp_(raw[name], 0.001)
            if update == 1000:
                anchor = (deepcopy(raw), deepcopy(original))
                restart = deepcopy(raw)
        anchor_raw = deepcopy(initial)
        anchor_raw.update(anchor[0])
        # Distinct buffers would expose accidental application of the formula.
        anchor_raw["counter"] = torch.tensor(-9)
        anchor_raw["position"] = torch.full((3,), -9., dtype=dtype)
        before = deepcopy((anchor_raw, anchor[1], original))
        result = reconstruct_ema_state(model, anchor_raw, anchor[1], original, 0.999, 2000)
        errors = [(result[name] - restart[name]).abs().max().item()
                  for name, _ in model.named_parameters()]
        for name, _ in model.named_buffers():
            torch.testing.assert_close(result[name], original[name], atol=0, rtol=0)
        for actual, expected in zip((anchor_raw, anchor[1], original), before):
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        return max(errors)

    def test_explicit_fp64_recurrence_matches_reconstruction(self):
        self.assertLess(self.sequence(torch.float64), 1e-12)

    def test_fp32_rounding_is_small_but_not_claimed_bitwise_exact(self):
        self.assertLess(self.sequence(torch.float32), 1e-5)

    def test_reconstruction_rejects_invalid_tensor_or_budget(self):
        model = nn.Linear(2, 2)
        state = model.state_dict()
        for case in ("nan", "keys", "shape", "decay", "steps"):
            bad = deepcopy(state)
            if case == "nan":
                bad["weight"][0, 0] = float("nan")
            if case == "keys":
                del bad["bias"]
            if case == "shape":
                bad["weight"] = torch.zeros(3, 2)
            with self.subTest(case=case), self.assertRaises(ValueError):
                reconstruct_ema_state(model, bad, state, state,
                                      1.0 if case == "decay" else 0.999,
                                      -1 if case == "steps" else 2000)

    def checkpoints(self):
        start = {"format_version": 1, "updates": 1000, "attempts": 1000, "skipped": 0,
                 "model_config": {"variant": "base"},
                 "train_config": {"ema_decay": 0.999}, "rng_device_type": "cuda",
                 "weather_scaler": {"mean": torch.zeros(4), "std": torch.ones(4)},
                 "metadata": {"data_kind": "real", "data_fingerprint": "data",
                              "protocol_fingerprint": "protocol",
                              "run_config": {"validation_limit": None, "scenarios": 100,
                                             "sampling_steps": 50, "validate_every": 250}}}
        end = deepcopy(start)
        end.update(updates=3000, attempts=3000)
        return start, end

    def test_checkpoint_pair_rejects_drift_failure_and_wrong_update_count(self):
        start, end = self.checkpoints()
        validate_checkpoint_pair(start, end)
        skipped_start, skipped_end = deepcopy(start), deepcopy(end)
        skipped_start.update(attempts=1010, skipped=10)
        skipped_end.update(attempts=3015, skipped=15)
        validate_checkpoint_pair(skipped_start, skipped_end)  # Interval is still 2000, not 2005.
        for case in ("failure", "decay", "data", "protocol", "model", "budget", "panel", "scaler", "attempts"):
            bad = deepcopy(end)
            if case == "failure":
                bad["metadata"]["failure"] = True
            elif case == "decay":
                bad["train_config"]["ema_decay"] = 0.99
            elif case == "data":
                bad["metadata"]["data_fingerprint"] = "other"
            elif case == "protocol":
                bad["metadata"]["protocol_fingerprint"] = "other"
            elif case == "model":
                bad["model_config"]["variant"] = "loop"
            elif case == "budget":
                bad["updates"] = 2999
            elif case == "panel":
                bad["metadata"]["run_config"]["validation_limit"] = 4
            elif case == "scaler":
                bad["weather_scaler"]["mean"][0] = 1
            elif case == "attempts":
                bad["attempts"] = 2999
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_checkpoint_pair(start, bad)

    def test_lineage_requires_actual_resume_evidence_not_just_equal_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, terminal = root / "start/checkpoint_last.pt", root / "end/checkpoint_last.pt"
            for path in (source, terminal):
                path.parent.mkdir()
                path.write_bytes(b"checkpoint")
                (path.parent / "checkpoint_best.pt").write_bytes(b"best")
                for weights in ("raw", "ema"):
                    report = path.parent / "validation/update_001000" / weights / "metrics.json"
                    report.parent.mkdir(parents=True)
                    report.write_text('{"crps": 0.1}')
            package = root / "data.pt"
            package.write_bytes(b"data")
            (source.parent / "config.json").write_text('{}')
            start, end = self.checkpoints()
            config = {"resume_from": str(source), "additional_successful_update_budget": 2000,
                      "evaluation": start["metadata"]["run_config"],
                      "training": start["train_config"], "model": start["model_config"],
                      "data_fingerprint": start["metadata"]["data_fingerprint"]}
            (terminal.parent / "config.json").write_text(json.dumps(config))
            history = [{"updates": u, "did_update": True, "skipped": 0}
                       for u in range(1001, 3001)]
            history_path = terminal.parent / "training_history.json"
            history_path.write_text(json.dumps(history))
            record = {"schema_version": 1, "run_exit_code": 0,
                      "additional_successful_updates": 2000,
                      "start_checkpoint": str(source), "end_checkpoint": str(terminal),
                      "start_sha256": file_hash(source), "end_sha256": file_hash(terminal),
                      "start_best_sha256": file_hash(source.parent / "checkpoint_best.pt"),
                      "start_config_sha256": file_hash(source.parent / "config.json"),
                      "data_package_sha256": file_hash(package),
                      "training_source_sha256": {
                          name: file_hash(Path(__file__).resolve().parents[1] / name)
                          for name in TRAINING_SOURCES}}
            validate_lineage(record, source, terminal, package, start, end)
            for changes in ({"run_exit_code": None}, {"training_source_sha256": {}}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_lineage({**record, **changes}, source, terminal, package, start, end)
            for field in ("start_sha256", "end_sha256", "data_package_sha256"):
                bad = deepcopy(record)
                bad[field] = "wrong"
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_lineage(bad, source, terminal, package, start, end)
            bad = deepcopy(config)
            bad["resume_from"] = str(root / "unrelated/checkpoint_last.pt")
            (terminal.parent / "config.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                validate_lineage(record, source, terminal, package, start, end)
            (terminal.parent / "config.json").write_text(json.dumps(config))
            history_path.write_text(json.dumps(history[1:]))
            with self.assertRaises(ValueError):
                validate_lineage(record, source, terminal, package, start, end)
            history_path.write_text(json.dumps(history))
            (terminal.parent / "validation/update_001000/raw/metrics.json").write_text('{}')
            with self.assertRaises(ValueError):
                validate_lineage(record, source, terminal, package, start, end)


if __name__ == "__main__":
    unittest.main()
