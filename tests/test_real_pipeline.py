from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import torch

from data import RealBatches, prepare_dataset, sample_id
from evaluate import align_artifacts, evaluate_artifacts, identified_artifact
from model import ModelConfig
from test_data import fixture_protocol, write_sources
from train import TrainConfig, Trainer, WeatherScaler
from train_real import load_real_checkpoint, rand_artifacts, validation_artifacts


class RealPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = fixture_protocol()
        write_sources(self.root, self.protocol)
        self.package = prepare_dataset(self.root, self.protocol)
        self.run_config = {"validation_limit": None, "validation_seed": 2026,
                           "rand_seed": 2027, "scenarios": 3, "sampling_steps": 2, "chunk_size": 3}

    def tearDown(self):
        self.temporary.cleanup()

    def trainer(self):
        data = self.package["splits"]["train"]
        return Trainer(ModelConfig(variant="base", hidden_dim=16, ffn_dim=32),
                       TrainConfig(microbatch_size=1, accumulation_steps=2), WeatherScaler.fit(data["nwp"]))

    def artifacts(self):
        values = self.package["splits"]["validation"]["y"].clone()
        # Different days must have different targets, otherwise a row-order bug
        # can leave aggregate scores unchanged and escape this regression test.
        values[1] = values[1] * 0.5 + 0.2
        manifest, meta = self.package["manifest"], self.package["metadata"]["validation"]
        return (identified_artifact(values[:, None].repeat(1, 3, 1, 1), manifest, meta),
                identified_artifact(values, manifest, meta))

    def test_reordered_predictions_are_joined_by_identity(self):
        predicted, target = self.artifacts()
        predicted["values"][1] += 0.1
        expected, expected_scores = evaluate_artifacts(predicted, target)
        predicted["values"] = predicted["values"].flip(0)
        predicted["samples"].reverse()
        actual, scores = evaluate_artifacts(predicted, target)
        self.assertEqual(actual, expected)
        torch.testing.assert_close(scores["crps"], expected_scores["crps"], atol=0, rtol=0)
        self.assertEqual(scores["crps"].shape, (2, 24))
        self.assertEqual(scores["energy_score"].shape, (2,))
        self.assertEqual(scores["intervals"][0]["covered"].shape, (2, 24))

    def test_duplicate_missing_extra_time_zone_unit_and_version_rejected(self):
        for case in ("duplicate", "missing", "extra", "time", "zone", "unit", "version", "split", "shape"):
            predicted, target = self.artifacts()
            if case == "duplicate":
                predicted["samples"][1] = deepcopy(predicted["samples"][0])
            elif case == "missing":
                predicted["samples"] = predicted["samples"][:1]
                predicted["values"] = predicted["values"][:1]
            elif case == "extra":
                issue = datetime(2012, 3, 2)
                record = predicted["samples"][1]
                record.update(sample_id=sample_id(1, issue), issue_time=issue.isoformat(),
                              valid_times=[(issue + timedelta(hours=h)).isoformat() for h in range(1, 25)])
            elif case == "time":
                predicted["samples"][0]["valid_times"][0] = "2012-02-29T02:00:00"
            elif case == "zone":
                predicted["samples"][0]["zone_id"] = 2
            elif case == "unit":
                predicted["power_unit"] = "MW"
            elif case == "version":
                predicted["data_fingerprint"] = "different_data_version"
            elif case == "shape":
                predicted["values"] = predicted["values"][:, :, :12]
                target["values"] = target["values"][:, :12]
            else:
                for row in predicted["samples"]:
                    row["split"] = "test"
            with self.subTest(case=case), self.assertRaises(ValueError):
                align_artifacts(predicted, target)

    def test_data_identity_is_required_before_trainer_construction(self):
        trainer = self.trainer()
        batches = RealBatches(self.package, 1)
        trainer.train_update(batches)
        path = self.root / "checkpoint.pt"
        identity = self.package["manifest"]["data_fingerprint"]
        trainer.save(path, batches, {"data_kind": "real", "data_fingerprint": identity})
        rng_before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            Trainer.load(path)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        def mutate(row, split, issue, lead):
            if split == "train":
                row["U10"] = str(float(row["U10"]) + 0.1)
            return row
        write_sources(self.root, self.protocol, mutate)
        other = prepare_dataset(self.root, self.protocol)
        self.assertEqual(other["manifest"]["counts"], self.package["manifest"]["counts"])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_real_checkpoint(path, other, "cpu")
        restored, _, _ = load_real_checkpoint(path, self.package, "cpu")
        self.assertEqual(restored.updates, 1)

    def test_validation_does_not_advance_training_or_batch_rng(self):
        trainer = self.trainer()
        batches = RealBatches(self.package, 1)
        trainer.train_update(batches)
        path = self.root / "checkpoint.pt"
        trainer.save(path, batches, {"data_kind": "real", "data_fingerprint": self.package["manifest"]["data_fingerprint"]})
        global_rng = torch.get_rng_state().clone()
        noise_rng = trainer.generator.get_state().clone()
        data_rng = batches.generator.get_state().clone()
        cursor = batches.cursor
        first, _, _ = validation_artifacts(trainer, self.package, self.run_config, "raw")
        second, _, _ = validation_artifacts(trainer, self.package, self.run_config, "raw")
        torch.testing.assert_close(first["values"], second["values"], atol=0, rtol=0)
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(noise_rng, trainer.generator.get_state()))
        self.assertTrue(torch.equal(data_rng, batches.generator.get_state()))
        self.assertEqual(cursor, batches.cursor)
        actual_step = trainer.train_update(batches)
        restored, restored_batches, _ = load_real_checkpoint(path, self.package, "cpu")
        expected_step = restored.train_update(restored_batches)
        self.assertEqual(actual_step, expected_step)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, restored.model.state_dict()[name], atol=0, rtol=0)

    def test_rand_only_draws_complete_training_trajectories(self):
        def distinct_heldout_targets(row, split, issue, lead):
            if split != "train":
                row["TARGETVAR"] = "0.12345"
            return row
        write_sources(self.root, self.protocol, distinct_heldout_targets)
        self.package = prepare_dataset(self.root, self.protocol)
        predicted, _, diagnostics = rand_artifacts(self.package, self.run_config)
        indices = torch.tensor(diagnostics["training_indices"])
        expected = self.package["splits"]["train"]["y"][indices]
        torch.testing.assert_close(predicted["values"], expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
