from copy import deepcopy
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data import file_hash, prepare_dataset
from history_data import (history_times, load_history_package, prepare_history,
                          read_history_targets, trace_history_sample,
                          validate_history_package, visible_history)
from prepare_history import write_history_package
from test_data import fixture_protocol, write_sources


class HistoryDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.protocol = fixture_protocol()

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, mutate_history=None, mutate_future=None):
        write_sources(self.root, self.protocol, mutate_future)
        path = self.root / "Train_W_Zone1.csv"
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, existing = reader.fieldnames, list(reader)
        issue = datetime.fromisoformat(self.protocol["splits"]["train"][0])
        earlier = []
        for i, time in enumerate(history_times(issue) + [issue]):
            row = {"ZONEID": "1", "TIMESTAMP": time.strftime("%Y%m%d %H:%M"),
                   "TARGETVAR": str(min(i / 23, 1)), "U10": "0", "V10": "0", "U100": "0", "V100": "0"}
            if mutate_history:
                row = mutate_history(row, i)
            if row is not None:
                earlier.append(row)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(earlier + existing)
        parent = prepare_dataset(self.root, self.protocol)
        protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/history_information_v1.json").read_text())
        protocol.update(zone_id=1, parent_data_fingerprint=parent["manifest"]["data_fingerprint"])
        return parent, protocol

    def test_delay_leap_year_year_boundary_and_no_future_power(self):
        for issue in [datetime(2012, 3, 1), datetime(2013, 1, 1)]:
            times = history_times(issue)
            self.assertEqual(times[0], issue - timedelta(hours=24))
            self.assertEqual(times[-1], issue - timedelta(hours=1))
            self.assertEqual(len(set(times)), 24)
            self.assertTrue(all(t + timedelta(hours=1) <= issue for t in times))
        parent, protocol = self.source()
        package = prepare_history(parent, self.root, protocol)
        for split, rows in package["metadata"].items():
            for row in rows:
                self.assertFalse(set(row["valid_times"]) & set(row["history_valid_times"]))
        train_last = parent["metadata"]["train"][-1]
        val_first = package["metadata"]["validation"][0]
        self.assertTrue(set(train_last["valid_times"]) & set(val_first["history_valid_times"]))
        # Rolling observations from an earlier day are legitimate; no target overlap.
        self.assertFalse(set(train_last["valid_times"]) & set(val_first["valid_times"]))

    def test_missing_na_and_out_of_range_history_drop_common_window(self):
        for value in [None, "NA", "nan", "2"]:
            with self.subTest(value=value):
                def mutate(row, i):
                    if i == 3:
                        if value is None:
                            return None
                        row["TARGETVAR"] = value
                    return row
                parent, protocol = self.source(mutate)
                package = prepare_history(parent, self.root, protocol)
                counts = package["manifest"]["counts"]["train"]
                self.assertEqual(counts["parent_future_kept"], 2)
                self.assertEqual(counts["common_kept"], 1)
                self.assertEqual(counts["additional_history_dropped"], 1)
                self.assertEqual(len(package["dropped"]["train"][0]["history_reasons"]), 1)
                # Latest power is valid, but H1/H0 must still use the full-history intersection.
                self.assertEqual(package["manifest"]["parent_indices"]["train"], [1])

    def test_true_zero_one_and_missing_history_weather(self):
        def mutate(row, i):
            row["U10"] = "NA"
            return row
        parent, protocol = self.source(mutate)
        package = prepare_history(parent, self.root, protocol)
        self.assertEqual(package["manifest"]["counts"]["train"]["common_kept"], 2)
        h = package["splits"]["train"]["history_power"][0]
        self.assertEqual(h[0].item(), 0)
        self.assertEqual(h[-1].item(), 1)

    def test_mask_blocks_invisible_values_including_nonfinite(self):
        h = torch.rand(2, 24, 1)
        a, mask = visible_history(h, "H1")
        changed = h.clone()
        changed[:, :-1] = float("nan")
        b, other = visible_history(changed, "H1")
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(mask, other))
        self.assertEqual(mask.sum().item(), 2)
        zero, mask = visible_history(changed, "H0")
        self.assertEqual(zero.count_nonzero().item(), 0)
        self.assertFalse(mask.any())
        full, mask = visible_history(h, "H24")
        torch.testing.assert_close(full, 2 * h - 1)
        self.assertTrue(mask.all())
        # Visible zero power is -1 and distinguishable from an invisible slot.
        zero, mask = visible_history(torch.zeros_like(h), "H1")
        self.assertEqual(zero[0, -1, 0].item(), -1)

    def test_validation_extremes_do_not_change_training_scaler(self):
        parent, protocol = self.source()
        original = prepare_history(parent, self.root, protocol)
        def mutate(row, split, issue, lead):
            if split != "train":
                row.update(TARGETVAR="1", U10="1000000", V10="-1000000")
            return row
        parent, protocol = self.source(mutate_future=mutate)
        changed = prepare_history(parent, self.root, protocol)
        self.assertEqual(original["manifest"]["scaler"], changed["manifest"]["scaler"])
        torch.testing.assert_close(original["splits"]["train"]["history_power"],
                                   changed["splits"]["train"]["history_power"])

    def test_refit_scaler_after_intersection(self):
        def mutate(row, split, issue, lead):
            if split == "train" and issue.day == 27:
                row["U10"] = "1000"
            return row
        parent, protocol = self.source(lambda row, i: None if i == 0 else row, mutate)
        package = prepare_history(parent, self.root, protocol)
        from train import WeatherScaler
        expected = WeatherScaler.fit(parent["splits"]["train"]["nwp"][[1]])
        self.assertEqual(package["manifest"]["scaler"],
                         {k: v.tolist() for k, v in expected.state_dict().items()})
        self.assertNotEqual(package["manifest"]["scaler"], parent["manifest"]["scaler"])

    def test_current_future_cannot_enter_current_history_but_can_enter_later_history(self):
        parent, protocol = self.source()
        original = prepare_history(parent, self.root, protocol)
        def mutate(row, split, issue, lead):
            if split == "validation" and issue.day == 29 and lead == 5:
                row["TARGETVAR"] = "0.123"
            return row
        parent, protocol = self.source(mutate_future=mutate)
        changed = prepare_history(parent, self.root, protocol)
        a, b = original["splits"]["validation"], changed["splits"]["validation"]
        torch.testing.assert_close(a["history_power"][0], b["history_power"][0], rtol=0, atol=0)
        self.assertFalse(torch.equal(a["y"][0], b["y"][0]))
        self.assertFalse(torch.equal(a["history_power"][1], b["history_power"][1]))

    def test_group_plans_share_recipe_and_only_visibility_differs(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "configs/history_information_v1"
        common = json.loads((directory / "common.json").read_text())
        for group, slots in (("H0", []), ("H1", [23]), ("H24", list(range(24)))):
            plan = json.loads((directory / (group + ".json")).read_text())
            self.assertEqual(file_hash(root / plan["common_config"]), plan["common_config_sha256"])
            self.assertEqual(plan["visible_history_slots"], slots)
            self.assertEqual(plan["history_group"], group)
        self.assertEqual(common["training"]["successful_updates"], 3000)
        self.assertEqual(common["training"]["supervision_mode"], "final-only")
        self.assertEqual(common["evaluation"]["windows"], len(common["sample_ids"]["validation"]))

    def test_trace_roundtrip_tamper_and_no_overwrite(self):
        parent, protocol = self.source()
        package = prepare_history(parent, self.root, protocol)
        copy = prepare_history(parent, self.root, protocol)
        self.assertEqual(package["manifest"], copy["manifest"])
        output = self.root / "prepared"
        write_history_package(package, parent, self.root, output)
        self.assertEqual(load_history_package(output / "data.pt")["manifest"], package["manifest"])
        with self.assertRaises(FileExistsError):
            write_history_package(package, parent, self.root, output)
        trace = trace_history_sample(package, parent, self.root, package["metadata"]["train"][0]["sample_id"])
        self.assertEqual(len(trace["history_rows"]), 24)
        self.assertEqual(len(trace["future_rows"]), 24)
        self.assertEqual(trace["history_rows"][0]["power"], 0)
        self.assertEqual(trace["history_rows"][-1]["power"], 1)
        tampered = deepcopy(package)
        tampered["splits"]["train"]["history_power"][0, 0, 0] += .1
        with self.assertRaisesRegex(ValueError, "tensor"):
            validate_history_package(tampered)
        path = self.root / "Train_W_Zone1.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            prepare_history(parent, self.root, protocol)

    def test_duplicate_history_key_rejected_and_parent_identity_bound(self):
        parent, protocol = self.source()
        bad = deepcopy(protocol)
        bad["parent_data_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "identity"):
            prepare_history(parent, self.root, bad)
        path = self.root / "Train_W_Zone1.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines + [lines[1]]) + "\n")
        parent["manifest"]["source_files"][path.name]["sha256"] = file_hash(path)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_history_targets(self.root, parent)


if __name__ == "__main__":
    unittest.main()
