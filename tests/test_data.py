from copy import deepcopy
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data import (calendar_features, issue_days, load_package,
                  prepare_dataset, trace_sample, validate_package,
                  validate_protocol)
from prepare_data import write_package


def fixture_protocol(zones=(1,)):
    path = Path(__file__).resolve().parents[1] / "configs/data_v1.json"
    protocol = json.loads(path.read_text())
    protocol["zone_selection"]["candidate_zones"] = list(zones)
    protocol["splits"] = {"train": ["2012-02-27", "2012-02-28"],
                          "validation": ["2012-02-29", "2012-03-01"],
                          "test": ["2012-03-02", "2012-03-03"]}
    return protocol


def write_sources(root, protocol, mutate=None, reverse=False):
    """Independent CSV fixtures, including endpoint power values and a leap day."""
    test_targets = []
    for zone in protocol["zone_selection"]["candidate_zones"]:
        history, forecasts = [], []
        for split, bounds in protocol["splits"].items():
            for issue in issue_days(bounds):
                for lead in range(1, 25):
                    time = issue + timedelta(hours=lead)
                    row = {"ZONEID": str(zone), "TIMESTAMP": time.strftime("%Y%m%d %H:%M"),
                           "TARGETVAR": str(0 if lead == 1 else 1 if lead == 24 else 0.5),
                           "U10": str(zone + lead / 24), "V10": str(lead / 12),
                           "U100": str(zone + lead / 6), "V100": str(lead / 3)}
                    if mutate:
                        row = mutate(row, split, issue, lead)
                    if row is None:
                        continue
                    if split == "test":
                        forecasts.append({key: value for key, value in row.items() if key != "TARGETVAR"})
                        test_targets.append({key: row[key] for key in ("ZONEID", "TIMESTAMP", "TARGETVAR")})
                    else:
                        history.append(row)
        for name, rows in (("Train_W_Zone{}.csv".format(zone), history),
                           ("TestPred_W_Zone{}.csv".format(zone), forecasts)):
            write_csv(root / name, list(reversed(rows)) if reverse else rows)
    write_csv(root / "TestTar_W.csv", list(reversed(test_targets)) if reverse else test_targets)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = fixture_protocol()

    def tearDown(self):
        self.temporary.cleanup()

    def test_midnight_leap_day_and_original_row_trace(self):
        write_sources(self.root, self.protocol)
        package = prepare_dataset(self.root, self.protocol)
        sample = package["metadata"]["validation"][0]
        self.assertEqual(sample["issue_time"], "2012-02-29T00:00:00")
        self.assertEqual(sample["valid_times"][0], "2012-02-29T01:00:00")
        self.assertEqual(sample["valid_times"][-1], "2012-03-01T00:00:00")
        trace = trace_sample(package, self.root, sample["sample_id"])
        self.assertEqual(len(trace["rows"]), 24)
        for lead, row in enumerate(trace["rows"], 1):
            self.assertEqual(row["lead_hour"], lead)
            for kind in ("target", "nwp"):
                raw_time = datetime.strptime(row["raw_records"][kind]["TIMESTAMP"], "%Y%m%d %H:%M")
                self.assertEqual(raw_time.isoformat(), row["valid_time"])
        self.assertEqual(package["splits"]["validation"]["y"][0, 0, 0].item(), 0)
        self.assertEqual(package["splits"]["validation"]["y"][0, -1, 0].item(), 1)

    def test_cross_year_and_calendar_use_actual_year_length(self):
        self.protocol["splits"] = {"train": ["2012-12-30", "2012-12-30"],
                                   "validation": ["2012-12-31", "2012-12-31"],
                                   "test": ["2013-01-01", "2013-01-01"]}
        write_sources(self.root, self.protocol)
        package = prepare_dataset(self.root, self.protocol)
        sample = package["metadata"]["validation"][0]
        self.assertEqual(sample["valid_times"][-1], "2013-01-01T00:00:00")
        self.assertEqual(calendar_features([datetime(2013, 1, 1)])[0], [0., 1., 0., 1.])
        import math
        fraction = (datetime(2012, 3, 1) - datetime(2012, 1, 1)).days / 366
        self.assertAlmostEqual(calendar_features([datetime(2012, 3, 1)])[0][2], math.sin(2 * math.pi * fraction))

    def test_missing_hour_and_na_drop_window_but_zero_and_one_are_kept(self):
        for kind in ("missing_row", "NA", "nan", "2.0"):
            with self.subTest(kind=kind):
                def mutate(row, split, issue, lead):
                    if split == "validation" and issue.day == 29 and lead == 4:
                        if kind == "missing_row":
                            return None
                        row["TARGETVAR"] = kind
                    return row
                write_sources(self.root, self.protocol, mutate)
                package = prepare_dataset(self.root, self.protocol)
                self.assertEqual(package["manifest"]["counts"]["validation"],
                                 {"kept": 1, "dropped": 1, "candidate": 2})
                self.assertEqual(package["dropped"]["validation"][0]["reasons"][0]["valid_time"],
                                 "2012-02-29T04:00:00")
                self.assertTrue((package["splits"]["validation"]["y"] == 0).any())
                self.assertTrue((package["splits"]["validation"]["y"] == 1).any())

    def test_duplicate_timestamp_is_fatal(self):
        write_sources(self.root, self.protocol)
        path = self.root / "Train_W_Zone1.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines + [lines[1]]) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_dataset(self.root, self.protocol)

    def test_csv_order_does_not_change_sample_values_or_primary_key_join(self):
        write_sources(self.root, self.protocol)
        original = prepare_dataset(self.root, self.protocol)
        write_sources(self.root, self.protocol, reverse=True)
        reversed_rows = prepare_dataset(self.root, self.protocol)
        for split in original["splits"]:
            for key in original["splits"][split]:
                torch.testing.assert_close(original["splits"][split][key], reversed_rows["splits"][split][key])
        self.assertEqual(original["manifest"]["sample_ids"], reversed_rows["manifest"]["sample_ids"])
        self.assertNotEqual(original["manifest"]["data_fingerprint"], reversed_rows["manifest"]["data_fingerprint"])

    def test_validation_test_extremes_cannot_change_training_scaler_or_zone(self):
        write_sources(self.root, self.protocol)
        original = prepare_dataset(self.root, self.protocol)
        def mutate(row, split, issue, lead):
            if split != "train":
                for field in self.protocol["nwp_fields"]:
                    row[field] = "1000000"
                row["TARGETVAR"] = "1"
            return row
        write_sources(self.root, self.protocol, mutate)
        changed = prepare_dataset(self.root, self.protocol)
        self.assertEqual(original["manifest"]["scaler"], changed["manifest"]["scaler"])
        self.assertEqual(original["manifest"]["zone_selection_audit"], changed["manifest"]["zone_selection_audit"])

    def test_zone_rule_uses_training_completeness_then_lowest_number(self):
        protocol = fixture_protocol((1, 2, 3))
        def mutate(row, split, issue, lead):
            if row["ZONEID"] == "1" and split == "train" and lead == 4:
                row["TARGETVAR"] = "NA"
            if row["ZONEID"] == "2" and split == "validation" and issue.day == 29:
                row["TARGETVAR"] = "NA"
            return row
        write_sources(self.root, protocol, mutate)
        package = prepare_dataset(self.root, protocol)
        self.assertEqual(package["manifest"]["selected_zone"], 2)

    def test_overlapping_splits_rejected(self):
        self.protocol["splits"]["validation"][0] = "2012-02-28"
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_protocol(self.protocol)

    def test_reproducible_package_fingerprint_and_tamper_detection(self):
        write_sources(self.root, self.protocol)
        package = prepare_dataset(self.root, self.protocol)
        self.assertEqual(package["manifest"], prepare_dataset(self.root, self.protocol)["manifest"])
        output = self.root / "prepared"
        write_package(package, self.root, output)
        self.assertEqual(load_package(output / "data.pt")["manifest"], package["manifest"])
        with self.assertRaises(FileExistsError):
            write_package(package, self.root, output)
        changed = deepcopy(package)
        changed["splits"]["train"]["nwp"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "tensor fingerprint"):
            validate_package(changed)
        source = self.root / "Train_W_Zone1.csv"
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source fingerprint"):
            trace_sample(package, self.root, package["metadata"]["train"][0]["sample_id"])

    def test_forecast_target_join_missing_key_is_logged(self):
        write_sources(self.root, self.protocol)
        path = self.root / "TestPred_W_Zone1.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join([lines[0]] + lines[2:]) + "\n")
        package = prepare_dataset(self.root, self.protocol)
        reason = package["dropped"]["test"][0]["reasons"][0]["reasons"]
        self.assertEqual(reason, ["missing_nwp_row"])

    def test_incomplete_package_is_rejected_even_when_remaining_tensors_match(self):
        write_sources(self.root, self.protocol)
        package = prepare_dataset(self.root, self.protocol)
        del package["splits"]["test"]
        with self.assertRaisesRegex(ValueError, "all three splits"):
            validate_package(package)


if __name__ == "__main__":
    unittest.main()
