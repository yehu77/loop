import json
import unittest

from data_audit.ens_only_v1.accept import HERE, ROOT, evaluate, inventory


class OfflineAcceptance(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "protocol.json").read_bytes())
        self.weather = json.loads((HERE / "ens.json").read_bytes())
        self.meter = json.loads((HERE / "b1610/2019-08-25.json").read_bytes())

    def result(self):
        return evaluate("2019-08-24", self.config, self.weather, self.meter)

    def test_two_clocks_and_last_anchor(self):
        r = self.result()
        self.assertTrue(r["offline_sample_pass"])
        self.assertIsNone(r["actual_available_time"])
        self.assertFalse(r["operational_availability_verified"])
        self.assertEqual([k["issue_lead_h"] for k in r["weather_knots"]], [12, 18, 24, 30, 36])
        self.assertEqual(r["weather_knots"][-1]["valid_time"], r["target_MW"][-1]["end"])
        self.assertEqual(len(r["target_MW"]), 24)

    def test_declaration_required(self):
        del self.config["access_assumption"]
        self.assertFalse(self.result()["offline_sample_pass"])

    def test_issue_required(self):
        del self.config["issue_hour_utc"]
        self.assertFalse(self.result()["time_relations_pass"])

    def test_member_required(self):
        self.weather["rows"].pop()
        self.assertFalse(self.result()["ens_panel_complete"])

    def test_weather_time_required(self):
        del self.weather["rows"][0]["valid_time"]
        result = self.result()
        self.assertFalse(result["time_relations_pass"])
        self.assertFalse(result["offline_sample_pass"])

    def test_signed_meter_and_incomplete_hour(self):
        self.meter[0]["quantity"] = -2.5
        self.meter[1]["quantity"] = 0
        self.assertEqual(self.result()["target_MW"][0]["MW"], -2.5)
        self.meter.pop(0)
        self.assertFalse(self.result()["power_target_complete"])

    def test_frozen_and_new_readiness(self):
        self.assertFalse(json.loads((HERE / "sample.json").read_bytes())["train_ready"])
        r = inventory()
        self.assertEqual(r, json.loads((ROOT / "acceptance.json").read_bytes()))
        self.assertEqual([run["offline_sample_pass"] for run in r["samples"]], [True, False, False])
        self.assertEqual(r["verified_complete_windows"], 1)
        self.assertFalse(r["dataset_train_ready"])
