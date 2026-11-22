import copy
import hashlib
import json
import unittest
from datetime import timedelta

from data_audit.audit import DAY, HALF, HERE, ORIGIN, available, check_ensemble, hourly, interval, report, weather_identity
from data_audit.compare import predicted_label


class BerryBurnAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((HERE / "b1610/2019-08-25.json").read_bytes())
        cls.ens = json.loads((HERE / "ens.json").read_bytes())

    def test_settlement_clocks_against_actual_rows(self):
        for day in ("2019-03-31", "2019-10-27", "2020-03-29", "2020-10-25",
                    "2020-01-15", "2023-07-01"):
            rows = json.loads((HERE / f"b1610/{day}.json").read_bytes())
            stamps = [interval(row)[0] for row in rows]
            self.assertEqual(len(stamps), 49)
            self.assertTrue(all(b - a == HALF for a, b in zip(stamps, stamps[1:])))
        bad = dict(self.rows[0], settlementDate="2019-03-31", settlementPeriod=47)
        with self.assertRaisesRegex(ValueError, "Invalid period"):
            interval(bad)

    def test_author_rule_is_seasonal(self):
        self.assertEqual(predicted_label("2020-01-15", 1).hour, 0)
        self.assertEqual(predicted_label("2019-08-25", 3).hour, 2)

    def test_measured_target_and_units(self):
        target = hourly(self.rows, ORIGIN + DAY)
        self.assertEqual(len(target), 24)
        self.assertAlmostEqual(target[0]["MW"], 5.248)
        self.assertEqual(target[-1]["end"], "2019-08-26T00:00:00+00:00")
        self.assertEqual([n for r in target for n in r["response_indices"]], list(range(48)))
        self.assertTrue(all(r["quality"] == "complete" for r in target))

    def test_missing_duplicate_zero(self):
        self.assertIsNone(hourly(self.rows[1:], ORIGIN + DAY)[0]["MW"])
        rows = [dict(row, quantity=0) for row in self.rows]
        self.assertEqual(hourly(rows, ORIGIN + DAY)[0]["MW"], 0)
        rows[1]["quantity"] = None
        self.assertIsNone(hourly(rows, ORIGIN + DAY)[0]["MW"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            hourly(rows + rows[:1], ORIGIN + DAY)

    def test_forecast_identity(self):
        check_ensemble(self.ens)
        hres = dict(self.ens, product="HRES")
        weather_identity(hres, "HRES", ORIGIN)
        with self.assertRaisesRegex(ValueError, "Mixed"):
            weather_identity(hres, "HRES", ORIGIN + DAY)
        bad = copy.deepcopy(self.ens)
        bad["rows"][0]["lead_hours"] += 1
        with self.assertRaisesRegex(ValueError, "valid time"):
            check_ensemble(bad)

    def test_member_and_order(self):
        for mutation in ("duplicate_id", "reverse_time", "missing"):
            bad = copy.deepcopy(self.ens)
            if mutation == "duplicate_id":
                bad["rows"][0]["member"] = 2
            elif mutation == "reverse_time":
                bad["rows"].reverse()
            else:
                bad["rows"].pop()
            with self.assertRaises(ValueError):
                check_ensemble(bad)

    def test_as_of_and_leakage(self):
        issue = ORIGIN + timedelta(hours=12)
        self.assertFalse(available(self.ens, issue))
        for hours, expected in ((4, True), (13, False), (-1, False)):
            item = dict(self.ens, available_time=(ORIGIN + timedelta(hours=hours)).isoformat())
            self.assertEqual(available(item, issue), expected)
        with self.assertRaises(ValueError):
            weather_identity(dict(self.ens, product="ERA5"), "ENS", ORIGIN)

    def test_offline_replay_and_hashes(self):
        frozen = (HERE / "sample.json").read_bytes()
        for _ in range(2):
            replay = (json.dumps(report(), indent=2, allow_nan=False) + "\n").encode()
            self.assertEqual(hashlib.sha256(replay).digest(), hashlib.sha256(frozen).digest())
        self.assertFalse(report()["train_ready"])
