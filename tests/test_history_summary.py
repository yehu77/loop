import unittest

from diagnostics.summarize_history_information import paired_rows


class HistorySummaryTests(unittest.TestCase):
    def test_pairing_uses_identity_not_row_order_and_preserves_mixed_signs(self):
        rows = [{"table": "primary", "group": g, "crps": v}
                for g, v in (("H24", .12), ("H0", .20), ("H1", .10))]
        pairs = {r["comparison"]: r["crps"] for r in paired_rows(rows, ("table",), ("crps",))}
        self.assertAlmostEqual(pairs["H1-H0"], -.10)
        self.assertAlmostEqual(pairs["H24-H0"], -.08)
        self.assertAlmostEqual(pairs["H24-H1"], .02)
        self.assertEqual(len(pairs), 3)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            paired_rows(rows[:2], ("table",), ("crps",))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_rows(rows + rows[:1], ("table",), ("crps",))

    def test_lead_groups_and_days_remain_separate_observations(self):
        rows = []
        for day in ("day1", "day2"):
            for lead in ("1-6", "7-12", "13-24"):
                for group, offset in (("H0", 0), ("H1", .01), ("H24", -.01)):
                    rows.append({"sample_id": day, "lead_group": lead, "group": group,
                                 "crps": .1 + offset * (1 if day == "day1" else -1)})
        result = paired_rows(rows, ("sample_id", "lead_group"), ("crps",))
        self.assertEqual(len(result), 18)
        self.assertTrue(any(r["crps"] > 0 for r in result))
        self.assertTrue(any(r["crps"] < 0 for r in result))
