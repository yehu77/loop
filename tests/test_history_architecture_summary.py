import unittest

from diagnostics.summarize_history_architecture import contrasts


class HistoryArchitectureSummaryTests(unittest.TestCase):
    def panel(self):
        values = {("base", "H0"): .080, ("base", "H24"): .060,
                  ("loop", "H0"): .079, ("loop", "H24"): .059,
                  ("untied", "H0"): .077, ("untied", "H24"): .058}
        return [{"table": "primary", "variant": v, "history_group": g, "crps": x}
                for (v, g), x in values.items()]

    def test_equal_history_improvement_is_not_a_special_loop_benefit(self):
        result = contrasts(self.panel(), ("table",), ("crps",))
        self.assertEqual(len(result), 11)
        effects = {(r["kind"], r["comparison"], r["condition"]): r["crps"] for r in result}
        self.assertAlmostEqual(effects["interaction", "loop-base", "history_effect_difference"], 0)
        self.assertAlmostEqual(effects["interaction", "loop-untied", "history_effect_difference"], -.001)
        # A better history interaction does not imply better absolute H24 quality.
        self.assertGreater(effects["same_input", "loop-untied", "H24"], 0)
        self.assertAlmostEqual(effects["history_effect", "loop", "H24-H0"], -.020)

    def test_identity_includes_variant_group_and_stratification(self):
        rows = self.panel()
        self.assertEqual(contrasts(rows, ("table",), ("crps",)), contrasts(rows[::-1], ("table",), ("crps",)))
        for bad in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                contrasts(bad, ("table",), ("crps",))
        stratified = []
        for day in ("day1", "day2"):
            for lead in ("1-6", "7-12", "13-24"):
                stratified.extend({**r, "sample_id": day, "lead_group": lead} for r in rows)
        result = contrasts(stratified, ("table", "sample_id", "lead_group"), ("crps",))
        self.assertEqual(len(result), 66)
