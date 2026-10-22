from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from summarize_runs import (COSTS, SCORES, paired_summary, select_nodes, stats,
                            stitch, summarize, verify_score_identity)


class RunSummaryTests(unittest.TestCase):
    def test_equal_scores_cannot_disguise_changed_window_identity(self):
        expected = {"sample_ids": ["day_a", "day_b"], "data_fingerprint": "frozen",
                    "crps": [0.1, 0.1]}
        verify_score_identity(deepcopy(expected), expected)
        for key, replacement in (("sample_ids", ["day_b", "day_a"]),
                                 ("data_fingerprint", "another_dataset")):
            saved = deepcopy(expected)
            saved[key] = replacement
            with self.assertRaisesRegex(ValueError, "identity differs"):
                verify_score_identity(saved, expected)

    def rows(self):
        rows = []
        for seed, difference in ((42, 1.0), (43, 2.0), (44, 3.0)):
            for variant, score in (("base", 10.0), ("loop", 10.0 + difference), ("untied", 9.0)):
                rows.append({"table": "primary", "seed": seed, "variant": variant,
                             **{metric: score for metric in SCORES + COSTS}})
        return rows

    def test_pair_within_seed_then_sample_standard_deviation(self):
        report = paired_summary(self.rows())
        item = next(row for row in report["paired_differences"]
                    if row["left"] == "loop" and row["right"] == "base" and row["metric"] == "crps")
        self.assertEqual(item["by_seed"], [{"seed": 42, "delta": 1.0},
                                          {"seed": 43, "delta": 2.0},
                                          {"seed": 44, "delta": 3.0}])
        self.assertEqual(item["n_seeds"], 3)
        self.assertEqual(item["mean"], 2.0)
        self.assertEqual(item["sample_std"], 1.0)
        self.assertIsNone(stats([1.0])["sample_std"])

    def test_reject_duplicate_rows_or_missing_architecture(self):
        rows = self.rows()
        for malformed in (rows + [rows[0]], rows[:-1]):
            with self.assertRaises(ValueError):
                paired_summary(malformed)
        with self.assertRaises(ValueError):
            stats([float("nan")])

    def updates(self, start, end):
        return [{"updates": update, "did_update": True, "skipped": 0}
                for update in range(start, end + 1)]

    def test_generic_segments_keep_only_saved_boundary(self):
        # The first segment logged update 4 but resumed from its saved update 3.
        merged = stitch([self.updates(1, 4), self.updates(4, 7)], [3, 7])
        self.assertEqual([r["updates"] for r in merged], list(range(1, 8)))

    def test_reject_missing_duplicate_or_reversed_updates(self):
        for histories, endpoints in (([self.updates(1, 3), self.updates(5, 7)], [3, 7]),
                                      ([self.updates(1, 3), self.updates(3, 7)], [3, 7]),
                                      ([self.updates(1, 3), self.updates(4, 7)], [3, 2])):
            with self.assertRaises(ValueError):
                stitch(histories, endpoints)

    def node(self, update, score):
        metric = {"crps": score, "energy_score": score, "variogram_score": score,
                  "boundary_fraction": 0.0, "sampling_seconds": 1.0}
        row = {"updates": update, "ema_initialized": update >= 2,
               "ema_num_averaged_updates": max(0, update - 2), "raw": metric}
        if update >= 2:
            row["ema"] = deepcopy(metric)
        return row

    def test_off_grid_resume_score_cannot_win_and_ties_choose_earlier(self):
        history = [[self.node(0, 3.0), self.node(2, 1.0)],
                   [self.node(3, -999.0), self.node(4, 1.0), self.node(6, 2.0)]]
        nodes, best = select_nodes(history, [3, 6], start=2, interval=2, endpoint=6)
        self.assertEqual([n["updates"] for n in nodes], [0, 2, 4, 6])
        self.assertEqual(best, 2)

    def test_resume_reevaluation_can_change_timing_but_not_scores(self):
        initial = [self.node(0, 3.0), self.node(2, 1.0)]
        resumed = [self.node(2, 1.0), self.node(4, 0.5)]
        resumed[0]["ema"]["sampling_seconds"] = 2.0
        select_nodes([initial, resumed], [2, 4], start=2, interval=2, endpoint=4)
        resumed[0]["ema"]["crps"] = 0.9
        with self.assertRaisesRegex(ValueError, "differs"):
            select_nodes([initial, resumed], [2, 4], start=2, interval=2, endpoint=4)

    def test_missing_nodes_and_invalid_ema_count_rejected(self):
        nodes = [self.node(0, 3.0), self.node(2, 1.0), self.node(4, 0.5)]
        with self.assertRaises(ValueError):
            select_nodes([nodes[1:]], [4], start=2, interval=2, endpoint=4)
        nodes[-1]["ema_num_averaged_updates"] = 999
        with self.assertRaises(ValueError):
            select_nodes([nodes], [4], start=2, interval=2, endpoint=4)

    def test_report_refuses_existing_or_overlapping_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"seeds": [42], "runs": [
                {"seed": 42, "variant": variant, "segments": [str(root / variant)]}
                for variant in ("base", "loop", "untied")]}
            for destination in (root, root / "base/report"):
                with self.assertRaises(ValueError):
                    summarize(manifest, destination)
            with self.assertRaisesRegex(ValueError, "Undeclared seed"):
                summarize(manifest, root / "report", [99])


if __name__ == "__main__":
    unittest.main()
