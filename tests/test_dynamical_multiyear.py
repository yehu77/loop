import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
from data_audit.dynamical import multiyear, spread


class Multiyear(unittest.TestCase):
    def test_retry_different_folder_for_mixed_snapshot(self):
        first = [{"snapshotId": s, "dataProduct": {"version": "1"}} for s in "aaab"]
        second = [first[0]] * 4
        with tempfile.TemporaryDirectory() as tmp, patch.object(multiyear, "fetch") as fetch, patch.object(
                multiyear, "cached", side_effect=[({"results": first}, {}), ({"results": second}, {})]):
            folder, _, _ = multiyear.daily_batch(Path(tmp), date(2021, 1, 1), True)
            self.assertEqual(folder.name, "attempt-2")
            self.assertNotEqual(fetch.call_args_list[0].args[0], fetch.call_args_list[1].args[0])

    def test_speed_mean_and_spread_use_members_only(self):
        wind = np.zeros((4, 50, 9, 4))
        wind[:, :25, :, 0] = 5
        wind[:, 25:, :, 0] = -5
        wind[1, :, :, 0] *= 2
        wind[0, :25, :, 2] = 2
        a, b = spread.features(wind, date(2021, 8, 1))
        self.assertEqual(a.shape, (24, 29))
        self.assertEqual(b.shape, (24, 8))
        np.testing.assert_array_equal(a[:, [0, 2, 8]], [[0, 5, 10]] * 24)
        np.testing.assert_array_equal(b[:, :3], [[0, 1, 0]] * 24)

    def test_score_constant_and_empirical_distribution(self):
        x = np.zeros((64, 1))
        y = np.repeat([0., 2.], 32)
        scores = spread.score(x, x[:1], y, np.array([1.]))
        self.assertEqual(scores["crps_MW"][0], .5)
        self.assertTrue(scores["coverage90"][0])
        self.assertEqual(scores["width90_MW"][0], 2.)

    def test_calendar_boundaries(self):
        rows = multiyear.calendar()
        self.assertEqual(len(rows), 1095)
        self.assertEqual([str(d) for d, r in rows.items() if r["boundary_holdout"]],
                         ["2021-12-31", "2022-12-31"])
