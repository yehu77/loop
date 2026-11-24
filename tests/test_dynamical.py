import copy
import unittest
from datetime import timedelta

from data_audit.dynamical.collect import DAYS, GRID, LEADS, PRODUCT, VARS, stamp
from data_audit.dynamical.prepare import point_values


class NativeAxes(unittest.TestCase):
    def setUp(self):
        self.day = DAYS[0]
        loc = dict(zip(("latitude", "longitude"), GRID[0]))
        self.forecast = dict(initTime=stamp(self.day), leadTimeHours=LEADS,
            validTimes=[stamp(self.day + timedelta(days=h//24), h%24) for h in LEADS],
            ensembleMembers=[dict(member=m, data={v: [m]*9 for v in VARS}) for m in reversed(range(51))],
            links={"canonical": "/example"})
        self.result = dict(dataProduct={"id": PRODUCT, "version": "1"}, snapshotId="s",
            variables={v: {"unit": "m s-1"} for v in VARS},
            point=dict(requested=loc, selected=copy.copy(loc), method="nearest"), forecasts=[self.forecast])

    def test_member_id_controls_alignment(self):
        values, _, _ = point_values(self.result, self.day, GRID[0])
        self.assertEqual(len(values), 50)
        self.assertEqual((values[0][0][0], values[-1][-1][-1]), (1, 50))

    def test_end_anchor_and_member(self):
        self.forecast["validTimes"].pop()
        with self.assertRaises(ValueError):
            point_values(self.result, self.day, GRID[0])

    def test_null_and_duplicate(self):
        self.forecast["ensembleMembers"][0]["data"][VARS[0]][0] = None
        with self.assertRaises(ValueError):
            point_values(self.result, self.day, GRID[0])
        self.forecast["ensembleMembers"][0]["member"] = 1
        with self.assertRaises(ValueError):
            point_values(self.result, self.day, GRID[0])

    def test_grid_and_units(self):
        self.result["point"]["selected"]["longitude"] = -4
        with self.assertRaises(ValueError):
            point_values(self.result, self.day, GRID[0])
        self.result["point"]["selected"]["longitude"] = GRID[0][1]
        self.result["variables"][VARS[-1]]["unit"] = "km/h"
        with self.assertRaises(ValueError):
            point_values(self.result, self.day, GRID[0])
