"""Paired seed accounting and configuration/RNG drift guards for R02 reports."""
from copy import deepcopy
from dataclasses import asdict
import unittest

import torch

from diagnostics.summarize_local_reference import (LABELS, METRICS, paired_differences,
                                                   verify_local_pair)
from model import ModelConfig


class LocalSummaryTests(unittest.TestCase):
    def test_pairing_uses_training_seeds_and_requires_complete_unique_panel(self):
        rows = []
        for table in ("primary", "best"):
            for seed in (42, 43, 44):
                for label in LABELS:
                    rows.append({"table": table, "seed": seed, "model": label,
                                 **{k: (seed - 43 if label == "local" else 0) for k in METRICS}})
        pairs, summary = paired_differences(rows[::-1], [42, 43, 44])
        self.assertEqual(len(pairs), 2 * 3 * len(METRICS))
        for row in summary:
            self.assertEqual(row["mean_local_minus_base"], 0)
            self.assertEqual(row["sample_std"], 1)
            self.assertEqual(row["n_training_seeds"], 3)
        for invalid in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                paired_differences(invalid, [42, 43, 44])

    def test_pair_rejects_scope_config_initialization_rng_and_position_drift(self):
        base = {"config": {"model": asdict(ModelConfig(variant="base")),
                           "training": {"seed": 42}, "evaluation": {"seed": 2026},
                           "data_fingerprint": "f", "validation_sample_ids": ["a", "b"]},
                "initialization": {"weight": "hash"},
                "checkpoint": {"updates": 3000, "noise_rng": torch.tensor([1, 2]),
                               "data": {"order": torch.tensor([2, 1]), "rng": torch.tensor([3]), "cursor": 1}}}
        del base["config"]["model"]["attention_scope"]  # Legacy full default.
        local = deepcopy(base)
        local["config"]["model"]["attention_scope"] = "diagonal"
        verify_local_pair(base, local)
        changes = [(["config", "model", "attention_scope"], "full"),
                   (["config", "model", "hidden_dim"], 32),
                   (["config", "training", "seed"], 43),
                   (["config", "evaluation", "seed"], 7),
                   (["config", "data_fingerprint"], "g"),
                   (["config", "validation_sample_ids"], ["b", "a"]),
                   (["initialization", "weight"], "other"),
                   (["checkpoint", "updates"], 2999),
                   (["checkpoint", "noise_rng"], torch.tensor([2, 3])),
                   (["checkpoint", "data", "rng"], torch.tensor([4])),
                   (["checkpoint", "data", "order"], torch.tensor([1, 2])),
                   (["checkpoint", "data", "cursor"], 2)]
        for path, value in changes:
            changed = deepcopy(local)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                verify_local_pair(base, changed)
