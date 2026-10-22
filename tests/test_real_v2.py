from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from data import prepare_dataset
from test_data import fixture_protocol, write_sources
from train_real import run, load_real_checkpoint, validation_artifacts


class RealV2Tests(unittest.TestCase):
    def test_all_variants_real_entry_resume_before_start_and_selection(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = fixture_protocol()
            write_sources(root, protocol)
            package = prepare_dataset(root, protocol)
            data = root / "data.pt"
            torch.save(package, data)
            for variant in ("base", "loop", "untied"):
                args = Namespace(data=data, output=root / variant, resume=None, variant=variant,
                                 training_protocol_version=2, ema_start_update=2, device="cpu",
                                 updates=1, amp=False, seed=42, microbatch_size=1,
                                 accumulation_steps=1, validate_every=2, validation_limit=None,
                                 scenarios=2, sampling_steps=2, chunk_size=4)
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                self.assertFalse((args.output / "checkpoint_best.pt").exists())
                trainer, _, metadata = load_real_checkpoint(args.output / "checkpoint_last.pt", package, "cpu")
                with self.assertRaisesRegex(ValueError, "not initialized"):
                    validation_artifacts(trainer, package, metadata["run_config"])
                args.resume = args.output / "checkpoint_last.pt"
                args.output = root / (variant + "_resumed")
                args.updates = 2
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                cost = json.loads((args.output / "cost.json").read_text())
                self.assertEqual(cost["primary_update"], 3)
                self.assertEqual(cost["auxiliary_best_update"], 2)
                history = json.loads((args.output / "validation_history.json").read_text())
                self.assertNotIn("ema", history[0])
                self.assertEqual(history[1]["ema_num_averaged_updates"], 0)
                self.assertEqual(history[1]["raw"]["crps"], history[1]["ema"]["crps"])
                self.assertTrue((args.output / "metrics.json").is_file())
                self.assertTrue((args.output / "selected/metrics.json").is_file())
                anchor = torch.load(args.output / "checkpoint_ema_start.pt", weights_only=True)
                self.assertEqual(anchor["updates"], 2)
                for key in anchor["model"]:
                    self.assertTrue(torch.equal(anchor["model"][key], anchor["ema"][key]))
                # Resume after start with a matching best; never re-copy the EMA.
                args.resume = args.output / "checkpoint_last.pt"
                args.output = root / (variant + "_after")
                args.updates = 1
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                trainer, _, _ = load_real_checkpoint(args.output / "checkpoint_last.pt", package, "cpu")
                self.assertEqual(trainer.ema_num_averaged_updates, 2)
