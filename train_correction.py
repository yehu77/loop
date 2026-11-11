"""Independent C-only optimizer/EMA/clock and identity-bound exact continuation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

import torch

from correction import CorrectedDenoiser, load_base, load_config, make_corrector, require
from data import RealBatches, file_hash, tensor_hash
from diffusion import LinearNoiseSchedule
from history_data import load_history_package
from train import synchronize


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_inputs(config, device):
    require(file_hash(config["data"]) == config["data_sha256"], "Data file hash mismatch")
    package = load_history_package(config["data"])
    require(package["manifest"]["data_fingerprint"] == config["data_fingerprint"], "Data identity mismatch")
    require([m["sample_id"] for m in package["metadata"]["validation"]] == config["validation_ids"],
            "Validation panel mismatch")
    base, scaler = load_base(config, package, device)
    return package, base, scaler


class CorrectionTrainer:
    def __init__(self, config, base, scaler, device):
        self.config, self.device = deepcopy(config), torch.device(device)
        self.base, self.scaler = base.eval().requires_grad_(False), scaler
        self.corrector = make_corrector(config, self.device)
        self.ema = deepcopy(self.corrector).eval().requires_grad_(False)
        self.schedule = LinearNoiseSchedule(1000).to(self.device)
        self.wrapper = CorrectedDenoiser(self.base, self.corrector, self.schedule, 1)
        c = config["training"]
        require(c["correction_steps"] == 1 and c["precision"] == "float32", "Train C once in FP32")
        self.optimizer = torch.optim.AdamW([
            {"params": [p for p in self.corrector.parameters() if p.ndim >= 2], "weight_decay": c["weight_decay"]},
            {"params": [p for p in self.corrector.parameters() if p.ndim < 2], "weight_decay": 0.},
        ], lr=c["learning_rate"], betas=(0.9, 0.999))
        self.noise = torch.Generator(device=self.device).manual_seed(config["seeds"]["training_noise"])
        self.updates = self.attempts = self.ema_num_averaged_updates = 0
        self.ema_initialized = False
        self.at_boundary = True

    def update(self, batches):
        self.at_boundary = False
        self.attempts += 1
        self.wrapper.train()
        self.optimizer.zero_grad(set_to_none=True)
        c, loss_sum = self.config["training"], 0.
        for _ in range(c["accumulation_steps"]):
            b = {k: v.to(self.device).float() for k, v in next(batches).items()}
            require(all(torch.isfinite(b[k]).all() for k in ("y", "nwp", "calendar")), "Nonfinite training data")
            require(bool(((b["y"] >= 0) & (b["y"] <= 1)).all()), "Invalid capacity power")
            x0 = 2 * b["y"] - 1
            t = torch.randint(1, 1001, (len(x0),), generator=self.noise, device=self.device)
            noise = torch.randn(x0.shape, generator=self.noise, device=self.device)
            xt = self.schedule.add_noise(x0, t, noise)
            epsilon = self.wrapper(xt, self.scaler.transform(b["nwp"]), b["calendar"], t, b["history_power"])
            loss = (epsilon.float() - noise).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite correction epsilon loss")
            (loss / c["accumulation_steps"]).backward()
            loss_sum += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(self.corrector.parameters(), c["max_grad_norm"], error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.update_ema()
        self.at_boundary = True
        return {"updates": self.updates, "attempts": self.attempts, "skipped": 0,
                "loss": loss_sum / c["accumulation_steps"], "grad_norm": norm.item()}

    @torch.no_grad()
    def update_ema(self):
        c = self.config["training"]
        if self.updates < c["ema_start_update"]:
            return
        if self.updates == c["ema_start_update"]:
            self.ema.load_state_dict(self.corrector.state_dict())
            self.ema_initialized = True
            return
        require(self.ema_initialized, "EMA start missed")
        for a, b in zip(self.ema.parameters(), self.corrector.parameters()):
            a.lerp_(b, 1 - c["ema_decay"])
        for a, b in zip(self.ema.buffers(), self.corrector.buffers()):
            a.copy_(b)
        self.ema_num_averaged_updates += 1

    def state(self, batches):
        return {"format": "prediction_correction_v1", "config": self.config,
                "model": self.corrector.state_dict(), "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(), "data": batches.state_dict(),
                "noise_rng": self.noise.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                "device_type": self.device.type, "updates": self.updates, "attempts": self.attempts,
                "ema_initialized": self.ema_initialized, "ema_num_averaged_updates": self.ema_num_averaged_updates,
                "metadata": {"failure": not self.at_boundary, "base_sha256": self.config["base"]["sha256"],
                             "data_fingerprint": self.config["data_fingerprint"]}}

    def save(self, path, batches):
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        torch.save(self.state(batches), temporary)
        temporary.replace(path)

    def restore(self, path, batches):
        s = torch.load(path, map_location="cpu", weights_only=True)
        require(s["format"] == "prediction_correction_v1" and not s["metadata"]["failure"], "Invalid/failure correction checkpoint")
        require(s["config"] == self.config and s["metadata"]["base_sha256"] == self.config["base"]["sha256"]
                and s["metadata"]["data_fingerprint"] == self.config["data_fingerprint"], "Correction checkpoint identity mismatch")
        require(s["device_type"] == self.device.type, "Resume device type mismatch")
        start = self.config["training"]["ema_start_update"]
        require(s["updates"] == s["attempts"] and s["ema_initialized"] == (s["updates"] >= start)
                and s["ema_num_averaged_updates"] == max(0, s["updates"] - start), "Correction clock mismatch")
        self.corrector.load_state_dict(s["model"])
        self.ema.load_state_dict(s["ema"])
        self.optimizer.load_state_dict(s["optimizer"])
        batches.load_state_dict(s["data"])
        self.noise.set_state(s["noise_rng"])
        torch.set_rng_state(s["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(s["cuda_rng"])
        for name in ("updates", "attempts", "ema_initialized", "ema_num_averaged_updates"):
            setattr(self, name, s[name])
        self.at_boundary = True


def run(args):
    torch.set_num_threads(1)
    config = load_config(args.config)
    package, base, scaler = load_inputs(config, args.device)
    trainer = CorrectionTrainer(config, base, scaler, args.device)
    batches = RealBatches(package, config["training"]["microbatch_size"], config["seeds"]["data_order"])
    if args.resume:
        trainer.restore(args.resume, batches)
    require(args.updates > 0 and trainer.updates + args.updates <= config["training"]["total_updates"], "Invalid additional update budget")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", config)
    write(output / "initialization.json", {k: tensor_hash(v) for k, v in trainer.corrector.state_dict().items()})
    lineage = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "resume_from": str(args.resume) if args.resume else None,
               "resume_sha256": file_hash(args.resume) if args.resume else None,
               "resumed_updates": trainer.updates, "started_unix": time.time(), "completed": False}
    write(output / "lineage.json", lineage)
    original_base = {k: tensor_hash(v) for k, v in base.state_dict().items()}
    history = []
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(args.updates):
            synchronize(trainer.device)
            start = time.perf_counter()
            record = trainer.update(batches)
            synchronize(trainer.device)
            record["training_seconds"] = time.perf_counter() - start
            history.append(record)
            if trainer.updates % config["training"]["save_every"] == 0:
                trainer.save(output / f"checkpoint_{trainer.updates:06d}.pt", batches)
                trainer.save(output / "checkpoint_last.pt", batches)
                write(output / "training_history.json", history)
            print(json.dumps(record), flush=True)
        require(original_base == {k: tensor_hash(v) for k, v in base.state_dict().items()}, "Frozen D changed")
        load_config(args.config)
        require(file_hash(config["base"]["checkpoint"]) == config["base"]["sha256"], "BASE file changed")
        trainer.save(output / "checkpoint_last.pt", batches)
        write(output / "training_history.json", history)
        lineage.update(completed=True, ended_unix=time.time(), successful_updates=trainer.updates,
                       checkpoint_sha256=file_hash(output / "checkpoint_last.pt"), frozen_base_unchanged=True)
        write(output / "lineage.json", lineage)
        write(output / "cost.json", {"D_parameters": sum(p.numel() for p in base.parameters()),
              "C_parameters": sum(p.numel() for p in trainer.corrector.parameters()),
              "training_seconds": sum(r["training_seconds"] for r in history),
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None})
    except BaseException:
        trainer.at_boundary = False
        trainer.save(output / "failure.pt", batches)
        write(output / "training_history.json", history)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--updates', type=int, required=True, help='Additional successful updates')
    p.add_argument('--resume', type=Path)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
