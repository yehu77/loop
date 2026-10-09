"""Minimal v1 training and synthetic smoke runner; not a real-data experiment."""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch

from diffusion import LinearNoiseSchedule, sample_scenarios
from model import ConditionalLoopDiT, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    microbatch_size: int = 4
    accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    diffusion_steps: int = 1000
    amp: bool = False
    seed: int = 42

    def __post_init__(self):
        if min(self.learning_rate, self.microbatch_size, self.accumulation_steps,
               self.max_grad_norm) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid training hyperparameters")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")


class WeatherScaler:
    def __init__(self, mean, std):
        self.mean, self.std = mean.cpu(), std.cpu()

    @classmethod
    def fit(cls, training_nwp):
        if training_nwp.ndim != 3 or not torch.isfinite(training_nwp).all():
            raise ValueError("Fit weather statistics on complete training windows only")
        data = training_nwp.float()
        return cls(data.mean(dim=(0, 1)), data.std(dim=(0, 1), correction=0).clamp_min(1e-6))

    def transform(self, nwp):
        return (nwp.float() - self.mean.to(nwp.device)) / self.std.to(nwp.device)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}


def make_synthetic_data(config, samples=128, seed=123):
    """Artificial correlated weather/power, with valid calendar sin/cos pairs."""
    generator = torch.Generator().manual_seed(seed)
    hours = torch.arange(config.horizon).float()[None, :]
    start = torch.randint(0, 365 * 24, (samples, 1), generator=generator)
    absolute_hour = start + hours
    daily = 2 * math.pi * absolute_hour / 24
    annual = 2 * math.pi * absolute_hour / (365 * 24)
    calendar = torch.stack((daily.sin(), daily.cos(), annual.sin(), annual.cos()), dim=-1)
    nwp = torch.randn(samples, config.horizon, config.nwp_features, generator=generator)
    nwp = nwp + 0.5 * calendar[:, :, :1]
    noise = torch.randn(samples, config.horizon, 1, generator=generator)
    power = torch.sigmoid(0.8 * nwp[:, :, :1] + 0.3 * calendar[:, :, 1:2] + 0.15 * noise)
    return {"y": power, "nwp": nwp, "calendar": calendar}


class SyntheticBatches:
    """Shuffled finite data with a saved permutation, cursor, and RNG."""
    def __init__(self, data, batch_size, seed=42):
        self.data, self.batch_size = data, batch_size
        self.size = len(data["y"])
        if not 1 <= batch_size <= self.size:
            raise ValueError("batch_size must be in [1, dataset size]")
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(self.size, generator=self.generator)
        self.cursor = 0

    def __next__(self):
        indices = []
        remaining = self.batch_size
        while remaining:
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.cursor = 0
            count = min(remaining, self.size - self.cursor)
            indices.append(self.order[self.cursor:self.cursor + count])
            self.cursor += count
            remaining -= count
        indices = torch.cat(indices)
        return {name: tensor[indices] for name, tensor in self.data.items()}

    def state_dict(self):
        return {"order": self.order, "cursor": self.cursor,
                "rng": self.generator.get_state(), "size": self.size,
                "batch_size": self.batch_size}

    def load_state_dict(self, state):
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("Dataset size or batch size differs from checkpoint")
        self.order, self.cursor = state["order"].cpu(), state["cursor"]
        self.generator.set_state(state["rng"].cpu())


class Trainer:
    def __init__(self, model_config, train_config, weather_scaler, device="cpu"):
        self.device = torch.device(device)
        self.config = train_config
        if train_config.amp and self.device.type != "cuda":
            raise ValueError("This v1 runner supports FP16 AMP on CUDA only")
        torch.manual_seed(train_config.seed)
        self.model = ConditionalLoopDiT(model_config).to(self.device)
        self.ema = deepcopy(self.model).eval().requires_grad_(False)
        self.schedule = LinearNoiseSchedule(train_config.diffusion_steps).to(self.device)
        self.weather_scaler = weather_scaler
        # Decay matrix weights, not biases; all LayerNorms have no affine parameters.
        decay = [p for p in self.model.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.model.parameters() if p.ndim < 2]
        self.optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": train_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=train_config.learning_rate, betas=(0.9, 0.999))
        self.amp_scaler = torch.amp.GradScaler("cuda", enabled=train_config.amp)
        self.generator = torch.Generator(device=self.device).manual_seed(train_config.seed + 1)
        self.attempts = self.updates = self.skipped = 0

    def train_update(self, batches):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(self.config.accumulation_steps):
            batch = {name: value.to(self.device).float() for name, value in next(batches).items()}
            y, nwp, calendar = batch["y"], batch["nwp"], batch["calendar"]
            if y.shape != (self.config.microbatch_size, self.model.config.horizon, 1):
                raise ValueError("Training requires full microbatches with shape [B, H, 1]")
            if not all(torch.isfinite(value).all().item() for value in (y, nwp, calendar)):
                raise ValueError("v1 requires complete finite windows; no zero-filled missing data")
            if (y < 0).any() or (y > 1).any():
                raise ValueError("Power must be capacity-normalized to [0, 1]")
            nwp = self.weather_scaler.transform(nwp)
            x0 = 2 * y - 1
            t = torch.randint(1, self.schedule.steps + 1, (len(y),),
                              device=self.device, generator=self.generator)
            noise = torch.randn(x0.shape, device=self.device, generator=self.generator)
            x_t = self.schedule.add_noise(x0, t, noise)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.config.amp):
                predicted = self.model(x_t, nwp, calendar, t)
            loss = (predicted.float() - noise).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite noise MSE; stop training")
            loss_sum += loss.item()
            self.amp_scaler.scale(loss / self.config.accumulation_steps).backward()

        # Unscale exactly once, after all microbatches, before clipping.
        self.amp_scaler.unscale_(self.optimizer)
        finite = all(p.grad is None or torch.isfinite(p.grad).all().item()
                     for p in self.model.parameters())
        grad_norm = None
        if finite:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
            )
            grad_norm = norm.item()
        elif not self.config.amp:
            raise FloatingPointError("Non-finite FP32 gradients; stop training")
        old_scale = self.amp_scaler.get_scale()
        self.amp_scaler.step(self.optimizer)  # GradScaler skips non-finite AMP gradients.
        self.amp_scaler.update()
        did_update = finite and self.amp_scaler.get_scale() >= old_scale
        self.attempts += 1
        if did_update:
            self.updates += 1
            with torch.no_grad():
                for ema_parameter, parameter in zip(self.ema.parameters(), self.model.parameters()):
                    ema_parameter.lerp_(parameter, 1 - self.config.ema_decay)
                for ema_buffer, buffer in zip(self.ema.buffers(), self.model.buffers()):
                    ema_buffer.copy_(buffer)
        else:
            self.skipped += 1
        return {"loss": loss_sum / self.config.accumulation_steps, "grad_norm": grad_norm,
                "did_update": did_update, "updates": self.updates, "skipped": self.skipped}

    def save(self, path, batches, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 1, "model_config": asdict(self.model.config),
            "train_config": asdict(self.config), "model": self.model.state_dict(),
            "ema": self.ema.state_dict(), "optimizer": self.optimizer.state_dict(),
            "amp_scaler": self.amp_scaler.state_dict(),
            "weather_scaler": self.weather_scaler.state_dict(),
            "noise_rng": self.generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "rng_device_type": self.device.type,
            "data": batches.state_dict(), "metadata": metadata or {},
            "attempts": self.attempts, "updates": self.updates, "skipped": self.skipped,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        if state["rng_device_type"] != torch.device(device).type:
            raise ValueError("Exact training resume requires the same CPU/CUDA device type")
        trainer = cls(ModelConfig(**state["model_config"]), TrainConfig(**state["train_config"]),
                      WeatherScaler(**state["weather_scaler"]), device)
        trainer.model.load_state_dict(state["model"])
        trainer.ema.load_state_dict(state["ema"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.amp_scaler.load_state_dict(state["amp_scaler"])
        trainer.generator.set_state(state["noise_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        trainer.attempts, trainer.updates, trainer.skipped = (
            state["attempts"], state["updates"], state["skipped"]
        )
        return trainer, state["data"], state["metadata"]


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def run_synthetic(args, variant):
    if args.resume:
        trainer, data_state, metadata = Trainer.load(args.resume, args.device)
        model_config, config = trainer.model.config, trainer.config
        variant = model_config.variant
    else:
        model_config = ModelConfig(variant=variant)
        config = TrainConfig(amp=args.amp, seed=args.seed,
                             microbatch_size=args.microbatch_size,
                             accumulation_steps=args.accumulation_steps)
        metadata = {"data_kind": "synthetic", "data_seed": 123, "samples": 128,
                    "train_samples": 96}
    data = make_synthetic_data(model_config, metadata["samples"], metadata["data_seed"])
    training = {name: value[:metadata["train_samples"]] for name, value in data.items()}
    if not args.resume:
        trainer = Trainer(model_config, config, WeatherScaler.fit(training["nwp"]), args.device)
    batches = SyntheticBatches(training, config.microbatch_size, config.seed)
    if args.resume:
        batches.load_state_dict(data_state)
    output_dir = args.output / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    device = trainer.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    history = []
    try:
        for _ in range(args.updates):
            history.append(trainer.train_update(batches))
            print("{} update={} loss={:.6f} skipped={}".format(
                variant, trainer.updates, history[-1]["loss"], trainer.skipped
            ), flush=True)
    except (FloatingPointError, ValueError):
        trainer.save(output_dir / "failure.pt", batches, {**metadata, "failure": True})
        raise
    synchronize(device)
    training_seconds = time.perf_counter() - started
    training_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    trainer.save(output_dir / "checkpoint.pt", batches, metadata)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Two held-out synthetic conditions; no checkpoint selection or quality claim.
    offset = metadata["train_samples"]
    nwp = trainer.weather_scaler.transform(data["nwp"][offset:offset + 2].to(device))
    calendar = data["calendar"][offset:offset + 2].to(device)
    sampling_generator = torch.Generator(device=device).manual_seed(2026)
    synchronize(device)
    started = time.perf_counter()
    trajectories, diagnostics = sample_scenarios(
        trainer.ema, trainer.schedule, nwp, calendar, args.scenarios, args.sampling_steps,
        args.chunk_size, sampling_generator,
    )
    synchronize(device)
    sampling_seconds = time.perf_counter() - started
    sampling_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    torch.save(trajectories.cpu(), output_dir / "scenarios.pt")
    report = {
        "purpose": "Synthetic execution check only; not forecasting performance",
        "variant": variant, "model_config": asdict(model_config), "train_config": asdict(config),
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "device": str(device), "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "history": history, "total_updates": trainer.updates, "skipped_updates": trainer.skipped,
        "training_seconds": training_seconds, "training_peak_allocated_bytes": training_memory,
        "sampling_seconds": sampling_seconds, "sampling_peak_allocated_bytes": sampling_memory,
        "sampling_steps": args.sampling_steps, "scenario_chunk_size": args.chunk_size,
        "scenario_shape": list(trajectories.shape),
        "mean_pointwise_scenario_std": trajectories.std(dim=1, correction=0).mean().item(),
        "diagnostics": diagnostics,
        "resource_scope": "Includes live model, EMA, optimizer and buffers; not total GPU use. "
                          "Training timing includes first-update warmup.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Report: {}".format((output_dir / "report.json").resolve()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("base", "loop", "untied", "all"), default="loop")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--updates", type=int, default=6, help="Additional optimizer attempts")
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.updates < 1 or args.scenarios < 1 or args.chunk_size < 1:
        parser.error("updates, scenarios and chunk-size must be positive")
    if args.resume and args.variant == "all":
        parser.error("Resume one checkpoint at a time")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this interpreter/execution environment")
    if args.device == "cpu":
        torch.set_num_threads(1)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output = Path("outputs/synthetic") / stamp
    for variant in (("base", "loop", "untied") if args.variant == "all" else (args.variant,)):
        run_synthetic(args, variant)


if __name__ == "__main__":
    main()
