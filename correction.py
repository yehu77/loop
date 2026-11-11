"""Frozen EMA proposal followed by shared prediction-space correction (FP32)."""
from pathlib import Path
import json

import torch
from torch import nn

from data import file_hash
from model import ConditionalLoopDiT, ModelConfig, sinusoidal_embedding
from train import WeatherScaler, TrainConfig


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_config(path):
    config = json.loads(Path(path).read_text())
    require(bool(config.get("source_sha256")), "Missing source freeze")
    for name, digest in config["source_sha256"].items():
        require(file_hash(name) == digest, "Frozen source mismatch: " + name)
    return config


def load_base(config, package, device):
    """Inference-only provenance check. Never restores D's optimizer or RNG."""
    spec = config["base"]
    require(file_hash(spec["checkpoint"]) == spec["sha256"], "BASE checkpoint hash mismatch")
    require(file_hash(spec["archive_config"]) == spec["archive_config_sha256"], "BASE archive mismatch")
    archive = json.loads(Path(spec["archive_config"]).read_text())
    state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    meta, manifest = state["metadata"], package["manifest"]
    require(not meta.get("failure", False), "Failure BASE checkpoint")
    require(state["model_config"] == archive["model"] and state["model_config"]["variant"] == "base"
            and state["model_config"]["history_group"] == "H0", "BASE architecture mismatch")
    require(state["updates"] == 3000 and state["ema_initialized"]
            and state["ema_num_averaged_updates"] == 2000, "BASE EMA clock mismatch")
    require(meta["data_kind"] == "real" and meta["data_fingerprint"] == manifest["data_fingerprint"]
            == config["data_fingerprint"] == archive["data_fingerprint"], "BASE data mismatch")
    require(meta["protocol_fingerprint"] == manifest["protocol_fingerprint"]
            == config["protocol_fingerprint"], "BASE protocol mismatch")
    require(meta["run_config"] == archive["evaluation"] and TrainConfig(**state["train_config"]) == TrainConfig(**archive["training"]),
            "BASE archived training/source identity mismatch")
    scaler = WeatherScaler(*(torch.tensor(manifest["scaler"][k]) for k in ("mean", "std")))
    require(all(torch.equal(state["weather_scaler"][k], v) for k, v in scaler.state_dict().items()),
            "BASE standardization mismatch")
    # Construct without consuming the new experiment's global initialization stream.
    with torch.random.fork_rng(devices=[]):
        base = ConditionalLoopDiT(ModelConfig(**state["model_config"]))
    base.load_state_dict(state["ema"], strict=True)
    return base.to(device).eval().requires_grad_(False), scaler


class Corrector(ConditionalLoopDiT):
    """One two-block Core, fresh hidden state each call; output is delta in x0 units."""
    def __init__(self, config=None):
        config = config or ModelConfig(variant="base", rounds=1)
        require(config.variant == "base" and config.history_group is None
                and config.rounds == 1 and config.attention_scope == "full", "C must be a single full-attention Core")
        super().__init__(config)
        self.position_encoding.copy_(sinusoidal_embedding(
            torch.arange(1, config.horizon + 1), config.hidden_dim).unsqueeze(0))

    def delta(self, proposal, nwp, calendar, t):
        return self.denoise_with_cached_condition(proposal, self.encode_condition(nwp, calendar), t)


def make_corrector(config, device):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seeds"]["corrector_initialization"])
        corrector = Corrector(ModelConfig(**config["corrector"]))
    return corrector.to(device)


class CorrectedDenoiser(nn.Module):
    """Flat sampler cache; K is independent of the old network's rounds."""
    def __init__(self, base, corrector, schedule, correction_steps=1):
        super().__init__()
        require(type(correction_steps) is int and correction_steps in (0, 1, 2), "K must be 0, 1 or 2")
        self.base = base.eval().requires_grad_(False)
        self.corrector, self.schedule = corrector, schedule
        self.correction_steps = correction_steps

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def encode_condition(self, nwp, calendar, history_power=None):
        with torch.no_grad():
            cache = self.base.encode_condition(nwp, calendar, history_power)
        flat = {"base_" + k: v for k, v in cache.items()}
        if self.correction_steps:
            flat["corrector_condition"] = self.corrector.encode_condition(nwp, calendar)
        return flat

    def denoise_with_cached_condition(self, x_t, condition, t):
        require(t.dtype == torch.long and t.shape == (len(x_t),)
                and bool(((t >= 1) & (t <= self.schedule.steps)).all()), "Correction requires t in [1,T]")
        with torch.no_grad():
            epsilon = self.base.denoise_with_cached_condition(
                x_t, {k[5:]: v for k, v in condition.items() if k.startswith("base_")}, t)
            if self.correction_steps == 0:
                return epsilon
            a, sigma = self.schedule.a[t, None, None], self.schedule.sigma[t, None, None]
            proposal0 = (x_t - sigma * epsilon) / a
        delta_sum = torch.zeros_like(proposal0)
        for _ in range(self.correction_steps):
            delta_sum = delta_sum + self.corrector.denoise_with_cached_condition(
                proposal0 + delta_sum, condition["corrector_condition"], t)
        return epsilon - (a / sigma) * delta_sum

    def forward(self, x_t, nwp, calendar, t, history_power=None):
        return self.denoise_with_cached_condition(x_t, self.encode_condition(nwp, calendar, history_power), t)
