"""FP32 DDPM forward process and deterministic DDIM with consistent clipping."""

import torch
from torch import nn


class LinearNoiseSchedule(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        if not isinstance(steps, int) or steps < 2:
            raise ValueError("Diffusion steps must be an integer >= 2")
        self.steps = steps
        beta = torch.linspace(1e-4, 0.02, steps, dtype=torch.float64)
        alpha_bar = torch.cat((torch.ones(1, dtype=torch.float64), (1 - beta).cumprod(0)))
        self.register_buffer("a", alpha_bar.sqrt().float())
        self.register_buffer("sigma", (1 - alpha_bar).sqrt().float())

    def add_noise(self, x0, t, noise):
        if t.shape != (x0.shape[0],) or t.dtype != torch.long:
            raise ValueError("t must be an int64 tensor with shape [B]")
        if (t < 1).any() or (t > self.steps).any():
            raise ValueError("Training timesteps must be in [1, T]")
        if x0.shape != noise.shape:
            raise ValueError("Noise and clean target shapes must match")
        return self.a[t, None, None] * x0.float() + self.sigma[t, None, None] * noise.float()

    def sampling_grid(self, sampling_steps):
        if not isinstance(sampling_steps, int) or not 1 <= sampling_steps <= self.steps:
            raise ValueError("sampling_steps must be an integer in [1, T]")
        grid = torch.linspace(0, self.steps, sampling_steps + 1).round().long().tolist()
        if any(s >= t for s, t in zip(grid, grid[1:])):
            raise ValueError("DDIM grid must be strictly increasing")
        return list(zip(grid[:0:-1], grid[-2::-1]))


def ddim_step(x_t, epsilon, schedule, t, s):
    if not 0 <= s < t <= schedule.steps:
        raise ValueError("DDIM step requires 0 <= s < t <= T")
    raw = (x_t.float() - schedule.sigma[t] * epsilon.float()) / schedule.a[t]
    clean = raw.clamp(-1, 1)
    consistent_noise = (x_t.float() - schedule.a[t] * clean) / schedule.sigma[t]
    x_s = schedule.a[s] * clean + schedule.sigma[s] * consistent_noise
    return x_s, raw


@torch.no_grad()
def sample_scenarios(model, schedule, nwp, calendar, scenarios=100, sampling_steps=50,
                     chunk_size=16, generator=None, initial_noise=None):
    """Return [B, M, H, 1] capacity-normalized trajectories and clipping diagnostics.

    nwp is already standardized with training-only statistics. Initial noise is
    drawn before chunking so a seed identifies the same trajectories for any chunk size.
    """
    if scenarios < 1 or chunk_size < 1:
        raise ValueError("scenarios and chunk_size must be positive")
    if not torch.isfinite(nwp).all() or not torch.isfinite(calendar).all():
        raise ValueError("Sampling requires complete finite conditions")
    pairs = schedule.sampling_grid(sampling_steps)
    batch, horizon = nwp.shape[:2]
    expected_shape = (batch, scenarios, horizon, 1)
    if initial_noise is None:
        initial_noise = torch.randn(expected_shape, generator=generator, device=nwp.device)
    if initial_noise.shape != expected_shape or not torch.isfinite(initial_noise).all():
        raise ValueError("initial_noise must be finite with shape [B, M, H, 1]")
    flat_noise = initial_noise.to(device=nwp.device, dtype=torch.float32).reshape(-1, horizon, 1)
    results = torch.empty_like(flat_noise)
    out_of_bounds = torch.zeros(len(pairs), device=nwp.device, dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        condition = model.encode_condition(nwp, calendar)  # Cached across all DDIM calls.
        for start in range(0, len(flat_noise), chunk_size):
            end = min(start + chunk_size, len(flat_noise))
            indices = torch.arange(start, end, device=nwp.device) // scenarios
            cached_condition = condition[indices]
            x = flat_noise[start:end]
            for index, (t, s) in enumerate(pairs):
                times = torch.full((len(x),), t, device=x.device, dtype=torch.long)
                epsilon = model.denoise_with_cached_condition(x, cached_condition, times)
                x, raw = ddim_step(x, epsilon, schedule, t, s)
                if not torch.isfinite(raw).all():
                    raise FloatingPointError("Non-finite DDIM prediction at t={}".format(t))
                out_of_bounds[index] += ((raw < -1) | (raw > 1)).sum()
            results[start:end] = (x + 1) / 2
    finally:
        model.train(was_training)
    trajectories = results.reshape(expected_shape)
    diagnostics = {
        "timesteps": [t for t, _ in pairs],
        "preclip_out_of_bounds_rate": (out_of_bounds.double() / results.numel()).tolist(),
        "zero_power_fraction": (trajectories == 0).float().mean().item(),
        "full_power_fraction": (trajectories == 1).float().mean().item(),
    }
    return trajectories, diagnostics
