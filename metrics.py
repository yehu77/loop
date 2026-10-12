"""Empirical-ensemble scores for single-farm trajectories; see docs/metrics.md."""

import argparse
import json
import math
from pathlib import Path

import torch


def _inputs(scenarios, targets):
    if not isinstance(scenarios, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError("Inputs must be tensors")
    if scenarios.ndim != 4 or scenarios.shape[-1] != 1 or min(scenarios.shape) < 1:
        raise ValueError("scenarios must have nonempty shape [B, M, H, 1]")
    if targets.shape != (scenarios.shape[0], scenarios.shape[2], 1):
        raise ValueError("targets must have shape [B, H, 1], aligned with scenarios")
    if scenarios.device != targets.device:
        raise ValueError("Inputs must be on the same device")
    if not scenarios.is_floating_point() or not targets.is_floating_point():
        raise ValueError("Inputs must be floating-point tensors")
    if not torch.isfinite(scenarios).all() or not torch.isfinite(targets).all():
        raise ValueError("Scores require complete finite windows; missing values are not masked")
    return scenarios[..., 0].double(), targets[..., 0].double()


@torch.no_grad()
def ensemble_crps(scenarios, targets):
    """Return [B, H] CRPS of the empirical distribution (M**2 denominator)."""
    x, y = _inputs(scenarios, targets)
    members = x.shape[1]
    ordered = x.sort(dim=1).values
    coefficients = 2 * torch.arange(1, members + 1, device=x.device) - members - 1
    # Half of the sum over all ordered pairs, computed without an M x M tensor.
    half_pair_sum = (ordered * coefficients[None, :, None]).sum(dim=1)
    return (x - y[:, None]).abs().mean(dim=1) - half_pair_sum / members**2


@torch.no_grad()
def energy_score(scenarios, targets, chunk_size=32):
    """Return [B] energy scores using the Euclidean norm over the H hours."""
    x, y = _inputs(scenarios, targets)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    members = x.shape[1]
    observation_term = torch.linalg.vector_norm(x - y[:, None], dim=-1).mean(dim=1)
    pair_sum = torch.zeros_like(observation_term)
    for start in range(0, members, chunk_size):
        # Direct distance computation avoids a nonzero diagonal from matmul roundoff.
        distances = torch.cdist(x[:, start:start + chunk_size], x, p=2,
                                compute_mode="donot_use_mm_for_euclid_dist")
        pair_sum += distances.sum(dim=(1, 2))
    return observation_term - pair_sum / (2 * members**2)


@torch.no_grad()
def variogram_score(scenarios, targets, power=0.5):
    """Return [B] VS: sum over ordered hour pairs, all off-diagonal weights = 1."""
    x, y = _inputs(scenarios, targets)
    if not math.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    if x.shape[2] < 2:
        raise ValueError("Temporal variogram score requires at least two hours")
    score = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for hour in range(x.shape[2] - 1):
        observed = (y[:, hour:hour + 1] - y[:, hour + 1:]).abs().pow(power)
        predicted = (x[:, :, hour:hour + 1] - x[:, :, hour + 1:]).abs().pow(power).mean(1)
        # Each unordered pair represents both (i,j) and (j,i) in the definition.
        score += 2 * (observed - predicted).square().sum(dim=-1)
    return score


@torch.no_grad()
def interval_statistics(scenarios, targets, level=0.9):
    """Central pointwise interval; inclusive coverage, linear empirical quantiles."""
    x, y = _inputs(scenarios, targets)
    if not math.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be finite and strictly between 0 and 1")
    tail = (1 - level) / 2
    quantiles = torch.tensor([tail, 1 - tail], device=x.device, dtype=x.dtype)
    lower, upper = torch.quantile(x, quantiles, dim=1, interpolation="linear")
    return {"lower": lower, "upper": upper, "width": upper - lower,
            "covered": (y >= lower) & (y <= upper)}


@torch.no_grad()
def evaluate_scenarios(scenarios, targets, levels=(0.8, 0.9), power=0.5, chunk_size=32):
    """JSON-ready summary; public score functions retain window-level results."""
    levels = tuple(levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("Provide at least one interval level, without duplicates")
    crps = ensemble_crps(scenarios, targets)
    energy = energy_score(scenarios, targets, chunk_size)
    variogram = variogram_score(scenarios, targets, power)
    intervals = []
    for level in levels:
        statistics = interval_statistics(scenarios, targets, level)
        covered = statistics["covered"].double()
        intervals.append({
            "nominal_level": level,
            "coverage": covered.mean().item(),
            "coverage_by_hour": covered.mean(dim=0).tolist(),
            "mean_width": statistics["width"].mean().item(),
            "mean_width_by_hour": statistics["width"].mean(dim=0).tolist(),
        })
    return {
        "schema_version": 1,
        "windows": scenarios.shape[0], "members": scenarios.shape[1],
        "horizon": scenarios.shape[2],
        "conventions": {
            "ensemble_estimator": "empirical distribution; ordered pairs / M^2",
            "score_direction": "CRPS, ES and VS: lower is better",
            "energy_norm": "Euclidean over H; no division by H or sqrt(H)",
            "variogram_power": power,
            "variogram_weights": "1 for all ordered off-diagonal hour pairs; no normalization",
            "intervals": "central, pointwise, linear quantiles, inclusive endpoints",
            "input_scale": "unchanged; caller must use the same power units for all models",
            "alignment": "caller must align farm, forecast issue and valid times before scoring",
        },
        "crps": crps.mean().item(), "crps_by_hour": crps.mean(dim=0).tolist(),
        "energy_score": energy.mean().item(), "variogram_score": variogram.mean().item(),
        "intervals": intervals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True, help="Tensor .pt: [B, M, H, 1]")
    parser.add_argument("--targets", type=Path, required=True, help="Aligned tensor .pt: [B, H, 1]")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.8, 0.9])
    parser.add_argument("--variogram-power", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are not overwritten")
    args = parser.parse_args()
    scenarios = torch.load(args.scenarios, map_location="cpu", weights_only=True)
    targets = torch.load(args.targets, map_location="cpu", weights_only=True)
    report = evaluate_scenarios(scenarios, targets, args.levels, args.variogram_power, args.chunk_size)
    report["inputs"] = {"scenarios": str(args.scenarios), "targets": str(args.targets)}
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
        print("Report: {}".format(args.output.resolve()))
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
