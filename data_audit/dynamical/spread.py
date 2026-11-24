"""Fixed 64-nearest-neighbour empirical power distributions, scored on validation."""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .freeze import digest, write

K = 64


def features(wind, day):
    # Each member's horizontal vector stays paired within its grid point and height.
    pair = np.asarray(wind, dtype=np.float64).reshape(4, 50, 9, 2, 2)
    speed = np.linalg.norm(pair, axis=-1)
    center = np.concatenate([pair.mean(axis=1), speed.mean(axis=1)[..., None]], axis=-1)
    spread = speed.std(axis=1, ddof=0)
    # Interpolate knot statistics to the midpoint of each hourly meter interval.
    mid = np.arange(24) + 0.5
    left = (mid / 3).astype(int)
    weight = (mid / 3 - left)[:, None]

    def hourly(data):
        data = np.moveaxis(data, 1, 0).reshape(9, -1)
        return data[left] * (1 - weight) + data[left + 1] * weight

    target = day + timedelta(days=1)
    annual = 2 * np.pi * ((target - date(target.year, 1, 1)).days + mid / 24) / (
        date(target.year + 1, 1, 1) - date(target.year, 1, 1)).days
    daily = mid * (2 * np.pi / 24)
    common = np.column_stack((np.sin(daily), np.cos(daily), np.sin(annual), np.cos(annual), mid + 12))
    return np.column_stack((hourly(center), common)), hourly(spread)


def load_split(root, summary, split):
    manifest = root / f"{split}.json"
    if digest(manifest) != summary["splits"][split]["sha256"]:
        raise ValueError("Split fingerprint mismatch")
    centers, spreads, targets, days = [], [], [], []
    for row in json.loads(manifest.read_bytes()):
        path = root / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Joint fingerprint mismatch")
        joint = json.loads(path.read_bytes())
        if joint["origin"][:10] != row["day"]:
            raise ValueError("Date mismatch")
        center, spread = features(joint["wind"], date.fromisoformat(row["day"]))
        centers.append(center)
        spreads.append(spread)
        targets.extend(h["MW"] for h in joint["target_MW"])
        days.append(row["day"])
    return np.vstack(centers), np.vstack(spreads), np.asarray(targets), days


def score(train, valid, y_train, y_valid):
    scale = train.std(axis=0)
    mean = train.mean(axis=0)
    x = (train - mean) / np.where(scale > 1e-12, scale, 1.)
    z = (valid - mean) / np.where(scale > 1e-12, scale, 1.)
    norm = (x*x).sum(axis=1)
    predictions = []
    for start in range(0, len(z), 64):
        distance = norm[None, :] - 2 * z[start:start+64] @ x.T
        # Stable sorting resolves ties by chronological training row.
        idx = np.argsort(distance, axis=1, kind="stable")[:, :K]
        predictions.append(np.sort(y_train[idx], axis=1))
    draws = np.vstack(predictions)
    ranks = 2*np.arange(K) - K + 1
    crps = np.abs(draws-y_valid[:, None]).mean(axis=1) - draws @ ranks / K**2
    lo, hi = np.quantile(draws, [0.05, 0.95], axis=1)
    return dict(crps_MW=crps, coverage90=(lo <= y_valid) & (y_valid <= hi), width90_MW=hi-lo)


def compare(root):
    summary = json.loads((root / "summary.json").read_bytes())
    if not summary["dataset_train_ready"]:
        raise ValueError("Dataset has unresolved windows")
    train, spread_t, y, _ = load_split(root, summary, "train")
    valid, spread_v, obs, days = load_split(root, summary, "validation")
    output = dict(split_fingerprints=summary["splits"], neighbors=K, groups={}, per_day={})
    for name, x, z in (("A", train, valid), ("B", np.column_stack((train, spread_t)),
                                             np.column_stack((valid, spread_v)))):
        scores = score(x, z, y, obs)
        output["groups"][name] = {k: float(v.mean()) for k,v in scores.items()}
        for i, day in enumerate(days):
            output["per_day"].setdefault(day, {})[name] = {
                k: float(v[24*i:24*i+24].mean()) for k,v in scores.items()}
    diffs = []
    for row in output["per_day"].values():
        delta = row["B"]["crps_MW"] - row["A"]["crps_MW"]
        row["B_minus_A_crps_MW"] = delta
        diffs.append(delta)
    output.update(mean_paired_delta_MW=float(np.mean(diffs)), days_B_better=sum(v<0 for v in diffs))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    write(args.root / "spread_validation.json", compare(args.root))
