"""Constrained greedy helper preserved from the original Experiment-A gate."""

from __future__ import annotations

import numpy as np


def constrained_greedy(
    candidate_idx: np.ndarray,
    acquisition_scores: np.ndarray,
    x_scaled: np.ndarray,
    suspicious_mask: np.ndarray,
    required_suspicious: list[bool],
    rng: np.random.Generator,
    diversity_lambda: float,
) -> tuple[np.ndarray, list[int]]:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    acq = np.asarray(acquisition_scores, dtype=float)
    if not len(candidate_idx) or not required_suspicious:
        return np.empty(0, dtype=np.int64), []

    acq = np.where(np.isfinite(acq), acq, -np.inf)
    acq = acq + rng.normal(0.0, 1e-12, size=len(acq))
    acq_z = (acq - np.nanmean(acq)) / (np.nanstd(acq) + 1e-12)
    local_group = suspicious_mask[candidate_idx]
    remaining = np.arange(len(candidate_idx))
    chosen_local: list[int] = []
    fallback_slots: list[int] = []

    for slot, required in enumerate(required_suspicious):
        if not len(remaining):
            break
        eligible = remaining[local_group[remaining] == required]
        if not len(eligible):
            eligible = remaining
            fallback_slots.append(slot + 1)

        if not chosen_local or diversity_lambda <= 0:
            best_local = eligible[np.argmax(acq_z[eligible])]
        else:
            x_chosen = x_scaled[candidate_idx[np.asarray(chosen_local)]]
            x_eligible = x_scaled[candidate_idx[eligible]]
            distances = np.sqrt(((x_eligible[:, None, :] - x_chosen[None, :, :]) ** 2).sum(axis=2))
            score = acq_z[eligible] + diversity_lambda * distances.min(axis=1)
            best_local = eligible[np.argmax(score)]
        chosen_local.append(int(best_local))
        remaining = remaining[remaining != best_local]

    return candidate_idx[np.asarray(chosen_local, dtype=int)], fallback_slots
