"""Batch-allocation policies using benchmark_core's unchanged PI values."""

from __future__ import annotations

import numpy as np


def _constrained_greedy(
    candidate_idx: np.ndarray,
    acquisition_scores: np.ndarray,
    x_scaled: np.ndarray,
    suspicious_mask: np.ndarray,
    required_suspicious: list[bool],
    rng: np.random.Generator,
    diversity_lambda: float,
) -> tuple[np.ndarray, list[int]]:
    """Original greedy score with a per-slot eligibility constraint."""
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


def select_gate_batch(
    candidate_idx: np.ndarray,
    acquisition_scores: np.ndarray,
    x_scaled: np.ndarray,
    suspicious_mask: np.ndarray,
    batch_size: int,
    exploration_slots: int,
    rng: np.random.Generator,
    diversity_lambda: float,
) -> tuple[np.ndarray, dict]:
    """Select T then S slots using unchanged PI and diversity scores."""
    n_suspicious = min(exploration_slots, batch_size)
    n_trust = batch_size - n_suspicious
    requirements = [False] * n_trust + [True] * n_suspicious
    selected, fallback_slots = _constrained_greedy(
        candidate_idx, acquisition_scores, x_scaled, suspicious_mask,
        requirements, rng, diversity_lambda,
    )
    selected_suspicious = int(suspicious_mask[selected].sum())
    return selected, {
        "requested_trustworthy": n_trust,
        "requested_suspicious": n_suspicious,
        "selected_trustworthy": int(len(selected) - selected_suspicious),
        "selected_suspicious": selected_suspicious,
        "gate_fallback": bool(fallback_slots),
        "fallback_slots": ";".join(map(str, fallback_slots)),
    }


def select_gate_2t1s_batch(
    candidate_idx: np.ndarray,
    acquisition_scores: np.ndarray,
    x_scaled: np.ndarray,
    suspicious_mask: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    diversity_lambda: float,
) -> tuple[np.ndarray, dict]:
    """Backward-compatible wrapper for the original 2T+1S policy."""
    return select_gate_batch(
        candidate_idx, acquisition_scores, x_scaled, suspicious_mask,
        batch_size, exploration_slots=1, rng=rng, diversity_lambda=diversity_lambda,
    )


def select_gate_2t1_random_exploration(
    candidate_idx: np.ndarray,
    acquisition_scores: np.ndarray,
    x_scaled: np.ndarray,
    suspicious_mask: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    diversity_lambda: float,
    remaining_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Two trustworthy PI selections plus one uniform random pool selection."""
    n_pi = max(0, batch_size - 1)
    selected, fallback_slots = _constrained_greedy(
        candidate_idx, acquisition_scores, x_scaled, suspicious_mask,
        [False] * n_pi, rng, diversity_lambda,
    )
    # The random slot is drawn from the full untried pool when supplied, not
    # merely from the PI-scored acquisition subsample.
    exploration_pool = candidate_idx if remaining_idx is None else np.asarray(remaining_idx, dtype=np.int64)
    remaining = np.setdiff1d(exploration_pool, selected, assume_unique=False)
    random_slots = min(batch_size - len(selected), len(remaining))
    if random_slots:
        selected = np.concatenate([selected, rng.choice(remaining, size=random_slots, replace=False)])
    selected_suspicious = int(suspicious_mask[selected].sum())
    return selected, {
        "requested_trustworthy": n_pi,
        "requested_suspicious": 0,
        "selected_trustworthy": int(len(selected) - selected_suspicious),
        "selected_suspicious": selected_suspicious,
        "gate_fallback": bool(fallback_slots),
        "fallback_slots": ";".join(map(str, fallback_slots)),
    }
