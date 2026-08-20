"""Validated Experiment-A imperfection-aware policy."""

from __future__ import annotations

import numpy as np

from ._constrained import constrained_greedy
from .base import PolicyContext, PolicyResult


class ImperfectionAwarePolicy:
    name = "imperfection_aware"
    family = "Imperfection-aware"

    def __init__(self, batch_size: int, exploration_slots: int):
        n_trustworthy = batch_size - exploration_slots
        self.method_label = f"GP_Matern52_PI_GATE_{n_trustworthy}T{exploration_slots}S"

    def select_batch(self, context: PolicyContext) -> PolicyResult:
        n_suspicious = min(context.exploration_slots, context.batch_size)
        n_trust = context.batch_size - n_suspicious
        requirements = [False] * n_trust + [True] * n_suspicious
        selected, fallback_slots = constrained_greedy(
            context.candidate_idx,
            context.pi,
            context.x_scaled,
            context.suspicious_mask,
            requirements,
            context.rng,
            context.diversity_lambda,
        )
        selected_suspicious = int(context.suspicious_mask[selected].sum())
        return PolicyResult(selected, {
            "requested_trustworthy": n_trust,
            "requested_suspicious": n_suspicious,
            "selected_trustworthy": int(len(selected) - selected_suspicious),
            "selected_suspicious": selected_suspicious,
            "gate_fallback": bool(fallback_slots),
            "fallback_slots": ";".join(map(str, fallback_slots)),
        })
