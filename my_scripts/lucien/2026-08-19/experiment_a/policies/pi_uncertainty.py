"""PI exploitation plus generic GP-uncertainty exploration."""

from __future__ import annotations

import numpy as np

from benchmark_core import greedy_diverse_batch

from .base import PolicyContext, PolicyResult


class PIUncertaintyPolicy:
    name = "pi_uncertainty"
    method_label = "GP_Matern52_PI_Uncertainty"
    family = "Exploration Ablation"

    def select_batch(self, context: PolicyContext) -> PolicyResult:
        n_explore = min(context.exploration_slots, context.batch_size)
        n_pi = context.batch_size - n_explore
        selected_parts: list[np.ndarray] = []
        if n_pi:
            selected_parts.append(
                np.asarray(
                    greedy_diverse_batch(
                        context.candidate_idx,
                        context.pi,
                        context.x_scaled,
                        n_pi,
                        context.rng,
                    ),
                    dtype=np.int64,
                )
            )

        selected = (
            np.concatenate(selected_parts).astype(np.int64)
            if selected_parts
            else np.empty(0, dtype=np.int64)
        )
        if n_explore:
            blocked = np.isin(context.candidate_idx, selected)
            eligible_local = np.flatnonzero(~blocked)
            n_take = min(n_explore, len(eligible_local))
            if n_take:
                order = np.argsort(context.sd[eligible_local])[::-1]
                selected = np.concatenate([selected, context.candidate_idx[eligible_local[order[:n_take]]]])

        return PolicyResult(selected.astype(np.int64), {
            "requested_trustworthy": n_pi,
            "requested_suspicious": n_explore,
            "gate_fallback": False,
            "fallback_slots": "",
        })
