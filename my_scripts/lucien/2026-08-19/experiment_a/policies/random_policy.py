"""Uniform random Experiment-A baseline."""

from __future__ import annotations

import numpy as np

from .base import PolicyContext, PolicyResult, default_diag


class RandomPolicy:
    name = "random"
    method_label = "Random"
    family = "Random Baseline"

    def select_batch(self, context: PolicyContext) -> PolicyResult:
        pool = np.asarray(context.remaining_idx, dtype=np.int64)
        k = min(context.batch_size, len(pool))
        selected = context.rng.choice(pool, size=k, replace=False) if k else np.empty(0, dtype=np.int64)
        return PolicyResult(np.asarray(selected, dtype=np.int64), default_diag())
