"""GP Matérn 5/2 Probability-of-Improvement baseline."""

from __future__ import annotations

import numpy as np

from benchmark_core import METHOD, greedy_diverse_batch

from .base import PolicyContext, PolicyResult, default_diag


class GPM52PIPolicy:
    name = "gp_m52_pi"
    method_label = METHOD
    family = "Acquisition Ablation"

    def select_batch(self, context: PolicyContext) -> PolicyResult:
        selected = greedy_diverse_batch(
            context.candidate_idx,
            context.pi,
            context.x_scaled,
            context.batch_size,
            context.rng,
        )
        return PolicyResult(np.asarray(selected, dtype=np.int64), default_diag())
