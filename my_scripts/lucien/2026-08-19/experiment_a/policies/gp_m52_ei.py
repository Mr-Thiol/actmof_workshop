"""GP Matérn 5/2 Expected-Improvement baseline."""

from __future__ import annotations

import numpy as np

from benchmark_core import greedy_diverse_batch

from .base import PolicyContext, PolicyResult, default_diag


class GPM52EIPolicy:
    name = "gp_m52_ei"
    method_label = "GP_Matern52_EI"
    family = "Acquisition Ablation"

    def select_batch(self, context: PolicyContext) -> PolicyResult:
        if context.ei is None:
            raise ValueError("EI acquisition values were not provided")
        selected = greedy_diverse_batch(
            context.candidate_idx,
            context.ei,
            context.x_scaled,
            context.batch_size,
            context.rng,
        )
        return PolicyResult(np.asarray(selected, dtype=np.int64), default_diag())
