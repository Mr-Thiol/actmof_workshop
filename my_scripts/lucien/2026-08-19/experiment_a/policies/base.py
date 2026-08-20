"""Common Experiment-A policy interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PolicyContext:
    candidate_idx: np.ndarray
    remaining_idx: np.ndarray
    x_scaled: np.ndarray
    suspicious_mask: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    pi: np.ndarray
    ei: np.ndarray | None
    batch_size: int
    exploration_slots: int
    rng: np.random.Generator
    diversity_lambda: float


@dataclass
class PolicyResult:
    indices: np.ndarray
    diag: dict = field(default_factory=dict)


class ExperimentAPolicy(Protocol):
    name: str
    method_label: str
    family: str

    def select_batch(self, context: PolicyContext) -> PolicyResult:
        ...


def default_diag() -> dict:
    return {
        "requested_trustworthy": np.nan,
        "requested_suspicious": np.nan,
        "gate_fallback": False,
        "fallback_slots": "",
    }
