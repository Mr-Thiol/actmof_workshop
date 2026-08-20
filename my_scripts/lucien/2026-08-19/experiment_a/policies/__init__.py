"""Experiment-A batch-selection policies."""

from __future__ import annotations

from .base import ExperimentAPolicy, PolicyContext, PolicyResult
from .gp_m52_ei import GPM52EIPolicy
from .gp_m52_pi import GPM52PIPolicy
from .imperfection_aware import ImperfectionAwarePolicy
from .pi_uncertainty import PIUncertaintyPolicy
from .random_policy import RandomPolicy

POLICY_NAMES = ("random", "gp_m52_pi", "gp_m52_ei", "pi_uncertainty", "imperfection_aware")


def make_policy(name: str, batch_size: int, exploration_slots: int) -> ExperimentAPolicy:
    if name == "random":
        return RandomPolicy()
    if name == "gp_m52_pi":
        return GPM52PIPolicy()
    if name == "gp_m52_ei":
        return GPM52EIPolicy()
    if name == "pi_uncertainty":
        return PIUncertaintyPolicy()
    if name == "imperfection_aware":
        return ImperfectionAwarePolicy(batch_size, exploration_slots)
    raise ValueError(f"Unknown Experiment-A policy: {name}")
