"""In-World Observer Agent for CHIMERA Phases 9 & 10.

Simulates in-world scientific observers that conduct their own sensor measurements,
execute local controlled experiments, formulate mathematical theories, and participate in peer review.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from packages.civilization.models import (
    InWorldObserver,
    CivilizationExperiment,
    CivilizationTheory,
)


class InWorldObserverAgent:
    """An autonomous in-world scientist agent that derives laws from internal measurements."""

    def __init__(self, observer: InWorldObserver, rng: Optional[np.random.Generator] = None):
        self.observer = observer
        self.rng = rng or np.random.default_rng(42)

    def conduct_harmonic_physics_experiment(
        self,
        true_k: float = 3.0,
        num_samples: int = 100,
    ) -> Tuple[CivilizationExperiment, CivilizationTheory]:
        """Conduct an in-world physical experiment on harmonic motion with instrument noise."""
        # Observer takes noisy observations: a = -k*x + noise
        x_obs = self.rng.uniform(-2.0, 2.0, size=num_samples)
        noise = self.rng.normal(0.0, self.observer.measurement_noise_std, size=num_samples)
        a_obs = -true_k * x_obs + noise

        # In-world linear regression
        slope, intercept = np.polyfit(x_obs, a_obs, 1)
        k_measured = -float(slope)

        # Compute R^2
        a_pred = -k_measured * x_obs
        ss_res = np.sum((a_obs - a_pred) ** 2)
        ss_tot = np.sum((a_obs - np.mean(a_obs)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

        exp = CivilizationExperiment(
            observer_id=self.observer.observer_id,
            target_phenomenon="Harmonic Restoring Force",
            intervention_type="Displacement Perturbation",
            sample_size=num_samples,
            measured_r_squared=round(r2, 4),
            conclusion=f"Acceleration is linearly proportional to displacement (k={k_measured:.3f}, R²={r2:.4f}).",
        )

        theory = CivilizationTheory(
            author_observer_id=self.observer.observer_id,
            title="Hookean Force Invariant",
            mathematical_formula=f"F = -{k_measured:.3f} * x",
            domain="PHYSICS",
            evidence_experiments=[exp.experiment_id],
            consensus_score=0.0,  # Awaiting peer review
            status="PEER_REVIEW",
        )

        self.observer.theories_formulated += 1
        return exp, theory

    def review_peer_theory(self, theory: CivilizationTheory, true_k: float = 3.0) -> bool:
        """Vote on a proposed theory by testing it against the reviewer's independent measurements."""
        # Independent replication
        x_test = self.rng.uniform(-2.0, 2.0, size=50)
        noise = self.rng.normal(0.0, self.observer.measurement_noise_std, size=50)
        a_true = -true_k * x_test + noise

        # Extract coefficient from formula if present
        try:
            # E.g. "F = -3.001 * x"
            k_val = float(theory.mathematical_formula.split("*")[0].replace("F = -", "").strip())
            a_pred = -k_val * x_test
            r2 = float(1.0 - np.sum((a_true - a_pred) ** 2) / (np.sum((a_true - np.mean(a_true)) ** 2) + 1e-12))
            return r2 > 0.90
        except Exception:
            return False
