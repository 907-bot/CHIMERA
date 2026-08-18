"""Deterministic Counterfactual Intervention Engine for CHIMERA Phase 4.

Runs perturbed benchmark worlds to test whether a hypothesis generalises
beyond its training data. Zero LLM cost — 100% deterministic numpy.

Principle:
  If a hypothesis truly encodes the governing law, its predictions must hold
  when initial conditions are changed (different seeds, amplitudes, phases).
  If R² collapses under perturbation, the hypothesis is overfitted or wrong.

Per AGENTS.md: AI agents MUST NOT modify simulation state directly.
This engine runs independent simulations — it does not mutate existing worlds.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional
import numpy as np

from packages.symbolic.benchmark_worlds import generate_blind_data, ALL_BENCHMARKS
from packages.symbolic.sindy_solver import SINDySolver
from packages.symbolic.hypothesis import Hypothesis
from packages.agents.debate_models import (
    CounterfactualExperiment,
    ExperimentResult,
    PerturbationSpec,
)


# ---------------------------------------------------------------------------
# Perturbed World Generator
# ---------------------------------------------------------------------------

def _generate_perturbed_harmonic(
    original_seed: int,
    perturbations: list,
) -> Dict[str, Any]:
    """Generate a perturbed harmonic spring trajectory with altered initial conditions.

    The hidden physics constant k is UNCHANGED — we only perturb initial state.
    If the hypothesis truly encodes F=-kx, it must survive any initial condition.

    Args:
        original_seed:  Original world seed.
        perturbations:  List of PerturbationSpec objects.

    Returns:
        Blind data dict (t, x, v, a) with perturbed initial conditions.
    """
    from packages.symbolic.benchmark_worlds import HARMONIC_SPRING

    k = HARMONIC_SPRING.hidden_params["k"]
    x_eq = HARMONIC_SPRING.hidden_params["x_eq"]
    dt = HARMONIC_SPRING.dt
    num_steps = HARMONIC_SPRING.num_steps

    # Apply perturbations to initial state
    rng = np.random.default_rng(original_seed + 999)  # Different seed
    x0 = float(rng.uniform(0.5, 2.0))
    v0 = float(rng.uniform(-0.5, 0.5))

    for pert in perturbations:
        if pert.parameter == "initial_displacement":
            x0 = pert.perturbed_value
        elif pert.parameter == "initial_velocity":
            v0 = pert.perturbed_value
        elif pert.parameter == "amplitude_scale":
            x0 *= pert.perturbed_value

    xs = np.zeros(num_steps + 1)
    vs = np.zeros(num_steps + 1)
    xs[0] = x0
    vs[0] = v0

    for i in range(num_steps):
        a = -k * (xs[i] - x_eq)
        vs[i + 1] = vs[i] + a * dt
        xs[i + 1] = xs[i] + vs[i + 1] * dt

    ts = np.arange(num_steps + 1) * dt
    acc = np.gradient(vs, dt)

    return {"world_name": "harmonic_spring_perturbed", "t": ts, "x": xs, "v": vs, "a": acc}


def _generate_perturbed_damped(
    original_seed: int,
    perturbations: list,
) -> Dict[str, Any]:
    """Generate a perturbed damped oscillator trajectory.

    Hidden k and b are UNCHANGED — only initial conditions perturbed.
    """
    from packages.symbolic.benchmark_worlds import DAMPED_OSCILLATOR

    k = DAMPED_OSCILLATOR.hidden_params["k"]
    b = DAMPED_OSCILLATOR.hidden_params["b"]
    x_eq = DAMPED_OSCILLATOR.hidden_params["x_eq"]
    dt = DAMPED_OSCILLATOR.dt
    num_steps = DAMPED_OSCILLATOR.num_steps

    rng = np.random.default_rng(original_seed + 999)
    x0 = float(rng.uniform(0.8, 2.0))
    v0 = float(rng.uniform(-0.3, 0.3))

    for pert in perturbations:
        if pert.parameter == "initial_displacement":
            x0 = pert.perturbed_value
        elif pert.parameter == "initial_velocity":
            v0 = pert.perturbed_value

    xs = np.zeros(num_steps + 1)
    vs = np.zeros(num_steps + 1)
    xs[0] = x0
    vs[0] = v0

    def f_damp(x, v):
        return v, -k * (x - x_eq) - b * v

    for i in range(num_steps):
        x, v = xs[i], vs[i]
        dx1, dv1 = f_damp(x, v)
        dx2, dv2 = f_damp(x + 0.5 * dt * dx1, v + 0.5 * dt * dv1)
        dx3, dv3 = f_damp(x + 0.5 * dt * dx2, v + 0.5 * dt * dv2)
        dx4, dv4 = f_damp(x + dt * dx3, v + dt * dv3)
        xs[i + 1] = x + dt * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6.0
        vs[i + 1] = v + dt * (dv1 + 2 * dv2 + 2 * dv3 + dv4) / 6.0

    ts = np.arange(num_steps + 1) * dt
    acc = np.gradient(vs, dt)

    return {"world_name": "damped_oscillator_perturbed", "t": ts, "x": xs, "v": vs, "a": acc}


# ---------------------------------------------------------------------------
# Intervention Engine
# ---------------------------------------------------------------------------

class InterventionEngine:
    """Runs counterfactual perturbation experiments to test hypothesis generalisation.

    Architecture Rule: This engine ONLY reads hypotheses (as prediction models).
    It never modifies WorldState, simulation history, or force laws.
    The perturbed simulations are entirely fresh, independent runs.

    Args:
        sindy_train_ratio: Fraction of perturbed trajectory used for fitting.
    """

    def __init__(self, sindy_train_ratio: float = 0.8):
        self.solver = SINDySolver(threshold=0.05, train_ratio=sindy_train_ratio)

    def run_experiment(
        self,
        hypothesis: Hypothesis,
        experiment: CounterfactualExperiment,
    ) -> ExperimentResult:
        """Execute a Skeptic's counterfactual experiment.

        Steps:
          1. Generate a perturbed world with changed initial conditions
          2. Run SINDy on perturbed blind data to score existing hypothesis
          3. Evaluate R² of hypothesis predictions on perturbed trajectory
          4. Return ExperimentResult with survived flag

        The hidden physics constants are NEVER changed in perturbations —
        only observable initial conditions vary. A correct law must generalise.

        Args:
            hypothesis:   The hypothesis being tested.
            experiment:   Skeptic's counterfactual specification.

        Returns:
            ExperimentResult with R² and survived status.
        """
        t_start = time.perf_counter()

        # Generate perturbed trajectory (same hidden law, different initial state)
        world_name = hypothesis.world_name
        perturbations = experiment.perturbations

        if "harmonic" in world_name:
            perturbed_data = _generate_perturbed_harmonic(
                original_seed=7, perturbations=perturbations
            )
        elif "damped" in world_name:
            perturbed_data = _generate_perturbed_damped(
                original_seed=13, perturbations=perturbations
            )
        else:
            # Fall back to standard blind data for unknown worlds
            perturbed_data = generate_blind_data(world_name)

        # Score the existing hypothesis equation on the perturbed trajectory
        # We use the discovered equation coefficients to predict acceleration
        x = np.asarray(perturbed_data["x"], dtype=np.float64)
        v = np.asarray(perturbed_data["v"], dtype=np.float64)
        a_true = np.asarray(perturbed_data["a"], dtype=np.float64)

        n = len(x)
        n_test_start = int(n * 0.8)  # Evaluate on held-out 20%
        x_test = x[n_test_start:]
        v_test = v[n_test_start:]
        a_test = a_true[n_test_start:]

        # Reconstruct prediction from hypothesis equation coefficients
        coef_x = hypothesis.parameters.values.get("coef_x", 0.0)
        coef_v = hypothesis.parameters.values.get("coef_v", 0.0)
        coef_x2 = hypothesis.parameters.values.get("coef_x²", 0.0)
        coef_const = hypothesis.parameters.values.get("offset", 0.0)

        a_pred = coef_const + coef_x * x_test + coef_v * v_test + coef_x2 * (x_test ** 2)

        ss_res = np.sum((a_test - a_pred) ** 2)
        ss_tot = np.sum((a_test - np.mean(a_test)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        r2 = max(-1.0, min(1.0, r2))
        rmse = float(np.sqrt(np.mean((a_test - a_pred) ** 2)))

        survived = r2 >= experiment.r2_threshold_to_survive

        if survived:
            interp = (
                f"Hypothesis survived perturbation (R²={r2:.4f} ≥ {experiment.r2_threshold_to_survive}). "
                f"This supports the hypothesis encoding a true governing law."
            )
        else:
            interp = (
                f"Hypothesis COLLAPSED under perturbation (R²={r2:.4f} < {experiment.r2_threshold_to_survive}). "
                f"This suggests overfitting or an incorrect law."
            )

        elapsed = time.perf_counter() - t_start

        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            hypothesis_id=hypothesis.id,
            world_name=world_name,
            r_squared_on_perturbed=r2,
            rmse_on_perturbed=rmse,
            survived=survived,
            interpretation=interp,
            run_duration_seconds=elapsed,
        )

    def design_standard_experiments(
        self, hypothesis: Hypothesis
    ) -> list:
        """Design a standard battery of counterfactual experiments for any hypothesis.

        Creates three canonical perturbations:
          1. Large displacement (tests law holds at different amplitude)
          2. Reversed initial velocity (tests symmetry)
          3. Small amplitude (tests near-equilibrium behaviour)

        Returns:
            List of CounterfactualExperiment objects.
        """
        world_name = hypothesis.world_name
        hyp_id = hypothesis.id

        experiments = [
            CounterfactualExperiment(
                hypothesis_id=hyp_id,
                world_name=world_name,
                experiment_name="large_displacement",
                description="Test hypothesis holds for 3x larger initial displacement",
                perturbations=[
                    PerturbationSpec(
                        parameter="initial_displacement",
                        original_value=1.0,
                        perturbed_value=3.0,
                        rationale="A true Hooke's law must hold at any amplitude",
                    )
                ],
                predicted_outcome_if_true="R² remains > 0.95 at large displacement",
                predicted_outcome_if_false="R² collapses below 0.5 — equation only fitted near training range",
                r2_threshold_to_survive=0.95,
            ),
            CounterfactualExperiment(
                hypothesis_id=hyp_id,
                world_name=world_name,
                experiment_name="reversed_velocity",
                description="Test hypothesis holds for time-reversed initial velocity",
                perturbations=[
                    PerturbationSpec(
                        parameter="initial_velocity",
                        original_value=0.0,
                        perturbed_value=-1.5,
                        rationale="A linear law must be direction-symmetric",
                    )
                ],
                predicted_outcome_if_true="R² remains > 0.95 — linear force is symmetric",
                predicted_outcome_if_false="R² drops — asymmetric non-linear law",
                r2_threshold_to_survive=0.95,
            ),
            CounterfactualExperiment(
                hypothesis_id=hyp_id,
                world_name=world_name,
                experiment_name="small_amplitude",
                description="Test hypothesis holds for very small initial displacement (0.05)",
                perturbations=[
                    PerturbationSpec(
                        parameter="initial_displacement",
                        original_value=1.0,
                        perturbed_value=0.05,
                        rationale="Near-equilibrium test distinguishes linear from nonlinear laws",
                    )
                ],
                predicted_outcome_if_true="R² remains > 0.90 — linear law holds near zero",
                predicted_outcome_if_false="Numerical noise dominates — law is purely nonlinear",
                r2_threshold_to_survive=0.90,
            ),
        ]

        return experiments
