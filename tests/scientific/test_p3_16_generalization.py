"""P3-16 — Generalization Test

Verifies that a discovered symbolic law accurately generalizes to independent
validation trajectories generated from different initial conditions.
"""

import pytest
import numpy as np
from packages.symbolic.benchmark_worlds import (
    BenchmarkWorldSpec,
    _integrate_harmonic,
    _integrate_damped,
)
from packages.symbolic.sindy_solver import SINDySolver


class TestSymbolicGeneralization:
    """Scientific test suite evaluating model generalization to independent trajectories."""

    def test_harmonic_oscillator_generalization(self):
        # 1. Generate training trajectory from seed 7
        spec_train = BenchmarkWorldSpec(
            name="harmonic_spring",
            description="Train harmonic",
            hidden_params={"k": 3.0, "x_eq": 0.0},
            num_particles=1,
            num_steps=500,
            dt=0.01,
            seed=7,
        )
        raw_train = _integrate_harmonic(spec_train)
        train_data = {
            "world_name": "harmonic_spring",
            "t": raw_train["t"],
            "x": raw_train["x"],
            "v": raw_train["v"],
            "a": raw_train["a"],
        }

        # 2. Discover law on training trajectory
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(train_data)
        assert hyp.metrics.r_squared > 0.99
        coef_x = hyp.parameters.values["coef_x"]

        # 3. Generate independent validation trajectory from completely different seed 999 with different initial displacement
        spec_val = BenchmarkWorldSpec(
            name="harmonic_spring",
            description="Val harmonic",
            hidden_params={"k": 3.0, "x_eq": 0.0},
            num_particles=1,
            num_steps=500,
            dt=0.01,
            seed=999,
        )
        raw_val = _integrate_harmonic(spec_val)
        x_val = raw_val["x"]
        a_val_true = raw_val["a"]

        # 4. Predict acceleration on independent validation trajectory using discovered law
        a_val_pred = coef_x * x_val

        # Compute generalization R^2
        ss_res = np.sum((a_val_true - a_val_pred) ** 2)
        ss_tot = np.sum((a_val_true - np.mean(a_val_true)) ** 2)
        generalization_r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

        # Scientific Acceptance Criteria: Generalization R^2 > 0.99 on unseen trajectory
        assert generalization_r2 > 0.99, f"Generalization failed: R^2 = {generalization_r2:.4f}"
