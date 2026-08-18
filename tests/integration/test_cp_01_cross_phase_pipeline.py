"""CP-01 — Phase 2 -> Phase 3 Integration

Validates seamless integration across Phase 2 Observatory and Phase 3 Discovery:
Phase 2 Simulation -> Event-Sourced Observatory -> Observation Data -> SINDy Solver -> Symbolic Hypothesis
"""

import pytest
import numpy as np
from packages.core.models import WorldConfig, WorldState, Particle, Vector2D, Boundary
from packages.physics.engine import DeterministicEngine
from packages.physics.forces import HarmonicForce
from packages.observatory.storage import ObservatoryStorageEngine
from packages.observatory.features import ObservationMask
from packages.symbolic.sindy_solver import SINDySolver
from packages.symbolic.hypothesis import Hypothesis
from packages.symbolic.registry import HypothesisRegistry


class TestCrossPhaseIntegration:
    """Integration test suite connecting Phase 2 Observatory with Phase 3 Discovery."""

    def test_observatory_to_symbolic_discovery_flow(self):
        # 1. Setup Phase 2 Physics World with Harmonic Restoring Force (k=2.0, center=(0,0))
        huge_boundary = Boundary(x_min=-1e5, x_max=1e5, y_min=-1e5, y_max=1e5)
        config = WorldConfig(
            world_id="cp01_harmonic_world",
            seed=42,
            num_particles=1,
            dt=0.01,
            boundary=huge_boundary,
            integrator_type="verlet",
        )
        engine = DeterministicEngine(config=config)
        # Configure spring force
        engine.force_field.forces = [HarmonicForce(k=2.0, center=Vector2D(x=0.0, y=0.0))]
        # Initial displacement
        p0 = Particle(id=1, mass=1.0, position=Vector2D(x=2.0, y=0.0), velocity=Vector2D(x=0.0, y=0.0))
        engine.current_state = WorldState(
            world_id=config.world_id,
            step=0,
            time=0.0,
            dt=config.dt,
            particles=[p0],
            boundary=huge_boundary,
            seed=config.seed,
            config_hash=engine.config_hash,
        )

        # 2. Run simulation for 500 steps
        history = engine.run(500)

        # 3. Store trajectory in Phase 2 DuckDB Columnar Store
        storage = ObservatoryStorageEngine(":memory:")
        storage.store_trajectory(history)

        # 4. Query trajectory slice back from Observatory DuckDB
        queried_states = storage.query_trajectory_slice("cp01_harmonic_world")
        assert len(queried_states) == 501

        # 5. Extract blind trajectory arrays
        t_arr = np.array([s.time for s in queried_states])
        x_arr = np.array([s.particles[0].position.x for s in queried_states])
        v_arr = np.array([s.particles[0].velocity.x for s in queried_states])
        # Compute acceleration via finite differences
        a_arr = np.gradient(v_arr, config.dt)

        blind_data = {
            "world_name": "cp01_harmonic_world",
            "t": t_arr,
            "x": x_arr,
            "v": v_arr,
            "a": a_arr,
        }

        # 6. Pass blind data to Phase 3 SINDy solver
        solver = SINDySolver(threshold=0.05)
        hyp = solver.solve(blind_data)

        # 7. Validate discovered law
        assert hyp.metrics.r_squared > 0.99
        coef_x = hyp.parameters.values.get("coef_x", 0.0)
        # Discovered spring constant should be close to -2.0
        assert abs(coef_x - (-2.0)) < 0.15

        # 8. Register in SQLite hypothesis registry
        registry = HypothesisRegistry(":memory:")
        validated_hyp = hyp.validate(hyp.metrics, threshold=0.99)
        reg_id = registry.register_hypothesis(validated_hyp)
        assert registry.count_all() == 1
        assert registry.get_by_id(reg_id).status == "VALIDATED"

        storage.close()
        registry.close()
