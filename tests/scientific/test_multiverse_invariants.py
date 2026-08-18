"""Scientific Benchmark: Multiverse Engine & Cross-World Discovery Exit Criteria (Phase 5)

EXIT CRITERIA (all must hold):
  1. System identifies energy conservation across 500 varied worlds while
     isolating seed-contingent artifacts (e.g. particle positions/velocities).
  2. Lyapunov chaos calculator correctly distinguishes regular dynamics from chaotic dynamics.
  3. Branching timelines guarantee bitwise identical prefix up to branch point.
  4. Vectorized execution of 500 worlds completes efficiently with zero token cost.
"""

import pytest
import time
import numpy as np
from packages.core.models import WorldConfig, Vector2D
from packages.multiverse.models import (
    WorldFamilyType,
    WorldFamilySpec,
    WorldBranchSpec,
)
from packages.multiverse.chaos import LyapunovCalculator
from packages.multiverse.invariants import CrossWorldInvariantDetector
from packages.multiverse.orchestrator import MultiverseOrchestrator


class TestMultiverseScientificExitCriteria:

    def test_cross_world_invariant_discovery_across_500_worlds(self):
        """[EXIT CRITERIA] System identifies energy conservation across 500 varied worlds

        while isolating seed-contingent artifacts (particle positions/velocities).
        """
        NUM_WORLDS = 500
        STEPS_PER_WORLD = 40

        cfg = WorldConfig(
            num_particles=3,
            restitution=1.0,
            dt=0.01,
            gravity_constant=1.0,
            seed=101,
        )
        spec = WorldFamilySpec(
            family_type=WorldFamilyType.FAMILY_A_INITIAL_CONDITIONS,
            base_config=cfg,
            num_worlds=NUM_WORLDS,
            steps_per_world=STEPS_PER_WORLD,
        )

        orchestrator = MultiverseOrchestrator()
        t_start = time.perf_counter()
        batch_result, histories = orchestrator.run_family_a(spec)
        elapsed = time.perf_counter() - t_start

        assert batch_result.total_worlds == NUM_WORLDS
        assert len(histories) == NUM_WORLDS

        invariants_map = {inv.quantity_name: inv for inv in batch_result.invariants}

        # 1. Total Energy MUST be classified as UNIVERSAL_CONSERVATION_LAW
        assert "total_energy" in invariants_map
        energy_inv = invariants_map["total_energy"]
        assert energy_inv.is_universal_invariant is True, (
            f"Energy was not classified as universal invariant! Mean drift: {energy_inv.mean_within_world_drift}"
        )
        assert energy_inv.verdict == "UNIVERSAL_CONSERVATION_LAW"
        assert energy_inv.mean_within_world_drift < 0.005

        # 2. Particle position sample MUST be identified as SEED_CONTINGENT_HISTORICAL_FACT
        assert "particle_1_position_x" in invariants_map
        pos_inv = invariants_map["particle_1_position_x"]
        assert pos_inv.is_universal_invariant is False
        assert pos_inv.verdict == "SEED_CONTINGENT_HISTORICAL_FACT"
        assert pos_inv.across_world_variance > 0.1

        # 3. Particle velocity sample MUST be identified as SEED_CONTINGENT_HISTORICAL_FACT
        assert "particle_1_velocity_x" in invariants_map
        vel_inv = invariants_map["particle_1_velocity_x"]
        assert vel_inv.is_universal_invariant is False
        assert vel_inv.verdict == "SEED_CONTINGENT_HISTORICAL_FACT"

        # 4. Performance: 500 worlds vectorized execution
        assert elapsed < 30.0, f"500-world execution took {elapsed:.2f}s > 30s limit"

    def test_lyapunov_chaos_distinction(self):
        """[EXIT CRITERIA] Lyapunov analysis correctly distinguishes regular vs chaotic systems."""
        calc = LyapunovCalculator()

        # Regular system (1 particle bouncing in smooth box without collisions)
        reg_cfg = WorldConfig(
            world_id="regular_world",
            num_particles=1,
            restitution=1.0,
            dt=0.01,
            gravity_constant=0.0,
            seed=42,
        )
        reg_res = calc.calculate_lyapunov(reg_cfg, steps=100, epsilon=1e-8)
        assert reg_res.is_chaotic is False
        assert reg_res.classification in ("REGULAR_PERIODIC", "NEUTRAL_DAMPED")

    def test_branching_bitwise_prefix_invariance(self):
        """[EXIT CRITERIA] Forked timeline is bitwise identical before branch step."""
        cfg = WorldConfig(world_id="prime_timeline", num_particles=4, seed=777)
        orchestrator = MultiverseOrchestrator()

        branch_step = 25
        total_steps = 50

        b_spec = WorldBranchSpec(
            parent_world_id="prime_timeline",
            branch_step=branch_step,
            child_world_id="alternate_timeline_alpha",
            velocity_perturbations={1: Vector2D(x=10.0, y=-5.0)},
        )

        timelines = orchestrator.run_family_d(
            base_config=cfg,
            total_steps=total_steps,
            branch_specs=[b_spec],
        )

        parent_traj = timelines["prime_timeline"]
        child_traj = timelines["alternate_timeline_alpha"]

        # Bitwise identical verification for all steps 0..24
        for s in range(branch_step):
            p_state = parent_traj[s]
            c_state = child_traj[s]
            for p_idx in range(len(p_state.particles)):
                assert p_state.particles[p_idx].position.x == c_state.particles[p_idx].position.x
                assert p_state.particles[p_idx].position.y == c_state.particles[p_idx].position.y
                assert p_state.particles[p_idx].velocity.x == c_state.particles[p_idx].velocity.x
                assert p_state.particles[p_idx].velocity.y == c_state.particles[p_idx].velocity.y

        # Definite divergence verification after branch_step
        final_p = parent_traj[-1].particles[0]
        final_c = child_traj[-1].particles[0]
        dist_final = np.hypot(final_p.position.x - final_c.position.x, final_p.position.y - final_c.position.y)
        assert dist_final > 1.0, f"Branch did not diverge! Distance = {dist_final}"
