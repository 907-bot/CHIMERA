"""Unit Tests for CHIMERA Multiverse Engine (Phase 5).

Covers:
  - WorldFamilySpec and WorldBranchSpec data models
  - Family A (varying initial conditions & invariant evaluation)
  - Family B (Lyapunov chaos computation)
  - Family C (Parameter sweep generation)
  - Family D (Checkpoint branching with bitwise prefix integrity)
  - CrossWorldInvariantDetector classification
"""

import pytest
import numpy as np
from packages.core.models import WorldConfig, Vector2D
from packages.multiverse.models import (
    WorldFamilyType,
    WorldFamilySpec,
    WorldBranchSpec,
    LyapunovResult,
    InvariantResult,
    MultiverseBatchResult,
)
from packages.multiverse.chaos import LyapunovCalculator
from packages.multiverse.invariants import CrossWorldInvariantDetector
from packages.multiverse.orchestrator import MultiverseOrchestrator


class TestMultiverseModels:

    def test_world_family_spec_creation(self):
        cfg = WorldConfig(num_particles=4, seed=42)
        spec = WorldFamilySpec(
            family_type=WorldFamilyType.FAMILY_A_INITIAL_CONDITIONS,
            base_config=cfg,
            num_worlds=5,
            steps_per_world=50,
        )
        assert spec.family_id.startswith("fam_")
        assert spec.num_worlds == 5
        assert spec.steps_per_world == 50

    def test_branch_spec_creation(self):
        b_spec = WorldBranchSpec(
            parent_world_id="world_parent",
            branch_step=10,
            velocity_perturbations={1: Vector2D(x=2.0, y=0.0)},
        )
        assert b_spec.parent_world_id == "world_parent"
        assert b_spec.branch_step == 10
        assert 1 in b_spec.velocity_perturbations


class TestLyapunovCalculator:

    def test_lyapunov_calculation_returns_result(self):
        cfg = WorldConfig(num_particles=3, seed=42, dt=0.01)
        calc = LyapunovCalculator()
        res = calc.calculate_lyapunov(cfg, steps=60, epsilon=1e-8)

        assert isinstance(res, LyapunovResult)
        assert res.base_world_id == cfg.world_id
        assert len(res.divergence_history) == 61
        assert res.classification in ("CHAOTIC", "REGULAR_PERIODIC", "NEUTRAL_DAMPED")


class TestCrossWorldInvariantDetector:

    def test_invariant_detector_on_elastic_worlds(self):
        """Total energy must be classified as UNIVERSAL_CONSERVATION_LAW across conservative worlds."""
        cfg = WorldConfig(num_particles=3, restitution=1.0, seed=10, dt=0.01)
        orchestrator = MultiverseOrchestrator()
        spec = WorldFamilySpec(
            family_type=WorldFamilyType.FAMILY_A_INITIAL_CONDITIONS,
            base_config=cfg,
            num_worlds=5,
            steps_per_world=50,
        )
        batch_res, _ = orchestrator.run_family_a(spec)

        assert batch_res.invariants is not None
        invariants_by_name = {inv.quantity_name: inv for inv in batch_res.invariants}

        # Energy should be conserved
        assert "total_energy" in invariants_by_name
        energy_inv = invariants_by_name["total_energy"]
        assert energy_inv.is_universal_invariant is True
        assert energy_inv.verdict == "UNIVERSAL_CONSERVATION_LAW"

        # Position of particle 1 should NOT be a universal invariant (seed-contingent)
        assert "particle_1_position_x" in invariants_by_name
        pos_inv = invariants_by_name["particle_1_position_x"]
        assert pos_inv.is_universal_invariant is False
        assert pos_inv.verdict == "SEED_CONTINGENT_HISTORICAL_FACT"


class TestMultiverseOrchestrator:

    def test_family_a_execution(self):
        cfg = WorldConfig(num_particles=2, seed=7)
        spec = WorldFamilySpec(
            family_type=WorldFamilyType.FAMILY_A_INITIAL_CONDITIONS,
            base_config=cfg,
            num_worlds=4,
            steps_per_world=30,
        )
        orchestrator = MultiverseOrchestrator()
        batch_res, histories = orchestrator.run_family_a(spec)

        assert batch_res.total_worlds == 4
        assert len(histories) == 4
        assert len(histories[0]) == 31
        assert batch_res.invariants is not None

    def test_family_b_chaos_execution(self):
        cfg = WorldConfig(num_particles=2, seed=7)
        spec = WorldFamilySpec(
            family_type=WorldFamilyType.FAMILY_B_CHAOS_LYAPUNOV,
            base_config=cfg,
            steps_per_world=40,
        )
        orchestrator = MultiverseOrchestrator()
        batch_res, lyap_res = orchestrator.run_family_b(spec)

        assert batch_res.total_worlds == 2
        assert lyap_res.base_world_id == cfg.world_id
        assert len(lyap_res.divergence_history) == 41

    def test_family_c_parameter_sweep(self):
        cfg = WorldConfig(num_particles=2, seed=7)
        spec = WorldFamilySpec(
            family_type=WorldFamilyType.FAMILY_C_PARAMETER_SWEEP,
            base_config=cfg,
            parameter_sweep_key="gravity_constant",
            parameter_sweep_values=[0.0, 5.0, 10.0],
            steps_per_world=20,
        )
        orchestrator = MultiverseOrchestrator()
        batch_res, histories = orchestrator.run_family_c(spec)

        assert batch_res.total_worlds == 3
        assert len(histories) == 3

    def test_family_d_checkpoint_branching(self):
        """Guarantees bitwise prefix equality before branch_step and divergence after."""
        cfg = WorldConfig(world_id="parent_world", num_particles=2, seed=100)
        orchestrator = MultiverseOrchestrator()

        branch_spec = WorldBranchSpec(
            parent_world_id="parent_world",
            branch_step=15,
            child_world_id="child_world_1",
            velocity_perturbations={1: Vector2D(x=5.0, y=5.0)},
        )

        world_dict = orchestrator.run_family_d(
            base_config=cfg,
            total_steps=30,
            branch_specs=[branch_spec],
        )

        assert "parent_world" in world_dict
        assert "child_world_1" in world_dict

        parent_hist = world_dict["parent_world"]
        child_hist = world_dict["child_world_1"]

        assert len(parent_hist) == 31
        assert len(child_hist) == 31

        # Check prefix: up to step 15, particle coordinates must be bitwise identical
        for step in range(15):
            p_parent = parent_hist[step].particles[0]
            p_child = child_hist[step].particles[0]
            assert p_parent.position.x == p_child.position.x
            assert p_parent.position.y == p_child.position.y
            assert p_parent.velocity.x == p_child.velocity.x
            assert p_parent.velocity.y == p_child.velocity.y

        # Check divergence: at step 30, state MUST diverge due to intervention
        final_parent = parent_hist[30].particles[0]
        final_child = child_hist[30].particles[0]
        assert (final_parent.position.x != final_child.position.x) or (final_parent.velocity.x != final_child.velocity.x)
