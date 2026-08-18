"""Multiverse Orchestrator for CHIMERA Phase 5.

Executes parallel World Families (A, B, C, D), branching timelines from checkpoints,
and orchestrates invariant & chaos analysis.
"""

from __future__ import annotations
import time
import copy
from typing import List, Dict, Tuple, Optional
from packages.core.models import WorldConfig, WorldState, Vector2D, Particle
from packages.physics.engine import DeterministicEngine
from packages.multiverse.models import (
    WorldFamilyType,
    WorldFamilySpec,
    WorldBranchSpec,
    MultiverseBatchResult,
    LyapunovResult,
    InvariantResult,
)
from packages.multiverse.chaos import LyapunovCalculator
from packages.multiverse.invariants import CrossWorldInvariantDetector


class MultiverseOrchestrator:
    """Orchestrates parallel multiversal world runs across World Families A, B, C, D."""

    def __init__(self):
        self.chaos_calculator = LyapunovCalculator()
        self.invariant_detector = CrossWorldInvariantDetector()

    def run_family_a(self, spec: WorldFamilySpec) -> Tuple[MultiverseBatchResult, List[List[WorldState]]]:
        """Family A: Identical physics laws, varying random seeds (initial conditions).

        Evaluates cross-world invariants across all generated worlds.
        """
        t_start = time.perf_counter()
        world_ids = []
        world_histories: List[List[WorldState]] = []

        base_seed = spec.base_config.seed

        for i in range(spec.num_worlds):
            w_id = f"{spec.family_id}_world_{i:03d}"
            # Unique deterministic seed per world
            cfg = spec.base_config.model_copy(update={
                "world_id": w_id,
                "seed": base_seed + (i * 1007) + 1,
            })
            engine = DeterministicEngine(cfg)
            history = engine.run(spec.steps_per_world)
            world_ids.append(w_id)
            world_histories.append(history)

        # Run invariant detector across all histories
        invariants = self.invariant_detector.detect_invariants(world_histories)
        elapsed = time.perf_counter() - t_start

        batch_result = MultiverseBatchResult(
            family_id=spec.family_id,
            family_type=WorldFamilyType.FAMILY_A_INITIAL_CONDITIONS,
            total_worlds=spec.num_worlds,
            steps_per_world=spec.steps_per_world,
            world_ids=world_ids,
            elapsed_seconds=round(elapsed, 4),
            invariants=invariants,
        )

        return batch_result, world_histories

    def run_family_b(self, spec: WorldFamilySpec) -> Tuple[MultiverseBatchResult, LyapunovResult]:
        """Family B: Chaos testing via micro-perturbations to compute Lyapunov exponent."""
        t_start = time.perf_counter()

        lyap_result = self.chaos_calculator.calculate_lyapunov(
            base_config=spec.base_config,
            steps=spec.steps_per_world,
            epsilon=spec.chaos_perturbation_epsilon,
        )

        elapsed = time.perf_counter() - t_start

        batch_result = MultiverseBatchResult(
            family_id=spec.family_id,
            family_type=WorldFamilyType.FAMILY_B_CHAOS_LYAPUNOV,
            total_worlds=2,
            steps_per_world=spec.steps_per_world,
            world_ids=[lyap_result.base_world_id, lyap_result.perturbed_world_id],
            elapsed_seconds=round(elapsed, 4),
            lyapunov_summary=lyap_result,
        )

        return batch_result, lyap_result

    def run_family_c(self, spec: WorldFamilySpec) -> Tuple[MultiverseBatchResult, List[List[WorldState]]]:
        """Family C: Systematic parameter sweeps (e.g. sweeping gravity_constant or restitution)."""
        if not spec.parameter_sweep_key or not spec.parameter_sweep_values:
            raise ValueError("Family C requires parameter_sweep_key and parameter_sweep_values.")

        t_start = time.perf_counter()
        world_ids = []
        world_histories: List[List[WorldState]] = []

        sweep_key = spec.parameter_sweep_key
        sweep_vals = spec.parameter_sweep_values

        for idx, val in enumerate(sweep_vals):
            w_id = f"{spec.family_id}_sweep_{sweep_key}_{val:.3f}"
            cfg = spec.base_config.model_copy(update={
                "world_id": w_id,
                sweep_key: val,
            })
            engine = DeterministicEngine(cfg)
            history = engine.run(spec.steps_per_world)
            world_ids.append(w_id)
            world_histories.append(history)

        elapsed = time.perf_counter() - t_start

        batch_result = MultiverseBatchResult(
            family_id=spec.family_id,
            family_type=WorldFamilyType.FAMILY_C_PARAMETER_SWEEP,
            total_worlds=len(sweep_vals),
            steps_per_world=spec.steps_per_world,
            world_ids=world_ids,
            elapsed_seconds=round(elapsed, 4),
        )

        return batch_result, world_histories

    def run_family_d(
        self,
        base_config: WorldConfig,
        total_steps: int,
        branch_specs: List[WorldBranchSpec],
    ) -> Dict[str, List[WorldState]]:
        """Family D: Branching worlds from parent checkpoints with counterfactual interventions.

        Guarantees:
          - Child worlds are bitwise identical to the parent up to branch_step.
          - Interventions are applied precisely at branch_step.
          - Child worlds evolve deterministically after branch_step.

        Returns:
            Dict mapping world_id -> full trajectory List[WorldState].
        """
        results: Dict[str, List[WorldState]] = {}

        # 1. Run parent world to completion
        parent_engine = DeterministicEngine(base_config)
        parent_history = parent_engine.run(total_steps)
        results[base_config.world_id] = parent_history

        # 2. Process each branch specification
        for b_spec in branch_specs:
            b_step = min(b_spec.branch_step, total_steps)
            checkpoint_state = parent_history[b_step]

            # Construct branched initial state by applying interventions
            branched_particles = []
            for p in checkpoint_state.particles:
                v_perturb = b_spec.velocity_perturbations.get(p.id, Vector2D(x=0.0, y=0.0))
                m_perturb = b_spec.mass_perturbations.get(p.id, p.mass)

                new_p = Particle(
                    id=p.id,
                    mass=m_perturb,
                    radius=p.radius,
                    position=Vector2D(x=p.position.x, y=p.position.y),
                    velocity=Vector2D(
                        x=p.velocity.x + v_perturb.x,
                        y=p.velocity.y + v_perturb.y,
                    ),
                )
                branched_particles.append(new_p)

            child_config = base_config.model_copy(update={
                "world_id": b_spec.child_world_id,
                "gravity_constant": b_spec.new_gravity if b_spec.new_gravity is not None else base_config.gravity_constant,
            })

            branched_state = WorldState(
                world_id=b_spec.child_world_id,
                step=checkpoint_state.step,
                time=checkpoint_state.time,
                dt=checkpoint_state.dt,
                particles=branched_particles,
                boundary=checkpoint_state.boundary,
                seed=checkpoint_state.seed,
                config_hash=parent_engine.config_hash,
            )

            # Child engine initialized and restored to branch state
            child_engine = DeterministicEngine(child_config)
            child_engine.restore_state(branched_state)

            # Trajectory = parent history up to branch_step (with updated world_id) + remaining branched steps
            child_history: List[WorldState] = [
                s.model_copy(update={"world_id": b_spec.child_world_id})
                for s in parent_history[:b_step]
            ]
            child_history.append(branched_state)

            remaining_steps = total_steps - b_step
            if remaining_steps > 0:
                for _ in range(remaining_steps):
                    child_history.append(child_engine.step())

            results[b_spec.child_world_id] = child_history

        return results

    def run_family(self, spec: WorldFamilySpec) -> MultiverseBatchResult:
        """Unified runner for any WorldFamilySpec."""
        if spec.family_type == WorldFamilyType.FAMILY_A_INITIAL_CONDITIONS:
            res, _ = self.run_family_a(spec)
            return res
        elif spec.family_type == WorldFamilyType.FAMILY_B_CHAOS_LYAPUNOV:
            res, _ = self.run_family_b(spec)
            return res
        elif spec.family_type == WorldFamilyType.FAMILY_C_PARAMETER_SWEEP:
            res, _ = self.run_family_c(spec)
            return res
        elif spec.family_type == WorldFamilyType.FAMILY_D_BRANCHING_CHECKPOINT:
            if not spec.branch_specs:
                raise ValueError("Family D requires branch_specs.")
            world_dict = self.run_family_d(
                base_config=spec.base_config,
                total_steps=spec.steps_per_world,
                branch_specs=spec.branch_specs,
            )
            return MultiverseBatchResult(
                family_id=spec.family_id,
                family_type=WorldFamilyType.FAMILY_D_BRANCHING_CHECKPOINT,
                total_worlds=len(world_dict),
                steps_per_world=spec.steps_per_world,
                world_ids=list(world_dict.keys()),
                elapsed_seconds=0.0,
            )
        else:
            raise ValueError(f"Unknown family type: {spec.family_type}")
