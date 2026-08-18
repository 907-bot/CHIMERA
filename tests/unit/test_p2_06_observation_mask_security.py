"""P2-06 — Observation Mask / Information Leakage

Verifies that observation masking sanitizes simulation states and withholds
internal/hidden parameters from public BlindObservation instances.
"""

import pytest
from packages.core.models import WorldConfig, WorldState
from packages.physics.engine import DeterministicEngine
from packages.observatory.features import ObservationMask, BlindObservation


class TestObservationMaskSecurity:
    """Test suite for observation masking and anti-information-leakage enforcement."""

    def test_mask_filters_hidden_and_internal_fields(self):
        config = WorldConfig(
            world_id="mask_sec_world",
            seed=42,
            num_particles=4,
            dt=0.01,
            gravity_constant=6.67430,
            softening=0.25,
        )
        engine = DeterministicEngine(config=config)
        state = engine.step()

        blind_obs = ObservationMask.mask_state(state)

        # 1. Verify allowed public fields
        allowed_fields = {
            "world_id",
            "step",
            "time",
            "dt",
            "particles_positions",
            "particles_velocities",
            "spatial_entropy",
            "kinetic_energy",
            "total_momentum",
        }
        obs_dict = blind_obs.model_dump()
        assert set(obs_dict.keys()) == allowed_fields

        # 2. Verify private/hidden internal simulation fields are completely absent
        forbidden_fields = [
            "gravity_constant",
            "G",
            "k",
            "GM",
            "softening",
            "force_field",
            "restitution",
            "forces",
            "force_x",
            "force_y",
            "integrator_type",
            "boundary",
            "config_hash",
            "seed",
        ]
        for f in forbidden_fields:
            assert f not in obs_dict, f"Forbidden internal parameter '{f}' leaked in BlindObservation dictionary!"
            assert not hasattr(blind_obs, f), f"Forbidden attribute '{f}' present on BlindObservation object!"

    def test_blind_observation_immutability(self):
        config = WorldConfig(world_id="immut_world", seed=1, num_particles=2)
        state = DeterministicEngine(config).current_state
        blind_obs = ObservationMask.mask_state(state)

        # ConfigDict(frozen=True) prevents mutating observable fields
        with pytest.raises(Exception):
            blind_obs.spatial_entropy = 999.0  # Attempt mutation
