"""Serialization, State Hashing, and Bitwise Equality Functions"""

import hashlib
import json
from typing import Dict, Any
from packages.core.models import WorldState, WorldConfig, Particle, Vector2D, Boundary


def particle_to_dict(particle: Particle) -> Dict[str, Any]:
    return {
        "id": particle.id,
        "mass": particle.mass,
        "radius": particle.radius,
        "position": {"x": particle.position.x, "y": particle.position.y},
        "velocity": {"x": particle.velocity.x, "y": particle.velocity.y},
        "force": {"x": particle.force.x, "y": particle.force.y},
    }


def particle_from_dict(data: Dict[str, Any]) -> Particle:
    return Particle(
        id=data["id"],
        mass=data["mass"],
        radius=data["radius"],
        position=Vector2D(x=data["position"]["x"], y=data["position"]["y"]),
        velocity=Vector2D(x=data["velocity"]["x"], y=data["velocity"]["y"]),
        force=Vector2D(x=data["force"]["x"], y=data["force"]["y"]),
    )


def world_state_to_dict(state: WorldState) -> Dict[str, Any]:
    return {
        "world_id": state.world_id,
        "step": state.step,
        "time": round(state.time, 10),
        "dt": state.dt,
        "particles": [particle_to_dict(p) for p in state.particles],
        "boundary": {
            "x_min": state.boundary.x_min,
            "x_max": state.boundary.x_max,
            "y_min": state.boundary.y_min,
            "y_max": state.boundary.y_max,
        },
        "seed": state.seed,
        "config_hash": state.config_hash,
    }


def world_state_from_dict(data: Dict[str, Any]) -> WorldState:
    boundary_data = data["boundary"]
    boundary = Boundary(
        x_min=boundary_data["x_min"],
        x_max=boundary_data["x_max"],
        y_min=boundary_data["y_min"],
        y_max=boundary_data["y_max"],
    )
    particles = [particle_from_dict(p) for p in data["particles"]]

    return WorldState(
        world_id=data["world_id"],
        step=data["step"],
        time=data["time"],
        dt=data["dt"],
        particles=particles,
        boundary=boundary,
        seed=data["seed"],
        config_hash=data.get("config_hash", ""),
    )


def hash_world_state(state: WorldState) -> str:
    """Generate SHA-256 hash of deterministic world state snapshot."""
    state_dict = world_state_to_dict(state)
    serialized = json.dumps(state_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_world_config(config: WorldConfig) -> str:
    """Generate SHA-256 hash of world configuration."""
    config_dict = config.model_dump()
    serialized = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
