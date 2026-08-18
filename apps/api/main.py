"""CHIMERA FastAPI Observatory API Gateway Microservice"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from packages.core.models import WorldConfig
from packages.physics.engine import DeterministicEngine
from packages.observatory.events import (
    SimEvent,
    EventType,
    SnapshotRecordedEvent,
    EnergyMeasuredEvent,
)
from packages.observatory.storage import ObservatoryStorageEngine
from packages.observatory.features import FeatureExtractor, ObservationMask, BlindObservation
from packages.physics.energy import EnergyMetrics

app = FastAPI(
    title="CHIMERA Scientific Observatory Gateway",
    description="Event-Sourced Universal Telemetry & Trajectory API Gateway",
    version="0.2a",
)

# Global storage engine instance
storage = ObservatoryStorageEngine(":memory:")


class SimRunRequest(BaseModel):
    world_id: str = "world_001"
    seed: int = 42
    steps: int = 500
    num_particles: int = 10
    dt: float = 0.01
    integrator_type: str = "verlet"


class SimRunResponse(BaseModel):
    world_id: str
    steps_recorded: int
    initial_hash: str
    final_hash: str
    status: str


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.2a", "service": "chimera-observatory-api"}


@app.post("/api/v1/sim/run", response_model=SimRunResponse)
def run_simulation(req: SimRunRequest):
    """Execute a deterministic simulation run and auto-record to Observatory."""
    config = WorldConfig(
        world_id=req.world_id,
        seed=req.seed,
        num_particles=req.num_particles,
        dt=req.dt,
        integrator_type=req.integrator_type,
    )
    engine = DeterministicEngine(config=config)
    history = engine.run(req.steps)

    # Store trajectory into DuckDB columnar storage
    storage.store_trajectory(history)

    # Emit & store milestone events
    for state in [history[0], history[-1]]:
        energy = EnergyMetrics.compute_all(state.particles)
        ev_energy = EnergyMeasuredEvent(
            world_id=state.world_id,
            step=state.step,
            time=state.time,
            payload=energy,
        )
        storage.store_event(ev_energy)

        ev_snap = SnapshotRecordedEvent(
            world_id=state.world_id,
            step=state.step,
            time=state.time,
            payload={"particle_count": len(state.particles)},
        )
        storage.store_event(ev_snap)

    from packages.core.serialization import hash_world_state
    init_hash = hash_world_state(history[0])
    final_hash = hash_world_state(history[-1])

    return SimRunResponse(
        world_id=req.world_id,
        steps_recorded=len(history),
        initial_hash=init_hash,
        final_hash=final_hash,
        status="RECORDED",
    )


@app.get("/api/v1/observatory/trajectory/{world_id}")
def get_trajectory(
    world_id: str,
    start_step: int = Query(0, ge=0),
    end_step: Optional[int] = Query(None, ge=0),
):
    """Query trajectory frame slices from DuckDB columnar storage."""
    states = storage.query_trajectory_slice(world_id, start_step=start_step, end_step=end_step)
    if not states:
        raise HTTPException(status_code=44, detail=f"No trajectory records found for world {world_id}")

    return {
        "world_id": world_id,
        "count": len(states),
        "states": [
            {
                "step": s.step,
                "time": s.time,
                "particles": [
                    {
                        "id": p.id,
                        "pos": [p.position.x, p.position.y],
                        "vel": [p.velocity.x, p.velocity.y],
                    }
                    for p in s.particles
                ],
            }
            for s in states
        ],
    }


@app.get("/api/v1/observatory/events/{world_id}")
def get_events(
    world_id: str,
    event_type: Optional[str] = None,
    start_step: int = Query(0, ge=0),
    end_step: Optional[int] = Query(None, ge=0),
):
    """Query logged simulation events from Observatory stream."""
    events = storage.query_events(world_id, event_type=event_type, start_step=start_step, end_step=end_step)
    return {
        "world_id": world_id,
        "count": len(events),
        "events": [
            {
                "event_id": ev.event_id,
                "step": ev.step,
                "time": ev.time,
                "type": ev.event_type,
                "payload": ev.payload,
            }
            for ev in events
        ],
    }


@app.get("/api/v1/observatory/features/{world_id}")
def get_derived_features(world_id: str):
    """Extract derived spatial entropy, energy, and MSD observables for a world trajectory."""
    states = storage.query_trajectory_slice(world_id)
    if not states:
        raise HTTPException(status_code=404, detail=f"No trajectory found for world {world_id}")

    entropy_series = [
        {"step": s.step, "time": s.time, "entropy": FeatureExtractor.spatial_entropy(s.particles)}
        for s in states
    ]

    msd_series = FeatureExtractor.mean_squared_displacement(states, particle_id=1)

    return {
        "world_id": world_id,
        "total_steps": len(states),
        "entropy_series": entropy_series,
        "msd_series": [{"time": t, "msd": val} for t, val in msd_series],
    }


@app.get("/api/v1/observatory/blind/{world_id}")
def get_blind_observation(world_id: str, step: int = Query(0, ge=0)):
    """Retrieve sanitized BlindObservation for AI Scientists (withholding hidden constants)."""
    states = storage.query_trajectory_slice(world_id, start_step=step, end_step=step)
    if not states:
        raise HTTPException(status_code=404, detail=f"No state snapshot found at step {step} for world {world_id}")

    blind_obs = ObservationMask.mask_state(states[0])
    return blind_obs.model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
