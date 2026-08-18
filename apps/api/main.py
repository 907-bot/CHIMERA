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
from packages.symbolic.benchmark_worlds import generate_blind_data, ALL_BENCHMARKS
from packages.symbolic.discovery_engine import DiscoveryEngine
from packages.symbolic.registry import HypothesisRegistry
from packages.agents.debate_engine import DebateEngine
from packages.agents.hypothesis_graph import HypothesisGraph

app = FastAPI(
    title="CHIMERA Scientific Observatory Gateway",
    description="Event-Sourced Universal Telemetry & Trajectory API Gateway",
    version="0.3",
)


# Global service instances
storage = ObservatoryStorageEngine(":memory:")
hypothesis_registry = HypothesisRegistry(":memory:")
discovery_engine = DiscoveryEngine(registry=hypothesis_registry)
hypothesis_graph = HypothesisGraph()
debate_engine = DebateEngine(graph=hypothesis_graph)


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
    return {"status": "ok", "version": "0.3", "service": "chimera-observatory-api"}


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
        raise HTTPException(status_code=404, detail=f"No trajectory records found for world {world_id}")

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


# ---------------------------------------------------------------------------
# Phase 3: Symbolic Discovery Routes (CHIMERA v0.2b)
# ---------------------------------------------------------------------------

@app.get("/api/v1/benchmark/list")
def list_benchmarks():
    """List all available hidden-law benchmark worlds."""
    return {
        "benchmarks": [
            {"name": name, "description": f"Hidden-law benchmark: {name}"}
            for name in ALL_BENCHMARKS
        ]
    }


@app.post("/api/v1/benchmark/run/{world_name}")
def run_benchmark_world(world_name: str):
    """Run a hidden-law benchmark world and return blind observable data only.

    The response contains ONLY position, velocity, and time arrays.
    Hidden physics parameters (k, GM, b) are NEVER included in the response.
    """
    if world_name not in ALL_BENCHMARKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown benchmark world '{world_name}'. Available: {list(ALL_BENCHMARKS.keys())}"
        )

    blind_data = generate_blind_data(world_name)

    # Strip numpy arrays to lists for JSON serialisation
    serialisable = {
        "world_name": blind_data["world_name"],
        "num_steps": int(len(blind_data["t"])),
        "t": blind_data["t"].tolist(),
        "x": blind_data["x"].tolist(),
    }
    if "v" in blind_data:
        serialisable["v"] = blind_data["v"].tolist()
    if "a" in blind_data:
        serialisable["a"] = blind_data["a"].tolist()
    if "y" in blind_data:
        serialisable["y"] = blind_data["y"].tolist()
        serialisable["vx"] = blind_data["vx"].tolist()
        serialisable["vy"] = blind_data["vy"].tolist()

    return serialisable


@app.post("/api/v1/symbolic/discover/{world_name}")
def discover_hidden_law(world_name: str):
    """Trigger the SINDy blind discovery pipeline for a benchmark world.

    Laws are derived mathematically from trajectory data — nothing is hardcoded.
    Returns the best discovered hypothesis with R², RMSE, and candidate equation.
    """
    if world_name not in ALL_BENCHMARKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown benchmark world '{world_name}'."
        )

    result = discovery_engine.run_discovery(world_name)
    best = result.best_hypothesis

    return {
        "world_name": world_name,
        "elapsed_seconds": round(result.elapsed_seconds, 4),
        "hypotheses_count": len(result.hypotheses),
        "best_hypothesis": {
            "id": best.id,
            "solver": best.solver,
            "candidate_equation": best.candidate_equation,
            "status": best.status,
            "r_squared": best.metrics.r_squared if best.metrics else None,
            "rmse": best.metrics.rmse if best.metrics else None,
            "parameters": best.parameters.values,
        } if best else None,
    }


@app.get("/api/v1/symbolic/hypotheses/{world_name}")
def list_hypotheses(world_name: str, status: Optional[str] = None):
    """List all hypotheses registered for a world (including falsified — immutable)."""
    hyps = hypothesis_registry.get_by_world(world_name, status_filter=status)
    return {
        "world_name": world_name,
        "count": len(hyps),
        "hypotheses": [
            {
                "id": h.id,
                "solver": h.solver,
                "candidate_equation": h.candidate_equation,
                "status": h.status,
                "r_squared": h.metrics.r_squared if h.metrics else None,
                "created_at": h.created_at,
            }
            for h in hyps
        ],
    }


# ---------------------------------------------------------------------------
# Phase 4: Adversarial Scientific Society Routes (CHIMERA v0.3)
# ---------------------------------------------------------------------------

@app.post("/api/v1/debate/{world_name}")
def run_full_debate(world_name: str):
    """Run full adversarial debate pipeline on the best discovered hypothesis.

    Pipeline: Discover (SINDy) → Bull → Bear → Skeptic → Intervention → Arbiter.
    Returns the complete DebateRecord with all arguments and the final verdict.
    """
    if world_name not in ALL_BENCHMARKS:
        raise HTTPException(status_code=404, detail=f"Unknown world '{world_name}'")

    # Step 1: Discover hypothesis via SINDy
    disc_result = discovery_engine.run_discovery(world_name)
    if not disc_result.best_hypothesis:
        raise HTTPException(status_code=500, detail="Discovery produced no hypothesis")

    hyp = disc_result.best_hypothesis

    # Step 2: Run full adversarial debate
    record = debate_engine.debate(hyp)

    return {
        "world_name": world_name,
        "hypothesis": {
            "id": hyp.id,
            "equation": hyp.candidate_equation,
            "r_squared": hyp.metrics.r_squared if hyp.metrics else None,
        },
        "bull": {
            "confidence": record.bull_argument.confidence_score,
            "strongest_claim": record.bull_argument.strongest_claim,
        },
        "bear": {
            "doubt": record.bear_argument.doubt_score,
            "critical_flaw": record.bear_argument.critical_flaw,
        },
        "experiment": {
            "name": record.skeptic_experiment.experiment_name,
            "r_squared_on_perturbed": record.experiment_result.r_squared_on_perturbed,
            "survived": record.experiment_result.survived,
            "interpretation": record.experiment_result.interpretation,
        },
        "verdict": {
            "decision": record.arbiter_verdict.verdict,
            "confidence": record.arbiter_verdict.bayesian_confidence,
            "reasoning": record.arbiter_verdict.reasoning,
        },
        "final_status": record.final_status,
        "duration_seconds": record.duration_seconds,
    }


@app.get("/api/v1/debate/graph/summary")
def get_graph_summary():
    """Return node counts for the hypothesis provenance graph."""
    return {
        "node_counts": hypothesis_graph.summary(),
        "accepted": hypothesis_graph.accepted_hypotheses(),
        "rejected": hypothesis_graph.rejected_hypotheses(),
    }


@app.get("/api/v1/debate/graph/lineage/{hypothesis_id}")
def get_hypothesis_lineage(hypothesis_id: str):
    """Return the full provenance DAG for a specific hypothesis."""
    lineage = hypothesis_graph.get_hypothesis_lineage(hypothesis_id)
    return lineage


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
