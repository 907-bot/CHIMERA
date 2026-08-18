import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
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
from packages.multiverse.models import (
    WorldFamilyType,
    WorldFamilySpec,
    WorldBranchSpec,
    MultiverseBatchResult,
    LyapunovResult,
    InvariantResult,
)
from packages.multiverse.orchestrator import MultiverseOrchestrator
from packages.chemistry.models import (
    ChemicalSpecies,
    Reaction,
    ReactionNetwork,
    ChemistryState,
    AutocatalyticCycleResult,
    KineticsSimulationResult,
)
from packages.chemistry.kinetics import MassActionKineticsSolver, BENCHMARK_NETWORKS
from packages.chemistry.detector import AutocatalysisDetector
from packages.chemistry.agent import ChemistAgent
from packages.alife.models import Environment, ALifeSimulationResult
from packages.alife.evolution import EvolutionaryEngine
from packages.alife.agent import BiologistAgent
from packages.intelligence.models import (
    NeuralPolicy,
    SensoryObservation,
    SocialSimulationResult,
)
from packages.intelligence.controller import NeuralAgentController
from packages.intelligence.information import EmergenceDetector
from packages.intelligence.agent import SocialScientistAgent
from packages.civilization.models import CivilizationSimulationResult
from packages.civilization.civilization import ScientificCivilizationEngine
from packages.civilization.agent import CivilizationArchivistAgent

app = FastAPI(
    title="CHIMERA Scientific Observatory Gateway",
    description="Event-Sourced Universal Telemetry & Trajectory API Gateway",
    version="1.0",
)


# Global service instances
storage = ObservatoryStorageEngine(":memory:")
hypothesis_registry = HypothesisRegistry(":memory:")
discovery_engine = DiscoveryEngine(registry=hypothesis_registry)
hypothesis_graph = HypothesisGraph()
debate_engine = DebateEngine(graph=hypothesis_graph)
multiverse_orchestrator = MultiverseOrchestrator()
chemist_agent = ChemistAgent()
biologist_agent = BiologistAgent()
social_scientist_agent = SocialScientistAgent()
civilization_archivist = CivilizationArchivistAgent()


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


# Mount Web Dashboard Static Assets
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


@app.get("/")
@app.get("/dashboard")
def serve_dashboard():
    """Serve the interactive CHIMERA Scientific Observatory Web Dashboard."""
    index_file = web_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "CHIMERA Scientific Observatory Gateway v1.0", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0", "service": "chimera-observatory-api"}


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
        "v": blind_data["v"].tolist(),
    }
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


# ---------------------------------------------------------------------------
# Phase 5: Multiverse & Cross-World Discovery Routes (CHIMERA v0.4)
# ---------------------------------------------------------------------------

@app.post("/api/v1/multiverse/run-family")
def run_world_family(spec: WorldFamilySpec):
    """Execute a parallel World Family (Family A, B, C, or D) and detect invariants/chaos."""
    try:
        result = multiverse_orchestrator.run_family(spec)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/multiverse/chaos-analysis")
def analyze_chaos(config: Optional[WorldConfig] = None, steps: int = 150, epsilon: float = 1e-8):
    """Run Lyapunov trajectory divergence analysis on a base world configuration."""
    cfg = config or WorldConfig()
    lyap_result = multiverse_orchestrator.chaos_calculator.calculate_lyapunov(
        base_config=cfg,
        steps=steps,
        epsilon=epsilon,
    )
    return lyap_result.model_dump()


class BranchTimelineRequest(BaseModel):
    base_config: WorldConfig = Field(default_factory=WorldConfig)
    total_steps: int = 100
    branch_specs: List[WorldBranchSpec] = Field(default_factory=list)


@app.post("/api/v1/multiverse/branch")
def branch_checkpoint_timeline(
    req: BranchTimelineRequest,
):
    """Branch multiple child timelines from a parent checkpoint at step k."""
    if not req.branch_specs:
        raise HTTPException(status_code=400, detail="Must provide at least one WorldBranchSpec.")
    
    timelines = multiverse_orchestrator.run_family_d(
        base_config=req.base_config,
        total_steps=req.total_steps,
        branch_specs=req.branch_specs,
    )
    
    summary = {
        w_id: {
            "total_steps": len(traj),
            "final_particle_count": len(traj[-1].particles) if traj else 0,
        }
        for w_id, traj in timelines.items()
    }
    return {"parent_world_id": base_config.world_id, "timelines": summary}


# ---------------------------------------------------------------------------
# Phase 6: Reaction-Network Chemistry Routes (CHIMERA v0.6)
# ---------------------------------------------------------------------------

@app.get("/api/v1/chemistry/networks/benchmark/{network_name}")
def get_benchmark_network(network_name: str):
    """Retrieve canonical benchmark reaction network definition (brusselator, formose, lotka_volterra)."""
    if network_name not in BENCHMARK_NETWORKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown benchmark network '{network_name}'. Available: {list(BENCHMARK_NETWORKS.keys())}"
        )
    net = BENCHMARK_NETWORKS[network_name]()
    return net.model_dump()


@app.post("/api/v1/chemistry/simulate")
def simulate_reaction_network(
    network: ReactionNetwork,
    total_time: float = 40.0,
    dt: float = 0.01,
):
    """Run deterministic mass-action kinetics simulation on a reaction network."""
    solver = MassActionKineticsSolver(network)
    sim_result = solver.simulate(total_time=total_time, dt=dt)
    
    # Run autocatalysis detector
    detector = AutocatalysisDetector(network)
    analysis = detector.analyze_trajectory(sim_result)
    
    return {
        "network_name": network.name,
        "time_points_count": len(sim_result.time_points),
        "species_names": network.get_species_names(),
        "final_concentrations": {
            sp: sim_result.concentrations[sp][-1]
            for sp in network.get_species_names()
        },
        "autocatalysis_analysis": analysis.model_dump(),
    }


@app.post("/api/v1/chemistry/analyze-stoichiometry")
def analyze_stoichiometry(network: ReactionNetwork):
    """Perform formal stoichiometric and pathway audit via ChemistAgent."""
    report = chemist_agent.analyze_network(network)
    return report.model_dump()


# ---------------------------------------------------------------------------
# Phase 7: Artificial Life & Evolutionary Dynamics Routes (CHIMERA v0.7-0.8)
# ---------------------------------------------------------------------------

@app.post("/api/v1/alife/simulate")
def simulate_artificial_life(
    initial_population: int = 15,
    total_steps: int = 80,
    seed: int = 42,
    speciation_threshold: float = 0.25,
):
    """Run an artificial life evolutionary simulation with speciation and metabolism."""
    engine = EvolutionaryEngine(seed=seed, speciation_threshold=speciation_threshold)
    sim_result = engine.run_simulation(initial_population_size=initial_population, total_steps=total_steps)
    report = biologist_agent.analyze_simulation(sim_result)
    
    return {
        "simulation_id": sim_result.simulation_id,
        "total_steps": sim_result.total_steps,
        "total_births": sim_result.total_births,
        "total_deaths": sim_result.total_deaths,
        "final_population": sim_result.final_population_size,
        "species_count": len(sim_result.phylogenetic_tree_nodes),
        "biologist_report": report.model_dump(),
    }


# ---------------------------------------------------------------------------
# Phase 8: Embodied Intelligence & Emergence Routes (CHIMERA v0.9)
# ---------------------------------------------------------------------------

@app.post("/api/v1/intelligence/evaluate-emergence")
def evaluate_collective_emergence(
    num_agents: int = 10,
    steps: int = 50,
):
    """Evaluate information-theoretic collective emergence (Transfer Entropy & Swarm Polarization)."""
    detector = EmergenceDetector()
    
    # Generate representative swarming test trajectory
    vel_history = []
    sig_history = [[float(np.sin(t * 0.1))] for t in range(steps)]
    
    for t in range(steps):
        heading = np.sin(t * 0.08)
        step_vels = [
            (float(np.cos(heading + i * 0.02)), float(np.sin(heading + i * 0.02)))
            for i in range(num_agents)
        ]
        vel_history.append(step_vels)
    
    # 2 signals for transfer entropy
    sig_0 = [float(np.sin(t * 0.1)) for t in range(steps)]
    sig_1 = [float(np.sin(t * 0.1 - 0.05)) for t in range(steps)]
    
    metrics = detector.evaluate_swarm_trajectory(vel_history, [sig_0, sig_1])
    
    sim_res = SocialSimulationResult(
        total_steps=steps,
        num_agents=num_agents,
        information_metrics=metrics,
        mean_energy_history=[30.0] * steps,
        polarization_history=[metrics.swarm_polarization] * steps,
    )
    
    report = social_scientist_agent.analyze_social_dynamics(sim_res)
    return report.model_dump()


# ---------------------------------------------------------------------------
# Phases 9 & 10: Scientific Civilization Routes (CHIMERA v1.0)
# ---------------------------------------------------------------------------

@app.post("/api/v1/civilization/simulate")
def simulate_scientific_civilization(
    generations: int = 5,
    num_observers: int = 5,
    ground_truth_k: float = 3.0,
):
    """Simulate an in-world scientific civilization conducting nested experiments and peer review."""
    engine = ScientificCivilizationEngine(seed=42, num_observers=num_observers)
    sim_result = engine.run_civilization(generations=generations, ground_truth_k=ground_truth_k)
    audit = civilization_archivist.audit_civilization(sim_result)
    
    return {
        "civilization_id": sim_result.civilization_id,
        "total_generations": sim_result.total_generations,
        "active_observers": len(sim_result.observers),
        "paradigm_count": sim_result.paradigm_count,
        "meta_accuracy": sim_result.accuracy_vs_ground_truth,
        "archivist_audit": audit.model_dump(),
        "accepted_theories": [
            t.model_dump() for t in sim_result.archived_theories
            if t.status == "ACCEPTED_PARADIGM"
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
