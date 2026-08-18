# CHIMERA System Architecture Document

## Overview

CHIMERA is an Artificial Multiverse Scientific Observatory platform designed to simulate deterministic artificial universes, capture event-sourced telemetry, infer hidden physical laws, and orchestrate adversarial AI scientific societies.

```
                            CHIMERA SYSTEM
                                   │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
    DETERMINISTIC SIMULATION                     AI SCIENTISTS
    • Vectorized Numerical Solvers              • Hypothesis Formulation
    • Deterministic Physics/Chemistry           • Adversarial Debate (Bull/Bear/Skeptic)
    • Exact Seed & Trajectory History           • Counterfactual Experiment Design
    • Independent of LLM (Zero Hallucination)   • Mathematical Intuition & Synthesis
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                          UNIVERSAL OBSERVATORY
                         (Event-Sourced Truth)
```

## Monorepo Layout

- `packages/core/`: Foundation data structures (`WorldState`, `Vector2D`, `Particle`, serialization, state hashing).
- `packages/physics/`: Integrators (RK4, Symplectic Verlet), force fields, elastic collision dynamics, energy metrics.
- `packages/observatory/`: Event-sourced logging pipeline, DuckDB/Parquet time-series telemetry store.
- `packages/agents/`: Multi-agent debate state machines (Bull, Bear, Skeptic, Arbiter).
- `packages/science/`: Hypothesis graph, theorem provenance, evidence verification engine.
- `apps/api/`: FastAPI microservice gateway.
- `apps/web/`: Web portal (Next.js, Three.js timeline scrubber).
- `apps/cli/`: Typer CLI runner for simulations and benchmarks.
- `configs/`: World configurations, physics initial conditions, sweep definitions.
- `experiments/`: Benchmark experiments (hidden laws, chaos, invariants).
- `tests/`: Unit tests, integration tests, and scientific verification benchmarks.

## Data Flow Pipeline

1. **Simulation**: `DeterministicEngine` steps `WorldState` through time using `VerletIntegrator` or `RK4Integrator`.
2. **Observatory**: Trajectory snapshots and interaction events (`COLLISION`, `ENERGY_MEASURED`) are emitted to the event pipeline.
3. **Symbolic Inference**: Discovery solvers (SINDy/PySR) extract equations from observations without knowing underlying hidden rules.
4. **Adversarial Debate**: LLM scientific agents formulate hypotheses, generate counterfactual interventions, and debate validity.
5. **Multiverse Sweeps**: Interventions branch into new world families (Family A: initial conditions, Family B: chaos/Lyapunov, Family C: parameter sweeps, Family D: checkpoint branches).
