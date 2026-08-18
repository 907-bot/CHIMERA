# CHIMERA — Multi-Agent Development & Governance Protocol

> **Core Principle:** The LLM is NEVER the simulator. The simulation engine must be 100% deterministic, testable, and independent of LLMs.

---

## 1. Non-Negotiable Architecture Rules

1. **Separation of Physics and Reasoning**: Numerical engine handles world evolution. AI agents reason over observations only. AI agents MUST NOT modify simulation state directly or inject non-deterministic code into physics integrators.
2. **Bitwise Reproducibility**: Every world trajectory is strictly a function of `(seed, world_config, code_version)`. Rerunning with identical inputs MUST produce bit-for-bit identical state histories.
3. **Immutable History**: Past simulation events and recorded observations in the Observatory are append-only. Never overwrite or delete telemetry records.
4. **Explicit Random Seeds**: All stochastic processes (initial state placement, perturbations) MUST derive from explicit, versioned random seeds (`numpy.random.SeedSequence` / `PRNGKey`).
5. **No Silent Changes to Physics**: Never alter force laws, integrator equations, or physical constants without creating or updating a corresponding scientific test benchmark in `tests/scientific/`.
6. **Failed Experiments are Immutable Evidence**: Failed hypotheses and counter-evidence MUST be retained in scientific records to prevent circular reasoning.

---

## 2. Multi-Agent Workstream Boundaries

| Agent / Model Role | Primary Domain | Permitted Actions | Forbidden Actions |
|---|---|---|---|
| **Codex** | Core Physics & Engine | `packages/core`, `packages/physics`, integrators, tests | Modifying agent LLM prompt flows or hypothesis state machines |
| **DeepSeek-V2** | Mathematics & Algorithms | Math solvers, symbolic regression, Lyapunov metrics, causal inference | Writing raw API endpoints or UI visualizer components |
| **Antigravity** | Microservice Architecture & UI | API Gateways, monorepo structure, Three.js/WebGPU visualizer, hypothesis graphs | Changing physics integrator equations or numerical kernels |
| **Human Lead** | Governance & Validation | Reviewing benchmarks, accepting PRs, defining phase goals | N/A |

---

## 3. Code Standards & Testing Requirements

- **Type Safety**: All code in `packages/` MUST use Python type hints (`Pydantic` models or standard Python types).
- **Docstrings**: Public functions and classes MUST include clear docstrings specifying inputs, outputs, and mathematical formulas where applicable.
- **Test Coverage**: Every new feature MUST include unit tests under `tests/unit/`. Physics changes MUST include scientific validation benchmarks under `tests/scientific/`.
- **Vectorization**: All physics calculations MUST use vectorized `NumPy` operations to ensure maximum computational efficiency ($0 LLM token cost).
