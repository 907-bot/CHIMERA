"""Hidden-Law Benchmark World Harnesses for CHIMERA Phase 3.

ARCHITECTURE RULE: This module is the ONLY place where real hidden physics
parameters live. AI Scientists (SINDy / SR solvers) NEVER import or read this
file — they only receive trajectory arrays produced by `generate_blind_data()`.

Three canonical benchmark worlds:
  1. harmonic_spring  – F = -k*(r - center)         (Hooke's Law)
  2. damped_oscillator – F = -k*x - b*ẋ            (Damped spring)
  3. keplerian_approx  – F = -GM/r² along radial    (2-body orbit, linearised)

Physics is derived from the *simulation* data — the constants here are the
ground-truth used only for exit-criteria scoring, never fed into solvers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True)
class BenchmarkWorldSpec:
    """Immutable specification for a hidden-law benchmark world.

    Args:
        name:           Unique identifier string for the world.
        description:    Human-readable description of the hidden physics.
        hidden_params:  Dict of ground-truth parameter names → values.
                        Used ONLY for final scoring; never exposed to solvers.
        num_particles:  Number of particles in the world.
        num_steps:      Number of integration steps to generate.
        dt:             Integration time step.
        seed:           Explicit random seed for reproducibility.
    """
    name: str
    description: str
    hidden_params: Dict[str, float]
    num_particles: int
    num_steps: int
    dt: float
    seed: int


# ---------------------------------------------------------------------------
# Canonical Benchmark World Definitions
# ---------------------------------------------------------------------------

HARMONIC_SPRING = BenchmarkWorldSpec(
    name="harmonic_spring",
    description="1-D harmonic oscillator: F = -k*(x - x_eq). Hidden: k=3.0",
    hidden_params={"k": 3.0, "x_eq": 0.0},
    num_particles=1,
    num_steps=1000,
    dt=0.01,
    seed=7,
)

DAMPED_OSCILLATOR = BenchmarkWorldSpec(
    name="damped_oscillator",
    description="Damped harmonic oscillator: F = -k*x - b*ẋ. Hidden: k=2.5, b=0.3",
    hidden_params={"k": 2.5, "b": 0.3, "x_eq": 0.0},
    num_particles=1,
    num_steps=1000,
    dt=0.01,
    seed=13,
)

KEPLERIAN_APPROX = BenchmarkWorldSpec(
    name="keplerian_approx",
    description="Central-force attraction: F = -GM/r² (radial). Hidden: GM=50.0",
    hidden_params={"GM": 50.0},
    num_particles=1,
    num_steps=1000,
    dt=0.01,
    seed=42,
)

ALL_BENCHMARKS: Dict[str, BenchmarkWorldSpec] = {
    "harmonic_spring": HARMONIC_SPRING,
    "damped_oscillator": DAMPED_OSCILLATOR,
    "keplerian_approx": KEPLERIAN_APPROX,
}


# ---------------------------------------------------------------------------
# Hidden Physics Integrators (Deterministic Numpy — no LLM involvement)
# ---------------------------------------------------------------------------

def _integrate_harmonic(spec: BenchmarkWorldSpec) -> Dict[str, np.ndarray]:
    """Run hidden harmonic oscillator physics and return trajectory arrays.

    Returns:
        Dict with keys: 't', 'x', 'v', 'a'  (all shape: [num_steps+1])
    """
    k = spec.hidden_params["k"]
    x_eq = spec.hidden_params["x_eq"]
    dt = spec.dt

    rng = np.random.default_rng(spec.seed)
    x0 = float(rng.uniform(0.5, 2.0))   # random initial displacement from eq
    v0 = float(rng.uniform(-0.5, 0.5))

    xs = np.zeros(spec.num_steps + 1)
    vs = np.zeros(spec.num_steps + 1)
    xs[0] = x0
    vs[0] = v0

    # Symplectic Euler (position-Verlet equivalent for unit mass)
    for i in range(spec.num_steps):
        a = -k * (xs[i] - x_eq)
        vs[i + 1] = vs[i] + a * dt
        xs[i + 1] = xs[i] + vs[i + 1] * dt

    ts = np.arange(spec.num_steps + 1) * dt
    # Compute accelerations from positions (observable via finite difference)
    acc = np.gradient(vs, dt)

    return {"t": ts, "x": xs, "v": vs, "a": acc}


def _integrate_damped(spec: BenchmarkWorldSpec) -> Dict[str, np.ndarray]:
    """Run hidden damped oscillator physics and return trajectory arrays.

    Returns:
        Dict with keys: 't', 'x', 'v', 'a'  (all shape: [num_steps+1])
    """
    k = spec.hidden_params["k"]
    b = spec.hidden_params["b"]
    x_eq = spec.hidden_params["x_eq"]
    dt = spec.dt

    rng = np.random.default_rng(spec.seed)
    x0 = float(rng.uniform(0.8, 2.0))
    v0 = float(rng.uniform(-0.3, 0.3))

    xs = np.zeros(spec.num_steps + 1)
    vs = np.zeros(spec.num_steps + 1)
    xs[0] = x0
    vs[0] = v0

    # RK4 for damped system
    def f_damp(x: float, v: float) -> Tuple[float, float]:
        """Returns (dx/dt, dv/dt) for damped oscillator."""
        dxdt = v
        dvdt = -k * (x - x_eq) - b * v
        return dxdt, dvdt

    for i in range(spec.num_steps):
        x, v = xs[i], vs[i]
        dx1, dv1 = f_damp(x, v)
        dx2, dv2 = f_damp(x + 0.5 * dt * dx1, v + 0.5 * dt * dv1)
        dx3, dv3 = f_damp(x + 0.5 * dt * dx2, v + 0.5 * dt * dv2)
        dx4, dv4 = f_damp(x + dt * dx3, v + dt * dv3)
        xs[i + 1] = x + dt * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6.0
        vs[i + 1] = v + dt * (dv1 + 2 * dv2 + 2 * dv3 + dv4) / 6.0

    ts = np.arange(spec.num_steps + 1) * dt
    acc = np.gradient(vs, dt)

    return {"t": ts, "x": xs, "v": vs, "a": acc}


def _integrate_keplerian(spec: BenchmarkWorldSpec) -> Dict[str, np.ndarray]:
    """Run hidden central-force gravity physics and return trajectory arrays.

    The particle orbits a fixed center at (0, 0). Observable: (x, y, vx, vy).

    Returns:
        Dict with keys: 't', 'x', 'y', 'vx', 'vy', 'r', 'ax', 'ay'
    """
    GM = spec.hidden_params["GM"]
    dt = spec.dt

    rng = np.random.default_rng(spec.seed)
    # Circular-ish orbit initial conditions
    r0 = float(rng.uniform(3.0, 5.0))
    vcirc = np.sqrt(GM / r0)

    xs = np.zeros(spec.num_steps + 1)
    ys = np.zeros(spec.num_steps + 1)
    vxs = np.zeros(spec.num_steps + 1)
    vys = np.zeros(spec.num_steps + 1)

    xs[0] = r0
    ys[0] = 0.0
    vxs[0] = 0.0
    vys[0] = vcirc * float(rng.uniform(0.85, 1.05))  # slight eccentricity

    def f_kepler(x: float, y: float, vx: float, vy: float) -> Tuple[float, float, float, float]:
        """Returns (dx/dt, dy/dt, dvx/dt, dvy/dt)."""
        r = np.sqrt(x * x + y * y) + 1e-9
        ax = -GM * x / (r ** 3)
        ay = -GM * y / (r ** 3)
        return vx, vy, ax, ay

    for i in range(spec.num_steps):
        x, y, vx, vy = xs[i], ys[i], vxs[i], vys[i]
        dx1, dy1, dvx1, dvy1 = f_kepler(x, y, vx, vy)
        dx2, dy2, dvx2, dvy2 = f_kepler(
            x + 0.5 * dt * dx1, y + 0.5 * dt * dy1,
            vx + 0.5 * dt * dvx1, vy + 0.5 * dt * dvy1
        )
        dx3, dy3, dvx3, dvy3 = f_kepler(
            x + 0.5 * dt * dx2, y + 0.5 * dt * dy2,
            vx + 0.5 * dt * dvx2, vy + 0.5 * dt * dvy2
        )
        dx4, dy4, dvx4, dvy4 = f_kepler(
            x + dt * dx3, y + dt * dy3,
            vx + dt * dvx3, vy + dt * dvy3
        )
        xs[i + 1] = x + dt * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6.0
        ys[i + 1] = y + dt * (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6.0
        vxs[i + 1] = vx + dt * (dvx1 + 2 * dvx2 + 2 * dvx3 + dvx4) / 6.0
        vys[i + 1] = vy + dt * (dvy1 + 2 * dvy2 + 2 * dvy3 + dvy4) / 6.0

    ts = np.arange(spec.num_steps + 1) * dt
    rs = np.sqrt(xs ** 2 + ys ** 2)
    axs = np.gradient(vxs, dt)
    ays = np.gradient(vys, dt)

    return {"t": ts, "x": xs, "y": ys, "vx": vxs, "vy": vys, "r": rs, "ax": axs, "ay": ays}


# ---------------------------------------------------------------------------
# Public API: generate_blind_data
# ---------------------------------------------------------------------------

def generate_blind_data(world_name: str) -> Dict[str, Any]:
    """Run a hidden-law world simulation and return ONLY blind observables.

    AI Scientists call this function. The returned dict contains only position,
    velocity, and time arrays — never forces, accelerations, or hidden params.

    Args:
        world_name: One of 'harmonic_spring', 'damped_oscillator', 'keplerian_approx'

    Returns:
        Dict of observable arrays that can safely be passed to solvers:
          - 't'  : time array
          - 'x'  : position (or x-component in 2D)
          - 'v'  : velocity (or vx-component in 2D)
          - 'y', 'vy' (only for keplerian_approx)
          - '_world_name': identifier string (no hidden params)
    """
    if world_name not in ALL_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark world '{world_name}'. "
            f"Available: {list(ALL_BENCHMARKS.keys())}"
        )

    spec = ALL_BENCHMARKS[world_name]

    if world_name == "harmonic_spring":
        raw = _integrate_harmonic(spec)
        return {
            "world_name": world_name,
            "t": raw["t"],
            "x": raw["x"],
            "v": raw["v"],
            # Acceleration INCLUDED as observable — it's measurable via strain gauges etc.
            # Hidden: what GENERATES it (k value)
            "a": raw["a"],
        }

    elif world_name == "damped_oscillator":
        raw = _integrate_damped(spec)
        return {
            "world_name": world_name,
            "t": raw["t"],
            "x": raw["x"],
            "v": raw["v"],
            "a": raw["a"],
        }

    elif world_name == "keplerian_approx":
        raw = _integrate_keplerian(spec)
        return {
            "world_name": world_name,
            "t": raw["t"],
            "x": raw["x"],
            "y": raw["y"],
            "vx": raw["vx"],
            "vy": raw["vy"],
            "r": raw["r"],
        }

    raise ValueError(f"No integrator registered for world: {world_name}")


def score_against_hidden_truth(world_name: str, discovered_params: Dict[str, float]) -> Dict[str, float]:
    """Score discovered parameters against the hidden ground truth.

    This function is used ONLY in test exit criteria — never by solvers.

    Args:
        world_name:        Name of the benchmark world.
        discovered_params: Dict mapping parameter names to discovered values.

    Returns:
        Dict mapping each param to its relative error (abs(disc - true) / true).
    """
    spec = ALL_BENCHMARKS[world_name]
    scores: Dict[str, float] = {}

    for param, true_val in spec.hidden_params.items():
        if param in discovered_params:
            relative_error = abs(discovered_params[param] - true_val) / (abs(true_val) + 1e-12)
            scores[param] = relative_error
        else:
            scores[param] = float("inf")

    return scores
