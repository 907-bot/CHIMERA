"""Immutable Core Models for Self-Evolving Simulation Architecture (CHIMERA v9.0 - Phase 17)"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class KernelCodeSpec(BaseModel):
    """Specification of an optimized numerical simulation kernel."""
    model_config = ConfigDict(frozen=True)

    kernel_id: str
    target_engine: str = "nbody_grav"  # 'nbody_grav', 'heat_diffusion', 'reaction_kinetics'
    vectorization_level: str = "numpy_simd"
    source_code: str
    optimization_flags: Dict[str, Any] = Field(default_factory=dict)


class OptimizationBenchmarkReport(BaseModel):
    """Benchmark comparing baseline vs optimized kernel execution."""
    model_config = ConfigDict(frozen=True)

    kernel_id: str
    baseline_time_ms: float
    optimized_time_ms: float
    speedup_factor: float
    bitwise_identical: bool = True
    max_relative_drift: float = 0.0


class PrecisionPolicy(BaseModel):
    """Dynamic precision assignment policy (FP16, FP32, FP64) driven by Lyapunov stability."""
    model_config = ConfigDict(frozen=True)

    current_precision: str = "float64"  # 'float16', 'float32', 'float64'
    lyapunov_exponent: float = 0.0
    stability_threshold: float = 0.5
