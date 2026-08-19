"""Adaptive Precision Controller (CHIMERA v9.0 - Phase 17)

Dynamically switches floating-point precision (FP16, FP32, FP64) based on localized Lyapunov stability metrics.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from packages.metaevolution.models import PrecisionPolicy


class AdaptivePrecisionController:
    """Selects precision dtype based on chaos/stability indicators."""

    def __init__(self, chaos_threshold: float = 0.8, calm_threshold: float = 0.1):
        self.chaos_threshold = chaos_threshold
        self.calm_threshold = calm_threshold

    def determine_optimal_precision(self, lyapunov_exponent: float) -> PrecisionPolicy:
        """Assigns FP64 for chaotic regimes (high sensitivity), FP32 for moderate, FP16 for quiescent equilibrium."""
        if lyapunov_exponent > self.chaos_threshold:
            dtype_str = "float64"
        elif lyapunov_exponent < self.calm_threshold:
            dtype_str = "float16"
        else:
            dtype_str = "float32"

        return PrecisionPolicy(
            current_precision=dtype_str,
            lyapunov_exponent=lyapunov_exponent,
            stability_threshold=self.chaos_threshold,
        )

    def cast_array(self, arr: np.ndarray, policy: PrecisionPolicy) -> np.ndarray:
        if policy.current_precision == "float64":
            return arr.astype(np.float64)
        elif policy.current_precision == "float32":
            return arr.astype(np.float32)
        elif policy.current_precision == "float16":
            return arr.astype(np.float16)
        return arr
