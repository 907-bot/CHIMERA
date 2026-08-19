"""Unit Tests for Phase 17: Self-Evolving Simulation Architecture (CHIMERA v9.0)"""

import pytest
import numpy as np
from packages.metaevolution.models import KernelCodeSpec, PrecisionPolicy
from packages.metaevolution.meta_compiler import MetaCompilerKernelOptimizer
from packages.metaevolution.precision_controller import AdaptivePrecisionController
from packages.metaevolution.verification_guard import BitwiseVerificationGuard


def test_meta_compiler_optimization():
    compiler = MetaCompilerKernelOptimizer()
    fast_acc = compiler.synthesize_vectorized_nbody_kernel()

    pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    masses = np.array([1.0, 1.0], dtype=np.float64)
    acc = fast_acc(pos, masses, 1.0, 0.1)

    assert acc.shape == (2, 2)
    assert acc[0, 0] > 0  # Particle 0 pulled towards particle 1


def test_adaptive_precision_controller():
    controller = AdaptivePrecisionController(chaos_threshold=0.8, calm_threshold=0.1)

    policy_high = controller.determine_optimal_precision(lyapunov_exponent=1.2)
    assert policy_high.current_precision == "float64"

    policy_low = controller.determine_optimal_precision(lyapunov_exponent=0.05)
    assert policy_low.current_precision == "float16"
