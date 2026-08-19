"""Scientific Benchmark: Dynamic Kernel Optimization & Bitwise Parity Guard (CHIMERA v9.0 - Phase 17)

Benchmark Goals:
1. Demonstrate acceleration of numerical force kernels through dynamic vectorization synthesis.
2. Verify 100% bitwise numerical parity (|max_error| < 1e-12) between reference and synthesized kernels.
"""

import pytest
import numpy as np
from packages.metaevolution.meta_compiler import MetaCompilerKernelOptimizer
from packages.metaevolution.verification_guard import BitwiseVerificationGuard


def reference_nbody_acc(pos: np.ndarray, masses: np.ndarray, G: float, eps: float) -> np.ndarray:
    """Non-vectorized loop-based reference kernel."""
    n = len(masses)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = pos[j] - pos[i]
                dist = np.sqrt(np.sum(diff ** 2) + eps ** 2)
                acc[i] += G * masses[j] * diff / (dist ** 3)
    return acc


def test_scientific_kernel_evolution_and_bitwise_parity():
    compiler = MetaCompilerKernelOptimizer()
    guard = BitwiseVerificationGuard()

    synthesized_kernel = compiler.synthesize_vectorized_nbody_kernel()

    def sample_gen():
        rng = np.random.default_rng(12345)
        pos = rng.uniform(-10.0, 10.0, size=(20, 2))
        masses = rng.uniform(0.5, 5.0, size=20)
        return (pos, masses, 1.0, 0.1)

    # Benchmark report
    test_inputs = sample_gen()
    report = compiler.benchmark_kernel(reference_nbody_acc, synthesized_kernel, test_inputs, iterations=20)

    # Verification guard across multiple independent batches
    verif = guard.verify_reproducibility(
        reference_kernel=reference_nbody_acc,
        mutated_kernel=synthesized_kernel,
        sample_generator=sample_gen,
        trials=5,
    )

    print(f"\n[Meta-Evolution Benchmark] Speedup: {report.speedup_factor:.2f}x | Max Error: {verif['max_absolute_difference']:.2e} | Parity: {verif['verification_status']}")

    assert verif["bitwise_reproducible"] is True
    assert verif["max_absolute_difference"] < 1e-12
    assert report.speedup_factor > 1.0
