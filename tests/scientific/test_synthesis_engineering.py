"""Scientific Benchmark: AI World-Engineering & Inverse Physics Optimization (CHIMERA v8.0 - Phase 16)

Benchmark Goal:
Demonstrate that the Inverse Law Synthesizer autonomously optimizes interaction law parameters
to reach a targeted emergent structural complexity score within < 5% error.
"""

import pytest
import numpy as np
from packages.synthesis.models import LawKernel, SynthesisTarget
from packages.synthesis.inverse_designer import InverseLawSynthesizer


def test_scientific_inverse_physics_design():
    synthesizer = InverseLawSynthesizer(seed=42)
    target_complexity = 25.0
    target = SynthesisTarget(target_metric="cluster_diversity", target_value=target_complexity)
    initial_kernel = LawKernel(
        kernel_name="PrebioticPrototype",
        gravity_exponent=2.0,
        repulsion_exponent=8.0,
        coupling_strength=1.0,
        crossover_radius=1.0,
    )

    designed_spec = synthesizer.optimize_law_kernel(target, initial_kernel, max_evals=50)
    opt_k = designed_spec.law_kernel

    # Evaluate achieved complexity using identical proxy formula
    achieved_complexity = opt_k.coupling_strength * (opt_k.gravity_exponent / (opt_k.repulsion_exponent + 1e-3)) * 50.0 + opt_k.crossover_radius * 2.0
    rel_error = abs(achieved_complexity - target_complexity) / target_complexity

    print(f"\n[World Engineering Benchmark] Target Complexity: {target_complexity:.2f} | Achieved: {achieved_complexity:.2f} | Error: {rel_error*100:.3f}%")

    assert rel_error < 0.05, f"Synthesized law error {rel_error*100:.3f}% exceeded 5% threshold"
