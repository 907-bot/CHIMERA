"""Unit Tests for Phase 16: World-Engineering & Causal Universe Synthesis (CHIMERA v8.0)"""

import pytest
from packages.synthesis.models import LawKernel, SynthesisTarget, UniverseSpecification
from packages.synthesis.inverse_designer import InverseLawSynthesizer
from packages.synthesis.causal_perturbation import CausalLawPerturbationEngine


def test_inverse_law_synthesizer():
    designer = InverseLawSynthesizer(seed=42)
    target = SynthesisTarget(target_metric="cluster_diversity", target_value=15.0)
    initial_kernel = LawKernel(kernel_name="NewtonLennard", gravity_exponent=2.0, repulsion_exponent=6.0)

    spec = designer.optimize_law_kernel(target, initial_kernel, max_evals=20)
    assert spec.law_kernel is not None
    assert spec.fitness_achieved <= 0.0  # negative loss (closer to 0 is better)


def test_causal_perturbation_stability():
    engine = CausalLawPerturbationEngine(delta_fraction=0.05)
    kernel = LawKernel(kernel_name="BaseKernel", gravity_exponent=2.0, coupling_strength=1.0)

    def simple_eval(k: LawKernel) -> float:
        return k.gravity_exponent * 10.0 + k.coupling_strength * 2.0

    report = engine.evaluate_bifurcation_stability(kernel, "gravity_exponent", simple_eval)
    assert report["is_causally_stable"] is True
    assert "bifurcation_metric" in report
