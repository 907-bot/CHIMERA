"""Causal Law Perturbation & Stability Engine (CHIMERA v8.0 - Phase 16)

Applies systematic counterfactual interventions to force laws to measure stability manifolds and bifurcation boundaries.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
from packages.synthesis.models import LawKernel


class CausalLawPerturbationEngine:
    """Evaluates stability of synthesized laws under counterfactual parameter perturbations."""

    def __init__(self, delta_fraction: float = 0.05):
        self.delta = delta_fraction

    def perturb_law_kernel(self, kernel: LawKernel, parameter_name: str) -> Tuple[LawKernel, LawKernel]:
        """Creates positive and negative counterfactual variants of a physical law."""
        val = getattr(kernel, parameter_name)
        val_plus = val * (1.0 + self.delta)
        val_minus = val * (1.0 - self.delta)

        k_plus = kernel.model_copy(update={parameter_name: val_plus})
        k_minus = kernel.model_copy(update={parameter_name: val_minus})
        return k_plus, k_minus

    def evaluate_bifurcation_stability(
        self,
        base_kernel: LawKernel,
        parameter_name: str,
        eval_fn: Any,
    ) -> Dict[str, Any]:
        """Tests whether small changes in law parameters cause abrupt discontinuous phase transitions."""
        k_plus, k_minus = self.perturb_law_kernel(base_kernel, parameter_name)
        res_base = eval_fn(base_kernel)
        res_plus = eval_fn(k_plus)
        res_minus = eval_fn(k_minus)

        grad_forward = (res_plus - res_base) / (self.delta * getattr(base_kernel, parameter_name))
        grad_backward = (res_base - res_minus) / (self.delta * getattr(base_kernel, parameter_name))
        second_diff = abs(grad_forward - grad_backward)
        is_stable = second_diff < 10.0

        return {
            "parameter": parameter_name,
            "base_response": float(res_base),
            "plus_response": float(res_plus),
            "minus_response": float(res_minus),
            "is_causally_stable": is_stable,
            "bifurcation_metric": float(second_diff),
        }
