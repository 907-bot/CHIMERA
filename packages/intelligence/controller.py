"""Vectorized Neural Agent Controller for CHIMERA Phase 8.

Executes sensory-motor mappings with zero token cost:
  y = tanh(W^T x + b)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from packages.intelligence.models import NeuralPolicy, SensoryObservation, AgentAction


class NeuralAgentController:
    """Evaluates neural policies to convert sensory inputs into motor actions."""

    @staticmethod
    def forward(policy: NeuralPolicy, observation: SensoryObservation, max_speed: float = 2.0) -> AgentAction:
        """Run forward inference through the neural policy."""
        x = observation.to_vector()
        W = np.array(policy.weights, dtype=np.float64)  # (5, 3)
        b = np.array(policy.bias, dtype=np.float64)     # (3,)

        # Linear projection with tanh activation
        raw_out = np.tanh(x @ W + b)

        move_dx = float(raw_out[0] * max_speed)
        move_dy = float(raw_out[1] * max_speed)
        signal = float(0.5 * (raw_out[2] + 1.0))  # Normalize to [0, 1]

        return AgentAction(
            move_dx=move_dx,
            move_dy=move_dy,
            broadcast_signal=signal,
        )
