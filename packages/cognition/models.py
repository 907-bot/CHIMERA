"""Immutable Core Models for Emergent Minds & Artificial Cognition (CHIMERA v6.0 - Phase 14)"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class NeuralGenome(BaseModel):
    """Genetic encoding for recurrent neural network controller."""
    model_config = ConfigDict(frozen=True)

    genome_id: str
    num_inputs: int = 4
    num_hidden: int = 8
    num_outputs: int = 2
    weights_input_hidden: Tuple[Tuple[float, ...], ...]
    weights_hidden_hidden: Tuple[Tuple[float, ...], ...]
    weights_hidden_output: Tuple[Tuple[float, ...], ...]
    bias_hidden: Tuple[float, ...]
    bias_output: Tuple[float, ...]
    fitness: float = 0.0


class CommunicationSignal(BaseModel):
    """Signal broadcasted by cognitive agent."""
    model_config = ConfigDict(frozen=True)

    sender_id: str
    signal_type: str = "acoustic"  # 'acoustic', 'chemical', 'visual'
    channel_values: Tuple[float, ...] = (0.0, 0.0)
    timestamp: float = 0.0


class CognitiveAgentState(BaseModel):
    """State of an embodied agent with internal neural state and communication capability."""
    model_config = ConfigDict(frozen=True)

    agent_id: str
    genome: NeuralGenome
    hidden_state: Tuple[float, ...]
    position: Tuple[float, float] = (0.0, 0.0)
    energy: float = 100.0
    last_signal_sent: Optional[CommunicationSignal] = None
    action_history: Tuple[Tuple[float, ...], ...] = Field(default_factory=tuple)
    signal_history: Tuple[Tuple[float, ...], ...] = Field(default_factory=tuple)
