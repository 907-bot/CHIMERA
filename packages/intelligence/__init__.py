"""CHIMERA Embodied Intelligence & Emergence Package (Phase 8 — v0.9)

Provides neural controllers, social interactions, collective swarming,
and information-theoretic emergence metrics (Transfer Entropy & Mutual Information).

Modules:
  models      : NeuralPolicy, SensoryObservation, AgentAction, InformationMetrics
  controller  : NeuralAgentController (vectorized sensor-to-action policy)
  information : EmergenceDetector (Transfer Entropy, Mutual Information, Swarm Polarization)
  agent       : SocialScientistAgent (analyzing collective intelligence and coordination)
"""

from packages.intelligence.models import (
    NeuralPolicy,
    SensoryObservation,
    AgentAction,
    InformationMetrics,
    SocialSimulationResult,
)
from packages.intelligence.controller import NeuralAgentController
from packages.intelligence.information import EmergenceDetector
from packages.intelligence.agent import SocialScientistAgent

__all__ = [
    "NeuralPolicy",
    "SensoryObservation",
    "AgentAction",
    "InformationMetrics",
    "SocialSimulationResult",
    "NeuralAgentController",
    "EmergenceDetector",
    "SocialScientistAgent",
]
