"""CHIMERA Scientific Civilization & In-World Observer Package (Phases 9 & 10 — v1.0)

Provides in-world observer agents that formulate their own hypotheses inside the simulation,
run autonomous experiments, build shared scientific repositories, and achieve nested discovery.

Modules:
  models       : InWorldObserver, CivilizationExperiment, CivilizationTheory, CivilizationState
  observers    : InWorldObserverAgent (nested sensor measurements, hypothesis formation)
  civilization : ScientificCivilizationEngine (multi-observer peer-review, consensus archive)
  agent        : CivilizationArchivistAgent (meta-scientific progress metrics vs ground truth)
"""

from packages.civilization.models import (
    InWorldObserver,
    CivilizationExperiment,
    CivilizationTheory,
    ScientificCivilizationState,
    CivilizationSimulationResult,
)
from packages.civilization.observers import InWorldObserverAgent
from packages.civilization.civilization import ScientificCivilizationEngine
from packages.civilization.agent import CivilizationArchivistAgent

__all__ = [
    "InWorldObserver",
    "CivilizationExperiment",
    "CivilizationTheory",
    "ScientificCivilizationState",
    "CivilizationSimulationResult",
    "InWorldObserverAgent",
    "ScientificCivilizationEngine",
    "CivilizationArchivistAgent",
]
