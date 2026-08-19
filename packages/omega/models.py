"""Immutable Core Models for The Omega Observatory (CHIMERA v10.0 - Phase 18)"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class RealityRecord(BaseModel):
    """Complete indexed catalog entry of a simulated synthetic universe."""
    model_config = ConfigDict(frozen=True)

    reality_id: str
    seed: int
    dimension_domain: str = "multiscale_cosmology"  # 'continuum', 'quantum', 'abiogenesis', 'cosmology', 'cognition'
    physical_constants: Dict[str, float] = Field(default_factory=dict)
    discovered_equations: List[str] = Field(default_factory=list)
    emergence_metrics: Dict[str, float] = Field(default_factory=dict)
    timestamp: float = 0.0


class ScientificPaperManifest(BaseModel):
    """Formal scientific publication generated autonomously by the Omega Observatory."""
    model_config = ConfigDict(frozen=True)

    paper_id: str
    title: str
    authors: List[str]  # e.g., ["AI-Scientist-Bull", "AI-Scientist-Skeptic", "CHIMERA-Omega-Engine"]
    abstract: str
    realities_referenced: List[str]
    latex_source: str
    markdown_content: str
    verified_invariants: List[str]


class RealityCatalogQuery(BaseModel):
    """Query filters for searching the infinite multiverse manifold."""
    domain: Optional[str] = None
    min_emergence_score: Optional[float] = None
    required_invariants: List[str] = Field(default_factory=list)
