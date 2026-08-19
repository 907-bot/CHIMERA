"""Unit Tests for Phase 12: Macro-Biochemistry & Abiogenesis Engine (CHIMERA v4.0)"""

import pytest
import numpy as np
from packages.abiogenesis.models import Monomer, Polymer3D, VesicleMembrane, HypercycleState
from packages.abiogenesis.folding import MolecularFolding3D
from packages.abiogenesis.autocatalysis import AutocatalyticHypercycleSolver
from packages.abiogenesis.compartments import VesicleCompartmentEngine


def test_3d_molecular_folding():
    folding = MolecularFolding3D()
    seq = "HPPHHPH"
    polymer = folding.fold_sequence(seq, max_steps=100, seed=42)
    assert len(polymer.coordinates) == len(seq)
    assert isinstance(polymer.energy, float)


def test_vesicle_compartment_assembly_and_metabolism():
    engine = VesicleCompartmentEngine(critical_micelle_concentration=10.0)

    # Sub-critical lipid pool does not assemble
    assert engine.assemble_vesicle("ves_0", lipid_pool=5.0) is None

    # Supra-critical lipid pool self-assembles
    vesicle = engine.assemble_vesicle("ves_1", lipid_pool=100.0)
    assert vesicle is not None
    assert vesicle.radius > 0.0

    # Step metabolism and verify growth
    grown = engine.step_metabolism(vesicle, external_substrate=2.0, dt=0.1)
    assert grown.lipid_count >= vesicle.lipid_count
