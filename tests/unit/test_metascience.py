"""Unit Tests for Phase 15: Inter-Civilization Science & Meta-Theories (CHIMERA v7.0)"""

import pytest
from packages.metascience.models import CrossUniverseMorphism, MetaInvariant
from packages.metascience.cross_universe_translator import CrossUniverseTranslator
from packages.metascience.meta_graph import MetaTheoreticalGraphEngine
from packages.metascience.inter_civilization_arena import InterCivilizationArena


def test_cross_universe_translation():
    morphism = CrossUniverseMorphism(
        morphism_id="morph_1_2",
        source_universe_id="u1",
        target_universe_id="u2",
        variable_mappings={"r": "radius"},
        scaling_factors={"time_dilation": 2.0},
    )
    translator = CrossUniverseTranslator(morphism)
    translated = translator.translate_symbolic_law("G * m / r**2")
    assert "radius" in translated


def test_meta_theoretic_graph_and_consensus():
    graph_engine = MetaTheoreticalGraphEngine()
    graph_engine.register_universe_theory("u1", "newton_grav", "F = G * m1 * m2 / r**2")
    graph_engine.register_universe_theory("u2", "modified_grav", "F = k * m1 * m2 / d**2")

    arena = InterCivilizationArena()
    consensus, inv = arena.evaluate_cross_world_invariant_consensus(
        candidate_invariant="Conservation of Energy: dE/dt = 0",
        universe_evaluations={"u1": 0.98, "u2": 0.97, "u3": 0.99},
    )

    assert consensus is True
    graph_engine.register_meta_invariant(inv)
    dag = graph_engine.export_dag()
    assert len(dag.meta_invariants) == 1
