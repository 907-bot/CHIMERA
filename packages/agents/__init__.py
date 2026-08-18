"""CHIMERA Adversarial Scientific Society (Phase 4 — v0.3)

Package providing the Bull / Bear / Skeptic / Arbiter debate state machine,
the Intervention Engine, and the Hypothesis Provenance Graph.

Module layout:
  debate_models      : Structured data models for all debate artifacts
  roles              : Bull, Bear, Skeptic, Arbiter role implementations
  intervention       : Deterministic intervention / counterfactual engine
  debate_engine      : State-machine orchestrating multi-round debates
  hypothesis_graph   : NetworkX DAG for hypothesis provenance

Design principle:
  - Agents output structured JSON (Pydantic models) only — no free-text loops.
  - Intervention Engine is 100% deterministic (zero token cost).
  - LLM calls are optional and isolated in `roles.py`. Default is rule-based.
  - Per AGENTS.md: AI agents MUST NOT modify simulation state directly.
"""
