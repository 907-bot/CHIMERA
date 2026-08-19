"""Scientific Benchmark: Information Flow & Transfer Entropy in Interacting Neural Agents (CHIMERA v6.0 - Phase 14)

Benchmark Goal:
Demonstrate statistically significant non-zero Transfer Entropy (T_{Leader -> Follower} > 0)
between interacting agents where one agent's past signals drive the future actions of another.
"""

import pytest
import numpy as np
from packages.cognition.information_dynamics import InformationDynamicsAnalyzer


def test_scientific_transfer_entropy_emergence():
    analyzer = InformationDynamicsAnalyzer()
    n_steps = 500
    rng = np.random.default_rng(42)

    # Leader generates a stochastic process
    leader_signal = rng.normal(0.0, 1.0, size=n_steps)

    # Follower responds with a 1-step lag to Leader + noise: Y_{t+1} = 0.8 * X_t + noise
    follower_action = np.zeros(n_steps)
    follower_action[1:] = 0.85 * leader_signal[:-1] + rng.normal(0.0, 0.2, size=n_steps - 1)

    # Compute directional Transfer Entropy
    te_leader_to_follower = analyzer.compute_transfer_entropy(leader_signal, follower_action, bins=4)
    te_follower_to_leader = analyzer.compute_transfer_entropy(follower_action, leader_signal, bins=4)

    print(f"\n[Cognitive Transfer Entropy Benchmark] T(Leader -> Follower): {te_leader_to_follower:.4f} bits | T(Follower -> Leader): {te_follower_to_leader:.4f} bits")

    # Directional causality check: T(Leader -> Follower) must be significantly higher than reverse
    assert te_leader_to_follower > 0.05, "Causal coupling must produce detectable positive transfer entropy"
    assert te_leader_to_follower > te_follower_to_leader, "Directional transfer entropy must correctly identify the causal driver"
