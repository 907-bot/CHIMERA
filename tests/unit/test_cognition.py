"""Unit Tests for Phase 14: Emergent Minds & Artificial Cognition (CHIMERA v6.0)"""

import pytest
import numpy as np
from packages.cognition.models import NeuralGenome, CommunicationSignal
from packages.cognition.neuroevolution import NeuroevolutionController
from packages.cognition.information_dynamics import InformationDynamicsAnalyzer
from packages.cognition.communication import CommunicationProtocolMiner


def test_neural_forward_and_mutation():
    genome = NeuroevolutionController.create_random_genome("gen_1", num_inputs=3, num_hidden=6, num_outputs=2, seed=42)
    inputs = np.array([1.0, 0.5, -0.5])
    hidden = np.zeros(6)

    outputs, next_hidden = NeuroevolutionController.forward(genome, inputs, hidden)
    assert len(outputs) == 2
    assert len(next_hidden) == 6

    mutated = NeuroevolutionController.mutate(genome, mutation_rate=0.5, seed=123)
    assert mutated.genome_id != genome.genome_id


def test_communication_protocol_miner():
    miner = CommunicationProtocolMiner(num_discrete_symbols=3)
    signals = [
        CommunicationSignal(sender_id="a1", channel_values=(1.0, 0.0), timestamp=0.1),
        CommunicationSignal(sender_id="a2", channel_values=(1.1, 0.1), timestamp=0.2),
        CommunicationSignal(sender_id="a3", channel_values=(0.0, 1.0), timestamp=0.3),
        CommunicationSignal(sender_id="a4", channel_values=(0.1, 0.9), timestamp=0.4),
        CommunicationSignal(sender_id="a5", channel_values=(-1.0, -1.0), timestamp=0.5),
        CommunicationSignal(sender_id="a6", channel_values=(-0.9, -1.1), timestamp=0.6),
    ]

    report = miner.mine_symbolic_lexicon(signals)
    assert report["num_symbols_discovered"] == 3
    assert len(report["cluster_centers"]) == 3
