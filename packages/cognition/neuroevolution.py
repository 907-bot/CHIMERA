"""Neuroevolutionary Controller Engine (CHIMERA v6.0 - Phase 14)

Implements Recurrent Neural Network (RNN) controller inference and evolutionary genetic mutation.
Forward Dynamics:
    h_{t+1} = tanh(W_ih * x_t + W_hh * h_t + b_h)
    y_{t+1} = tanh(W_ho * h_{t+1} + b_o)
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np
from packages.cognition.models import NeuralGenome, CognitiveAgentState, CommunicationSignal


class NeuroevolutionController:
    """Evaluates and mutates recurrent neural controllers."""

    @staticmethod
    def create_random_genome(
        genome_id: str,
        num_inputs: int = 4,
        num_hidden: int = 8,
        num_outputs: int = 2,
        seed: int = 42,
    ) -> NeuralGenome:
        """Initialize random Gaussian weights."""
        rng = np.random.default_rng(seed)
        w_ih = rng.normal(0.0, 0.5, size=(num_hidden, num_inputs))
        w_hh = rng.normal(0.0, 0.3, size=(num_hidden, num_hidden))
        w_ho = rng.normal(0.0, 0.5, size=(num_outputs, num_hidden))
        b_h = np.zeros(num_hidden)
        b_o = np.zeros(num_outputs)

        return NeuralGenome(
            genome_id=genome_id,
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            weights_input_hidden=tuple(tuple(float(x) for x in row) for row in w_ih),
            weights_hidden_hidden=tuple(tuple(float(x) for x in row) for row in w_hh),
            weights_hidden_output=tuple(tuple(float(x) for x in row) for row in w_ho),
            bias_hidden=tuple(float(x) for x in b_h),
            bias_output=tuple(float(x) for x in b_o),
            fitness=0.0,
        )

    @staticmethod
    def forward(
        genome: NeuralGenome,
        inputs: np.ndarray,
        prev_hidden: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute single RNN forward step."""
        w_ih = np.array(genome.weights_input_hidden)
        w_hh = np.array(genome.weights_hidden_hidden)
        w_ho = np.array(genome.weights_hidden_output)
        b_h = np.array(genome.bias_hidden)
        b_o = np.array(genome.bias_output)

        h_next = np.tanh(w_ih @ inputs + w_hh @ prev_hidden + b_h)
        outputs = np.tanh(w_ho @ h_next + b_o)
        return outputs, h_next

    @staticmethod
    def mutate(genome: NeuralGenome, mutation_rate: float = 0.1, mutation_std: float = 0.2, seed: int = 42) -> NeuralGenome:
        """Mutate genome weights with Gaussian perturbations."""
        rng = np.random.default_rng(seed)

        def mutate_matrix(m: np.ndarray) -> np.ndarray:
            mask = rng.random(m.shape) < mutation_rate
            noise = rng.normal(0.0, mutation_std, size=m.shape)
            return m + mask * noise

        w_ih = mutate_matrix(np.array(genome.weights_input_hidden))
        w_hh = mutate_matrix(np.array(genome.weights_hidden_hidden))
        w_ho = mutate_matrix(np.array(genome.weights_hidden_output))
        b_h = mutate_matrix(np.array(genome.bias_hidden))
        b_o = mutate_matrix(np.array(genome.bias_output))

        return NeuralGenome(
            genome_id=f"mut_{genome.genome_id}_{seed}",
            num_inputs=genome.num_inputs,
            num_hidden=genome.num_hidden,
            num_outputs=genome.num_outputs,
            weights_input_hidden=tuple(tuple(float(x) for x in row) for row in w_ih),
            weights_hidden_hidden=tuple(tuple(float(x) for x in row) for row in w_hh),
            weights_hidden_output=tuple(tuple(float(x) for x in row) for row in w_ho),
            bias_hidden=tuple(float(x) for x in b_h),
            bias_output=tuple(float(x) for x in b_o),
            fitness=0.0,
        )
