"""Communication Protocol Miner & Vocabulary Extractor (CHIMERA v6.0 - Phase 14)"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.cluster import KMeans
from packages.cognition.models import CommunicationSignal


class CommunicationProtocolMiner:
    """Discovers emergent symbolic lexicons and protocol correlations from agent signaling."""

    def __init__(self, num_discrete_symbols: int = 4):
        self.num_symbols = num_discrete_symbols

    def mine_symbolic_lexicon(self, signals: List[CommunicationSignal]) -> Dict[str, Any]:
        """Clusters continuous signal channels into discrete emergent symbolic tokens."""
        if len(signals) < self.num_symbols:
            return {
                "num_symbols_discovered": 0,
                "cluster_centers": [],
                "token_frequencies": {},
            }

        vectors = np.array([s.channel_values for s in signals], dtype=np.float64)
        kmeans = KMeans(n_clusters=self.num_symbols, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        unique, counts = np.unique(labels, return_counts=True)
        freqs = {f"SYMBOL_{k}": int(v) for k, v in zip(unique, counts)}

        return {
            "num_symbols_discovered": self.num_symbols,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "token_frequencies": freqs,
        }
