"""Information Dynamics and Cognitive Emergence Analyzer (CHIMERA v6.0 - Phase 14)

Calculates:
1. Transfer Entropy T_{X -> Y} measuring directional information flow between agents:
   T_{X -> Y} = sum p(y_{t+1}, y_t, x_t) * log2( p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t) )

2. Mutual Information I(X; Y) = H(X) + H(Y) - H(X, Y)

3. Integrated Information Phi (IIT approximation) measuring causal interconnectedness.
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np


class InformationDynamicsAnalyzer:
    """Computes transfer entropy, mutual information, and cognitive emergence indicators."""

    @staticmethod
    def _discretize(data: np.ndarray, bins: int = 4) -> np.ndarray:
        """Discretize continuous time series into discrete bin tokens."""
        min_v = np.min(data)
        max_v = np.max(data)
        if max_v - min_v < 1e-9:
            return np.zeros(len(data), dtype=int)
        normalized = (data - min_v) / (max_v - min_v + 1e-9)
        disc = np.clip(np.floor(normalized * bins).astype(int), 0, bins - 1)
        return disc

    def compute_transfer_entropy(
        self,
        source_series: np.ndarray,
        target_series: np.ndarray,
        bins: int = 4,
    ) -> float:
        """Computes Transfer Entropy T_{source -> target}."""
        x = self._discretize(source_series, bins=bins)
        y = self._discretize(target_series, bins=bins)

        n = len(x) - 1
        if n < 5:
            return 0.0

        x_t = x[:-1]
        y_t = y[:-1]
        y_next = y[1:]

        # Joint 3D distribution: p(y_{t+1}, y_t, x_t)
        hist_3d, _ = np.histogramdd((y_next, y_t, x_t), bins=(bins, bins, bins))
        p_3d = hist_3d / float(n)

        # 2D distributions: p(y_{t+1}, y_t) and p(y_t, x_t)
        p_y_next_yt = np.sum(p_3d, axis=2)  # (bins, bins)
        p_yt_xt = np.sum(p_3d, axis=0)       # (bins, bins)
        p_yt = np.sum(p_y_next_yt, axis=0)   # (bins,)

        te = 0.0
        for i in range(bins):       # y_{t+1}
            for j in range(bins):   # y_t
                for k in range(bins):  # x_t
                    p_joint = p_3d[i, j, k]
                    if p_joint > 1e-12:
                        p_cond_full = p_joint / (p_yt_xt[j, k] + 1e-12)
                        p_cond_target_only = p_y_next_yt[i, j] / (p_yt[j] + 1e-12)

                        if p_cond_full > 1e-12 and p_cond_target_only > 1e-12:
                            ratio = p_cond_full / p_cond_target_only
                            te += p_joint * np.log2(ratio)

        return float(max(0.0, te))

    def compute_mutual_information(self, series_a: np.ndarray, series_b: np.ndarray, bins: int = 4) -> float:
        """Compute Mutual Information I(A; B)."""
        a = self._discretize(series_a, bins=bins)
        b = self._discretize(series_b, bins=bins)
        n = len(a)
        if n < 5:
            return 0.0

        hist_2d, _, _ = np.histogram2d(a, b, bins=(bins, bins))
        p_ab = hist_2d / float(n)
        p_a = np.sum(p_ab, axis=1)
        p_b = np.sum(p_ab, axis=0)

        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_ab[i, j] > 1e-12 and p_a[i] > 1e-12 and p_b[j] > 1e-12:
                    mi += p_ab[i, j] * np.log2(p_ab[i, j] / (p_a[i] * p_b[j]))

        return float(max(0.0, mi))

    def compute_integrated_information_phi(self, neural_trajectories: np.ndarray) -> float:
        """Approximates Integrated Information (Phi) across neural subsystem partitions."""
        # neural_trajectories: (time_steps, num_neurons)
        n_neurons = neural_trajectories.shape[1]
        if n_neurons < 2:
            return 0.0

        # Whole-system covariance / determinant
        cov_whole = np.cov(neural_trajectories, rowvar=False) + 1e-6 * np.eye(n_neurons)
        h_whole = 0.5 * np.linalg.slogdet(cov_whole)[1]

        # Bipartition cut in half
        mid = n_neurons // 2
        cov_part1 = np.cov(neural_trajectories[:, :mid], rowvar=False) + 1e-6 * np.eye(mid)
        cov_part2 = np.cov(neural_trajectories[:, mid:], rowvar=False) + 1e-6 * np.eye(n_neurons - mid)

        h_part1 = 0.5 * np.linalg.slogdet(cov_part1)[1]
        h_part2 = 0.5 * np.linalg.slogdet(cov_part2)[1]

        # Integration Phi = (H(part1) + H(part2)) - H(whole)
        phi = (h_part1 + h_part2) - h_whole
        return float(max(0.0, phi))
