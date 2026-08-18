"""SINDy Sparse Identification of Nonlinear Dynamics Solver.

Zero-token symbolic discovery: identifies governing ODEs from trajectory
observations alone using Sparse Identification of Nonlinear Dynamics (SINDy).

Algorithm:
  1. Build feature library Θ(X) = [1, x, ẋ, x², xẋ, ẋ², sin(x), ...]
  2. Solve sparse regression: ẍ ≈ Θ(X) · ξ
  3. Return non-zero ξ coefficients as candidate equation

Laws are DERIVED from data — not hardcoded into this module.
The hidden physics constants (e.g. k) are never imported or known here.

Reference:
  Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016).
  Discovering governing equations from data by sparse identification of
  nonlinear dynamical systems. PNAS 113(15), 3932–3937.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from packages.symbolic.hypothesis import (
    Hypothesis,
    HypothesisParameters,
    PredictionMetrics,
)


class FeatureLibrary:
    """Builds the candidate feature library Θ(x, v) for SINDy.

    Supported feature terms:
        - constant  : 1
        - linear    : x, v
        - quadratic : x², xv, v²
        - cubic     : x³, v³
        - trig      : sin(x), cos(x)

    Args:
        include_trig:   Include sin/cos terms (default True)
        include_cubic:  Include cubic terms (default False)
    """

    def __init__(self, include_trig: bool = False, include_cubic: bool = False):
        self.include_trig = include_trig
        self.include_cubic = include_cubic

    def build(self, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Construct feature matrix and corresponding label strings.

        Args:
            x: Position array, shape [N]
            v: Velocity array, shape [N]

        Returns:
            Tuple of (Theta matrix [N, D], feature_names [D])
        """
        features: List[np.ndarray] = []
        names: List[str] = []

        # Constant term
        features.append(np.ones_like(x))
        names.append("1")

        # Linear terms
        features.append(x)
        names.append("x")
        features.append(v)
        names.append("v")

        # Quadratic terms
        features.append(x ** 2)
        names.append("x²")
        features.append(x * v)
        names.append("xv")
        features.append(v ** 2)
        names.append("v²")

        # Cubic terms (optional)
        if self.include_cubic:
            features.append(x ** 3)
            names.append("x³")
            features.append(v ** 3)
            names.append("v³")

        # Trigonometric terms (optional)
        if self.include_trig:
            features.append(np.sin(x))
            names.append("sin(x)")
            features.append(np.cos(x))
            names.append("cos(x)")

        theta = np.column_stack(features)
        return theta, names


def _stlsq(
    theta: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.05,
    max_iter: int = 20,
    alpha: float = 1e-5,
) -> np.ndarray:
    """Sequentially Thresholded Least Squares (STLSQ) for sparse regression.

    Identifies the sparse coefficient vector ξ such that target ≈ Θ · ξ
    by iteratively zeroing coefficients below `threshold`.

    Args:
        theta:     Feature matrix [N, D]
        target:    Target array (acceleration) [N]
        threshold: Coefficient magnitude below which a term is pruned.
        max_iter:  Maximum STLSQ iterations.
        alpha:     Ridge regularisation coefficient.

    Returns:
        Sparse coefficient array ξ [D]
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(theta, target)
    xi = ridge.coef_.copy()

    for _ in range(max_iter):
        # Identify small (near-zero) coefficients
        small_mask = np.abs(xi) < threshold
        xi[small_mask] = 0.0

        # Refit on remaining active features only
        active = ~small_mask
        if active.sum() == 0:
            break

        ridge.fit(theta[:, active], target)
        xi[active] = ridge.coef_

    return xi


def _equation_string(xi: np.ndarray, names: List[str], tol: float = 1e-4) -> str:
    """Build a human-readable equation string from sparse coefficients.

    Args:
        xi:    Coefficient array.
        names: Feature name strings (same order as xi).
        tol:   Coefficients below this are treated as zero.

    Returns:
        String like '-2.99*x + 0.01*v' or '0' if all terms zero.
    """
    terms = []
    for coef, name in zip(xi, names):
        if abs(coef) > tol:
            if name == "1":
                terms.append(f"{coef:.4f}")
            else:
                terms.append(f"{coef:.4f}*{name}")

    return " + ".join(terms) if terms else "0"


class SINDySolver:
    """SINDy solver that identifies governing equations from blind trajectory data.

    Args:
        threshold:     STLSQ sparsity threshold.
        train_ratio:   Fraction of trajectory steps used for fitting [0, 1).
        include_trig:  Include sin/cos terms in feature library.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        train_ratio: float = 0.8,
        include_trig: bool = False,
    ):
        self.threshold = threshold
        self.train_ratio = train_ratio
        self.library = FeatureLibrary(include_trig=include_trig)

    def solve(self, blind_data: Dict[str, Any]) -> Hypothesis:
        """Run SINDy sparse identification on blind observable data.

        Derives the governing equation purely from (x, v, a) time series.
        The hidden parameters (e.g., spring constant k) are NEVER given here.

        Args:
            blind_data: Dict returned by `generate_blind_data()`, containing:
                        't', 'x', 'v', 'a' (and optionally 'y', 'vy').

        Returns:
            Hypothesis in CANDIDATE state with discovered equation and R².
        """
        world_name: str = blind_data["world_name"]
        t: np.ndarray = np.asarray(blind_data["t"], dtype=np.float64)
        x: np.ndarray = np.asarray(blind_data["x"], dtype=np.float64)
        v: np.ndarray = np.asarray(blind_data["v"], dtype=np.float64)
        a: np.ndarray = np.asarray(blind_data["a"], dtype=np.float64)

        N = len(x)
        n_train = int(N * self.train_ratio)

        # Split: train on first 80%, hold out last 20%
        x_train, v_train, a_train = x[:n_train], v[:n_train], a[:n_train]
        x_test, v_test, a_test = x[n_train:], v[n_train:], a[n_train:]

        # Build feature library from BLIND observables only
        theta_train, feature_names = self.library.build(x_train, v_train)

        # Scale features for numerical stability
        scaler = StandardScaler()
        theta_train_scaled = scaler.fit_transform(theta_train)

        # Scale target acceleration
        a_scaler = StandardScaler()
        a_train_scaled = a_scaler.fit_transform(a_train.reshape(-1, 1)).ravel()

        # STLSQ sparse regression
        xi_scaled = _stlsq(theta_train_scaled, a_train_scaled, threshold=self.threshold)

        # Recover coefficients in original (unscaled) space
        # xi_orig[j] = xi_scaled[j] * (a_std / feature_std[j])
        a_std = a_scaler.scale_[0]
        a_mean = a_scaler.mean_[0]
        feat_std = scaler.scale_
        feat_mean = scaler.mean_

        xi_orig = xi_scaled * a_std / feat_std

        # Constant offset correction
        offset = a_mean - np.dot(feat_mean * a_std / feat_std, xi_scaled)

        # Validate predictions on held-out test set
        theta_test, _ = self.library.build(x_test, v_test)
        a_pred = theta_test @ xi_orig + offset

        ss_res = np.sum((a_test - a_pred) ** 2)
        ss_tot = np.sum((a_test - np.mean(a_test)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        rmse = float(np.sqrt(np.mean((a_test - a_pred) ** 2)))
        mae = float(np.mean(np.abs(a_test - a_pred)))

        equation_str = _equation_string(xi_orig, feature_names)

        # Build parameter dict from non-zero terms (derived, not hardcoded)
        derived_params: Dict[str, float] = {}
        for coef, name in zip(xi_orig, feature_names):
            if abs(coef) > 1e-4:
                derived_params[f"coef_{name}"] = float(coef)
        if abs(offset) > 1e-4:
            derived_params["offset"] = float(offset)

        metrics = PredictionMetrics(
            r_squared=max(-1.0, min(1.0, r2)),
            rmse=rmse,
            mae=mae,
            train_steps=n_train,
            test_steps=N - n_train,
        )

        hypothesis = Hypothesis(
            world_name=world_name,
            solver="SINDy-STLSQ",
            candidate_equation=equation_str,
            parameters=HypothesisParameters(values=derived_params),
            metrics=metrics,
            evidence_step_range=(0, n_train),
            status="CANDIDATE",
        )

        return hypothesis
