"""P3-14 — Feature Library Validation

Verifies feature matrix construction, column names, ordering, trigonometric terms,
polynomial terms, and deterministic evaluation.
"""

import pytest
import numpy as np
from packages.symbolic.sindy_solver import FeatureLibrary


class TestFeatureLibraryValidation:
    """Test suite for SINDy FeatureLibrary matrix construction."""

    def test_default_feature_library_shape_and_names(self):
        lib = FeatureLibrary(include_trig=False, include_cubic=False)
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -0.5, 1.0])

        theta, names = lib.build(x, v)

        # Expected default columns: ['1', 'x', 'v', 'x²', 'xv', 'v²'] (6 features)
        assert names == ["1", "x", "v", "x²", "xv", "v²"]
        assert theta.shape == (3, 6)

        # Verify numerical values
        assert np.array_equal(theta[:, 0], [1.0, 1.0, 1.0])          # Constant 1
        assert np.array_equal(theta[:, 1], x)                        # x
        assert np.array_equal(theta[:, 2], v)                        # v
        assert np.allclose(theta[:, 3], x ** 2)                      # x^2
        assert np.allclose(theta[:, 4], x * v)                       # xv
        assert np.allclose(theta[:, 5], v ** 2)                      # v^2

    def test_extended_cubic_and_trig_library(self):
        lib = FeatureLibrary(include_trig=True, include_cubic=True)
        x = np.array([0.0, np.pi / 2.0, np.pi])
        v = np.array([1.0, 0.0, -1.0])

        theta, names = lib.build(x, v)

        # Expected: 6 base + 2 cubic (x³, v³) + 2 trig (sin(x), cos(x)) = 10 features
        assert len(names) == 10
        assert "x³" in names
        assert "v³" in names
        assert "sin(x)" in names
        assert "cos(x)" in names
        assert theta.shape == (3, 10)

        # Verify trig values
        sin_idx = names.index("sin(x)")
        cos_idx = names.index("cos(x)")
        assert np.allclose(theta[:, sin_idx], [0.0, 1.0, 0.0], atol=1e-7)
        assert np.allclose(theta[:, cos_idx], [1.0, 0.0, -1.0], atol=1e-7)
