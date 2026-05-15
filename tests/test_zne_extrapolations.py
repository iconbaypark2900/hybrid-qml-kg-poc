"""Unit tests for the linear and Richardson zero-noise extrapolation helpers
in ``quantum_layer.advanced_error_mitigation``.

Verifies:
  - Linear extrapolation returns the intercept of the best-fit line.
  - Richardson at preregistered scales [1, 3, 5] gives the closed-form
    coefficients (1.875, -1.25, 0.375).
  - Richardson exactly recovers a low-degree polynomial's zero-noise value.
  - Both helpers reject mismatched-shape inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantum_layer.advanced_error_mitigation import (
    linear_zero_noise_extrapolation,
    richardson_zero_noise_extrapolation,
)


def test_linear_zne_recovers_intercept():
    # y = 0.7 + 0.1 * lambda
    scales = np.array([1.0, 2.0, 3.0])
    meas = 0.7 + 0.1 * scales
    assert linear_zero_noise_extrapolation(scales, meas) == pytest.approx(0.7, abs=1e-10)


def test_linear_zne_handles_noise_via_least_squares():
    rng = np.random.default_rng(0)
    scales = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    true_intercept = 0.55
    meas = true_intercept + 0.08 * scales + rng.normal(0, 0.01, size=scales.shape)
    intercept = linear_zero_noise_extrapolation(scales, meas)
    assert intercept == pytest.approx(true_intercept, abs=0.05)


def test_richardson_zne_three_points_matches_closed_form():
    """At scales [1, 3, 5] the Richardson coefficients are (1.875, -1.25, 0.375).

    f(0) ≈ 1.875 * f(1) − 1.25 * f(3) + 0.375 * f(5).
    """
    scales = np.array([1.0, 3.0, 5.0])
    meas = np.array([0.6, 0.4, 0.25])
    expected = 1.875 * 0.6 - 1.25 * 0.4 + 0.375 * 0.25
    got = richardson_zero_noise_extrapolation(scales, meas)
    assert got == pytest.approx(expected, abs=1e-10)


def test_richardson_zne_recovers_quadratic_zero_value():
    """Richardson with 3 points cancels error terms up to order 2."""
    # y = 0.42 + 0.05 * lambda + 0.02 * lambda**2 — Richardson with 3 scales
    # should recover the constant term 0.42 exactly.
    scales = np.array([1.0, 3.0, 5.0])
    meas = 0.42 + 0.05 * scales + 0.02 * scales ** 2
    assert richardson_zero_noise_extrapolation(scales, meas) == pytest.approx(0.42, abs=1e-10)


def test_richardson_zne_recovers_higher_order_with_more_points():
    """With 4 points it should cancel up to lambda^3."""
    scales = np.array([1.0, 2.0, 3.0, 4.0])
    meas = 0.31 + 0.04 * scales + 0.01 * scales ** 2 + 0.005 * scales ** 3
    assert richardson_zero_noise_extrapolation(scales, meas) == pytest.approx(0.31, abs=1e-10)


def test_extrapolators_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="!="):
        linear_zero_noise_extrapolation(np.array([1.0, 2.0]), np.array([0.5, 0.4, 0.3]))
    with pytest.raises(ValueError, match="!="):
        richardson_zero_noise_extrapolation(np.array([1.0, 2.0]), np.array([0.5, 0.4, 0.3]))


def test_extrapolators_reject_too_few_points():
    with pytest.raises(ValueError, match="at least 2"):
        linear_zero_noise_extrapolation(np.array([1.0]), np.array([0.5]))
    with pytest.raises(ValueError, match="at least 2"):
        richardson_zero_noise_extrapolation(np.array([1.0]), np.array([0.5]))


def test_richardson_zne_two_points_is_linear():
    """With 2 noise scales, Richardson collapses to linear extrapolation."""
    scales = np.array([1.0, 3.0])
    meas = np.array([0.6, 0.4])
    rich = richardson_zero_noise_extrapolation(scales, meas)
    lin = linear_zero_noise_extrapolation(scales, meas)
    assert rich == pytest.approx(lin, abs=1e-10)
