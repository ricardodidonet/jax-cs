"""Integrators against analytical ODE solutions and SciPy reference."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from jaxcs import (
    AnalyticalField,
    double_gyre,
    flowmap,
    integrate,
    integrate_ensemble,
    integrate_trajectory,
)

ROTATION = AnalyticalField(lambda t, x: jnp.stack([-x[1], x[0]]))
EXP_GROWTH = AnalyticalField(lambda t, x: 0.5 * x)


@pytest.mark.parametrize("method,tol", [("rk4", 1e-8), ("dopri5", 1e-10)])
def test_rotation_analytical(method, tol):
    """x' = (-y, x): solution rotates by angle T."""
    x0 = jnp.array([1.0, 0.0])
    T = 2.0
    xT = integrate(ROTATION, x0, 0.0, T, n_steps=400, method=method)
    expected = np.array([np.cos(T), np.sin(T)])
    np.testing.assert_allclose(np.asarray(xT), expected, atol=tol)


@pytest.mark.parametrize("method", ["rk4", "dopri5"])
def test_backward_integration_sign(method):
    """T < 0 integrates backward: rotation by -|T|."""
    x0 = jnp.array([1.0, 0.0])
    T = -1.5
    xT = integrate(ROTATION, x0, 0.0, T, n_steps=300, method=method)
    expected = np.array([np.cos(T), np.sin(T)])
    np.testing.assert_allclose(np.asarray(xT), expected, atol=1e-8)


def test_forward_backward_roundtrip():
    field = double_gyre()
    x0 = jnp.array([1.3, 0.4])
    fwd = integrate(field, x0, 0.0, 5.0, n_steps=500)
    back = integrate(field, fwd, 5.0, -5.0, n_steps=500)
    np.testing.assert_allclose(np.asarray(back), np.asarray(x0), atol=1e-7)


def test_exponential_growth():
    x0 = jnp.array([2.0])
    xT = integrate(EXP_GROWTH, x0, 0.0, 3.0, n_steps=300)
    np.testing.assert_allclose(float(xT[0]), 2.0 * np.exp(1.5), rtol=1e-9)


def test_trajectory_endpoints_and_times():
    ts, xs = integrate_trajectory(ROTATION, jnp.array([1.0, 0.0]), 1.0, 2.0, n_steps=100)
    assert ts.shape == (101,) and xs.shape == (101, 2)
    assert float(ts[0]) == 1.0
    np.testing.assert_allclose(float(ts[-1]), 3.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(xs[-1]),
        np.asarray(integrate(ROTATION, jnp.array([1.0, 0.0]), 1.0, 2.0, n_steps=100)),
    )


def test_ensemble_matches_single():
    x0s = jnp.array([[1.0, 0.0], [0.5, 0.5], [1.7, 0.9]])
    field = double_gyre()
    batched = integrate_ensemble(field, x0s, 0.0, 2.0, n_steps=200)
    for k in range(3):
        single = integrate(field, x0s[k], 0.0, 2.0, n_steps=200)
        np.testing.assert_allclose(np.asarray(batched[k]), np.asarray(single))


@pytest.mark.parametrize("T", [5.0, -5.0])
def test_double_gyre_vs_scipy(T):
    """Independent reference: SciPy adaptive RK45 at tight tolerance."""
    field = double_gyre()
    x0s = np.array([[0.3, 0.3], [1.0, 0.5], [1.6, 0.2], [0.7, 0.8]])

    def rhs(t, xy):
        return np.asarray(field(jnp.asarray(t), jnp.asarray(xy)))

    ours = np.asarray(flowmap(field, jnp.asarray(x0s), 0.0, T, dt=0.01))
    for k, x0 in enumerate(x0s):
        sol = solve_ivp(rhs, (0.0, T), x0, rtol=1e-11, atol=1e-11, dense_output=True)
        np.testing.assert_allclose(ours[k], sol.y[:, -1], atol=5e-7)


def test_checkpoint_matches_plain():
    field = double_gyre()
    x0 = jnp.array([0.8, 0.6])
    a = integrate(field, x0, 0.0, 4.0, n_steps=400, checkpoint=False)
    b = integrate(field, x0, 0.0, 4.0, n_steps=400, checkpoint=True)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_flowmap_composition_matches_direct():
    """Composing 4 short flow maps ≈ one long flow map (interpolation-
    error tolerance on a fine grid)."""
    from jaxcs import flowmap_composition, flowmap_grid

    field = double_gyre()
    x = jnp.linspace(0.0, 2.0, 201)
    y = jnp.linspace(0.0, 1.0, 101)
    h = 1.0
    maps = jnp.stack([
        flowmap_grid(field, x, y, k * h, h, dt=0.05) for k in range(4)
    ])
    composed = flowmap_composition(maps, x, y)
    direct = flowmap_grid(field, x, y, 0.0, 4 * h, dt=0.05)
    err = np.abs(np.asarray(composed) - np.asarray(direct))
    assert np.median(err) < 5e-4
    assert np.percentile(err, 95) < 2e-2
