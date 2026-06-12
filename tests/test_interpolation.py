"""Interpolators against known functions; NaN-mask workflow."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxcs import GriddedField2D, fill_nans_and_get_mask
from jaxcs.interpolation import bilinear, bilinear_manual, trilinear


def _grid(n0, x0, x1):
    return jnp.linspace(x0, x1, n0)


def test_bilinear_exact_for_bilinear_function():
    """Bilinear interpolation reproduces a + bx + cy + dxy exactly."""
    x = _grid(11, 0.0, 2.0)
    y = _grid(7, -1.0, 1.0)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    f = 1.0 + 2.0 * X - 0.5 * Y + 3.0 * X * Y
    rng = np.random.default_rng(0)
    qx = rng.uniform(0.0, 2.0, 50)
    qy = rng.uniform(-1.0, 1.0, 50)
    vals = jax.vmap(lambda a, b: bilinear(f, a, b, x, y))(jnp.asarray(qx), jnp.asarray(qy))
    expected = 1.0 + 2.0 * qx - 0.5 * qy + 3.0 * qx * qy
    np.testing.assert_allclose(np.asarray(vals), expected, atol=1e-12)


def test_manual_matches_map_coordinates():
    x = _grid(21, 0.0, 1.0)
    y = _grid(31, 0.0, 3.0)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    f = jnp.sin(3 * X) * jnp.cos(Y)
    rng = np.random.default_rng(1)
    qx = jnp.asarray(rng.uniform(0.0, 1.0, 40))
    qy = jnp.asarray(rng.uniform(0.0, 3.0, 40))
    a = jax.vmap(lambda px, py: bilinear(f, px, py, x, y))(qx, qy)
    ix = (qx - x[0]) / (x[1] - x[0])
    iy = (qy - y[0]) / (y[1] - y[0])
    b = jax.vmap(lambda px, py: bilinear_manual(f, px, py))(ix, iy)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-13)


def test_trilinear_exact_for_trilinear_function():
    t = _grid(5, 0.0, 4.0)
    x = _grid(9, 0.0, 2.0)
    y = _grid(7, 0.0, 1.0)
    T, X, Y = jnp.meshgrid(t, x, y, indexing="ij")
    f = 2.0 * T + X * Y - T * X * Y + 0.5
    rng = np.random.default_rng(2)
    qt = jnp.asarray(rng.uniform(0, 4, 30))
    qx = jnp.asarray(rng.uniform(0, 2, 30))
    qy = jnp.asarray(rng.uniform(0, 1, 30))
    vals = jax.vmap(lambda a, b, c: trilinear(f, a, b, c, t, x, y))(qt, qx, qy)
    expected = 2.0 * qt + qx * qy - qt * qx * qy + 0.5
    np.testing.assert_allclose(np.asarray(vals), np.asarray(expected), atol=1e-12)


def test_time_clamping():
    """Queries outside the data time span clamp to first/last frame."""
    t = _grid(3, 0.0, 2.0)
    x = _grid(5, 0.0, 1.0)
    y = _grid(5, 0.0, 1.0)
    f = jnp.stack([jnp.full((5, 5), v) for v in [1.0, 2.0, 3.0]])
    early = trilinear(f, jnp.asarray(-5.0), 0.5, 0.5, t, x, y, mode="nearest")
    late = trilinear(f, jnp.asarray(10.0), 0.5, 0.5, t, x, y, mode="nearest")
    assert float(early) == 1.0 and float(late) == 3.0


def test_spatial_extrapolation_modes():
    x = _grid(5, 0.0, 1.0)
    y = _grid(5, 0.0, 1.0)
    f = jnp.ones((5, 5)) * 7.0
    out_const = bilinear(f, jnp.asarray(2.0), jnp.asarray(0.5), x, y, mode="constant")
    out_near = bilinear(f, jnp.asarray(2.0), jnp.asarray(0.5), x, y, mode="nearest")
    assert float(out_const) == 0.0 and float(out_near) == 7.0


def test_gridded_field_constant_advection():
    """A uniform gridded velocity advects exactly linearly."""
    from jaxcs import integrate

    t = _grid(3, 0.0, 10.0)
    x = _grid(11, 0.0, 10.0)
    y = _grid(11, 0.0, 10.0)
    u = jnp.ones((3, 11, 11)) * 0.7
    v = jnp.ones((3, 11, 11)) * -0.3
    field = GriddedField2D(u=u, v=v, tvals=t, xvals=x, yvals=y)
    xT = integrate(field, jnp.array([5.0, 5.0]), 0.0, 4.0, n_steps=40)
    np.testing.assert_allclose(np.asarray(xT), [5.0 + 0.7 * 4, 5.0 - 0.3 * 4], atol=1e-12)


def test_fill_nans_and_get_mask():
    u = np.ones((2, 4, 4))
    u[0, 1, 1] = np.nan
    v = np.ones((2, 4, 4))
    v[1, 2, 3] = np.nan
    (uf, vf), mask = fill_nans_and_get_mask([u, v], fill_value=0.0)
    assert not np.any(np.isnan(np.asarray(uf)))
    assert mask[1, 1] and mask[2, 3] and mask.sum() == 2


def test_interpolation_is_differentiable():
    x = _grid(11, 0.0, 1.0)
    y = _grid(11, 0.0, 1.0)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    f = 2.0 * X + 3.0 * Y
    g = jax.grad(lambda p: bilinear(f, p[0], p[1], x, y))(jnp.array([0.43, 0.57]))
    np.testing.assert_allclose(np.asarray(g), [2.0, 3.0], atol=1e-12)


def test_steady_field_nt1():
    """nt == 1 gridded data: time is ignored, no NaN from zero spacing."""
    x = _grid(11, 0.0, 10.0)
    y = _grid(11, 0.0, 10.0)
    field = GriddedField2D(
        u=jnp.ones((1, 11, 11)) * 2.0, v=jnp.ones((1, 11, 11)) * -1.0,
        tvals=jnp.array([0.0]), xvals=x, yvals=y,
    )
    out = field(jnp.asarray(123.4), jnp.array([5.0, 5.0]))
    np.testing.assert_allclose(np.asarray(out), [2.0, -1.0], atol=1e-14)
