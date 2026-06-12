"""HYCOM-style gridded ocean-velocity case.

Synthetic 'ocean data': a time-dependent double-gyre velocity sampled on
a coarse (t, x, y) grid with a land mask, fed through GriddedField2D.
The gridded FTLE must agree with the analytical-field FTLE to the
interpolation-error tolerance.
"""

import jax.numpy as jnp
import numpy as np

from jaxcs import (
    GriddedField2D,
    binary_dilation,
    double_gyre,
    fill_nans_and_get_mask,
    flowmap,
    flowmap_grid,
    ftle_grid,
    integrate,
)


def _sampled_double_gyre(nt=33, nx=201, ny=101, t1=8.0):
    field = double_gyre()
    tvals = np.linspace(0.0, t1, nt)
    xvals = np.linspace(0.0, 2.0, nx)
    yvals = np.linspace(0.0, 1.0, ny)
    T, X, Y = np.meshgrid(tvals, xvals, yvals, indexing="ij")
    a = 0.25 * np.sin(0.2 * np.pi * T)
    b = 1 - 2 * a
    f = a * X**2 + b * X
    df = 2 * a * X + b
    U = -np.pi * 0.1 * np.sin(np.pi * f) * np.cos(np.pi * Y)
    V = np.pi * 0.1 * np.cos(np.pi * f) * np.sin(np.pi * Y) * df
    return field, tvals, xvals, yvals, U, V


def test_gridded_field_matches_analytical_trajectories():
    field, tvals, xvals, yvals, U, V = _sampled_double_gyre()
    gridded = GriddedField2D(
        u=jnp.asarray(U), v=jnp.asarray(V),
        tvals=jnp.asarray(tvals), xvals=jnp.asarray(xvals), yvals=jnp.asarray(yvals),
    )
    x0s = jnp.array([[0.4, 0.4], [1.2, 0.6], [1.7, 0.3]])
    a = flowmap(field, x0s, 0.0, 6.0, dt=0.05)
    b = flowmap(gridded, x0s, 0.0, 6.0, dt=0.05)
    # error budget: bilinear interpolation on a 201x101 grid over T=6
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0.02)


def test_gridded_ftle_matches_analytical():
    field, tvals, xvals, yvals, U, V = _sampled_double_gyre()
    gridded = GriddedField2D(
        u=jnp.asarray(U), v=jnp.asarray(V),
        tvals=jnp.asarray(tvals), xvals=jnp.asarray(xvals), yvals=jnp.asarray(yvals),
    )
    x = jnp.linspace(0.1, 1.9, 61)
    y = jnp.linspace(0.1, 0.9, 31)
    T = 6.0
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    f_ref = ftle_grid(flowmap_grid(field, x, y, 0.0, T, dt=0.05), T, dx, dy)
    f_grd = ftle_grid(flowmap_grid(gridded, x, y, 0.0, T, dt=0.05), T, dx, dy)
    ref = np.asarray(f_ref)
    grd = np.asarray(f_grd)
    assert np.corrcoef(ref.ravel(), grd.ravel())[0, 1] > 0.98
    assert np.median(np.abs(ref - grd)) < 0.02


def test_land_mask_workflow():
    """NaN land cells: fill, dilate, and verify masked FTLE zeros land."""
    _, tvals, xvals, yvals, U, V = _sampled_double_gyre(nx=101, ny=51)
    U[:, 40:48, 20:28] = np.nan
    V[:, 40:48, 20:28] = np.nan
    (u, v), mask = fill_nans_and_get_mask([U, V])
    assert mask.shape == (101, 51) and mask.sum() == 8 * 8
    dilated = binary_dilation(jnp.asarray(mask))
    assert int(dilated.sum()) > int(mask.sum())

    gridded = GriddedField2D(
        u=u, v=v, tvals=jnp.asarray(tvals), xvals=jnp.asarray(xvals),
        yvals=jnp.asarray(yvals),
    )
    x = jnp.asarray(xvals)
    y = jnp.asarray(yvals)
    fmap = flowmap_grid(gridded, x, y, 0.0, 2.0, dt=0.1)
    ftle = ftle_grid(fmap, 2.0, float(x[1] - x[0]), float(y[1] - y[0]),
                     mask=dilated)
    assert float(jnp.abs(ftle[44, 24])) == 0.0
    assert np.all(np.isfinite(np.asarray(ftle)))


def test_spherical_velocity_conversion():
    """spherical=True: u (km/day) at latitude 60° moves lon at
    u/(r cos 60°) rad/day = 2u/r — checked against an exact great-circle
    displacement over a short time."""
    r = 6371.0
    tvals = jnp.linspace(0.0, 10.0, 3)
    lon = jnp.linspace(0.0, 10.0, 21)
    lat = jnp.linspace(50.0, 70.0, 21)
    u = jnp.full((3, 21, 21), 100.0)  # km/day eastward
    v = jnp.zeros((3, 21, 21))
    field = GriddedField2D(u=u, v=v, tvals=tvals, xvals=lon, yvals=lat,
                           spherical=True, r=r)
    x0 = jnp.array([5.0, 60.0])
    xT = integrate(field, x0, 0.0, 1.0, n_steps=50)
    dlon_expected = np.rad2deg(100.0 / (r * np.cos(np.deg2rad(60.0))))
    np.testing.assert_allclose(float(xT[0]) - 5.0, dlon_expected, rtol=1e-9)
    np.testing.assert_allclose(float(xT[1]), 60.0, atol=1e-12)
