"""LCS extraction: ridges, tensorlines, elliptic vortices.

Includes the two canonical validation cases: double-gyre hyperbolic LCS
and Bickley-jet LAVD vortices.
"""

import jax.numpy as jnp
import numpy as np

from jaxcs import (
    ADVorticity,
    bickley_jet,
    double_gyre,
    eig_cauchy_green,
    cauchy_green_grid,
    flowmap_grid,
    ftle_grid,
    ftle_ordered_ridges,
    ftle_ridge_points,
    hyperbolic_lcs,
    lavd_grid,
    rotcohvrt,
    tensorlines,
    uniform_grid,
)


# --------------------------------------------------------------------------
# Ridge detection on synthetic fields
# --------------------------------------------------------------------------

def _synthetic_ridge():
    """f = exp(-y²/2σ²): a ridge along y = 0; transverse direction (0,1)."""
    x = jnp.linspace(0.0, 4.0, 81)
    y = jnp.linspace(-1.0, 1.0, 41)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    f = jnp.exp(-(Y**2) / (2 * 0.2**2))
    ev = jnp.zeros((81, 41, 2)).at[..., 1].set(1.0)
    return f, ev, x, y


def test_ridge_points_synthetic():
    f, ev, x, y = _synthetic_ridge()
    pts, ok = ftle_ridge_points(f, ev, x, y, sdd_thresh=1.0)
    ridge = np.asarray(pts)[np.asarray(ok)]
    assert len(ridge) > 50
    np.testing.assert_allclose(ridge[:, 1], 0.0, atol=1e-6)


def test_ordered_ridges_synthetic():
    f, ev, x, y = _synthetic_ridge()
    ridges = ftle_ordered_ridges(f, ev, x, y, sdd_thresh=1.0)
    assert len(ridges) == 1
    r = ridges[0]
    # spans (almost) the full detectable x-extent, ordered monotonically
    assert r[:, 0].max() - r[:, 0].min() > 3.5
    assert np.all(np.abs(np.diff(r[:, 0])) > 0)


def test_tensorlines_straight_field():
    """Constant eigenvector field (1, 0): tensorlines are straight x-lines."""
    x = jnp.linspace(0.0, 1.0, 11)
    y = jnp.linspace(0.0, 1.0, 11)
    ev = jnp.zeros((11, 11, 2)).at[..., 0].set(1.0)
    seeds = jnp.array([[[0.1, 0.5], [1.0, 0.0]]])
    lines, alive = tensorlines(ev, x, y, seeds, h=0.05, steps=30)
    line = np.asarray(lines[0])[np.asarray(alive[0])]
    np.testing.assert_allclose(line[:, 1], 0.5, atol=1e-12)
    assert line[-1, 0] > 0.9  # marched until the domain edge froze it


def test_tensorlines_sign_ambiguity():
    """Eigenvector field with random per-node signs must integrate the
    same line as the consistently-oriented field."""
    x = jnp.linspace(0.0, 1.0, 11)
    y = jnp.linspace(0.0, 1.0, 11)
    ev = jnp.zeros((11, 11, 2)).at[..., 0].set(1.0)
    rng = np.random.default_rng(3)
    signs = jnp.asarray(rng.choice([-1.0, 1.0], size=(11, 11, 1)))
    seeds = jnp.array([[[0.1, 0.5], [1.0, 0.0]]])
    a, _ = tensorlines(ev, x, y, seeds, h=0.05, steps=15)
    b, _ = tensorlines(ev * signs, x, y, seeds, h=0.05, steps=15)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)


# --------------------------------------------------------------------------
# Elliptic extraction on synthetic LAVD
# --------------------------------------------------------------------------

def test_rotcohvrt_gaussian_bump():
    x = np.linspace(-2.0, 2.0, 101)
    y = np.linspace(-2.0, 2.0, 101)
    X, Y = np.meshgrid(x, y, indexing="ij")
    lavd = np.exp(-(X**2 + Y**2) / 0.5)
    # exclusion radius must cover the bump's footprint above the 80th
    # percentile, else ring points seed spurious maxima (numbacs behaves
    # identically)
    rcv = rotcohvrt(lavd, x, y, r=1.5)
    assert len(rcv) == 1
    boundary, center = rcv[0]
    np.testing.assert_allclose(center, [0.0, 0.0], atol=0.05)
    # boundary is a closed, roughly circular contour around the center
    radii = np.hypot(boundary[:, 0], boundary[:, 1])
    assert radii.std() / radii.mean() < 0.05
    np.testing.assert_allclose(boundary[0], boundary[-1])


# --------------------------------------------------------------------------
# Canonical cases
# --------------------------------------------------------------------------

def test_double_gyre_ftle_ridges():
    """Forward-FTLE ridge of the double gyre at t0=0, T=8 passes near the
    known repelling structure emanating from (1, 0) (Shadden et al. 2005)."""
    field = double_gyre()
    x, y = uniform_grid(field.domain, 201, 101)
    T = 8.0
    fmap = flowmap_grid(field, x, y, 0.0, T, dt=0.05)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    f = ftle_grid(fmap, T, dx, dy)
    C = cauchy_green_grid(fmap, dx, dy)
    _, eigvecs = eig_cauchy_green(C)
    ridges = ftle_ordered_ridges(f, eigvecs[..., :, -1], x, y, percentile=80,
                                 min_ridge_pts=10)
    assert len(ridges) >= 1
    allpts = np.concatenate(ridges)
    # the dominant ridge attaches near the bottom boundary around x ≈ 1
    near_anchor = allpts[(allpts[:, 1] < 0.2)]
    assert len(near_anchor) > 0
    assert np.any(np.abs(near_anchor[:, 0] - 1.0) < 0.35)


def test_double_gyre_hyperbolic_lcs():
    field = double_gyre()
    x, y = uniform_grid(field.domain, 201, 101)
    T = 8.0
    fmap = flowmap_grid(field, x, y, 0.0, T, dt=0.05)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    C = cauchy_green_grid(fmap, dx, dy)
    eigvals, eigvecs = eig_cauchy_green(C)
    curves = hyperbolic_lcs(eigvals[..., -1], eigvecs[..., :, 0], x, y,
                            r=0.4, steps=1000, n_seeds=4, min_len=0.2)
    assert len(curves) >= 1
    # strongest curve stays in the domain
    c = curves[0]
    assert c[:, 0].min() >= -1e-9 and c[:, 0].max() <= 2.0 + 1e-9


def test_bickley_jet_lavd_vortices():
    """Bickley jet: LAVD level sets give rotationally coherent vortices
    centered in the recirculation regions away from the meandering jet
    core (|y| around 1.5–3 Mm), cf. Hadjighasem et al. 2017."""
    field = bickley_jet()
    vort = ADVorticity(field)
    x = jnp.linspace(0.0, field.period_x, 241)
    y = jnp.linspace(-3.0, 3.0, 73)
    T = 20.0
    lavd = lavd_grid(field, vort, x, y, 0.0, T, n_steps=400)
    rcv = rotcohvrt(np.asarray(lavd), np.asarray(x), np.asarray(y), r=2.0,
                    convexity_deficiency=2e-2)
    assert len(rcv) >= 2
    centers = np.array([c for _, c in rcv])
    # vortices live off the jet core, inside the channel
    assert np.all(np.abs(centers[:, 1]) > 0.5)
    assert np.all(np.abs(centers[:, 1]) < 3.0)
