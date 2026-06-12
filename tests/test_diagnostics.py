"""FTLE / LAVD / iLE against analytical fields and an independent
SciPy-based reference implementation."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from jaxcs import (
    ADVorticity,
    AnalyticalField,
    S_eig_grid,
    cauchy_green_grid,
    double_gyre,
    eig_cauchy_green,
    flowmap_aux_grid,
    flowmap_grid,
    flowmap_trajectory_grid,
    ftle_ad_grid,
    ftle_aux_grid,
    ftle_grid,
    ile_field,
    ile_grid,
    ivd_grid,
    lavd_from_trajectories,
    lavd_grid,
    uniform_grid,
    vorticity_grid,
)

A = 0.3
SADDLE = AnalyticalField(lambda t, x: jnp.stack([A * x[0], -A * x[1]]))


def _saddle_grid():
    x = jnp.linspace(-1.0, 1.0, 21)
    y = jnp.linspace(-1.0, 1.0, 21)
    return x, y


# --------------------------------------------------------------------------
# FTLE: analytical saddle, forward and backward, all three Jacobian paths
# --------------------------------------------------------------------------

@pytest.mark.parametrize("T", [2.0, -2.0], ids=["forward", "backward"])
def test_ftle_grid_saddle(T):
    """Linear saddle u=(ax,-ay): FTLE = a exactly, forward AND backward
    (backward FTLE positive — sign convention check)."""
    x, y = _saddle_grid()
    fmap = flowmap_grid(SADDLE, x, y, 0.0, T, n_steps=200)
    ftle = ftle_grid(fmap, T, float(x[1] - x[0]), float(y[1] - y[0]))
    np.testing.assert_allclose(np.asarray(ftle), A, rtol=1e-6)


@pytest.mark.parametrize("T", [2.0, -2.0], ids=["forward", "backward"])
def test_ftle_ad_saddle(T):
    x, y = _saddle_grid()
    ftle = ftle_ad_grid(SADDLE, x, y, 0.0, T, n_steps=200)
    np.testing.assert_allclose(np.asarray(ftle), A, rtol=1e-6)


def test_ftle_aux_grid_saddle():
    x, y = _saddle_grid()
    fmap_aux = flowmap_aux_grid(SADDLE, x, y, 0.0, 2.0, h=1e-5, n_steps=200)
    ftle = ftle_aux_grid(fmap_aux, 2.0, h=1e-5)
    np.testing.assert_allclose(np.asarray(ftle), A, rtol=1e-5)


def test_cauchy_green_eigvecs_saddle():
    """Dominant stretch of the forward saddle is the x-direction; eigh
    returns it as the LAST column of eigvecs."""
    x, y = _saddle_grid()
    fmap = flowmap_grid(SADDLE, x, y, 0.0, 1.0, n_steps=100)
    C = cauchy_green_grid(fmap, float(x[1] - x[0]), float(y[1] - y[0]))
    eigvals, eigvecs = eig_cauchy_green(C)
    np.testing.assert_allclose(np.asarray(eigvals[..., -1]), np.exp(2 * A), rtol=1e-5)
    vmax = np.abs(np.asarray(eigvecs[5, 5, :, -1]))
    np.testing.assert_allclose(vmax, [1.0, 0.0], atol=1e-8)


def test_ftle_fd_vs_ad_consistency_double_gyre():
    """FD-grid and jacrev FTLE agree on a smooth flow at moderate T."""
    field = double_gyre()
    x, y = uniform_grid(field.domain, 81, 41)
    T = 4.0
    fmap = flowmap_grid(field, x, y, 0.0, T, dt=0.05)
    fd = np.asarray(ftle_grid(fmap, T, float(x[1] - x[0]), float(y[1] - y[0])))
    ad = np.asarray(ftle_ad_grid(field, x, y, 0.0, T, n_steps=80))
    # interior only (FD uses one-sided stencils at boundaries); correlation
    # below 1 reflects FD smoothing of sharp ridges, not error — both paths
    # are exact on the analytical saddle above
    diff = np.abs(fd[1:-1, 1:-1] - ad[1:-1, 1:-1])
    assert np.median(diff) < 0.01
    assert np.corrcoef(fd[1:-1, 1:-1].ravel(), ad[1:-1, 1:-1].ravel())[0, 1] > 0.95


def test_ftle_double_gyre_vs_scipy_reference():
    """Full independent reference: flow map by SciPy RK45 (rtol 1e-10),
    FTLE by NumPy central differences."""
    field = double_gyre()
    x = np.linspace(0.0, 2.0, 25)
    y = np.linspace(0.0, 1.0, 13)
    T = 6.0

    def rhs(t, xy):
        return np.asarray(field(jnp.asarray(t), jnp.asarray(xy)))

    ref_fmap = np.zeros((25, 13, 2))
    for i in range(25):
        for j in range(13):
            sol = solve_ivp(rhs, (0, T), [x[i], y[j]], rtol=1e-10, atol=1e-10)
            ref_fmap[i, j] = sol.y[:, -1]
    dx, dy = x[1] - x[0], y[1] - y[0]
    dXdx, dXdy = np.gradient(ref_fmap[..., 0], dx, dy)
    dYdx, dYdy = np.gradient(ref_fmap[..., 1], dx, dy)
    C11 = dXdx**2 + dYdx**2
    C12 = dXdx * dXdy + dYdx * dYdy
    C22 = dXdy**2 + dYdy**2
    lam = 0.5 * (C11 + C22) + np.sqrt(0.25 * (C11 - C22) ** 2 + C12**2)
    ref_ftle = np.maximum(np.log(np.maximum(lam, 1.0)) / (2 * T), 0.0)

    fmap = flowmap_grid(field, jnp.asarray(x), jnp.asarray(y), 0.0, T, dt=0.01)
    ours = np.asarray(ftle_grid(fmap, T, dx, dy))
    np.testing.assert_allclose(ours, ref_ftle, atol=2e-5)


def test_ftle_mask():
    x, y = _saddle_grid()
    fmap = flowmap_grid(SADDLE, x, y, 0.0, 2.0, n_steps=100)
    mask = jnp.zeros((21, 21), bool).at[3, 4].set(True)
    ftle = ftle_grid(fmap, 2.0, float(x[1] - x[0]), float(y[1] - y[0]), mask=mask)
    assert float(ftle[3, 4]) == 0.0 and float(ftle[10, 10]) > 0.0


# --------------------------------------------------------------------------
# LAVD
# --------------------------------------------------------------------------

def test_lavd_zero_velocity_linear_vorticity():
    """With zero advection and ω(x) = x, LAVD = |x0 - mean(x)| * |T|."""
    still = AnalyticalField(lambda t, x: jnp.zeros(2))
    # curl of (0, x^2/2) is x
    omega_src = AnalyticalField(lambda t, x: jnp.stack([0.0 * x[0], x[0] ** 2 / 2]))
    vort = ADVorticity(omega_src)
    x = jnp.linspace(0.0, 1.0, 11)
    y = jnp.linspace(0.0, 1.0, 5)
    T = 3.0
    lavd = lavd_grid(still, vort, x, y, 0.0, T, n_steps=60)
    expected = np.abs(np.asarray(x)[:, None] - 0.5) * T * np.ones((1, 5))
    np.testing.assert_allclose(np.asarray(lavd), expected, atol=1e-10)


def test_lavd_rigid_rotation_is_zero():
    """Rigid rotation: ω constant in space → deviation ≡ 0 → LAVD = 0."""
    rot = AnalyticalField(lambda t, x: jnp.stack([-x[1], x[0]]))
    vort = ADVorticity(rot)
    x = jnp.linspace(-0.5, 0.5, 9)
    y = jnp.linspace(-0.5, 0.5, 9)
    lavd = lavd_grid(rot, vort, x, y, 0.0, 2.0, n_steps=100)
    np.testing.assert_allclose(np.asarray(lavd), 0.0, atol=1e-12)


def test_lavd_two_stage_matches_fused():
    """lavd_from_trajectories with explicit grid-average vorticity matches
    the fused single-scan lavd_grid."""
    field = double_gyre()
    vort = ADVorticity(field)
    x = jnp.linspace(0.0, 2.0, 31)
    y = jnp.linspace(0.0, 1.0, 16)
    T, n = 4.0, 80
    fused = lavd_grid(field, vort, x, y, 0.0, T, n_steps=n)

    ts, trajs = flowmap_trajectory_grid(field, x, y, 0.0, T, n_steps=n)
    import jax

    X, Y = jnp.meshgrid(x, y, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    vort_at = jax.vmap(jax.vmap(vort, in_axes=(None, 0)), in_axes=(0, None))
    vort_avg = jnp.mean(vort_at(ts, pts), axis=1)
    two_stage = lavd_from_trajectories(ts, trajs, vort, vort_avg=vort_avg)
    np.testing.assert_allclose(np.asarray(fused), np.asarray(two_stage), atol=1e-10)


# --------------------------------------------------------------------------
# iLE / S tensor / IVD / vorticity
# --------------------------------------------------------------------------

def test_ile_linear_strain():
    """u = (ax, -ay): S = diag(a, -a), iLE = a everywhere; AD and gridded
    paths agree."""
    pts = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, 5), jnp.linspace(-1, 1, 5),
                                 indexing="ij"), axis=-1)
    ile = ile_field(SADDLE, 0.0, pts)
    np.testing.assert_allclose(np.asarray(ile), A, atol=1e-12)

    x = np.linspace(-1, 1, 11)
    X, Y = np.meshgrid(x, x, indexing="ij")
    ile_d = ile_grid(jnp.asarray(A * X), jnp.asarray(-A * Y), x[1] - x[0], x[1] - x[0])
    np.testing.assert_allclose(np.asarray(ile_d), A, atol=1e-10)


def test_s_eig_grid_shear():
    """Pure shear u=(cy, 0): S eigenvalues ±c/2."""
    c = 0.8
    x = np.linspace(0, 1, 11)
    X, Y = np.meshgrid(x, x, indexing="ij")
    eigvals, _ = S_eig_grid(jnp.asarray(c * Y), jnp.zeros_like(jnp.asarray(X)),
                            x[1] - x[0], x[1] - x[0])
    np.testing.assert_allclose(np.asarray(eigvals[..., -1]), c / 2, atol=1e-10)
    np.testing.assert_allclose(np.asarray(eigvals[..., 0]), -c / 2, atol=1e-10)


def test_vorticity_grid_and_ivd():
    x = np.linspace(0, 2 * np.pi, 64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = jnp.asarray(np.sin(Y))
    v = jnp.asarray(np.zeros_like(X))
    vort = vorticity_grid(u, v, x[1] - x[0], x[1] - x[0])
    np.testing.assert_allclose(np.asarray(vort[2:-2, 2:-2]),
                               -np.cos(Y)[2:-2, 2:-2], atol=2e-3)
    ivd = ivd_grid(vort, jnp.mean(vort))
    assert float(jnp.min(ivd)) >= 0.0


def test_ftle_is_differentiable_wrt_params():
    """End-to-end differentiability: gradient of mean FTLE w.r.t. the
    double-gyre amplitude parameter exists and is finite."""
    import jax

    def loss(amp):
        field = double_gyre(A=amp)
        x = jnp.linspace(0.5, 1.5, 8)
        y = jnp.linspace(0.25, 0.75, 6)
        return jnp.mean(ftle_ad_grid(field, x, y, 0.0, 2.0, n_steps=40))

    g = jax.grad(loss)(jnp.asarray(0.1))
    assert np.isfinite(float(g)) and float(g) != 0.0
