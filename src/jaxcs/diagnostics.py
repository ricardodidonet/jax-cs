"""LCS diagnostics: FTLE, Cauchy–Green tensor, LAVD, iLE, IVD.

Conventions (full statement in ``docs/conventions.md``):

* Flow-map gradient ``F = dΦ/dx0``; Cauchy–Green tensor ``C = Fᵀ F``.
* FTLE ``σ = log(λ_max(C)) / (2 |T|)``, clipped below at 0 (numbacs
  convention: cells with ``λ_max ≤ 1`` report 0).  The ``|T|`` makes the
  *backward* FTLE (computed from a flow map with ``T < 0``) positive on
  attracting structures — same sign convention as numbacs.
* Grid arrays are ``(nx, ny)`` with x first; eigenvalue/vector outputs of
  the batched ``jnp.linalg.eigh`` follow NumPy semantics: for input
  ``(..., 2, 2)``, ``eigvals[..., i]`` is the i-th *ascending* eigenvalue
  and ``eigvecs[..., :, i]`` (a **column**, not a row) is its eigenvector.

FTLE Jacobian: two paths
------------------------
``ftle_grid`` differentiates the *gridded* flow map by central finite
differences (cheap: one trajectory per node; accuracy limited by grid
spacing — O(dx²) and severely degraded once neighbouring trajectories
separate nonlinearly).  ``ftle_ad`` instead differentiates each
trajectory's flow map with :func:`jax.jacrev` (exact derivative of the
*discrete* integrator; ~3–5× the cost of the plain flow map for 2D, and
independent of grid spacing).  See ``docs/conventions.md`` for benchmark
numbers and guidance; the regression tests check both paths against an
analytical saddle flow, forward and backward.

Improvement over numbacs (documented deviation): interior derivatives are
2nd-order central differences as in numbacs, but boundary nodes use
one-sided differences (via ``jnp.gradient``) instead of being zeroed.
``ile`` computes ``∇v`` by automatic differentiation rather than numbacs's
``h``-stencil finite differences, so it is exact for analytical fields.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .fields import Field, ScalarField2D
from .integration import Method, integrate

__all__ = [
    "cauchy_green_grid",
    "cauchy_green_aux",
    "cauchy_green_ad",
    "eig_cauchy_green",
    "ftle_from_eigval",
    "ftle_grid",
    "ftle_aux_grid",
    "ftle_ad",
    "ftle_ad_grid",
    "lavd_grid",
    "lavd_from_trajectories",
    "ile_grid",
    "ile_field",
    "S_tensor_field",
    "S_eig_field",
    "S_eig_grid",
    "ivd_grid",
]


# --------------------------------------------------------------------------
# Cauchy–Green tensor
# --------------------------------------------------------------------------

def _flowmap_gradient_grid(fmap: Array, dx: float, dy: float) -> Array:
    """Finite-difference gradient of a gridded flow map.

    ``fmap``: ``(nx, ny, 2)``.  Returns ``F`` with shape ``(nx, ny, 2, 2)``
    where ``F[..., a, b] = d fmap_a / d x0_b``.
    """
    dXdx, dXdy = jnp.gradient(fmap[..., 0], dx, dy)
    dYdx, dYdy = jnp.gradient(fmap[..., 1], dx, dy)
    return jnp.stack(
        [jnp.stack([dXdx, dXdy], axis=-1), jnp.stack([dYdx, dYdy], axis=-1)],
        axis=-2,
    )


def cauchy_green_grid(fmap: Array, dx: float, dy: float) -> Array:
    """Cauchy–Green tensor ``C = FᵀF`` from a gridded flow map.

    Parameters
    ----------
    fmap
        Flow map on the grid, shape ``(nx, ny, 2)``.
    dx, dy
        Grid spacing.

    Returns
    -------
    ``C`` with shape ``(nx, ny, 2, 2)`` (symmetric positive semidefinite).
    """
    F = _flowmap_gradient_grid(fmap, dx, dy)
    return jnp.einsum("...ka,...kb->...ab", F, F)


def cauchy_green_aux(fmap_aux: Array, h: float) -> Array:
    """Cauchy–Green tensor from an auxiliary-grid flow map.

    ``fmap_aux`` has shape ``(nx, ny, 5, 2)`` ordered
    ``[center, +x, -x, +y, -y]`` (output of
    :func:`jaxcs.flowmap.flowmap_aux_grid`); derivatives are central
    differences over the ``2h`` aux spacing, decoupling Jacobian accuracy
    from the main grid spacing (numbacs's ``C_tensor_2D``).
    """
    dFdx = (fmap_aux[..., 1, :] - fmap_aux[..., 2, :]) / (2 * h)
    dFdy = (fmap_aux[..., 3, :] - fmap_aux[..., 4, :]) / (2 * h)
    F = jnp.stack([dFdx, dFdy], axis=-1)  # (nx, ny, 2(a), 2(b))
    return jnp.einsum("...ka,...kb->...ab", F, F)


def _flowmap_jacobian_point(
    field: Field, x0: Array, t0: Array, T: Array, n_steps: int,
    method: Method, checkpoint: bool,
) -> Array:
    fn = lambda x: integrate(field, x, t0, T, n_steps, method=method,
                             checkpoint=checkpoint)
    return jax.jacrev(fn)(x0)


def cauchy_green_ad(
    field: Field,
    x0s: Array,
    t0: float,
    T: float,
    n_steps: int,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Array:
    """Cauchy–Green tensor at arbitrary points via ``jacrev`` of the flow map.

    ``x0s``: ``(n, 2)`` (or any leading batch shape ``(..., 2)``).
    Returns ``C`` of shape ``(..., 2, 2)``.
    """
    batch = x0s.shape[:-1]
    pts = x0s.reshape(-1, x0s.shape[-1])
    jac = jax.vmap(
        partial(_flowmap_jacobian_point, field, t0=jnp.asarray(t0),
                T=jnp.asarray(T), n_steps=n_steps, method=method,
                checkpoint=checkpoint)
    )(pts)
    C = jnp.einsum("...ka,...kb->...ab", jac, jac)
    return C.reshape(*batch, x0s.shape[-1], x0s.shape[-1])


def eig_cauchy_green(C: Array) -> Tuple[Array, Array]:
    """Batched eigendecomposition of (symmetric) ``C``.

    Accepts ``C`` of shape ``(..., d, d)``.  Returns
    ``(eigvals, eigvecs)`` with ``eigvals[..., i]`` ascending and the
    matching eigenvector in the **column** ``eigvecs[..., :, i]`` — so the
    dominant stretch direction is ``eigvecs[..., :, -1]`` and the
    contracting direction ``eigvecs[..., :, 0]``.
    """
    return jnp.linalg.eigh(C)


# --------------------------------------------------------------------------
# FTLE
# --------------------------------------------------------------------------

def ftle_from_eigval(eigval_max: Array, T: float) -> Array:
    """FTLE from the maximum Cauchy–Green eigenvalue.

    ``σ = log(λ_max) / (2 |T|)``, clipped below at 0 (numbacs convention).
    Pass the *signed* ``T`` used for the flow map; backward FTLE
    (``T < 0``) is positive on attracting material lines.
    """
    sigma = jnp.log(jnp.maximum(eigval_max, 1.0)) / (2.0 * jnp.abs(T))
    return sigma


def _apply_mask(arr: Array, mask: Optional[Array]) -> Array:
    if mask is None:
        return arr
    return jnp.where(mask, 0.0, arr)


@eqx.filter_jit
def ftle_grid(
    fmap: Array, T: float, dx: float, dy: float, mask: Optional[Array] = None
) -> Array:
    """FTLE field from a gridded flow map via finite differences.

    Parameters
    ----------
    fmap
        Flow map, shape ``(nx, ny, 2)`` (from
        :func:`jaxcs.flowmap.flowmap_grid`).
    T
        Signed integration horizon used to produce ``fmap``.
    dx, dy
        Seeding-grid spacing.
    mask
        Optional boolean ``(nx, ny)`` array, True on invalid (e.g. land)
        cells; masked cells return 0.  Dilate the mask first
        (:func:`jaxcs.utils.binary_dilation`) so stencils never straddle
        the boundary.

    Returns
    -------
    FTLE, shape ``(nx, ny)``.
    """
    C = cauchy_green_grid(fmap, dx, dy)
    eigvals = jnp.linalg.eigvalsh(C)
    return _apply_mask(ftle_from_eigval(eigvals[..., -1], T), mask)


def ftle_aux_grid(
    fmap_aux: Array, T: float, h: float, mask: Optional[Array] = None
) -> Array:
    """FTLE from an auxiliary-grid flow map (``(nx, ny, 5, 2)``)."""
    C = cauchy_green_aux(fmap_aux, h)
    eigvals = jnp.linalg.eigvalsh(C)
    return _apply_mask(ftle_from_eigval(eigvals[..., -1], T), mask)


def ftle_ad(
    field: Field,
    x0s: Array,
    t0: float,
    T: float,
    n_steps: int,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Array:
    """FTLE at arbitrary points via automatic differentiation.

    Differentiates each point's flow map with ``jacrev`` — exact for the
    discrete trajectory, no grid-resolution error, and meaningful at
    isolated (Lagrangian) particles where no finite-difference neighbours
    exist.  Costs roughly the equivalent of a few extra trajectories per
    point; see module docstring for the FD-vs-AD trade-off.

    ``x0s``: shape ``(..., 2)``; returns shape ``(...,)``.
    """
    C = cauchy_green_ad(field, x0s, t0, T, n_steps, method=method,
                        checkpoint=checkpoint)
    eigvals = jnp.linalg.eigvalsh(C)
    return ftle_from_eigval(eigvals[..., -1], T)


def ftle_ad_grid(
    field: Field,
    x: Array,
    y: Array,
    t0: float,
    T: float,
    n_steps: int,
    method: Method = "rk4",
    checkpoint: bool = False,
    mask: Optional[Array] = None,
) -> Array:
    """:func:`ftle_ad` evaluated on the tensor grid ``x × y`` → ``(nx, ny)``."""
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    pts = jnp.stack([X, Y], axis=-1)
    return _apply_mask(
        ftle_ad(field, pts, t0, T, n_steps, method=method, checkpoint=checkpoint),
        mask,
    )


# --------------------------------------------------------------------------
# LAVD
# --------------------------------------------------------------------------

def lavd_grid(
    field: Field,
    vorticity: ScalarField2D,
    x: Array,
    y: Array,
    t0: float,
    T: float,
    n_steps: int,
    method: Method = "rk4",
    mask: Optional[Array] = None,
) -> Array:
    """Lagrangian-averaged vorticity deviation on a grid, in one scan.

    .. math:: \\mathrm{LAVD}(x_0) = \\int_{t_0}^{t_0+T}
        \\bigl|\\,\\omega(x(t), t) - \\bar\\omega(t)\\,\\bigr|\\, dt

    where ``ω̄(t)`` is the spatial mean of vorticity over the seeding grid
    at time ``t`` (numbacs convention; no ``1/|T|`` normalisation).  The
    advection *and* the deviation quadrature run inside the same
    ``lax.scan`` — trajectories are never materialised.

    Documented deviation from numbacs: the time integral uses the
    trapezoidal rule (O(dt²)) instead of composite Simpson, which is what
    fusing the quadrature into the scan naturally yields; with the small
    ``dt`` a fixed-step integrator needs anyway, the difference is far
    below the vorticity-interpolation error.

    Parameters
    ----------
    field
        Velocity field to advect particles in.
    vorticity
        Scalar interpolant ``ω(t, x)`` (build with
        :func:`jaxcs.utils.vorticity_field` or from data); set its
        ``period_x``/``period_y`` for periodic domains.
    x, y
        Seeding grid (also where ``ω̄`` is averaged).
    t0, T, n_steps, method
        Integration controls as in :func:`jaxcs.flowmap.flowmap_grid`.
    mask
        Optional boolean ``(nx, ny)``; masked cells return 0.

    Returns
    -------
    LAVD, shape ``(nx, ny)``.
    """
    from .integration import _STEPPERS

    X, Y = jnp.meshgrid(x, y, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (npts, 2)
    dt = jnp.asarray(T) / n_steps
    ts = t0 + dt * jnp.arange(n_steps + 1)

    # Eulerian mean vorticity over the seeding grid at every step time.
    vort_at = jax.vmap(jax.vmap(vorticity, in_axes=(None, 0)), in_axes=(0, None))
    vort_avg = jnp.mean(vort_at(ts, pts), axis=1)  # (n_steps + 1,)

    step = _STEPPERS[method]
    w = jnp.abs(dt)  # quadrature weight is unsigned time

    def dev(k, p):
        return jnp.abs(jax.vmap(vorticity, in_axes=(None, 0))(ts[k], p) - vort_avg[k])

    def body(carry, k):
        p, acc = carry
        d0 = dev(k, p)
        p_next = jax.vmap(lambda q: step(field, ts[k], q, dt))(p)
        d1 = dev(k + 1, p_next)
        return (p_next, acc + 0.5 * w * (d0 + d1)), None

    init = (pts, jnp.zeros(pts.shape[0], dtype=pts.dtype))
    (_, lavd), _ = jax.lax.scan(body, init, jnp.arange(n_steps))
    return _apply_mask(lavd.reshape(x.shape[0], y.shape[0]), mask)


def lavd_from_trajectories(
    ts: Array,
    trajs: Array,
    vorticity: ScalarField2D,
    vort_avg: Optional[Array] = None,
) -> Array:
    """LAVD from precomputed trajectories (two-stage path).

    Parameters
    ----------
    ts
        Trajectory times, shape ``(n,)``.
    trajs
        Trajectories, shape ``(..., n, 2)`` (e.g. the
        ``(nx, ny, n, 2)`` output of
        :func:`jaxcs.flowmap.flowmap_trajectory_grid`).
    vorticity
        Scalar interpolant ``ω(t, x)``.
    vort_avg
        Optional precomputed spatial-mean vorticity at ``ts``; if None it
        is averaged over the trajectory positions at each time (close to,
        but not exactly, the Eulerian grid mean — pass it explicitly to
        reproduce :func:`lavd_grid`).

    Returns
    -------
    LAVD with shape ``trajs.shape[:-2]``.
    """
    batch = trajs.shape[:-2]
    p = trajs.reshape(-1, ts.shape[0], 2)
    vort = jax.vmap(
        jax.vmap(vorticity, in_axes=(0, 0)), in_axes=(None, 0)
    )(ts, p)  # (npts, n)
    if vort_avg is None:
        vort_avg = jnp.mean(vort, axis=0)
    dev = jnp.abs(vort - vort_avg[None, :])
    lavd = jnp.trapezoid(dev, jnp.abs(ts - ts[0]), axis=1)
    return lavd.reshape(batch)


# --------------------------------------------------------------------------
# Instantaneous diagnostics: iLE, S tensor, IVD
# --------------------------------------------------------------------------

def S_tensor_field(field: Field, t0: float, pts: Array) -> Array:
    """Rate-of-strain tensor ``S = (∇v + ∇vᵀ)/2`` at points ``(..., 2)``.

    ``∇v`` is computed exactly by forward-mode AD of the field (numbacs
    uses finite differences with step ``h``; AD removes that parameter).
    Returns shape ``(..., 2, 2)``.
    """
    batch = pts.shape[:-1]
    p = pts.reshape(-1, pts.shape[-1])
    t = jnp.asarray(t0)
    grad_v = jax.vmap(jax.jacfwd(lambda q: field(t, q)))(p)
    S = 0.5 * (grad_v + jnp.swapaxes(grad_v, -1, -2))
    return S.reshape(*batch, pts.shape[-1], pts.shape[-1])


def S_eig_field(field: Field, t0: float, pts: Array) -> Tuple[Array, Array]:
    """Eigendecomposition of the rate-of-strain tensor at ``pts``.

    Returns ``(eigvals, eigvecs)`` in :func:`eig_cauchy_green`'s
    (ascending, column-vector) layout.
    """
    return jnp.linalg.eigh(S_tensor_field(field, t0, pts))


def ile_field(field: Field, t0: float, pts: Array) -> Array:
    """Instantaneous Lyapunov exponent ``s_max = λ_max(S)`` at ``pts``."""
    return jnp.linalg.eigvalsh(S_tensor_field(field, t0, pts))[..., -1]


def _S_from_data(u: Array, v: Array, dx: float, dy: float) -> Array:
    dudx, dudy = jnp.gradient(u, dx, dy)
    dvdx, dvdy = jnp.gradient(v, dx, dy)
    return jnp.stack(
        [
            jnp.stack([dudx, 0.5 * (dudy + dvdx)], axis=-1),
            jnp.stack([0.5 * (dudy + dvdx), dvdy], axis=-1),
        ],
        axis=-2,
    )


def ile_grid(
    u: Array, v: Array, dx: float, dy: float, mask: Optional[Array] = None
) -> Array:
    """iLE from a gridded velocity snapshot ``u, v`` of shape ``(nx, ny)``."""
    S = _S_from_data(u, v, dx, dy)
    return _apply_mask(jnp.linalg.eigvalsh(S)[..., -1], mask)


def S_eig_grid(
    u: Array, v: Array, dx: float, dy: float
) -> Tuple[Array, Array]:
    """Rate-of-strain eigendecomposition from a gridded snapshot."""
    return jnp.linalg.eigh(_S_from_data(u, v, dx, dy))


def ivd_grid(vort: Array, vort_avg: Array) -> Array:
    """Instantaneous vorticity deviation ``|ω - ω̄|`` (numbacs ``ivd_grid_2D``)."""
    return jnp.abs(vort - vort_avg)
