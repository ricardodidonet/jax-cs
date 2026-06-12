"""Pure-JAX interpolation on uniform rectilinear grids.

Grid convention (see ``docs/conventions.md`` for the full statement):

* Scalar fields on a 2D spatial grid are stored as arrays of shape
  ``(nx, ny)`` — the **x axis comes first** (``ij`` indexing), matching
  numbacs.  Time-dependent fields are ``(nt, nx, ny)`` and 3D fields are
  ``(nt, nx, ny, nz)``.
* Coordinate arrays ``xvals``/``yvals``/``tvals`` must be uniformly spaced
  and strictly increasing.  Physical coordinates are converted to fractional
  index space via ``(q - vals[0]) / (vals[1] - vals[0])`` and looked up with
  multilinear interpolation (``jax.scipy.ndimage.map_coordinates`` with
  ``order=1``, plus a hand-rolled variant where finer control is needed).
* All interpolators are pure functions of their inputs, differentiable, and
  written so that the *query* arguments carry no batch dimension — wrap them
  in :func:`jax.vmap` for ensembles.  The Jacobian of an interpolated flow
  map with respect to the query point is therefore exactly the Jacobian
  downstream FTLE code (``diagnostics.ftle_ad``) assumes.

NaN / land-mask handling
------------------------
JAX has no masked arrays and NaNs poison gradients, so gridded data must be
made NaN-free *before* it enters the hot path:
:func:`jaxcs.utils.fill_nans_and_get_mask` replaces NaNs with a fill value
and returns a boolean mask which downstream diagnostics use to blank out
invalid cells.  This mirrors numbacs's recommended workflow
(``fill_nans_and_get_mask`` + dilated masks for finite differencing).
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates

ExtrapMode = Literal["constant", "nearest"]

__all__ = [
    "phys_to_index",
    "bilinear",
    "trilinear",
    "quadrilinear",
    "bilinear_manual",
]


def phys_to_index(q: Array, v0: Array, dv: Array) -> Array:
    """Convert physical coordinate(s) ``q`` to fractional grid-index space.

    Parameters
    ----------
    q
        Physical coordinate(s), any shape.
    v0
        Coordinate of grid index 0.
    dv
        (Uniform) grid spacing.
    """
    return (q - v0) / dv


def _time_index(f: Array, t: Array, tvals: Array) -> Array:
    """Fractional time index, clamped to the data's span.

    Handles steady data (``nt == 1``), where ``tvals[1]`` would silently
    clamp to ``tvals[0]`` under JAX indexing and yield a 0/0 NaN.
    """
    if f.shape[0] == 1:
        return jnp.zeros_like(jnp.asarray(t, dtype=f.dtype))
    it = phys_to_index(t, tvals[0], tvals[1] - tvals[0])
    return jnp.clip(it, 0.0, f.shape[0] - 1.0)


def _map_coords(f: Array, idx: list[Array], mode: ExtrapMode, cval: float) -> Array:
    if mode == "nearest":
        return map_coordinates(f, idx, order=1, mode="nearest")
    return map_coordinates(f, idx, order=1, mode="constant", cval=cval)


def bilinear(
    f: Array,
    x: Array,
    y: Array,
    xvals: Array,
    yvals: Array,
    mode: ExtrapMode = "constant",
    cval: float = 0.0,
) -> Array:
    """Bilinear interpolation of a steady 2D field ``f(x, y)``.

    Parameters
    ----------
    f
        Field values, shape ``(nx, ny)``.
    x, y
        Scalar physical query coordinates (vmap for batches).
    xvals, yvals
        Uniformly spaced coordinate arrays of length ``nx`` / ``ny``.
    mode
        Out-of-domain behaviour: ``"constant"`` returns ``cval``,
        ``"nearest"`` clamps to the boundary value.
    """
    ix = phys_to_index(x, xvals[0], xvals[1] - xvals[0])
    iy = phys_to_index(y, yvals[0], yvals[1] - yvals[0])
    return _map_coords(f, [ix, iy], mode, cval)


def trilinear(
    f: Array,
    t: Array,
    x: Array,
    y: Array,
    tvals: Array,
    xvals: Array,
    yvals: Array,
    mode: ExtrapMode = "constant",
    cval: float = 0.0,
) -> Array:
    """Trilinear interpolation of an unsteady 2D field ``f(t, x, y)``.

    ``f`` has shape ``(nt, nx, ny)``.  In time the lookup always clamps to
    the first/last frame (numbacs behaviour: velocity data is held constant
    outside its time span); ``mode`` only governs *spatial* extrapolation.
    Steady data may be passed with ``nt == 1`` (time is then ignored).
    """
    it = _time_index(f, t, tvals)
    ix = phys_to_index(x, xvals[0], xvals[1] - xvals[0])
    iy = phys_to_index(y, yvals[0], yvals[1] - yvals[0])
    if mode == "nearest":
        return map_coordinates(f, [it, ix, iy], order=1, mode="nearest")
    # Spatial constant-extrapolation with time clamping: clamp t above, let
    # x/y fall out of range so map_coordinates substitutes cval.
    return map_coordinates(f, [it, ix, iy], order=1, mode="constant", cval=cval)


def quadrilinear(
    f: Array,
    t: Array,
    x: Array,
    y: Array,
    z: Array,
    tvals: Array,
    xvals: Array,
    yvals: Array,
    zvals: Array,
    mode: ExtrapMode = "constant",
    cval: float = 0.0,
) -> Array:
    """Quadrilinear interpolation of an unsteady 3D field ``f(t, x, y, z)``.

    ``f`` has shape ``(nt, nx, ny, nz)``; time clamps as in
    :func:`trilinear`.
    """
    it = _time_index(f, t, tvals)
    ix = phys_to_index(x, xvals[0], xvals[1] - xvals[0])
    iy = phys_to_index(y, yvals[0], yvals[1] - yvals[0])
    iz = phys_to_index(z, zvals[0], zvals[1] - zvals[0])
    return _map_coords(f, [it, ix, iy, iz], mode, cval)


def bilinear_manual(
    f: Array,
    ix: Array,
    iy: Array,
    mode: ExtrapMode = "constant",
    cval: float = 0.0,
) -> Array:
    """Hand-rolled bilinear interpolation in *index space*.

    Equivalent to ``map_coordinates(f, [ix, iy], order=1)`` but with the
    corner gather spelled out, for cases needing finer control (e.g.
    custom corner weighting or NaN-aware variants built on top).

    Parameters
    ----------
    f
        Field values, shape ``(nx, ny)``.
    ix, iy
        Scalar fractional indices (vmap for batches).
    """
    nx, ny = f.shape
    i0f = jnp.floor(ix)
    j0f = jnp.floor(iy)
    wx = ix - i0f
    wy = iy - j0f
    i0 = jnp.clip(i0f.astype(jnp.int32), 0, nx - 1)
    j0 = jnp.clip(j0f.astype(jnp.int32), 0, ny - 1)
    i1 = jnp.clip(i0 + 1, 0, nx - 1)
    j1 = jnp.clip(j0 + 1, 0, ny - 1)
    val = (
        f[i0, j0] * (1 - wx) * (1 - wy)
        + f[i1, j0] * wx * (1 - wy)
        + f[i0, j1] * (1 - wx) * wy
        + f[i1, j1] * wx * wy
    )
    if mode == "nearest":
        return val
    in_dom = (ix >= 0) & (ix <= nx - 1) & (iy >= 0) & (iy <= ny - 1)
    return jnp.where(in_dom, val, cval)
