"""Utilities: precision control, grids, masking, vorticity, geometry.

The geometry helpers at the bottom (shoelace, winding number, arclength,
``max_in_radius``) are deliberately NumPy: they serve the extraction
post-processing in :mod:`jaxcs.lcs`, which is not in the hot path (see
``docs/conventions.md``).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .fields import Field

__all__ = [
    "enable_x64",
    "uniform_grid",
    "pad_domain",
    "fill_nans_and_get_mask",
    "binary_dilation",
    "vorticity_grid",
    "ADVorticity",
    "shoelace",
    "arclength",
    "pts_in_poly",
    "max_in_radius",
]


def enable_x64() -> None:
    """Enable double precision globally.

    jax-cs does **not** flip this on import (that would silently change
    the host program's precision); call this — or set
    ``JAX_ENABLE_X64=1`` — before building fields.  FTLE is
    log-of-eigenvalue of a product of long Jacobian chains and is the
    diagnostic most sensitive to rounding: default to float64 for FTLE
    and use float32 only when memory/GPU throughput demands it (expect
    visible noise in ridge-scale FTLE detail at float32).
    """
    jax.config.update("jax_enable_x64", True)


def uniform_grid(
    domain: Sequence[Sequence[float]], nx: int, ny: int
) -> Tuple[Array, Array]:
    """Uniform grid over ``domain = ((x0, x1), (y0, y1))`` → ``(x, y)``."""
    (x0, x1), (y0, y1) = domain
    return jnp.linspace(x0, x1, nx), jnp.linspace(y0, y1, ny)


def pad_domain(
    domain: Sequence[Sequence[float]], pad: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Shrink-or-grow a domain by ``pad`` on every side (negative shrinks)."""
    (x0, x1), (y0, y1) = domain
    return ((x0 - pad, x1 + pad), (y0 - pad, y1 + pad))


def fill_nans_and_get_mask(
    arrs: Sequence[np.ndarray], fill_value: float = 0.0
) -> Tuple[list, np.ndarray]:
    """Replace NaNs with ``fill_value``; return filled arrays and a mask.

    The mask is True wherever *any* of the input arrays is NaN at any
    leading (time) index — i.e. the spatial union over time, matching
    numbacs.  Gridded data must be passed through this before building a
    :class:`jaxcs.fields.GriddedField2D` (NaNs poison both interpolation
    and gradients in JAX).
    """
    mask = None
    filled = []
    for a in arrs:
        a = np.asarray(a)
        nan = np.isnan(a)
        m = nan.any(axis=0) if a.ndim == 3 else nan
        mask = m if mask is None else (mask | m)
        filled.append(jnp.asarray(np.where(nan, fill_value, a)))
    return filled, mask


def binary_dilation(mask: Array, corners: bool = False, iterations: int = 1) -> Array:
    """Dilate a boolean mask by one cell per iteration (pure JAX).

    Use on land masks before finite-difference diagnostics so stencils
    never straddle the data boundary (numbacs ``binary_mask_dilation``).
    ``corners=True`` also dilates diagonally (8-connectivity).
    """
    m = jnp.asarray(mask, bool)
    for _ in range(iterations):
        out = m
        shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if corners:
            shifts += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for sx, sy in shifts:
            shifted = jnp.roll(m, (sx, sy), axis=(0, 1))
            # roll wraps; blank the wrapped border
            if sx == 1:
                shifted = shifted.at[0, :].set(False)
            elif sx == -1:
                shifted = shifted.at[-1, :].set(False)
            if sy == 1:
                shifted = shifted.at[:, 0].set(False)
            elif sy == -1:
                shifted = shifted.at[:, -1].set(False)
            out = out | shifted
        m = out
    return m


def vorticity_grid(u: Array, v: Array, dx: float, dy: float) -> Array:
    """Vorticity ``ω = ∂v/∂x − ∂u/∂y`` from gridded velocity.

    Accepts ``(nx, ny)`` snapshots or ``(nt, nx, ny)`` series (numbacs
    ``curl_vel`` / ``curl_vel_tspan``); derivative axes are the spatial
    ones in either case.
    """
    ax = u.ndim - 2
    dvdx = jnp.gradient(v, dx, axis=ax)
    dudy = jnp.gradient(u, dy, axis=ax + 1)
    return dvdx - dudy


class ADVorticity(eqx.Module):
    """Pointwise vorticity ``ω(t, x)`` of a 2D field, by AD.

    Wraps any :class:`Field`; ``__call__(t, x)`` returns the scalar curl
    ``∂v/∂x − ∂u/∂y`` computed exactly with ``jacfwd`` (numbacs's
    ``curl_func`` uses an ``h``-stencil instead).  Pass directly as the
    ``vorticity`` argument of :func:`jaxcs.diagnostics.lavd_grid` for
    analytical flows — no interpolant needed.
    """

    field: Field

    def __call__(self, t: Array, x: Array) -> Array:
        J = jax.jacfwd(lambda q: self.field(t, q))(x)
        return J[1, 0] - J[0, 1]


# --------------------------------------------------------------------------
# NumPy geometry helpers (extraction post-processing)
# --------------------------------------------------------------------------

def shoelace(polygon: np.ndarray) -> float:
    """Unsigned area of a closed polygon, shape ``(n, 2)``."""
    x, y = np.asarray(polygon).T
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def arclength(pts: np.ndarray) -> float:
    """Total arc length of a polyline, shape ``(n, 2)``."""
    d = np.diff(np.asarray(pts), axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def pts_in_poly(polygon: np.ndarray, pts: np.ndarray) -> int:
    """Index of the first point inside ``polygon`` (winding number), or −1."""
    poly = np.asarray(polygon)
    pts = np.atleast_2d(pts)
    x0, y0 = poly[:-1, 0], poly[:-1, 1]
    x1, y1 = poly[1:, 0], poly[1:, 1]
    for i, (px, py) in enumerate(pts):
        up = (y0 <= py) & (y1 > py)
        down = (y0 > py) & (y1 <= py)
        cross = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)
        wn = np.sum(up & (cross > 0)) - np.sum(down & (cross < 0))
        if wn != 0:
            return i
    return -1


def max_in_radius(
    arr: np.ndarray,
    r: float,
    dx: float,
    dy: float,
    n: int = -1,
    min_val: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy local maxima with exclusion radius ``r`` (numbacs port).

    Repeatedly takes the global maximum of ``arr`` (above ``min_val``)
    and zeros a disc of radius ``r`` around it; at most ``n`` maxima if
    ``n > 0``.  Returns ``(values, indices)`` with indices of shape
    ``(k, 2)`` into the ``(nx, ny)`` array.
    """
    a = np.array(arr, dtype=float, copy=True)
    nx, ny = a.shape
    ri = int(np.ceil(r / dx))
    rj = int(np.ceil(r / dy))
    vals, inds = [], []
    while True:
        ij = np.unravel_index(np.argmax(a), a.shape)
        v = a[ij]
        if v <= min_val or (n > 0 and len(vals) >= n):
            break
        vals.append(v)
        inds.append(ij)
        i0, i1 = max(ij[0] - ri, 0), min(ij[0] + ri + 1, nx)
        j0, j1 = max(ij[1] - rj, 0), min(ij[1] + rj + 1, ny)
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        disc = ((ii - ij[0]) * dx) ** 2 + ((jj - ij[1]) * dy) ** 2 <= r**2
        a[i0:i1, j0:j1][disc] = -np.inf
    return np.array(vals), np.array(inds, dtype=int).reshape(-1, 2)
