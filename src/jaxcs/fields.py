"""Velocity-field abstraction.

A :class:`Field` is anything with a pure ``__call__(t, x) -> dx/dt`` where
``t`` is a scalar and ``x`` is a state vector of shape ``(d,)``.  Both
analytical callables and gridded velocity datasets are wrapped behind this
single interface, so integrators, flow maps and diagnostics are agnostic to
where the velocity comes from.  Fields are `equinox` Modules, hence valid
JAX Pytrees: they can be passed through ``jit``/``vmap``/``grad`` and their
array leaves live on device.

Time direction is handled by the *integrator* (via the sign of ``T``), not
by the field; fields always return the physical velocity.  This differs
from numbacs, where the integration direction is baked into the RHS through
``params[0]`` — see ``docs/conventions.md``.
"""

from __future__ import annotations

from typing import Callable, Optional

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .interpolation import ExtrapMode, trilinear, quadrilinear

__all__ = [
    "Field",
    "AnalyticalField",
    "GriddedField2D",
    "GriddedField3D",
    "ScalarField2D",
]

#: Mean Earth radius in km, used for spherical velocity conversion.
EARTH_RADIUS_KM = 6371.0


class Field(eqx.Module):
    """Abstract base: a velocity field ``v(t, x)``.

    Subclasses implement ``__call__`` for a *single* scalar time and a
    *single* state vector; batching is the caller's job via ``vmap``.
    """

    def __call__(self, t: Array, x: Array) -> Array:  # pragma: no cover
        raise NotImplementedError


class AnalyticalField(Field):
    """Wraps an analytical callable ``f(t, x) -> dx/dt``.

    The callable must be pure and built from JAX primitives.  Any
    parameters it closes over are static from JAX's point of view; for
    differentiable / swappable parameters prefer an `equinox` Module with
    array fields (see :mod:`jaxcs.flows` for examples).
    """

    f: Callable[[Array, Array], Array]

    def __call__(self, t: Array, x: Array) -> Array:
        return jnp.asarray(self.f(t, x))


class GriddedField2D(Field):
    """Velocity from gridded 2D(+t) data ``u(t, x, y)``, ``v(t, x, y)``.

    Arrays follow the ``(nt, nx, ny)`` layout (x first; see
    ``docs/conventions.md``).  Steady data may be passed with ``nt == 1``.
    Spatial interpolation is bilinear, temporal interpolation linear with
    clamping outside the data's time span.

    For geophysical data on a longitude/latitude grid set
    ``spherical=True``: coordinates are then degrees lon/lat and ``u, v``
    are linear velocities (e.g. km/day with ``r`` in km); the returned
    rate-of-change of (lon, lat) in degrees includes the ``1/cos(lat)``
    metric factor, exactly as numbacs's ``spherical=1`` mode.

    NaNs must be removed first (``utils.fill_nans_and_get_mask``).
    """

    u: Array
    v: Array
    tvals: Array
    xvals: Array
    yvals: Array
    spherical: bool = eqx.field(static=True, default=False)
    r: float = eqx.field(static=True, default=EARTH_RADIUS_KM)
    extrap_mode: ExtrapMode = eqx.field(static=True, default="constant")

    def __check_init__(self):
        if self.u.ndim != 3 or self.v.ndim != 3:
            raise ValueError("u, v must have shape (nt, nx, ny)")

    def __call__(self, t: Array, x: Array) -> Array:
        ui = trilinear(
            self.u, t, x[0], x[1], self.tvals, self.xvals, self.yvals,
            mode=self.extrap_mode,
        )
        vi = trilinear(
            self.v, t, x[0], x[1], self.tvals, self.xvals, self.yvals,
            mode=self.extrap_mode,
        )
        if self.spherical:
            deg = 180.0 / jnp.pi
            lat = jnp.deg2rad(x[1])
            ui = deg * ui / (self.r * jnp.cos(lat))
            vi = deg * vi / self.r
        return jnp.stack([ui, vi])


class GriddedField3D(Field):
    """Velocity from gridded 3D(+t) data with ``(nt, nx, ny, nz)`` layout."""

    u: Array
    v: Array
    w: Array
    tvals: Array
    xvals: Array
    yvals: Array
    zvals: Array
    extrap_mode: ExtrapMode = eqx.field(static=True, default="constant")

    def __call__(self, t: Array, x: Array) -> Array:
        args = (t, x[0], x[1], x[2], self.tvals, self.xvals, self.yvals, self.zvals)
        ui = quadrilinear(self.u, *args, mode=self.extrap_mode)
        vi = quadrilinear(self.v, *args, mode=self.extrap_mode)
        wi = quadrilinear(self.w, *args, mode=self.extrap_mode)
        return jnp.stack([ui, vi, wi])


class ScalarField2D(eqx.Module):
    """Interpolant for a scalar quantity ``f(t, x, y)`` (e.g. vorticity).

    Same grid conventions as :class:`GriddedField2D`.  Optional
    ``period_x`` / ``period_y`` wrap query points into the fundamental
    domain before interpolation (used for LAVD on periodic domains, like
    the Bickley jet channel).
    """

    f: Array
    tvals: Array
    xvals: Array
    yvals: Array
    period_x: Optional[float] = eqx.field(static=True, default=None)
    period_y: Optional[float] = eqx.field(static=True, default=None)
    extrap_mode: ExtrapMode = eqx.field(static=True, default="nearest")

    def __call__(self, t: Array, x: Array) -> Array:
        px, py = x[0], x[1]
        if self.period_x is not None:
            px = self.xvals[0] + jnp.mod(px - self.xvals[0], self.period_x)
        if self.period_y is not None:
            py = self.yvals[0] + jnp.mod(py - self.yvals[0], self.period_y)
        return trilinear(
            self.f, t, px, py, self.tvals, self.xvals, self.yvals,
            mode=self.extrap_mode,
        )
