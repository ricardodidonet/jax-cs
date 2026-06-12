"""Flow maps over points, ensembles and grids.

The flow map :math:`\\Phi_{t_0}^{t_0+T}(x_0)` is the position at time
``t0 + T`` of the trajectory seeded at ``x0`` at time ``t0``.  Everything
here is a thin, ``vmap``-batched layer over :mod:`jaxcs.integration`; all
functions accept any :class:`jaxcs.fields.Field` (analytical or gridded).

Grid outputs follow the ``(nx, ny, ...)`` layout with x first, matching
numbacs (``flowmap_grid_2D`` etc.).
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .fields import Field
from .integration import Method, integrate, integrate_trajectory
from .interpolation import bilinear

__all__ = [
    "flowmap",
    "flowmap_grid",
    "flowmap_trajectory",
    "flowmap_trajectory_grid",
    "flowmap_aux_grid",
    "flowmap_composition",
]


def _resolve_steps(T: float, dt: Optional[float], n_steps: Optional[int]) -> int:
    if (dt is None) == (n_steps is None):
        raise ValueError("specify exactly one of dt or n_steps")
    if n_steps is not None:
        return int(n_steps)
    n = int(round(abs(float(T)) / abs(float(dt))))
    return max(n, 1)


def flowmap(
    field: Field,
    x0: Array,
    t0: float,
    T: float,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Array:
    """Flow map for one or many initial conditions.

    Parameters
    ----------
    field
        Velocity field.
    x0
        Initial condition(s): shape ``(d,)`` or batched ``(n, d)``.
    t0, T
        Start time and signed horizon (``T < 0`` → backward in time).
    dt, n_steps
        Step control — pass exactly one.  ``dt`` is an unsigned step size;
        the actual signed step is ``T / n_steps``.
    method
        ``"rk4"`` (default) or ``"dopri5"``.
    checkpoint
        Use :func:`jax.checkpoint` on the scan body (memory-bound reverse
        AD on long horizons).

    Returns
    -------
    Final position(s) ``Phi(x0)`` with the same shape as ``x0``.
    """
    n = _resolve_steps(T, dt, n_steps)
    fn = partial(
        integrate, field, t0=jnp.asarray(t0), T=jnp.asarray(T), n_steps=n,
        method=method, checkpoint=checkpoint,
    )
    if x0.ndim == 1:
        return fn(x0)
    return jax.vmap(fn)(x0)


def flowmap_trajectory(
    field: Field,
    x0: Array,
    t0: float,
    T: float,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Tuple[Array, Array]:
    """Full trajectories (for LAVD and diagnostics).

    Returns ``(ts, xs)`` with ``ts`` of shape ``(n_steps+1,)`` and ``xs``
    of shape ``(n_steps+1, d)`` for a single point, or
    ``(n, n_steps+1, d)`` for batched ``x0``.
    """
    n = _resolve_steps(T, dt, n_steps)
    fn = partial(
        integrate_trajectory, field, t0=jnp.asarray(t0), T=jnp.asarray(T),
        n_steps=n, method=method, checkpoint=checkpoint,
    )
    if x0.ndim == 1:
        return fn(x0)
    ts, xs = jax.vmap(fn)(x0)
    return ts[0], xs


def _grid_points(x: Array, y: Array) -> Array:
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return jnp.stack([X.ravel(), Y.ravel()], axis=-1)


def flowmap_grid(
    field: Field,
    x: Array,
    y: Array,
    t0: float,
    T: float,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Array:
    """Flow map seeded on the tensor grid ``x × y``.

    Returns shape ``(nx, ny, 2)`` (x-first layout).
    """
    pts = _grid_points(x, y)
    out = flowmap(field, pts, t0, T, dt=dt, n_steps=n_steps, method=method,
                  checkpoint=checkpoint)
    return out.reshape(x.shape[0], y.shape[0], 2)


def flowmap_trajectory_grid(
    field: Field,
    x: Array,
    y: Array,
    t0: float,
    T: float,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Tuple[Array, Array]:
    """Trajectories seeded on a grid.

    Returns ``(ts, xs)`` with ``ts`` of shape ``(n+1,)`` and ``xs`` of
    shape ``(nx, ny, n+1, 2)`` — the layout numbacs's ``flowmap_n_grid_2D``
    uses for LAVD.
    """
    pts = _grid_points(x, y)
    ts, xs = flowmap_trajectory(field, pts, t0, T, dt=dt, n_steps=n_steps,
                                method=method, checkpoint=checkpoint)
    return ts, xs.reshape(x.shape[0], y.shape[0], ts.shape[0], 2)


def flowmap_aux_grid(
    field: Field,
    x: Array,
    y: Array,
    t0: float,
    T: float,
    h: float = 1e-5,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Array:
    """Flow map on an auxiliary grid: each node plus 4 offsets ``±h``.

    Returns shape ``(nx, ny, 5, 2)`` ordered
    ``[center, +x, -x, +y, -y]``, the input expected by
    :func:`jaxcs.diagnostics.cauchy_green_aux`.  Mirrors numbacs's
    ``flowmap_aux_grid_2D`` (with the center point always included).
    """
    pts = _grid_points(x, y)  # (npts, 2)
    offsets = jnp.array(
        [[0.0, 0.0], [h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]]
    )
    aux = pts[:, None, :] + offsets[None, :, :]  # (npts, 5, 2)
    out = flowmap(field, aux.reshape(-1, 2), t0, T, dt=dt, n_steps=n_steps,
                  method=method, checkpoint=checkpoint)
    return out.reshape(x.shape[0], y.shape[0], 5, 2)


def flowmap_composition(
    flowmaps: Array,
    x: Array,
    y: Array,
) -> Array:
    """Compose a sequence of short-horizon flow maps into a long one.

    Given ``k`` flow maps over the *same* grid, where ``flowmaps[i]`` maps
    positions at ``t_i`` to ``t_{i+1}`` (shape ``(k, nx, ny, 2)``), the
    composition :math:`\\Phi_k \\circ \\dots \\circ \\Phi_1` is evaluated by
    repeatedly bilinearly interpolating the next map at the current
    positions (numbacs's ``flowmap_composition`` strategy).

    Interpolation error accumulates with ``k``; use a grid fine enough for
    the flow map to be well-resolved.  Points that leave the grid are
    clamped to the boundary (``nearest`` extrapolation).

    Returns the composed flow map, shape ``(nx, ny, 2)``.
    """
    pts0 = _grid_points(x, y)  # (npts, 2)

    interp = jax.vmap(
        lambda f, px, py: bilinear(f, px, py, x, y, mode="nearest"),
        in_axes=(None, 0, 0),
    )

    def body(pts, fmap):
        px = interp(fmap[:, :, 0], pts[:, 0], pts[:, 1])
        py = interp(fmap[:, :, 1], pts[:, 0], pts[:, 1])
        return jnp.stack([px, py], axis=-1), None

    pts, _ = jax.lax.scan(body, pts0, flowmaps)
    return pts.reshape(x.shape[0], y.shape[0], 2)
