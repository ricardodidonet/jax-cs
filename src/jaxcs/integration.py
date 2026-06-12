"""Fixed-step ODE integrators built on ``jax.lax.scan``.

Two methods are provided:

* ``"rk4"`` — classical 4th-order Runge–Kutta.
* ``"dopri5"`` — the 5th-order Dormand–Prince stage combination, used here
  as a *fixed-step* method (the embedded 4th-order solution is not used for
  step-size control).

Deviation from numbacs (documented, see ``docs/conventions.md``): numbacs
integrates with adaptive LSODA / DOP853 via numbalsoda.  Adaptive stepping
inside ``lax.scan`` costs compilation complexity and ragged work per
particle, which destroys the SIMD batching that makes ``vmap`` fast, so
jax-cs uses fixed steps.  Choose ``n_steps`` (or ``dt``) so the step is
small relative to the velocity field's time scales; the validation tests
compare against SciPy's adaptive ``solve_ivp`` at tight tolerances.

Time direction: integration runs from ``t0`` to ``t0 + T``.  Backward-time
integration is requested with ``T < 0`` — the step ``dt = T / n_steps`` is
then negative and the same update rule marches backwards.  Velocity fields
are *never* sign-flipped (unlike numbacs's ``params[0]`` convention).

Ensembles: every function here takes a single initial condition; batching
is exposed separately (``integrate_ensemble``) via ``vmap``, never via
Python loops.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from .fields import Field

Method = Literal["rk4", "dopri5"]

__all__ = [
    "rk4_step",
    "dopri5_step",
    "integrate",
    "integrate_trajectory",
    "integrate_ensemble",
]


def rk4_step(f: Callable[[Array, Array], Array], t: Array, x: Array, dt: Array) -> Array:
    """One classical RK4 step from ``(t, x)`` with (signed) step ``dt``."""
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Dormand–Prince 5(4) Butcher tableau (5th-order weights only).
_DP_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_DP_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_DP_B = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)


def dopri5_step(f: Callable[[Array, Array], Array], t: Array, x: Array, dt: Array) -> Array:
    """One fixed-size 5th-order Dormand–Prince step."""
    ks = []
    for ci, ai in zip(_DP_C, _DP_A):
        xi = x
        for aij, kj in zip(ai, ks):
            xi = xi + dt * aij * kj
        ks.append(f(t + ci * dt, xi))
    out = x
    for bi, ki in zip(_DP_B, ks):
        out = out + dt * bi * ki
    return out


_STEPPERS = {"rk4": rk4_step, "dopri5": dopri5_step}


@eqx.filter_jit
def integrate(
    field: Field,
    x0: Array,
    t0: Array,
    T: Array,
    n_steps: int,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Array:
    """Integrate ``dx/dt = field(t, x)`` from ``t0`` to ``t0 + T``.

    Parameters
    ----------
    field
        Velocity field (pytree; any ``Field``).
    x0
        Initial state, shape ``(d,)``.
    t0
        Initial time (scalar).
    T
        Signed integration horizon; ``T < 0`` integrates backward in time.
    n_steps
        Number of fixed steps, ``dt = T / n_steps``.
    method
        ``"rk4"`` or ``"dopri5"``.
    checkpoint
        Rematerialize each step on the backward pass
        (:func:`jax.checkpoint`) — use for long horizons where storing all
        intermediate states would be memory-bound under reverse-mode AD.

    Returns
    -------
    Final state ``x(t0 + T)``, shape ``(d,)``.
    """
    step = _STEPPERS[method]
    dt = T / n_steps

    def body(carry, i):
        t = t0 + i * dt
        return step(field, t, carry, dt), None

    if checkpoint:
        body = jax.checkpoint(body)
    xT, _ = lax.scan(body, x0, jnp.arange(n_steps))
    return xT


@eqx.filter_jit
def integrate_trajectory(
    field: Field,
    x0: Array,
    t0: Array,
    T: Array,
    n_steps: int,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Tuple[Array, Array]:
    """Like :func:`integrate` but returns the whole trajectory.

    Returns
    -------
    ts
        Times, shape ``(n_steps + 1,)``, from ``t0`` to ``t0 + T``
        inclusive (decreasing if ``T < 0``).
    xs
        States at those times, shape ``(n_steps + 1, d)``.
    """
    step = _STEPPERS[method]
    dt = T / n_steps

    def body(carry, i):
        t = t0 + i * dt
        nxt = step(field, t, carry, dt)
        return nxt, nxt

    if checkpoint:
        body = jax.checkpoint(body)
    _, xs = lax.scan(body, x0, jnp.arange(n_steps))
    ts = t0 + dt * jnp.arange(n_steps + 1)
    return ts, jnp.concatenate([x0[None], xs], axis=0)


def integrate_ensemble(
    field: Field,
    x0s: Array,
    t0: Array,
    T: Array,
    n_steps: int,
    method: Method = "rk4",
    checkpoint: bool = False,
) -> Array:
    """``vmap`` of :func:`integrate` over a batch of initial conditions.

    ``x0s`` has shape ``(n, d)``; returns shape ``(n, d)``.
    """
    fn = partial(
        integrate, field, t0=t0, T=T, n_steps=n_steps, method=method,
        checkpoint=checkpoint,
    )
    return jax.vmap(fn)(x0s)
