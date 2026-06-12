"""Predefined analytical flows (numbacs's ``get_predefined_flow``).

Each flow is an `equinox` Module: parameters are array leaves of the
Pytree (differentiable, swappable with ``eqx.tree_at``), and the module
itself is a :class:`jaxcs.fields.Field` usable directly by the
integrators.  Defaults reproduce numbacs's defaults exactly.

Time direction: unlike numbacs there is no ``int_direction`` parameter —
request backward integration with ``T < 0`` at the integrator level.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .fields import Field

__all__ = ["DoubleGyre", "BickleyJet", "double_gyre", "bickley_jet"]


class DoubleGyre(Field):
    """Time-periodic double gyre (Shadden, Lekien & Marsden 2005).

    Stream function ``ψ = A sin(π f(x, t)) sin(π y)`` with
    ``f = ε sin(ω t + ψ₀) x² + (1 − 2 ε sin(ω t + ψ₀)) x`` on the domain
    ``[0, 2] × [0, 1]``; ``alpha`` adds the optional linear damping term
    numbacs includes (default 0).
    """

    A: Array
    eps: Array
    alpha: Array
    omega: Array
    psi: Array

    def __init__(self, A=0.1, eps=0.25, alpha=0.0, omega=0.2 * jnp.pi, psi=0.0):
        self.A = jnp.asarray(A)
        self.eps = jnp.asarray(eps)
        self.alpha = jnp.asarray(alpha)
        self.omega = jnp.asarray(omega)
        self.psi = jnp.asarray(psi)

    @property
    def domain(self):
        return ((0.0, 2.0), (0.0, 1.0))

    def __call__(self, t: Array, x: Array) -> Array:
        a = self.eps * jnp.sin(self.omega * t + self.psi)
        b = 1.0 - 2.0 * a
        f = a * x[0] ** 2 + b * x[0]
        df = 2.0 * a * x[0] + b
        u = -jnp.pi * self.A * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * x[1]) - self.alpha * x[0]
        v = jnp.pi * self.A * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * x[1]) * df - self.alpha * x[1]
        return jnp.stack([u, v])


class BickleyJet(Field):
    """Idealised Bickley jet with three Rossby waves.

    Standard test case for elliptic LCS (Rypina et al. 2007; Hadjighasem
    et al. 2017).  Default units: time in days, length in Mm, matching
    numbacs (domain ``[0, π r_e] × [-3, 3]`` Mm, x-periodic with period
    ``π r_e ≈ 20.01 Mm``).
    """

    U0: Array
    L: Array
    A: Array   # wave amplitudes (3,)
    k: Array   # wavenumbers (3,)
    c: Array   # phase speeds (3,)

    def __init__(self, U0=None, L=None, A=None, k=None, c=None):
        r_e = 6371.0e-3  # Earth radius in Mm
        U0_d = 86400 * 62.66e-6  # 62.66 m/s in Mm/day
        self.U0 = jnp.asarray(U0 if U0 is not None else U0_d)
        self.L = jnp.asarray(L if L is not None else 1770.0e-3)
        self.A = jnp.asarray(A if A is not None else [0.0075, 0.15, 0.3])
        self.k = jnp.asarray(k if k is not None else [2.0 / r_e, 4.0 / r_e, 6.0 / r_e])
        if c is None:
            c2 = 0.205 * U0_d
            c3 = 0.461 * U0_d
            c1 = c3 + (jnp.sqrt(5.0) - 1.0) * (c2 - c3)
            c = [c1, c2, c3]
        self.c = jnp.asarray(c)

    @property
    def domain(self):
        return ((0.0, 6371.0e-3 * jnp.pi), (-3.0, 3.0))

    @property
    def period_x(self) -> float:
        return float(6371.0e-3 * jnp.pi)

    def __call__(self, t: Array, x: Array) -> Array:
        Y = x[1] / self.L
        sech2 = 1.0 / jnp.cosh(Y) ** 2
        phase = self.k * (x[0] - self.c * t)  # (3,)
        u = self.U0 * sech2 + (
            2.0 * self.U0 * jnp.tanh(Y) * sech2 * jnp.sum(self.A * jnp.cos(phase))
        )
        v = -self.U0 * self.L * sech2 * jnp.sum(self.A * self.k * jnp.sin(phase))
        return jnp.stack([u, v])


def double_gyre(**kwargs) -> DoubleGyre:
    """Construct a :class:`DoubleGyre` with numbacs's default parameters."""
    return DoubleGyre(**kwargs)


def bickley_jet(**kwargs) -> BickleyJet:
    """Construct a :class:`BickleyJet` with numbacs's default parameters."""
    return BickleyJet(**kwargs)
