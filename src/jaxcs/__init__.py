"""jax-cs: Lagrangian Coherent Structures in JAX.

A JAX reimplementation of `numbacs <https://github.com/numbacs>`_:
flow maps, FTLE, LAVD, iLE and hyperbolic/elliptic LCS extraction built
on ``jit`` / ``vmap`` / ``lax.scan`` / ``jacrev`` — CPU/GPU/TPU
compatible, functionally composable and differentiable end-to-end.

Quick start::

    import jaxcs

    jaxcs.enable_x64()          # float64 strongly recommended for FTLE
    field = jaxcs.double_gyre()
    x, y = jaxcs.uniform_grid(field.domain, 401, 201)
    fmap = jaxcs.flowmap_grid(field, x, y, t0=0.0, T=16.0, dt=0.05)
    ftle = jaxcs.ftle_grid(fmap, T=16.0, dx=x[1] - x[0], dy=y[1] - y[0])

Conventions (grid layout, time direction, backward-FTLE sign, NaN
handling) are documented in ``docs/conventions.md``.
"""

from .fields import (
    AnalyticalField,
    Field,
    GriddedField2D,
    GriddedField3D,
    ScalarField2D,
)
from .flows import BickleyJet, DoubleGyre, bickley_jet, double_gyre
from .integration import (
    dopri5_step,
    integrate,
    integrate_ensemble,
    integrate_trajectory,
    rk4_step,
)
from .flowmap import (
    flowmap,
    flowmap_aux_grid,
    flowmap_composition,
    flowmap_grid,
    flowmap_trajectory,
    flowmap_trajectory_grid,
)
from .diagnostics import (
    S_eig_field,
    S_eig_grid,
    S_tensor_field,
    cauchy_green_ad,
    cauchy_green_aux,
    cauchy_green_grid,
    eig_cauchy_green,
    ftle_ad,
    ftle_ad_grid,
    ftle_aux_grid,
    ftle_from_eigval,
    ftle_grid,
    ile_field,
    ile_grid,
    ivd_grid,
    lavd_from_trajectories,
    lavd_grid,
)
from .lcs import (
    ftle_ordered_ridges,
    ftle_ridge_points,
    hyperbolic_lcs,
    hyperbolic_oecs,
    rotcohvrt,
    tensorlines,
)
from . import plot
from .utils import (
    ADVorticity,
    binary_dilation,
    enable_x64,
    fill_nans_and_get_mask,
    pad_domain,
    uniform_grid,
    vorticity_grid,
)

__version__ = "0.1.0"

__all__ = [
    # fields
    "Field", "AnalyticalField", "GriddedField2D", "GriddedField3D",
    "ScalarField2D",
    # flows
    "DoubleGyre", "BickleyJet", "double_gyre", "bickley_jet",
    # integration
    "rk4_step", "dopri5_step", "integrate", "integrate_trajectory",
    "integrate_ensemble",
    # flow maps
    "flowmap", "flowmap_grid", "flowmap_trajectory",
    "flowmap_trajectory_grid", "flowmap_aux_grid", "flowmap_composition",
    # diagnostics
    "cauchy_green_grid", "cauchy_green_aux", "cauchy_green_ad",
    "eig_cauchy_green", "ftle_from_eigval", "ftle_grid", "ftle_aux_grid",
    "ftle_ad", "ftle_ad_grid", "lavd_grid", "lavd_from_trajectories",
    "ile_field", "ile_grid", "S_tensor_field", "S_eig_field", "S_eig_grid",
    "ivd_grid",
    # lcs
    "ftle_ridge_points", "ftle_ordered_ridges", "tensorlines",
    "hyperbolic_lcs", "hyperbolic_oecs", "rotcohvrt",
    # utils / plot
    "enable_x64", "uniform_grid", "pad_domain", "fill_nans_and_get_mask",
    "binary_dilation", "vorticity_grid", "ADVorticity", "plot",
]
