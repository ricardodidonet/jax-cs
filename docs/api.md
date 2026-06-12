# jax-cs public API

All names below are stable and importable from the top-level `jaxcs`
namespace. Full parameter documentation lives in the docstrings; numbacs
counterparts are listed for migration.

## Fields (`jaxcs.fields`)

| jax-cs | numbacs counterpart |
|---|---|
| `Field` | — (cfunc pointers) |
| `AnalyticalField(f)` | `get_predefined_callable` |
| `GriddedField2D(u, v, tvals, xvals, yvals, spherical=, r=, extrap_mode=)` | `get_interp_arrays_2D` + `get_flow_2D` |
| `GriddedField3D(u, v, w, ...)` | `flowmap_grid_ND` machinery |
| `ScalarField2D(f, tvals, xvals, yvals, period_x=, period_y=)` | `get_callable_scalar` |

## Predefined flows (`jaxcs.flows`)

`double_gyre()` / `DoubleGyre`, `bickley_jet()` / `BickleyJet` —
numbacs `get_predefined_flow("double_gyre" | "bickley_jet")` with the same
default parameters; parameters are equinox Module fields (differentiable).

## Integration (`jaxcs.integration`)

`rk4_step`, `dopri5_step`, `integrate`, `integrate_trajectory`,
`integrate_ensemble` — fixed-step `lax.scan` integrators (numbacs:
adaptive numbalsoda; see `docs/conventions.md` for the rationale).

## Flow maps (`jaxcs.flowmap`)

| jax-cs | numbacs counterpart |
|---|---|
| `flowmap` | `flowmap` |
| `flowmap_grid` | `flowmap_grid_2D` |
| `flowmap_trajectory`, `flowmap_trajectory_grid` | `flowmap_n`, `flowmap_n_grid_2D` |
| `flowmap_aux_grid` | `flowmap_aux_grid_2D` |
| `flowmap_composition` | `flowmap_composition` |

## Diagnostics (`jaxcs.diagnostics`)

| jax-cs | numbacs counterpart |
|---|---|
| `cauchy_green_grid`, `cauchy_green_aux`, `cauchy_green_ad` | `C_tensor_2D` (aux only) |
| `eig_cauchy_green` | `C_eig_2D` / `C_eig_aux_2D` |
| `ftle_grid`, `ftle_aux_grid`, `ftle_ad`, `ftle_ad_grid`, `ftle_from_eigval` | `ftle_grid_2D`, `ftle_from_eig` |
| `lavd_grid` (fused scan), `lavd_from_trajectories` | `lavd_grid_2D` |
| `ile_field`, `ile_grid` | `ile_2D_func`, `ile_2D_data` |
| `S_tensor_field`, `S_eig_field`, `S_eig_grid` | `S_2D_func`, `S_eig_2D_func`, `S_eig_2D_data` |
| `ivd_grid` | `ivd_grid_2D` |

## LCS extraction (`jaxcs.lcs`)

| jax-cs | numbacs counterpart |
|---|---|
| `ftle_ridge_points` | `ftle_ridge_pts` |
| `ftle_ordered_ridges` | `ftle_ordered_ridges` / `ftle_ridges` |
| `tensorlines` | `rk4_tensorlines` |
| `hyperbolic_lcs` | `hyperbolic_lcs` |
| `hyperbolic_oecs` | `hyperbolic_oecs` |
| `rotcohvrt` | `rotcohvrt` |

## Utilities (`jaxcs.utils`, `jaxcs.plot`)

`enable_x64`, `uniform_grid`, `pad_domain`, `fill_nans_and_get_mask`,
`binary_dilation`, `vorticity_grid` (numbacs `curl_vel`), `ADVorticity`
(numbacs `curl_func`, by AD), `shoelace`, `arclength`, `pts_in_poly`,
`max_in_radius`; plotting: `plot.plot_field`, `plot.plot_curves`,
`plot.plot_vortices` (NumPy/matplotlib, never JAX-traced).
