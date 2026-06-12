# jax-cs conventions

This page is the authoritative statement of the conventions every module
assumes. Where jax-cs deviates from numbacs, the deviation is stated here
(and in the relevant docstring) with the reason.

## Grid layout

* 2D scalar fields are arrays of shape `(nx, ny)` — **x is the first
  axis** (`numpy.meshgrid(..., indexing="ij")`), matching numbacs. A
  field plotted with matplotlib therefore needs a transpose
  (`jaxcs.plot.plot_field` does this for you).
* Time-dependent gridded data is `(nt, nx, ny)`; 3D data is
  `(nt, nx, ny, nz)`.
* Coordinate arrays (`tvals`, `xvals`, `yvals`, …) must be **uniformly
  spaced and strictly increasing**. Flow maps on grids return
  `(nx, ny, 2)`; trajectories on grids return `(nx, ny, n_steps + 1, 2)`.

## Index space vs. physical space

All public APIs take **physical** coordinates. Internally, interpolation
converts physical to fractional index space via
`i = (q - vals[0]) / (vals[1] - vals[0])` and evaluates with
`jax.scipy.ndimage.map_coordinates(order=1)` (multilinear). The
hand-rolled `interpolation.bilinear_manual` operates directly in index
space for cases that need custom corner handling.

Because the interpolators are differentiable JAX functions of the query
point, the Jacobian convention downstream FTLE assumes —
`F[a, b] = ∂Φ_a/∂x0_b` in *physical* units — holds identically for
analytical and gridded fields; no index/physical Jacobian factor ever
appears in user code.

## Time direction and the sign of backward FTLE

* Integration always runs from `t0` to `t0 + T`. **Backward time is
  requested with `T < 0`**; the step `dt = T / n_steps` is then negative.
* Velocity fields are never sign-flipped. *Deviation from numbacs*: numbacs
  bakes an `int_direction` parameter into the RHS (`params[0]`) and keeps
  `T > 0`; baking direction into the field makes the field impure with
  respect to its physical meaning and complicates composing diagnostics, so
  jax-cs moved the convention to the integrator.
* FTLE is `σ = log(max(λ_max, 1)) / (2 |T|)` with the **absolute** `T`.
  Backward FTLE (from a `T < 0` flow map) is therefore **positive on
  attracting structures**, the same sign convention as numbacs. The clip at
  `λ_max = 1` (FTLE ≥ 0) is also numbacs's convention. Both are verified
  against the analytical saddle `u = (ax, −ay)`, whose forward *and*
  backward FTLE equal `a` exactly (`tests/test_diagnostics.py`).
* LAVD integrates `|ω(x(t), t) − ω̄(t)| dt` with unsigned `dt`, so backward
  LAVD is also positive.

## Units

jax-cs is unit-agnostic: FTLE has units 1/[time of `T`], LAVD has units of
vorticity × time. The predefined `BickleyJet` uses numbacs's units (days,
Mm). For lon/lat ocean data, `GriddedField2D(spherical=True)` takes
coordinates in **degrees**, velocities in length/time consistent with `r`
(default km, `r = 6371`), and converts to degrees/time internally with the
`1/cos(lat)` metric factor — numbacs's `spherical=1` convention.

## NaN handling / land masks

JAX has no masked arrays, and a single NaN poisons every value (and
gradient) it touches, so NaNs must never enter the hot path:

1. `utils.fill_nans_and_get_mask(arrs)` replaces NaNs with a fill value
   (default 0 = no-slip "land") and returns the spatial union mask.
2. Build `GriddedField2D` from the *filled* arrays.
3. Dilate the mask (`utils.binary_dilation`) so finite-difference stencils
   never straddle the land boundary, and pass it to the diagnostics
   (`ftle_grid(..., mask=...)` etc.). Masked cells return 0, as in numbacs.

## Precision: float32 vs float64

`jaxcs.enable_x64()` (or `JAX_ENABLE_X64=1`) enables double precision;
jax-cs does **not** flip the global flag on import. Both paths work
throughout.

**Default to float64 for FTLE.** FTLE takes the log of eigenvalues of
`FᵀF`, where `F` accumulates over thousands of multiplicative steps; in
float32 the Jacobian's condition number quickly approaches `1/eps`, which
shows up as grid-scale noise on ridges (where you care most). float32 is
acceptable for exploratory work, trajectories/LAVD (an integral — errors
average out), and GPU-throughput-bound ensembles; halve memory, roughly
2–8× faster on consumer GPUs.

## FTLE Jacobian: finite differences vs `jacrev`

Two independent paths are provided and cross-validated in the tests:

| | `ftle_grid` (FD on gridded flow map) | `ftle_ad` (`jacrev` per point) |
|---|---|---|
| cost | 1 trajectory per node | ≈ 3–4 trajectories-equivalent per node (reverse pass over the scan) |
| accuracy | O(dx²), degrades when neighbours separate nonlinearly (long `T`, sharp ridges) | exact derivative of the discrete flow map; independent of grid spacing |
| needs a grid | yes | no — works on scattered Lagrangian particles |
| memory | O(grid) | O(grid × steps) for reverse AD; use `checkpoint=True` on long horizons |

Measured on the double gyre (CPU, float64, 201×101 grid, T=8, dt=0.05;
reproduce with `examples/benchmark_ftle.py`): FD 0.85 s, aux-grid 4.1 s,
AD 5.5 s. The AD and aux-grid paths agree on the ridge-peak FTLE to four
digits (1.0548 vs 1.0547) while grid-FD smooths the same peak down to
0.61 — yet off-ridge the median |FD − AD| is only 4 × 10⁻⁴. In short: the
~6× cost of AD buys exact ridge values. Rule of thumb: FD for
dense-grid visualisation at
moderate `T`; AD for quantitative ridge values, long horizons, scattered
particles, or whenever you also need parameter gradients.
*Deviation from numbacs*: numbacs's third option (auxiliary grid) is kept
as `ftle_aux_grid`, and interior FD stencils match numbacs's central
differences, but boundary nodes use one-sided differences
(`jnp.gradient`) instead of being zeroed.

## Integrators

Fixed-step RK4 (default) and fixed-step Dormand–Prince 5 over
`lax.scan`. *Deviation from numbacs*, which uses adaptive LSODA/DOP853
through numbalsoda: adaptive stepping gives every particle a different
step sequence, which destroys `vmap` batching and complicates reverse-mode
AD; fixed steps keep the computation rectangular and differentiable.
Validation: `tests/test_integration.py` matches SciPy's adaptive RK45 at
`rtol=1e-11` to `5e-7` absolute on double-gyre trajectories with
`dt = 0.01`.

## What stays NumPy and why

The hot path (integration, interpolation, flow maps, FTLE/LAVD/iLE
fields, ridge-point detection, tensorline integration) is pure JAX.
Extraction *assembly* — linking ridge points into ordered curves, walking
LAVD contour levels (contourpy), convex hulls (SciPy) — is sequential,
data-dependent, and returns ragged Python lists; it runs in NumPy exactly
as numbacs runs it outside its Numba kernels. These steps are O(number of
extracted points), not O(grid × steps), so they are never the bottleneck.

## Scope relative to numbacs

All 2D planar methods are ported: flow maps (point/grid/aux/trajectory/
composition), forward+backward FTLE, Cauchy–Green eigenfields, LAVD, IVD,
iLE and rate-of-strain eigenfields, FTLE ridges (point detection and
ordered curves), variational hyperbolic LCS/OECS tensorlines, and elliptic
LCS (`rotcohvrt`). Spherical *velocity* handling for lon/lat data is
ported (`spherical=True`). **Not ported** (explicitly, not silently):
numbacs's icosphere/S2 global-FTLE machinery (`ftle_grid_S2`,
`ftle_icosphere`, icosphere mesh utilities) — it is a specialised
unstructured-mesh pipeline whose scatter/gather style would need a
ground-up JAX redesign; planar + spherical-velocity covers the regional
ocean/atmosphere use cases. Simplifications in the hyperbolic-LCS
post-filters are documented in `jaxcs/lcs.py` docstrings.
