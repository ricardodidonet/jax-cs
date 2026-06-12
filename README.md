# jax-cs

Lagrangian Coherent Structures in [JAX](https://github.com/jax-ml/jax) — a
JAX reimplementation of [numbacs](https://github.com/numbacs). Numba JIT is
replaced by JAX transformations (`jit`, `vmap`, `lax.scan`, `jacrev`), so
everything runs on CPU/GPU/TPU, composes functionally, and is
**differentiable end-to-end** (you can take gradients of FTLE with respect
to flow parameters).

## Features

- **Integrators** — fixed-step RK4 and Dormand–Prince 5 over `lax.scan`;
  forward/backward time via the sign of `T`; particle ensembles via `vmap`;
  `jax.checkpoint` for memory-bound reverse AD on long horizons.
- **Fields** — one `Field` interface for analytical callables and gridded
  velocity data (bilinear/trilinear interpolation, NaN/land-mask workflow,
  spherical lon/lat support); fields are equinox Modules (Pytrees).
- **Flow maps** — point/ensemble/grid/auxiliary-grid/trajectory variants,
  plus flow-map composition for time series.
- **Diagnostics** — forward & backward FTLE with *three* Jacobian paths
  (grid finite differences, auxiliary grid, exact `jacrev`); Cauchy–Green
  tensor + batched `eigh`; LAVD fused into the advection scan; iLE and
  rate-of-strain eigenfields via AD; IVD.
- **LCS extraction** — FTLE ridge points and ordered ridges; variational
  hyperbolic LCS/OECS tensorlines; elliptic LCS (rotationally coherent
  vortices from LAVD level sets).

## Install

```bash
pip install -e ".[dev]"     # or: uv pip install -e ".[dev]"
```

## Quick start

```python
import jaxcs

jaxcs.enable_x64()                       # float64: recommended for FTLE

field = jaxcs.double_gyre()              # Shadden et al. 2005
x, y = jaxcs.uniform_grid(field.domain, 401, 201)
dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

# forward FTLE (T < 0 gives backward FTLE, positive on attracting LCS)
fmap = jaxcs.flowmap_grid(field, x, y, t0=0.0, T=16.0, dt=0.05)
ftle = jaxcs.ftle_grid(fmap, T=16.0, dx=dx, dy=dy)

# exact per-particle FTLE by automatic differentiation (no grid needed)
ftle_pts = jaxcs.ftle_ad(field, x0s, t0=0.0, T=16.0, n_steps=320)

# LAVD vortices, vorticity by AD, accumulated inside the advection scan
lavd = jaxcs.lavd_grid(field, jaxcs.ADVorticity(field), x, y, 0.0, 16.0, 320)
vortices = jaxcs.rotcohvrt(lavd, x, y, r=0.3)
```

See `examples/` for the double gyre, Bickley jet, and a HYCOM-style gridded
ocean case, and **`docs/conventions.md`** for the conventions every module
assumes (grid layout, time direction, backward-FTLE sign, NaN handling,
float32/float64 trade-offs, FD-vs-AD Jacobian guidance, scope vs numbacs).

## Testing

```bash
python -m pytest            # integrators vs analytical ODEs & SciPy,
                            # interpolation vs known functions,
                            # FTLE/LAVD vs analytical references,
                            # double gyre / Bickley jet / gridded regressions
```
