# Task: JAX port of `numbacs`

Implement a JAX-based reimplementation of the `numbacs` Python package (a Numba-accelerated library for Lagrangian Coherent Structures). 
The goal is to replace Numba JIT with JAX transformations (`jit`, `vmap`, `scan`, `jacrev`/`jacfwd`) so the code runs on CPU/GPU/TPU, composes functionally, and is differentiable end-to-end. Preserve the scientific intent and the breadth of LCS methods covered by numbacs; you are free — and encouraged — to redesign internals, module layout, and APIs where JAX idioms suggest a cleaner solution.

**Upstream reference:** https://github.com/ricardodidonet/numbacs.git (mirror its scope; do not mirror its imperative style).

## Modules to implement

1. **Integrators (`integration.py`)**
   - RK4 (and optionally Dormand–Prince) built on `jax.lax.scan` over time steps
   - Forward and backward integration; explicit sign conventions
   - Particle ensembles via `vmap`, not Python loops
   - Support both analytical vector fields (callables `f(t, x) -> dx/dt`) and gridded velocity data (via the interpolation module)
   - Use `jax.checkpoint` where memory-bound on long horizons

2. **Interpolation (`interpolation.py`)**
   - Bilinear (2D+t) and trilinear (3D+t) interpolators as pure, `vmap`-friendly JAX functions
   - Prefer `jax.scipy.ndimage.map_coordinates`; provide a hand-rolled index-space version where finer control is needed
   - Configurable handling of NaN / land masks
   - Document the grid convention explicitly: index-space vs. physical-space, and the Jacobian convention that downstream FTLE assumes

3. **Flow map (`flowmap.py`)**
   - `flowmap(field, x0, t0, T, dt) -> xT`, with optional trajectory output for LAVD and diagnostics
   - A `Field` abstraction wrapping either an analytical callable or a gridded velocity dataset behind one interface

4. **Diagnostics (`diagnostics.py`)**
   - **FTLE**: provide *both* (a) finite-difference Jacobian on the gridded flow map and (b) `jacrev`-based per-point Jacobian. Benchmark and document the trade-off. Test backward FTLE sign convention against an analytical case.
   - **LAVD**: trajectory-averaged vorticity deviation computed inside the same `scan` as the flow map
   - **iLE**: instantaneous Lyapunov exponent from `∇v`
   - **Cauchy–Green tensor** + eigendecomposition via `vmap`'d `jnp.linalg.eigh` (be careful with batch shape semantics)

5. **LCS extraction (`lcs.py`)**
   - Hyperbolic LCS ridges from FTLE fields
   - Elliptic LCS / rotationally coherent vortices from LAVD level sets
   - Both gridded and Lagrangian-particle variants

6. **Utilities**: grid construction, domain padding, masking, plotting helpers. Plotting stays NumPy/matplotlib; do not JAX-ify.

## Improvements over numbacs you should make

- Replace every per-particle Python `for` loop with `vmap`
- Replace every time-stepping Python loop with `lax.scan` (prefer `scan` over `fori_loop` for cleaner gradients)
- All functions pure; no in-place mutation, no global state
- Use `equinox.Module` (or `flax.struct.dataclass`) Pytrees for parameter bundles instead of dicts
- Provide both `float32` and `float64` paths; default to `float64` for FTLE; document the trade-off
- Type hints everywhere; public API stable and documented
- Avoid Python control flow inside JIT regions; use `lax.cond` / `lax.switch`

## Validation

Reproduce canonical examples and check numerical agreement with numbacs and/or published references, to documented tolerances:
- Double gyre (Shadden et al.) — forward and backward FTLE
- Bickley jet — LAVD-based vortices
- A small gridded ocean-velocity example (HYCOM-style `u(t,y,x), v(t,y,x)`)

Each becomes a `pytest` regression test.

## Deliverables

- A `pip`/`uv`-installable package (suggested name `jax-cs`) with `pyproject.toml`
- Module layout mirroring the responsibilities above
- `examples/` with runnable scripts for double gyre, Bickley jet, and the HYCOM-like case
- `tests/` covering integrators against analytical ODE solutions, interpolators against known functions, and FTLE/LAVD against reference fields
- `docs/conventions.md` covering: grid layout, index-vs-physical space, time direction, sign of backward FTLE, units, NaN handling

## Constraints

- Pure JAX in the hot path; NumPy/SciPy only for I/O and plotting
- Any deviation from numbacs's behavior must be called out in a docstring or `docs/` note explaining why
- Keep the LCS coverage at least as broad as numbacs — do not silently drop methods to simplify the port
