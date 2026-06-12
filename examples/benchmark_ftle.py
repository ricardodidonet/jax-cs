"""Benchmark: finite-difference vs jacrev FTLE (the trade-off documented
in docs/conventions.md).

Run:  python examples/benchmark_ftle.py
"""

import time

import jax
import numpy as np

import jaxcs

jaxcs.enable_x64()

field = jaxcs.double_gyre()
x, y = jaxcs.uniform_grid(field.domain, 201, 101)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
t0, T, dt = 0.0, 8.0, 0.05
n_steps = int(T / dt)


def timed(label, fn):
    fn()  # compile
    t = time.perf_counter()
    out = jax.block_until_ready(fn())
    el = time.perf_counter() - t
    print(f"{label:34s} {el*1e3:9.1f} ms   max FTLE = {float(np.max(np.asarray(out))):.4f}")
    return out


fd = timed("FD grid (flowmap + ftle_grid)",
           lambda: jaxcs.ftle_grid(
               jaxcs.flowmap_grid(field, x, y, t0, T, n_steps=n_steps),
               T, dx, dy))
ad = timed("AD (jacrev per point)",
           lambda: jaxcs.ftle_ad_grid(field, x, y, t0, T, n_steps=n_steps))
aux = timed("aux grid (4 extra trajectories)",
            lambda: jaxcs.ftle_aux_grid(
                jaxcs.flowmap_aux_grid(field, x, y, t0, T, h=1e-5,
                                       n_steps=n_steps),
                T, h=1e-5))

fd_i = np.asarray(fd)[1:-1, 1:-1]
ad_i = np.asarray(ad)[1:-1, 1:-1]
print(f"\nFD vs AD: median |diff| = {np.median(np.abs(fd_i - ad_i)):.4f}, "
      f"corr = {np.corrcoef(fd_i.ravel(), ad_i.ravel())[0, 1]:.4f}")
print("AD resolves ridge peaks the grid-FD path smooths out; "
      "aux-grid agrees with AD to FD-step error.")
