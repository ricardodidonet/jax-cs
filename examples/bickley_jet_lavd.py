"""Bickley jet: LAVD field and rotationally coherent vortices.

LAVD is accumulated inside the same lax.scan as the advection; vorticity
comes from automatic differentiation of the analytical field (no
finite-difference stencil, no interpolant).

Run:  python examples/bickley_jet_lavd.py
"""

import pathlib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxcs

jaxcs.enable_x64()

field = jaxcs.bickley_jet()          # units: days, Mm
vorticity = jaxcs.ADVorticity(field)

x = jnp.linspace(0.0, field.period_x, 481)
y = jnp.linspace(-3.0, 3.0, 145)
t0, T = 0.0, 40.0

lavd = jaxcs.lavd_grid(field, vorticity, x, y, t0, T, n_steps=800)

rcv = jaxcs.rotcohvrt(np.asarray(lavd), np.asarray(x), np.asarray(y),
                      r=2.0, convexity_deficiency=1e-2)
print(f"found {len(rcv)} rotationally coherent vortices")
for _, center in rcv:
    print(f"  center: ({center[0]:.2f}, {center[1]:.2f}) Mm")

fig, ax = plt.subplots(figsize=(11, 4))
jaxcs.plot.plot_field(lavd, x, y, ax=ax, cmap="magma",
                      title=f"Bickley jet LAVD, T = {T} days")
jaxcs.plot.plot_vortices(rcv, ax=ax, color="cyan")
out = pathlib.Path(__file__).with_name("bickley_jet_lavd.png")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"wrote {out}")
