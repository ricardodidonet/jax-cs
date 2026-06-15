"""Bickley jet: forward and backward FTLE.

The Bickley jet (Rypina et al. 2007) is x-periodic with period
``π r_e ≈ 20.01 Mm``; the seeding grid spans one full period in x and the
channel ``[-3, 3]`` Mm in y. Forward FTLE (``T > 0``) highlights repelling
material lines; backward FTLE (``T < 0``) highlights attracting ones — the
filaments that organise the meandering jet and its vortices.

Run:  python examples/bickley_jet_ftle.py
Writes bickley_jet_ftle.png next to this script.
"""

import pathlib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxcs

jaxcs.enable_x64()                       # float64: recommended for FTLE

field = jaxcs.bickley_jet()              # units: days, Mm
x = jnp.linspace(0.0, field.period_x, 481)
y = jnp.linspace(-3.0, 3.0, 145)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
t0, T = 0.0, 6.0                         # days

fig, axes = plt.subplots(2, 1, figsize=(11, 7))

for ax, Tsigned, label in [(axes[0], T, "forward"), (axes[1], -T, "backward")]:
    # backward FTLE: seed at the end time and integrate back to t0
    fmap = jaxcs.flowmap_grid(field, x, y, t0 if Tsigned > 0 else t0 + T,
                              Tsigned, dt=0.05)
    ftle = jaxcs.ftle_grid(fmap, Tsigned, dx, dy)
    jaxcs.plot.plot_field(ftle, x, y, ax=ax, cmap="inferno",
                          title=f"Bickley jet {label} FTLE, |T| = {T} days")
    print(f"{label}: max FTLE = {float(np.max(np.asarray(ftle))):.4f} / day")

out = pathlib.Path(__file__).with_name("bickley_jet_ftle.png")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"wrote {out}")
