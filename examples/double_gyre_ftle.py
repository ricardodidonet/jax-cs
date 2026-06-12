"""Double gyre (Shadden et al. 2005): forward and backward FTLE + ridges.

Run:  python examples/double_gyre_ftle.py
Writes double_gyre_ftle.png next to this script.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

import jaxcs

jaxcs.enable_x64()

field = jaxcs.double_gyre()
x, y = jaxcs.uniform_grid(field.domain, 401, 201)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
t0, T = 0.0, 16.0

fig, axes = plt.subplots(2, 1, figsize=(9, 9))

for ax, Tsigned, label in [(axes[0], T, "forward"), (axes[1], -T, "backward")]:
    fmap = jaxcs.flowmap_grid(field, x, y, t0 if Tsigned > 0 else t0 + T,
                              Tsigned, dt=0.05)
    ftle = jaxcs.ftle_grid(fmap, Tsigned, dx, dy)
    C = jaxcs.cauchy_green_grid(fmap, dx, dy)
    _, eigvecs = jaxcs.eig_cauchy_green(C)
    ridges = jaxcs.ftle_ordered_ridges(ftle, eigvecs[..., :, -1], x, y,
                                       percentile=90, min_ridge_pts=15)
    jaxcs.plot.plot_field(ftle, x, y, ax=ax, title=f"{label} FTLE, |T| = {T}")
    jaxcs.plot.plot_curves(ridges[:6], ax=ax, color="r", lw=1.0)
    print(f"{label}: max FTLE = {float(np.max(np.asarray(ftle))):.4f}, "
          f"{len(ridges)} ridges")

out = pathlib.Path(__file__).with_name("double_gyre_ftle.png")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"wrote {out}")
