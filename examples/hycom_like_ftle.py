"""HYCOM-style gridded ocean velocity: masked FTLE on a lon/lat grid.

Builds a synthetic 'ocean' dataset in HYCOM layout — u(t, y, x), v(t, y, x)
in m/s on a lon/lat grid with NaN land cells — and runs the full gridded
workflow: transpose to jax-cs's (t, x, y) layout, fill NaNs + mask,
spherical velocity conversion, backward FTLE (attracting fronts), masked
diagnostics.  Substitute real HYCOM/Copernicus arrays for `make_dataset`.

Run:  python examples/hycom_like_ftle.py
"""

import pathlib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxcs

jaxcs.enable_x64()

R_KM = 6371.0
KM_PER_DAY = 86.4  # 1 m/s in km/day


def make_dataset(nt=11, nlat=80, nlon=120):
    """Synthetic mesoscale eddy field, HYCOM-style axes u(t, lat, lon) [m/s]."""
    tvals = np.linspace(0.0, 10.0, nt)                # days
    lon = np.linspace(-5.0, 5.0, nlon)
    lat = np.linspace(35.0, 42.0, nlat)
    T, LAT, LON = np.meshgrid(tvals, lat, lon, indexing="ij")
    # two drifting counter-rotating eddies (stream function in deg^2/day)
    cx1, cx2 = -2.0 + 0.15 * T, 2.0 - 0.15 * T
    cy = 38.5
    psi = (np.exp(-(((LON - cx1) ** 2 + (LAT - cy) ** 2) / 1.5))
           - np.exp(-(((LON - cx2) ** 2 + (LAT - cy) ** 2) / 1.5))) * 8.0
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]
    u_deg = -np.gradient(psi, dlat, axis=1)          # deg/day
    v_deg = np.gradient(psi, dlon, axis=2)
    # convert to m/s as a real dataset would provide
    coslat = np.cos(np.deg2rad(LAT))
    u = u_deg * np.deg2rad(1.0) * R_KM * coslat / KM_PER_DAY * 1.0
    v = v_deg * np.deg2rad(1.0) * R_KM / KM_PER_DAY * 1.0
    # a land mass in the corner
    land = (LON[0] > 3.2) & (LAT[0] < 36.5)
    u[:, land] = np.nan
    v[:, land] = np.nan
    return tvals, lon, lat, u, v


tvals, lon, lat, u_tyx, v_tyx = make_dataset()

# HYCOM layout (t, lat, lon) -> jax-cs layout (t, x=lon, y=lat)
u = np.transpose(u_tyx, (0, 2, 1)) * KM_PER_DAY      # km/day
v = np.transpose(v_tyx, (0, 2, 1)) * KM_PER_DAY

(u, v), land_mask = jaxcs.fill_nans_and_get_mask([u, v])
mask = jaxcs.binary_dilation(jnp.asarray(land_mask), iterations=2)

field = jaxcs.GriddedField2D(
    u=u, v=v,
    tvals=jnp.asarray(tvals), xvals=jnp.asarray(lon), yvals=jnp.asarray(lat),
    spherical=True, r=R_KM,
)

x = jnp.asarray(lon)
y = jnp.asarray(lat)
t0, T = 10.0, -8.0   # backward FTLE from the last day: attracting fronts
fmap = jaxcs.flowmap_grid(field, x, y, t0, T, dt=0.02)
ftle = jaxcs.ftle_grid(fmap, T, float(x[1] - x[0]), float(y[1] - y[0]),
                       mask=mask)
print(f"backward FTLE: max = {float(jnp.max(ftle)):.3f} 1/day")

fig, ax = plt.subplots(figsize=(9, 6))
jaxcs.plot.plot_field(ftle, x, y, ax=ax, cmap="inferno",
                      title="Backward FTLE (1/day), synthetic HYCOM-like data")
ax.contourf(np.asarray(x), np.asarray(y), np.asarray(mask).T,
            levels=[0.5, 1.5], colors=["0.6"])
out = pathlib.Path(__file__).with_name("hycom_like_ftle.png")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"wrote {out}")
