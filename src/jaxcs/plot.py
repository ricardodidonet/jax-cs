"""Matplotlib plotting helpers (NumPy only — never JAX-traced).

Arrays are converted with ``np.asarray``; remember the package's
``(nx, ny)`` x-first layout means fields are transposed for ``pcolormesh``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["plot_field", "plot_curves", "plot_vortices"]


def plot_field(
    f,
    x,
    y,
    ax=None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    colorbar: bool = True,
    **kwargs,
):
    """``pcolormesh`` of an ``(nx, ny)`` field on grid ``x × y``."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    pc = ax.pcolormesh(
        np.asarray(x), np.asarray(y), np.asarray(f).T, cmap=cmap,
        shading="gouraud", **kwargs,
    )
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(pc, ax=ax)
    return ax


def plot_curves(curves: Sequence, ax=None, color: str = "r", lw: float = 1.5, **kwargs):
    """Overlay extracted curves (list of ``(k, 2)`` arrays)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for c in curves:
        c = np.asarray(c)
        ax.plot(c[:, 0], c[:, 1], color=color, lw=lw, **kwargs)
    return ax


def plot_vortices(
    rcv: Sequence[Tuple[np.ndarray, np.ndarray]],
    ax=None,
    color: str = "r",
    lw: float = 2.0,
):
    """Overlay :func:`jaxcs.lcs.rotcohvrt` output (boundary + center)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for boundary, center in rcv:
        ax.plot(boundary[:, 0], boundary[:, 1], color=color, lw=lw)
        ax.plot(center[0], center[1], "*", color=color)
    return ax
