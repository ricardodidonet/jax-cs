"""LCS extraction: hyperbolic ridges/tensorlines and elliptic vortices.

Hot path vs post-processing
---------------------------
Pointwise/field computations (ridge-point detection, tensorline
integration) are pure JAX — vectorised over the grid, no Python loops.
The final *assembly* steps (linking ridge points into ordered curves,
walking contour levels, convexity tests) are inherently sequential,
data-dependent graph operations with ragged outputs; they run in NumPy /
contourpy / SciPy, exactly as numbacs runs them outside its Numba
kernels.  This is a documented deviation from the "pure JAX" rule — see
``docs/conventions.md`` ("What stays NumPy and why").

Methods
-------
* :func:`ftle_ridge_points` + :func:`ftle_ordered_ridges` — hyperbolic
  LCS as FTLE ridges (numbacs ``ftle_ridge_pts`` / ``ftle_ordered_ridges``).
* :func:`tensorlines` + :func:`hyperbolic_lcs` — variational hyperbolic
  LCS: integrate lines tangent to the *minimum* Cauchy–Green eigenvector
  through maxima of the maximum eigenvalue (numbacs ``hyperbolic_lcs``).
* :func:`hyperbolic_oecs` — instantaneous (objective Eulerian) analogue
  from the rate-of-strain tensor (numbacs ``hyperbolic_oecs``).
* :func:`rotcohvrt` — elliptic LCS / rotationally coherent vortices as
  outermost convex closed LAVD contours (numbacs ``rotcohvrt``).

Simplifications relative to numbacs are called out in each docstring.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .interpolation import bilinear
from .utils import arclength, max_in_radius, pts_in_poly, shoelace

__all__ = [
    "ftle_ridge_points",
    "ftle_ordered_ridges",
    "tensorlines",
    "hyperbolic_lcs",
    "hyperbolic_oecs",
    "rotcohvrt",
]


# --------------------------------------------------------------------------
# FTLE ridges
# --------------------------------------------------------------------------

def ftle_ridge_points(
    f: Array,
    eigvec_max: Array,
    x: Array,
    y: Array,
    sdd_thresh: float = 0.0,
    f_min: Optional[float] = None,
    percentile: float = 0.0,
) -> Tuple[Array, Array]:
    """Sub-grid FTLE ridge points (vectorised numbacs ``ftle_ridge_pts``).

    A grid node is a ridge candidate when the second directional
    derivative of ``f`` along the dominant Cauchy–Green eigenvector ``e``
    satisfies ``eᵀ H e < −sdd_thresh`` and the 1D Newton step
    ``t = −(∇f·e)/(eᵀHe)`` lands within half a cell — the ridge point is
    then ``p + t e``.

    Parameters
    ----------
    f
        FTLE field, ``(nx, ny)``.
    eigvec_max
        Dominant eigenvector per node, ``(nx, ny, 2)`` (take
        ``eigvecs[..., :, -1]`` from
        :func:`jaxcs.diagnostics.eig_cauchy_green`).
    x, y
        Grid coordinate arrays.
    sdd_thresh
        Ridge-strength threshold on the (negative) second directional
        derivative.
    f_min, percentile
        Keep only nodes with ``f > f_min``; if ``f_min`` is None it is
        the given percentile of ``f`` (0 → no threshold).

    Returns
    -------
    points
        ``(nx*ny, 2)`` candidate locations (junk rows where invalid).
    is_ridge
        ``(nx*ny,)`` boolean validity mask.  ``points[is_ridge]`` are the
        ridge points (boolean indexing — do outside ``jit``).
    """
    nx, ny = f.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    if f_min is None:
        f_min = jnp.percentile(f, percentile) if percentile > 0 else 0.0

    fx, fy = jnp.gradient(f, dx, dy)
    fxx, fxy = jnp.gradient(fx, dx, dy)
    _, fyy = jnp.gradient(fy, dx, dy)

    ex = eigvec_max[..., 0]
    ey = eigvec_max[..., 1]
    sdd = ex * (fxx * ex + fxy * ey) + ey * (fxy * ex + fyy * ey)
    t = -(fx * ex + fy * ey) / jnp.where(sdd == 0.0, 1.0, sdd)

    X, Y = jnp.meshgrid(x, y, indexing="ij")
    pts = jnp.stack([X + t * ex, Y + t * ey], axis=-1)

    interior = jnp.zeros((nx, ny), bool).at[2:-2, 2:-2].set(True)
    ok = (
        interior
        & (f > f_min)
        & (sdd < -sdd_thresh)
        & (jnp.abs(t * ex) <= dx / 2)
        & (jnp.abs(t * ey) <= dy / 2)
    )
    return pts.reshape(-1, 2), ok.ravel()


def ftle_ordered_ridges(
    f: Array,
    eigvec_max: Array,
    x: Array,
    y: Array,
    sdd_thresh: float = 0.0,
    percentile: float = 0.0,
    dist_tol: Optional[float] = None,
    min_ridge_pts: int = 3,
) -> List[np.ndarray]:
    """Ordered FTLE ridge curves.

    Detection is :func:`ftle_ridge_points` (JAX); the linking is a NumPy
    greedy chain: starting from each unvisited point, the curve is
    extended in both directions to the nearest unvisited point within
    ``dist_tol``, requiring direction continuity (no reversals sharper
    than 90°).  Simpler than numbacs's eigenvector-guided
    stepper-plus-endpoint-merging, but the same intent; expect small
    differences in how nearby ridges are split or joined.

    Returns a list of ``(k, 2)`` arrays, longest first.
    """
    pts_all, ok = ftle_ridge_points(
        f, eigvec_max, x, y, sdd_thresh=sdd_thresh, percentile=percentile
    )
    pts = np.asarray(pts_all[np.asarray(ok)])
    if len(pts) == 0:
        return []
    if dist_tol is None:
        dist_tol = 1.5 * float(np.hypot(x[1] - x[0], y[1] - y[0]))

    n = len(pts)
    unused = np.ones(n, bool)

    def nearest(p, direction):
        idx = np.nonzero(unused)[0]
        if len(idx) == 0:
            return -1
        d = np.hypot(*(pts[idx] - p).T)
        order = np.argsort(d)
        for k in order:
            if d[k] > dist_tol:
                break
            j = idx[k]
            step = pts[j] - p
            if direction is None or np.dot(step, direction) > 0:
                return j
        return -1

    ridges = []
    for start in range(n):
        if not unused[start]:
            continue
        unused[start] = False
        curve = [pts[start]]
        for sign in (1, -1):
            direction = None
            p = pts[start]
            while True:
                j = nearest(p, direction)
                if j < 0:
                    break
                unused[j] = False
                q = pts[j]
                direction = q - p
                p = q
                if sign == 1:
                    curve.append(q)
                else:
                    curve.insert(0, q)
        if len(curve) >= min_ridge_pts:
            ridges.append(np.array(curve))
    ridges.sort(key=len, reverse=True)
    return ridges


# --------------------------------------------------------------------------
# Tensorlines (variational hyperbolic LCS / OECS)
# --------------------------------------------------------------------------

def _oriented_eigvec(ev: Array, x: Array, y: Array, p: Array, prev: Array) -> Array:
    """Interpolate a (sign-ambiguous) unit eigenvector field at ``p``.

    The four corner vectors are sign-aligned with ``prev`` before the
    bilinear combination (numbacs ``_reorient_eigvec``), the result is
    renormalised and again aligned with ``prev``.
    """
    nx, ny = ev.shape[:2]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ix = (p[0] - x[0]) / dx
    iy = (p[1] - y[0]) / dy
    i0 = jnp.clip(jnp.floor(ix).astype(int), 0, nx - 2)
    j0 = jnp.clip(jnp.floor(iy).astype(int), 0, ny - 2)
    wx = jnp.clip(ix - i0, 0.0, 1.0)
    wy = jnp.clip(iy - j0, 0.0, 1.0)

    corners = jnp.stack(
        [ev[i0, j0], ev[i0 + 1, j0], ev[i0, j0 + 1], ev[i0 + 1, j0 + 1]]
    )  # (4, 2)
    sign = jnp.where(corners @ prev < 0, -1.0, 1.0)
    corners = corners * sign[:, None]
    w = jnp.array([(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy])
    v = w @ corners
    v = v / jnp.maximum(jnp.linalg.norm(v), 1e-12)
    return jnp.where(v @ prev < 0, -v, v)


def tensorlines(
    eigvec: Array,
    x: Array,
    y: Array,
    seeds: Array,
    h: float,
    steps: int,
    stop_field: Optional[Array] = None,
    stop_value: float = -jnp.inf,
) -> Tuple[Array, Array]:
    """Integrate tensorlines of a unit eigenvector field (pure JAX).

    RK4 in arclength with per-stage orientation correction; all seeds run
    batched under ``vmap`` inside one ``lax.scan``.  A line *freezes* (its
    position stops updating) once it leaves the grid or once
    ``stop_field`` interpolated at its position drops below
    ``stop_value`` (numbacs stops inside its Python loop; freezing is the
    JAX-native equivalent).

    Parameters
    ----------
    eigvec
        Unit eigenvector field, ``(nx, ny, 2)`` (sign-ambiguous is fine).
    x, y
        Grid coordinates.
    seeds
        Seed points *and* initial orientations, shape ``(ns, 2, 2)``:
        ``seeds[:, 0]`` positions, ``seeds[:, 1]`` initial direction
        (use ``±eigvec`` at the seed to integrate both ways).
    h
        Arclength step.
    steps
        Number of steps.
    stop_field, stop_value
        Optional ``(nx, ny)`` scalar field and threshold for freezing
        (e.g. ``λ_max`` with the domain-average ``λ̄`` as in numbacs).

    Returns
    -------
    lines
        ``(ns, steps + 1, 2)`` positions (constant after freezing).
    alive
        ``(ns, steps + 1)`` False once frozen.
    """
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]

    def stop_ok(p):
        in_dom = (p[0] >= xmin) & (p[0] <= xmax) & (p[1] >= ymin) & (p[1] <= ymax)
        if stop_field is None:
            return in_dom
        val = bilinear(stop_field, p[0], p[1], x, y, mode="nearest")
        return in_dom & (val >= stop_value)

    def step_one(p, d):
        k1 = _oriented_eigvec(eigvec, x, y, p, d)
        k2 = _oriented_eigvec(eigvec, x, y, p + 0.5 * h * k1, k1)
        k3 = _oriented_eigvec(eigvec, x, y, p + 0.5 * h * k2, k2)
        k4 = _oriented_eigvec(eigvec, x, y, p + h * k3, k3)
        v = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        v = v / jnp.maximum(jnp.linalg.norm(v), 1e-12)
        return p + h * v, v

    def body(carry, _):
        p, d, alive = carry
        p_new, d_new = step_one(p, d)
        alive_new = alive & stop_ok(p_new)
        p_out = jnp.where(alive_new, p_new, p)
        d_out = jnp.where(alive_new, d_new, d)
        return (p_out, d_out, alive_new), (p_out, alive_new)

    def run(seed):
        p0, d0 = seed[0], seed[1]
        d0 = d0 / jnp.maximum(jnp.linalg.norm(d0), 1e-12)
        (_, _, _), (ps, alive) = jax.lax.scan(
            body, (p0, d0, jnp.asarray(True)), None, length=steps
        )
        return (
            jnp.concatenate([p0[None], ps], axis=0),
            jnp.concatenate([jnp.array([True]), alive]),
        )

    return jax.vmap(run)(seeds)


def _eigvec_at_points(eigvec: Array, x: Array, y: Array, pts: np.ndarray) -> np.ndarray:
    ix = np.clip(np.round((pts[:, 0] - float(x[0])) / float(x[1] - x[0])), 0, eigvec.shape[0] - 1)
    iy = np.clip(np.round((pts[:, 1] - float(y[0])) / float(y[1] - y[0])), 0, eigvec.shape[1] - 1)
    return np.asarray(eigvec)[ix.astype(int), iy.astype(int)]


def _tensorline_curves(
    eigvec: Array,
    x: Array,
    y: Array,
    seed_pts: np.ndarray,
    h: float,
    steps: int,
    stop_field: Optional[Array],
    stop_value: float,
    min_len: float,
) -> List[np.ndarray]:
    """Run tensorlines both ways from each seed; join into NumPy curves."""
    d0 = _eigvec_at_points(eigvec, x, y, seed_pts)
    seeds = np.concatenate(
        [np.stack([seed_pts, d0], axis=1), np.stack([seed_pts, -d0], axis=1)]
    )
    lines, alive = tensorlines(
        eigvec, x, y, jnp.asarray(seeds), h, steps,
        stop_field=stop_field, stop_value=stop_value,
    )
    lines = np.asarray(lines)
    alive = np.asarray(alive)
    ns = len(seed_pts)
    curves = []
    for i in range(ns):
        fwd = lines[i][alive[i]]
        bwd = lines[i + ns][alive[i + ns]]
        curve = np.concatenate([bwd[::-1], fwd[1:]], axis=0)
        if len(curve) >= 2 and arclength(curve) >= min_len:
            curves.append(curve)
    return curves


def hyperbolic_lcs(
    eigval_max: Array,
    eigvec_min: Array,
    x: Array,
    y: Array,
    r: float,
    h: float = -1.0,
    steps: int = 2000,
    n_seeds: int = -1,
    lambda_min: Optional[float] = None,
    min_len: float = 0.0,
) -> List[np.ndarray]:
    """Variational hyperbolic (repelling) LCS candidates.

    Shrink lines: curves tangent to the *minimum* Cauchy–Green
    eigenvector seeded at local maxima of ``λ_max`` (exclusion radius
    ``r``), integrated until they leave the domain or enter the region
    where ``λ_max`` falls below ``lambda_min`` (default: domain mean, as
    numbacs uses).  For attracting LCS, run on the *backward* flow map.

    Documented simplification vs numbacs ``hyperbolic_lcs``: numbacs
    additionally deduplicates near-parallel tensorlines and merges nearby
    endpoints; here each seed yields one curve, ranked by seed strength.

    Parameters
    ----------
    eigval_max, eigvec_min
        ``λ_max`` ``(nx, ny)`` and the minimum eigenvector
        ``(nx, ny, 2)`` of the Cauchy–Green tensor
        (``eigvecs[..., :, 0]``).
    r
        Exclusion radius for seed maxima.
    h
        Arclength step; if negative, half the grid diagonal is used.
    steps
        Steps per direction.
    n_seeds
        Maximum number of seeds (−1 → all maxima above the threshold).
    lambda_min
        Freeze threshold on ``λ_max`` (None → mean of ``λ_max``).
    min_len
        Discard curves shorter than this.

    Returns
    -------
    List of ``(k, 2)`` curves, strongest seed first.
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if h <= 0:
        h = 0.5 * float(np.hypot(dx, dy))
    lam = np.asarray(eigval_max)
    if lambda_min is None:
        lambda_min = float(lam.mean())
    _, inds = max_in_radius(lam, r, dx, dy, n=n_seeds, min_val=lambda_min)
    if len(inds) == 0:
        return []
    seed_pts = np.column_stack([np.asarray(x)[inds[:, 0]], np.asarray(y)[inds[:, 1]]])
    return _tensorline_curves(
        eigvec_min, x, y, seed_pts, h, steps, eigval_max, lambda_min, min_len
    )


def hyperbolic_oecs(
    s_max: Array,
    eigvec_min: Array,
    x: Array,
    y: Array,
    r: float,
    h: float = -1.0,
    steps: int = 500,
    n_seeds: int = -1,
    s_min: Optional[float] = None,
    min_len: float = 0.0,
) -> List[np.ndarray]:
    """Objective Eulerian coherent structures (instantaneous analogue).

    Same construction as :func:`hyperbolic_lcs` applied to the
    rate-of-strain tensor: repelling OECS are curves tangent to the
    minimum eigenvector of ``S`` through maxima of ``s_max = λ_max(S)``
    (Serra & Haller 2016).  For attracting OECS pass ``−λ_min(S)`` as
    ``s_max`` and the maximum eigenvector as ``eigvec_min``.
    """
    return hyperbolic_lcs(
        s_max, eigvec_min, x, y, r, h=h, steps=steps, n_seeds=n_seeds,
        lambda_min=s_min, min_len=min_len,
    )


# --------------------------------------------------------------------------
# Elliptic LCS: rotationally coherent vortices from LAVD
# --------------------------------------------------------------------------

def rotcohvrt(
    lavd: Array,
    x: Array,
    y: Array,
    r: float,
    convexity_deficiency: float = 5e-3,
    min_val: float = -1.0,
    nlevs: int = 20,
    start_level: float = 0.0,
    end_level: float = 0.0,
    min_len: float = 0.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Rotationally coherent vortices from the LAVD field.

    Direct port of numbacs ``rotcohvrt`` (``convex_hull`` method): find
    LAVD maxima (exclusion radius ``r``), then for increasing contour
    levels keep the first closed contour around each maximum whose
    relative convexity deficiency — ``(hull area − area)/area`` — is
    below the tolerance.  Pure NumPy/contourpy/SciPy post-processing.

    Parameters mirror numbacs: ``min_val`` defaults to the 80th
    percentile of LAVD, ``start_level`` to the 70th percentile,
    ``end_level`` to the LAVD maximum.

    Returns
    -------
    List of ``(boundary, center)`` pairs: boundary is the convex hull of
    the accepted contour, ``(k, 2)``; center is the LAVD maximum, ``(2,)``.
    """
    from contourpy import contour_generator
    from scipy.spatial import ConvexHull

    lavd = np.asarray(lavd)
    x = np.asarray(x)
    y = np.asarray(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    if min_val == -1.0:
        min_val = float(np.percentile(lavd, 80))
    max_vals, max_inds = max_in_radius(lavd, r, dx, dy, min_val=min_val)
    if len(max_vals) == 0:
        return []
    if start_level == 0.0:
        start_level = float(np.percentile(lavd, 70))
    if end_level == 0.0:
        end_level = float(max_vals.max())
    clevels = np.linspace(start_level, end_level, nlevs)

    rem = np.column_stack([x[max_inds[:, 0]], y[max_inds[:, 1]]])
    rcv: List[Tuple[np.ndarray, np.ndarray]] = []
    cgen = contour_generator(x=x, y=y, z=lavd.T)
    for lev in clevels:
        for contour in cgen.lines(lev):
            closed = np.allclose(contour[0], contour[-1])
            if not closed:
                continue
            if min_len and arclength(contour) <= min_len:
                continue
            ind = pts_in_poly(contour, rem)
            if ind < 0:
                continue
            hull = ConvexHull(contour)
            ch = contour[np.concatenate([hull.vertices, hull.vertices[:1]])]
            area = shoelace(contour)
            if (hull.volume - area) / area < convexity_deficiency:
                rcv.append((ch, rem[ind]))
                rem = np.delete(rem, ind, axis=0)
                if len(rem) == 0:
                    return rcv
    return rcv
