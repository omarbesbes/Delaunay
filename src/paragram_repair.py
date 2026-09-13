"""Exact CPU fallback for Paragram cells that failed (non-zero status).

Paragram builds each Voronoi cell on the GPU in float32 with a fixed plane/vertex budget; a cell
that overflows or whose boundary walk becomes inconsistent returns a truncated neighbour list
(status != 0).  Every Delaunay tetrahedron through a missing edge is then unrecoverable by any
Voronoi->Delaunay conversion.  The original GPU Voronoi paper handled such cells with a CPU
fallback; this module does the same in Python, in two stages:

  1. local: for a failed cell p take its k nearest points, compute their Delaunay triangulation
     (Qhull), read the star of p and hence its neighbours.  Certificate (security radius): if the
     k-th neighbour distance >= 2 x the largest circumradius of the star, no point outside the
     patch can change the star, so the result is exact.  Otherwise double k up to k_max.
     Volumetric point clouds certify at k = 64..256 almost always.
  2. global: cells that cannot be certified (typically surface samples, whose stars contain tets
     with huge circumspheres) are resolved from ONE exact Delaunay triangulation of the whole
     point set (CGAL tool `cgal_delaunay` if CGAL_DELAUNAY_BIN is set, else the CGAL Python
     bindings, else Qhull).  A short probe skips stage 1 when it is hopeless.

The repaired lists replace the failed cells' entries in the CSR adjacency (the union with the
other cells' lists is symmetrised by the converter).

    from paragram_repair import repair_failed_cells
    adjacency, offsets, stats = repair_failed_cells(points, diag.adjacency, diag.offsets, diag.status)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Callable

import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree

__all__ = ["delaunay_star", "exact_delaunay", "repair_failed_cells"]


# ----------------------------------------------------------------------------------------
# exact global triangulation (best available backend)
# ----------------------------------------------------------------------------------------


def exact_delaunay(points: np.ndarray) -> tuple[np.ndarray, str]:
    """Finite tetrahedra (T, 4) of the Delaunay triangulation of `points` and the backend used."""
    binary = os.environ.get("CGAL_DELAUNAY_BIN")
    if binary and os.path.exists(binary):
        with tempfile.TemporaryDirectory() as d:
            pin, pout = os.path.join(d, "points.f64"), os.path.join(d, "tets.i32")
            np.ascontiguousarray(points, dtype="<f8").tofile(pin)
            subprocess.run([binary, pin, pout], check=True, capture_output=True)
            return np.fromfile(pout, dtype="<i4").reshape(-1, 4).astype(np.int64), "cgal_delaunay"
    try:
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3

        index = {tuple(p): i for i, p in enumerate(points.tolist())}
        dt = Delaunay_triangulation_3()
        dt.insert([Point_3(*p) for p in points.tolist()])
        tets = np.empty((dt.number_of_finite_cells(), 4), dtype=np.int64)
        for k, c in enumerate(dt.finite_cells()):
            for i in range(4):
                p = c.vertex(i).point()
                tets[k, i] = index[(p.x(), p.y(), p.z())]
        return tets, "CGAL bindings"
    except ImportError:
        pass
    return Delaunay(points, qhull_options="Qbb Qc Qz Q12").simplices.astype(
        np.int64
    ), "scipy.Delaunay"


def _neighbour_lists(tets: np.ndarray, n: int, cells: np.ndarray) -> dict[int, np.ndarray]:
    """Delaunay neighbours of the given cells, read from a tetrahedron list."""
    t = tets.astype(np.int64)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    u = np.concatenate([t[:, i] for i, j in pairs] + [t[:, j] for i, j in pairs])
    v = np.concatenate([t[:, j] for i, j in pairs] + [t[:, i] for i, j in pairs])
    want = np.zeros(n, bool)
    want[cells] = True
    m = want[u]
    u, v = u[m], v[m]
    keys = np.unique(u * n + v)
    uu, vv = keys // n, keys % n
    out: dict[int, np.ndarray] = {}
    starts = np.searchsorted(uu, cells, side="left")
    ends = np.searchsorted(uu, cells, side="right")
    for c, s, e in zip(cells.tolist(), starts.tolist(), ends.tolist()):
        out[c] = vv[s:e]
    return out


# ----------------------------------------------------------------------------------------
# local star with security-radius certificate
# ----------------------------------------------------------------------------------------


def delaunay_star(local_pts: np.ndarray, centre: int = 0) -> tuple[np.ndarray, float]:
    """Neighbours of `centre` in the Delaunay triangulation of `local_pts` and the largest
    circumradius of its incident tets (inf if degenerate)."""
    try:
        tri = Delaunay(local_pts, qhull_options="Qbb Qc Qz Q12")
    except Exception:  # noqa: BLE001 - degenerate patch: joggle as a last resort
        tri = Delaunay(local_pts, qhull_options="QJ")
    simp = tri.simplices
    star = simp[(simp == centre).any(1)]
    if len(star) == 0:
        return np.empty(0, dtype=np.int64), np.inf
    nb = np.unique(star)
    nb = nb[nb != centre]
    a, b, c, d = (local_pts[star[:, i]] for i in range(4))
    B, C, D = b - a, c - a, d - a
    cxd, dxb, bxc = np.cross(C, D), np.cross(D, B), np.cross(B, C)
    det = np.einsum("ij,ij->i", B, cxd)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = (
            a
            + (
                (B * B).sum(1)[:, None] * cxd
                + (C * C).sum(1)[:, None] * dxb
                + (D * D).sum(1)[:, None] * bxc
            )
            / (2 * det)[:, None]
        )
    r = np.linalg.norm(o - local_pts[centre], axis=1)
    r_max = float(np.max(r)) if np.all(np.isfinite(r)) else float("inf")
    return nb.astype(np.int64), r_max


def _local_star(
    pts: np.ndarray, tree: cKDTree, p: int, k_start: int, k_max: int
) -> tuple[np.ndarray, bool, int]:
    n = len(pts)
    k = min(k_start, n)
    while True:
        d, idx = tree.query(pts[p], k=k)
        idx = np.asarray(idx).reshape(-1)
        d = np.asarray(d).reshape(-1)
        order = np.argsort(d, kind="stable")
        idx, d = idx[order], d[order]
        if idx[0] != p:  # duplicates: make sure p is the centre
            pos = np.nonzero(idx == p)[0]
            if len(pos):
                idx[[0, pos[0]]] = idx[[pos[0], 0]]
            else:
                idx = np.concatenate([[p], idx[:-1]])
        local_nb, r_max = delaunay_star(pts[idx], 0)
        nb = idx[local_nb]
        if d[-1] >= 2.0 * r_max:
            return nb, True, k
        if k >= min(k_max, n):
            return nb, False, k
        k = min(2 * k, n, k_max)


# ----------------------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------------------


@torch.no_grad()
def repair_failed_cells(
    points: torch.Tensor,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    status: torch.Tensor,
    *,
    k_start: int = 64,
    k_max: int = 512,
    probe: int = 40,
    local_max_cells: int = 200,
    global_fallback: Callable[[np.ndarray], tuple[np.ndarray, str]] | None = exact_delaunay,
    cells: np.ndarray | None = None,
    include_hull: bool = True,
    hull_tol: float = 1e-6,
    hull_max_fraction: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Replace the neighbour lists of failed cells (status != 0, or the given `cells`) by exact
    CPU-computed Delaunay neighbours.  Returns (adjacency int32, offsets int32, stats) on the
    device of `points`.

    A probe of `probe` cells decides whether the local stage is worth running (>= 50 % certified);
    it is skipped altogether when more than `local_max_cells` cells need repair, because one exact
    global triangulation is then cheaper.  Cells left uncertified are resolved by `global_fallback`
    (one exact triangulation of all points), or kept as their best local star if that is None.

    include_hull: also repair the cells of points lying on the convex hull (within `hull_tol` x
    extent of a hull facet), unless more than `hull_max_fraction` of all points are on the hull
    (then the "repair" would replace Paragram's whole result by the CPU triangulation).  Their Voronoi cells are unbounded prisms with nearly parallel
    bisector planes, which Paragram's float32 clipping resolves poorly *without* reporting a
    failure; this is where the remaining silent edge losses concentrate.
    """
    dev = points.device
    t0 = time.perf_counter()
    pts = points.detach().to("cpu", torch.float64).numpy()
    n = len(pts)
    adj = adjacency.detach().to("cpu", torch.long).numpy()
    off = offsets.detach().to("cpu", torch.long).numpy()
    st = status.detach().to("cpu").numpy().reshape(-1)
    failed = np.nonzero(st != 0)[0] if cells is None else np.asarray(cells, dtype=np.int64)
    hull_cells = 0
    hull_skipped = 0
    if include_hull and n >= 4:
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(pts)
            eq = hull.equations  # (F, 4): n . x + d = 0, outward normals
            extent = float(np.ptp(pts, axis=0).max())
            dist = np.full(n, np.inf)
            for s in range(0, len(eq), 256):  # chunked N x F distance to the facet planes
                dist = np.minimum(
                    dist, np.abs(pts @ eq[s : s + 256, :3].T + eq[s : s + 256, 3]).min(1)
                )
            on_hull = np.nonzero(dist <= hull_tol * extent)[0]
            hull_cells = len(np.setdiff1d(on_hull, failed))
            if hull_cells > hull_max_fraction * n:
                # e.g. points sampled on a sphere: everything is on the hull, and "repairing" it all
                # would just replace Paragram's result by the CPU triangulation.  Leave it.
                hull_skipped = hull_cells
                hull_cells = 0
            else:
                failed = np.union1d(failed, on_hull)
        except Exception as exc:  # noqa: BLE001 - degenerate (flat) input: no hull repair
            hull_cells = -1
            print(f"paragram_repair: hull detection skipped ({exc})")
    stats = {
        "failed_cells": int((st != 0).sum()) if cells is None else len(cells),
        "hull_cells": hull_cells,
        "hull_cells_skipped": hull_skipped,
        "repaired_cells": len(failed),
        "repaired_fraction": len(failed) / max(n, 1),
        "local_certified": 0,
        "local_uncertified": 0,
        "resolved_globally": 0,
        "global_backend": None,
        "local_skipped": False,
        "k_hist": {},
        "local_seconds": 0.0,
        "global_seconds": 0.0,
        "seconds": 0.0,
    }
    if len(failed) == 0:
        return adjacency, offsets, stats

    lists = [adj[off[i] : off[i + 1]] for i in range(n)]
    tree = cKDTree(pts)
    pending = list(failed.tolist())
    uncertified: list[int] = []

    # ---- stage 1: local stars, after a probe ------------------------------------------------
    t1 = time.perf_counter()
    if len(pending) > local_max_cells and global_fallback is not None:
        stats["local_skipped"] = True
        stats["local_uncertified"] = len(pending)
        uncertified = list(pending)
        pending = []
    # spread the probe over the index range: failed indices are sorted, and low/high indices tend to
    # sit on the same side of the bounding box (np.unique sorts by x), i.e. near the hull
    probe_idx = (
        np.unique(np.linspace(0, len(pending) - 1, min(probe, len(pending))).astype(int))
        if pending
        else np.empty(0, dtype=int)
    )
    probe_cells = [pending[i] for i in probe_idx]
    probe_ok = 0
    probe_results: dict[int, tuple[np.ndarray, bool, int]] = {}
    for p in probe_cells:
        res = _local_star(pts, tree, p, k_start, k_max)
        probe_results[p] = res
        probe_ok += res[1]
    run_local = bool(probe_cells) and probe_ok >= 0.5 * len(probe_cells)
    if pending:
        stats["local_skipped"] = not run_local
        uncertified = []
    for p in pending:
        if p in probe_results:
            nb, ok, k = probe_results[p]
        elif run_local:
            nb, ok, k = _local_star(pts, tree, p, k_start, k_max)
        else:
            uncertified.append(p)
            continue
        if ok:
            lists[p] = nb.astype(np.int64)
            stats["local_certified"] += 1
            stats["k_hist"][k] = stats["k_hist"].get(k, 0) + 1
        else:
            uncertified.append(p)
            lists[p] = nb.astype(np.int64)  # best effort, overwritten by the global stage if any
    stats["local_uncertified"] = len(uncertified)
    stats["local_seconds"] = time.perf_counter() - t1

    # ---- stage 2: one exact global triangulation for the rest -------------------------------
    if uncertified and global_fallback is not None:
        t2 = time.perf_counter()
        tets, backend = global_fallback(pts)
        for c, nb in _neighbour_lists(tets, n, np.asarray(uncertified, dtype=np.int64)).items():
            lists[c] = nb.astype(np.int64)
        stats["resolved_globally"] = len(uncertified)
        stats["global_backend"] = backend
        stats["global_seconds"] = time.perf_counter() - t2

    new_off = np.zeros(n + 1, dtype=np.int64)
    new_off[1:] = np.cumsum([len(x) for x in lists])
    new_adj = np.concatenate(lists) if n else np.empty(0, dtype=np.int64)
    stats["seconds"] = time.perf_counter() - t0
    stats["k_hist"] = {int(k): v for k, v in sorted(stats["k_hist"].items())}
    return (
        torch.from_numpy(new_adj.astype(np.int32)).to(dev),
        torch.from_numpy(new_off.astype(np.int32)).to(dev),
        stats,
    )
