"""Convert a Paragram Voronoi diagram into the 3D Delaunay triangulation.

Paragram returns the Voronoi *adjacency* (CSR neighbour lists), which is exactly the
edge graph (1-skeleton) of the Delaunay triangulation.  Every Delaunay tetrahedron is a
Voronoi vertex, and a Voronoi vertex of cell `a` is the point equidistant from `a` and
three of its neighbours.  This module recovers the tetrahedra from the adjacency alone,
fully vectorised in torch (runs on the same CUDA device as the diagram, no host round
trip):

  1. build the undirected, sorted edge set E of the adjacency graph
  2. enumerate triangles  a<b<c  with ab, ac, bc in E
  3. enumerate 4-cliques  a<b<c<d  with ad, bd, cd in E
  4. keep a 4-clique iff no Delaunay neighbour of a, b, c or d lies strictly inside its
     circumsphere (insphere determinant in float64)

Step 4 is exact, not a heuristic: the Voronoi cell of `a` is the intersection of the
bisector half-spaces of its Delaunay neighbours, so the circumcentre `o` of {a,b,c,d}
lies in Vor(a) iff no neighbour of `a` is strictly closer to `o` than `a` is.  Since
a, b, c, d are all at the same distance from `o`, that already makes `o` a Voronoi
vertex shared by all four cells, i.e. {a,b,c,d} is a Delaunay tetrahedron.  Only the
*local* neighbour lists ever have to be consulted, no global search.  Checking all four
cells instead of just `a` only adds numerical robustness.

Caveats
  * Degenerate input (5+ exactly co-spherical points, e.g. regular grids) has no unique
    Delaunay triangulation; every co-spherical 4-clique passes the test, so overlapping
    tetrahedra are returned.  Jitter such inputs.
  * Paragram clips cells to the bounding box of the points, so faces lying entirely
    outside that box are absent from the adjacency.  The interior is exact; a few
    tetrahedra of the convex hull whose circumcentre is far outside the box can be
    missing.  Cells with a non-zero `status` are unreliable as well.

The in-sphere test of step 4 is the analogue of a filtered predicate: it is evaluated in
float64 and compared against a Shewchuk-style rounding-error bound, so a determinant smaller
than the bound is *undecidable in double precision* -- an exact predicate would have to take
over there (this module has no exact fallback; such a 4-clique is kept and reported).
`last_insphere_stats()` returns the counts of the last conversion, which is how this method
enters the exact-vs-filtered comparison against gDel3D and CGAL.

Usage
    import paragram
    from voronoi_to_delaunay import delaunay_from_diagram

    diag = paragram.voronoi_diagram(points)
    tets, circumcentres = delaunay_from_diagram(points, diag)   # (T, 4) int64, (T, 3) float64

CLI
    python voronoi_to_delaunay.py --n 200000            # run paragram (CUDA) and convert
    python voronoi_to_delaunay.py --n 5000 --check      # also compare against scipy
    python voronoi_to_delaunay.py --n 5000 --scipy-only # no CUDA: test on scipy's graph
"""

from __future__ import annotations

import argparse
import time
import warnings

import torch

__all__ = [
    "adjacency_from_tets",
    "delaunay_from_adjacency",
    "delaunay_from_diagram",
    "last_insphere_stats",
]

_LAST_INSPHERE: dict[str, float | int] = {}


def last_insphere_stats() -> dict[str, float | int]:
    """Counters of the last `delaunay_from_adjacency` call:

        cliques    4-cliques of the adjacency graph that reached the in-sphere test
        tests      in-sphere determinants evaluated (one per clique per candidate 5th point)
        uncertain  of those, how many landed inside the float64 rounding-error bound, i.e.
                   could not be decided in double precision
        cospherical_tets  tetrahedra kept although a 5th point sits on their circumsphere
    """
    return dict(_LAST_INSPHERE)


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def _expand_neighbours(
    rows: torch.Tensor, deg: torch.Tensor, offs: torch.Tensor, adj: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For every vertex in `rows`, emit (row_index, neighbour) for all its neighbours."""
    d = deg[rows]
    total = int(d.sum())
    if total == 0:
        e = torch.empty(0, dtype=torch.long, device=rows.device)
        return e, e
    row_idx = torch.repeat_interleave(torch.arange(rows.numel(), device=rows.device), d)
    start = torch.cumsum(d, 0) - d  # exclusive prefix sum
    local = torch.arange(total, device=rows.device) - start[row_idx]
    nb = adj[offs[rows][row_idx] + local]
    return row_idx, nb


def _has_edge(ekeys: torch.Tensor, u: torch.Tensor, v: torch.Tensor, n: int) -> torch.Tensor:
    """Membership of undirected edge (u,v) (any order) in the sorted key array `ekeys`."""
    lo = torch.minimum(u, v)
    hi = torch.maximum(u, v)
    q = lo * n + hi
    pos = torch.searchsorted(ekeys, q).clamp_(max=ekeys.numel() - 1)
    return ekeys[pos] == q


def _weighted_chunks(weights: torch.Tensor, budget: int):
    """Yield (start, end) slices such that sum(weights[start:end]) <= budget (>= 1 item each)."""
    n = weights.numel()
    if n == 0:
        return
    cs = torch.cumsum(weights, 0)
    s = 0
    while s < n:
        base = cs[s - 1 : s] if s > 0 else torch.zeros(1, dtype=cs.dtype, device=cs.device)
        e = int(torch.searchsorted(cs, base + budget, right=True))
        e = min(max(e, s + 1), n)
        yield s, e
        s = e


def _det3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (x * torch.cross(y, z, dim=1)).sum(1)


def _circumcentre(p: torch.Tensor, tets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Circumcentre (float64) and |6 V| (orientation volume) of each tetrahedron."""
    a = p[tets[:, 0]]
    B = p[tets[:, 1]] - a
    C = p[tets[:, 2]] - a
    D = p[tets[:, 3]] - a
    cxd = torch.cross(C, D, dim=1)
    dxb = torch.cross(D, B, dim=1)
    bxc = torch.cross(B, C, dim=1)
    det = (B * cxd).sum(1)  # 6 * signed volume
    num = (
        (B * B).sum(1, keepdim=True) * cxd
        + (C * C).sum(1, keepdim=True) * dxb
        + (D * D).sum(1, keepdim=True) * bxc
    )
    o = a + num / (2.0 * det).unsqueeze(1)
    return o, det


# ----------------------------------------------------------------------------------------
# main entry points
# ----------------------------------------------------------------------------------------


@torch.no_grad()
def delaunay_from_adjacency(
    points: torch.Tensor,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    status: torch.Tensor | None = None,
    *,
    rel_tol: float = 1e-13,
    budget: int = 4_000_000,
    check_all_cells: bool = True,
    return_circumcentres: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Recover Delaunay tetrahedra from a Voronoi adjacency graph.

    Args:
        points:    (N, 3) float tensor (any float dtype; math is done in float64).
        adjacency: flattened int neighbour list (Paragram `Diagram.adjacency`).
        offsets:   (N + 1,) CSR offsets (Paragram `Diagram.offsets`).
        status:    optional (N,) per-cell status; non-zero cells trigger a warning.
        rel_tol:   tolerance of the insphere test relative to its rounding-error bound; a
                   5th point closer than that to the circumsphere counts as *on* it.  The
                   float64 error is below ~1e-14 of the bound, Paragram's own float32
                   arithmetic resolves only ~1e-7.
        budget:    max number of candidate rows materialised per chunk (memory knob; peak
                   GPU temporaries are roughly 500 bytes per row, i.e. ~2 GB at the default).
        check_all_cells: test the neighbours of all four cells (default) or only of the
                   first one.  Both are exact in exact arithmetic; checking all four is
                   ~4x more work in step 4 but tolerates an imperfect (asymmetric or
                   slightly incomplete) input adjacency.
        return_circumcentres: also return the Voronoi vertex of each tetrahedron.

    Returns:
        tets: (T, 4) int64, each row sorted ascending, rows sorted lexicographically.
        circumcentres: (T, 3) float64 or None.
    """
    dev = points.device
    n = points.shape[0]
    _LAST_INSPHERE.clear()
    if n < 4:
        return torch.empty(0, 4, dtype=torch.long, device=dev), (
            torch.empty(0, 3, dtype=torch.float64, device=dev) if return_circumcentres else None
        )
    if status is not None:
        bad = int((status != 0).sum())
        if bad:
            warnings.warn(f"{bad} cells have non-zero status; tetrahedra around them may be wrong")

    adjacency = adjacency.to(dev, torch.long)
    offsets = offsets.to(dev, torch.long)
    counts = offsets[1:] - offsets[:-1]

    # ---- 1. undirected, deduplicated, sorted edge set -----------------------------------
    src = torch.repeat_interleave(torch.arange(n, device=dev), counts)
    dst = adjacency[: int(offsets[-1])]
    ok = (dst >= 0) & (dst < n) & (dst != src)
    src, dst = src[ok], dst[ok]
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    ekeys = torch.unique(lo * n + hi)  # sorted
    eu = ekeys // n
    ev = ekeys - eu * n

    # symmetric CSR with sorted neighbour lists (needed for expansion + cell test)
    dsrc = torch.cat([eu, ev])
    ddst = torch.cat([ev, eu])
    order = torch.argsort(dsrc * n + ddst)
    dsrc, adj = dsrc[order], ddst[order]
    deg = torch.bincount(dsrc, minlength=n)
    offs = torch.zeros(n + 1, dtype=torch.long, device=dev)
    offs[1:] = torch.cumsum(deg, 0)

    # ---- 2-4. streamed: edges -> triangles -> 4-cliques -> tetrahedra ---------------------
    # Every stage is chunked by the *exact* number of rows it will materialise (sum of the
    # degrees involved), so peak memory is bounded by `budget` rows whatever the input.
    p = points.to(torch.float64)
    check_k = 4 if check_all_cells else 1
    tet_parts = []
    n_degenerate = 0
    # Predicate counters.  `n_tests` is free (it is a tensor *shape*); `n_uncertain` is
    # accumulated on the device and read once at the end, so no chunk forces a sync.
    n_cliques = n_tests = 0
    n_uncertain = torch.zeros((), dtype=torch.long, device=dev)

    def triangles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a<b<c with ab, ac, bc edges (c drawn from the neighbours of a)."""
        ridx, c = _expand_neighbours(a, deg, offs, adj)
        keep = c > b[ridx]
        ridx, c = ridx[keep], c[keep]
        keep = _has_edge(ekeys, b[ridx], c, n)
        ridx, c = ridx[keep], c[keep]
        return torch.stack([a[ridx], b[ridx], c], 1)

    def cliques(t: torch.Tensor) -> torch.Tensor:
        """a<b<c<d with ad, bd, cd edges (d drawn from the neighbours of a)."""
        ridx, d = _expand_neighbours(t[:, 0], deg, offs, adj)
        keep = d > t[ridx, 2]
        ridx, d = ridx[keep], d[keep]
        keep = _has_edge(ekeys, t[ridx, 1], d, n) & _has_edge(ekeys, t[ridx, 2], d, n)
        ridx, d = ridx[keep], d[keep]
        return torch.cat([t[ridx], d[:, None]], 1)

    def empty_sphere(t: torch.Tensor) -> torch.Tensor:
        """Keep cliques whose circumsphere contains no Delaunay neighbour of a, b, c (, d).

        Exact criterion (see module docstring); decided with the insphere determinant, which
        needs no division and is far better conditioned than testing distances against a
        computed circumcentre (slivers!)."""
        nonlocal n_degenerate, n_cliques, n_tests
        n_cliques += t.shape[0]
        pa, pb, pc, pd = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]], p[t[:, 3]]
        B, C, D = pb - pa, pc - pa, pd - pa
        orient = _det3(B, C, D)  # 6 * signed volume
        scale = B.norm(dim=1) * C.norm(dim=1) * D.norm(dim=1)
        keep = orient.abs() > 1e-12 * scale  # flat cliques are not tetrahedra
        sgn = torch.sign(orient)
        cosph = torch.zeros(t.shape[0], dtype=torch.bool, device=dev)
        for k in range(check_k):
            ridx, nb = _expand_neighbours(t[:, k], deg, offs, adj)
            own = (nb == t[ridx, 0]) | (nb == t[ridx, 1]) | (nb == t[ridx, 2]) | (nb == t[ridx, 3])
            ridx, nb = ridx[~own], nb[~own]
            if ridx.numel() == 0:
                continue
            n_tests += ridx.numel()
            pe = p[nb]
            A_, B_, C_, D_ = pa[ridx] - pe, pb[ridx] - pe, pc[ridx] - pe, pd[ridx] - pe
            del pe
            na, nb_, nc, nd = A_.norm(dim=1), B_.norm(dim=1), C_.norm(dim=1), D_.norm(dim=1)
            la, lb, lc, ld = na * na, nb_ * nb_, nc * nc, nd * nd
            t0 = la * _det3(B_, C_, D_)
            t1 = lb * _det3(A_, C_, D_)
            t2 = lc * _det3(A_, B_, D_)
            t3 = ld * _det3(A_, B_, C_)
            del A_, B_, C_, D_
            # insphere < 0  <=>  e strictly inside the circumsphere of positively oriented abcd
            insphere = -t0 + t1 - t2 + t3
            # Rounding-error bound.  The 3x3 determinants cancel catastrophically for slivers,
            # so the bound must scale with the products of the row norms (as in Shewchuk's
            # adaptive predicates), not with the magnitude of the computed terms.
            bound = la * nb_ * nc * nd + lb * na * nc * nd + lc * na * nb_ * nd + ld * na * nb_ * nc
            inside = sgn[ridx] * insphere < -rel_tol * bound
            killed = torch.zeros(t.shape[0], dtype=torch.bool, device=dev)
            killed[ridx[inside]] = True
            keep &= ~killed
            # Determinants inside the rounding-error bound: undecidable in float64.  An exact
            # predicate would decide them; here they are counted and the clique is kept.
            undecided = insphere.abs() <= rel_tol * bound
            n_uncertain.add_(undecided.sum())
            if k == 0:
                # a 5th point *on* the circumsphere: co-spherical group, triangulation not unique
                cosph[ridx[undecided]] = True
        n_degenerate += int((keep & cosph).sum())
        return t[keep]

    for s, e in _weighted_chunks(deg[eu], budget):
        tris = triangles(eu[s:e], ev[s:e])
        for s2, e2 in _weighted_chunks(deg[tris[:, 0]], budget):
            k4 = cliques(tris[s2:e2])
            if k4.shape[0] == 0:
                continue
            for s3, e3 in _weighted_chunks(deg[k4[:, :check_k]].sum(1), budget):
                tet_parts.append(empty_sphere(k4[s3:e3]))
            del k4
        del tris

    _LAST_INSPHERE.clear()
    _LAST_INSPHERE.update(
        cliques=n_cliques,
        tests=n_tests,
        uncertain=int(n_uncertain),
        cospherical_tets=n_degenerate,
        rel_tol=rel_tol,
    )

    if n_degenerate:
        warnings.warn(
            f"{n_degenerate} tetrahedra belong to co-spherical point groups (5+ points on one sphere): "
            "the Delaunay triangulation is not unique there and overlapping tetrahedra are returned. "
            "Add a tiny random jitter to the input points to get a proper triangulation.",
            stacklevel=2,
        )

    tets = torch.cat(tet_parts) if tet_parts else torch.empty(0, 4, dtype=torch.long, device=dev)
    cc = _circumcentre(p, tets)[0] if return_circumcentres else None
    return tets, cc


def delaunay_from_diagram(points: torch.Tensor, diagram, **kwargs):
    """Convenience wrapper taking a `paragram.Diagram`."""
    return delaunay_from_adjacency(
        points, diagram.adjacency, diagram.offsets, getattr(diagram, "status", None), **kwargs
    )


def adjacency_from_tets(tets: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Paragram-style CSR adjacency (int32) from a (T,4) tetrahedron list.

    Used for testing without CUDA: turns scipy's Delaunay into a Voronoi adjacency graph.
    """
    dev = tets.device
    tets = tets.long()
    pairs = torch.combinations(torch.arange(4, device=dev), 2)  # 6 edges per tet
    u = tets[:, pairs[:, 0]].reshape(-1)
    v = tets[:, pairs[:, 1]].reshape(-1)
    src = torch.cat([u, v])
    dst = torch.cat([v, u])
    keys = torch.unique(src * n + dst)
    src = keys // n
    dst = keys - src * n
    deg = torch.bincount(src, minlength=n)
    offsets = torch.zeros(n + 1, dtype=torch.long, device=dev)
    offsets[1:] = torch.cumsum(deg, 0)
    return dst.to(torch.int32), offsets.to(torch.int32)


# ----------------------------------------------------------------------------------------
# CLI / self-test
# ----------------------------------------------------------------------------------------


def _canon(t: torch.Tensor) -> torch.Tensor:
    t, _ = torch.sort(t, dim=1)
    n = int(t.max()) + 1 if t.numel() else 1
    key = ((t[:, 0] * n + t[:, 1]) * n + t[:, 2]) * n + t[:, 3]
    return torch.unique(key)


def _compare_with_scipy(points: torch.Tensor, tets: torch.Tensor) -> None:
    import numpy as np
    from scipy.spatial import Delaunay

    ref = torch.from_numpy(Delaunay(points.double().cpu().numpy()).simplices.astype(np.int64))
    a = _canon(tets.cpu())
    b = _canon(ref)
    missing = int((~torch.isin(b, a)).sum())
    extra = int((~torch.isin(a, b)).sum())
    print(f"scipy tets: {b.numel()}   ours: {a.numel()}   missing: {missing}   extra: {extra}")
    if missing == 0 and extra == 0:
        print("MATCH: identical triangulation")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=100_000, help="number of random points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="compare against scipy.spatial.Delaunay")
    ap.add_argument(
        "--scipy-only", action="store_true", help="build the graph with scipy (no CUDA)"
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="save tets (and circumcentres) to this .pt file")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    use_paragram = not args.scipy_only and torch.cuda.is_available()
    device = torch.device(args.device or ("cuda" if use_paragram else "cpu"))
    points = torch.rand(args.n, 3, device=device, dtype=torch.float32)

    if use_paragram:
        import paragram

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        diag = paragram.voronoi_diagram(points)
        torch.cuda.synchronize()
        print(f"paragram.voronoi_diagram: {time.perf_counter() - t0:.3f}s")
        adjacency, offsets, status = diag.adjacency, diag.offsets, diag.status
    else:
        from scipy.spatial import Delaunay

        print("CUDA not available or --scipy-only: building the adjacency graph from scipy")
        ref = torch.from_numpy(Delaunay(points.double().cpu().numpy()).simplices.astype("int64"))
        adjacency, offsets = adjacency_from_tets(ref.to(device), args.n)
        status = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    tets, cc = delaunay_from_adjacency(points, adjacency, offsets, status)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"voronoi -> delaunay: {time.perf_counter() - t0:.3f}s   tets: {tets.shape[0]}")

    if args.check or not use_paragram:
        _compare_with_scipy(points, tets)
    if args.out:
        torch.save(
            {"points": points.cpu(), "tets": tets.cpu(), "circumcentres": cc.cpu()}, args.out
        )
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
