# ruff: noqa: B023  (closures below are invoked immediately inside the same loop iteration)
"""Compare `voronoi_to_delaunay` against CGAL on point sets sampled from manifold surfaces.

For every dataset the script
  1. builds the Voronoi adjacency (Paragram on CUDA if available; without CUDA the edges of
     Qhull's Delaunay triangulation, which in the generic case are exactly the face
     adjacencies Paragram returns; `--adjacency ref-edges` uses CGAL's edges instead),
  2. converts it to tetrahedra with `delaunay_from_adjacency`,
  3. computes the reference 3D Delaunay triangulation with CGAL (exact predicates), preferably
     the *parallel* CGAL (Parallel_tag + TBB) through the compiled `cgal_delaunay` tool
     (--cgal-bin / CGAL_DELAUNAY_BIN, source in cgal_delaunay.cpp).  Otherwise the sequential
     `cgal` Python bindings are used, in this interpreter or, via CGAL_PYTHON=/path/to/python,
     in another one (wheels exist for Python 3.8-3.12 only); last resort is scipy/Qhull,
  4. optionally runs gDel3D (GPU Delaunay by flipping + star splaying, double precision with
     exact predicates and symbolic perturbation) through the pyGDel3D bindings
     (https://github.com/half-potato/pyGDel3D) as a second GPU method,
  5. reports correctness metrics (set difference of tetrahedra vs the reference, empty-
     circumsphere violations, total volume vs convex-hull volume, face manifoldness, Euler
     characteristic) and quality metrics (volume statistics, radius ratio, dihedral angles,
     sliver count) for every method.

Datasets: cube corners, hollow cube (2000 surface samples), sphere (Fibonacci & random), torus
(grid & random), Klein bottle, Möbius strip, trefoil tube, famous meshes (Stanford bunny, Spot,
teapot, cow, Suzanne, Armadillo ...) downloaded from alecjacobson/common-3d-test-models and
cached, plus any PLY files given with --ply (vertices only).

Several of these are *degenerate* on purpose: the 8 cube corners and points on an exact
sphere are all co-spherical, a torus grid has many co-circular points.  There the
Delaunay triangulation is not unique and is NOT determined by the Voronoi adjacency
(e.g. the cube's Voronoi cells are the 8 octants, sharing faces only along cube edges,
so no diagonal, hence no tetrahedron, is encoded).  CGAL resolves such cases by symbolic
perturbation.  Expect mismatches there; `--jitter` shows that any generic perturbation
removes them.

Usage
    python test_delaunay_surfaces.py                       # all built-in datasets
    python test_delaunay_surfaces.py --models stanford-bunny spot --no-analytic
    python test_delaunay_surfaces.py --jitter 1e-6 --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from voronoi_to_delaunay import delaunay_from_adjacency

VERBOSE = False


def log(msg: str) -> None:
    """Timestamped progress line on stderr (enabled with --verbose)."""
    if VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def summarize_runs(runs: list[dict]) -> dict:
    """Aggregate per-run timing dicts (keys: total, gpu, cpu, + phases) into mean/std/min."""
    import statistics

    out = {"runs": len(runs), "totals": [r["total"] for r in runs]}
    tot = out["totals"]
    out["mean"] = statistics.fmean(tot)
    out["std"] = statistics.pstdev(tot) if len(tot) > 1 else 0.0
    out["min"] = min(tot)
    for key in sorted({k for r in runs for k in r} - {"total"}):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = statistics.fmean(vals)
    return out


TIMING_LABELS = {
    "paragram": "paragram",
    "gdel3d": "gdel3d",
    "dewall": "dewall",
    "cgal_parallel": "cgal parallel",
    "cgal_sequential": "cgal sequential",
}


def format_timing_table(timing: dict, indent: str = "   ") -> str:
    """Per-method wall time over the repeated runs plus the CPU / GPU split and phase breakdown."""
    lines = [
        f"{indent}timing over repeated runs (seconds; mean = GPU + CPU, excl.I/O is measured but NOT in the mean):",
        (
            f"{indent}  {'method':16s} {'runs':>4s} {'mean':>9s} {'std':>8s} {'min':>9s} {'GPU':>9s} "
            f"{'CPU':>9s} {'excl.I/O':>9s}  breakdown"
        ),
    ]
    for label in ("paragram", "gdel3d", "dewall", "cgal_parallel", "cgal_sequential"):
        t = timing.get(label)
        if not t:
            continue
        gpu = t.get("gpu_mean")
        cpu = t.get("cpu_mean")
        io = t.get("io_mean")
        lines.append(
            f"{indent}  {TIMING_LABELS[label]:16s} {t['runs']:4d} {t['mean']:9.4f} {t['std']:8.4f} {t['min']:9.4f} "
            f"{(f'{gpu:9.4f}' if gpu is not None else '        -')} "
            f"{(f'{cpu:9.4f}' if cpu is not None else '        -')} "
            f"{(f'{io:9.4f}' if io is not None else '        -')}  {t.get('breakdown', '')}"
        )
    return "\n".join(lines)


def unit_cube(points: np.ndarray) -> np.ndarray:
    """Local-DeWall's normalisation: (p - min) / (largest extent * 1.001), i.e. into [0, 1/1.001]^3.
    Applied to every dataset up front (--unit-cube) so that all methods see the same float32 points."""
    lo = points.min(0)
    maxside = float(np.ptp(points, axis=0).max()) * 1.001
    return (points - lo) / maxside


MODEL_URL = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/{}.obj"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "paragram_test_models")
DEFAULT_MODELS = ["stanford-bunny", "spot", "teapot", "cow", "suzanne", "armadillo"]


# ----------------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------------


def cube8() -> np.ndarray:
    g = np.array([-1.0, 1.0])
    return np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T


def hollow_cube(n: int, rng) -> np.ndarray:
    """n points uniformly distributed on the surface of the cube [-1, 1]^3 (6 faces)."""
    face = rng.integers(0, 6, n)
    uv = rng.uniform(-1, 1, (n, 2))
    pts = np.empty((n, 3))
    axis = face // 2
    sign = np.where(face % 2 == 0, -1.0, 1.0)
    for a in range(3):
        m = axis == a
        others = [k for k in range(3) if k != a]
        pts[m, a] = sign[m]
        pts[m, others[0]] = uv[m, 0]
        pts[m, others[1]] = uv[m, 1]
    return pts


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1
    )


def random_sphere(n: int, rng) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def torus(u, v, R=1.0, r=0.4) -> np.ndarray:
    return np.stack(
        [
            (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v),
        ],
        1,
    )


def torus_grid(nu: int, nv: int) -> np.ndarray:
    u, v = np.meshgrid(
        np.linspace(0, 2 * np.pi, nu, endpoint=False),
        np.linspace(0, 2 * np.pi, nv, endpoint=False),
    )
    return torus(u.ravel(), v.ravel())


def torus_random(n: int, rng) -> np.ndarray:
    return torus(rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n))


def klein_bottle(n: int, rng, a=2.0) -> np.ndarray:  # figure-8 immersion
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    w = a + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)
    return np.stack(
        [
            w * np.cos(u),
            w * np.sin(u),
            np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v),
        ],
        1,
    )


def mobius(n: int, rng) -> np.ndarray:
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(-1, 1, n)
    w = 1 + 0.5 * v * np.cos(u / 2)
    return np.stack([w * np.cos(u), w * np.sin(u), 0.5 * v * np.sin(u / 2)], 1)


def trefoil_tube(n: int, rng, r=0.35) -> np.ndarray:
    t = rng.uniform(0, 2 * np.pi, n)
    c = np.stack(
        [np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)],
        1,
    )
    tan = np.stack(
        [
            np.cos(t) + 4 * np.cos(2 * t),
            -np.sin(t) + 4 * np.sin(2 * t),
            -3 * np.cos(3 * t),
        ],
        1,
    )
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    ref = np.where(np.abs(tan[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    n1 = np.cross(tan, ref)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 = np.cross(tan, n1)
    phi = rng.uniform(0, 2 * np.pi, n)
    return c + r * (np.cos(phi)[:, None] * n1 + np.sin(phi)[:, None] * n2)


def _download(url: str, path: str) -> None:
    """urllib first; some Python builds (python.org macOS) lack root certificates, so fall back
    to a certifi SSL context and finally to curl."""
    tmp = path + ".part"
    errors = []
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:  # noqa: BLE001 - any failure just moves on to the next strategy
        errors.append(str(exc))
        try:
            import ssl

            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, context=ctx) as r, open(tmp, "wb") as f:
                f.write(r.read())
        except Exception as exc2:  # noqa: BLE001
            errors.append(str(exc2))
            if subprocess.call(["curl", "-sSLf", "-o", tmp, url]) != 0:
                raise OSError("download failed: " + " | ".join(errors)) from None
    os.replace(tmp, path)


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def load_ply_vertices(path: str) -> np.ndarray:
    """Vertex positions of a PLY file (ascii, binary_little_endian or binary_big_endian).
    Only the x/y/z properties of the `vertex` element are used; other elements are skipped."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header.append(line.decode("ascii", "replace").strip())
            if header[-1] == "end_header":
                break
        fmt = next(h.split()[1] for h in header if h.startswith("format"))
        elements: list[tuple[str, int, list[tuple[str, str]]]] = []
        for h in header:
            tok = h.split()
            if not tok:
                continue
            if tok[0] == "element":
                elements.append((tok[1], int(tok[2]), []))
            elif tok[0] == "property" and elements:
                if tok[1] == "list":
                    elements[-1][2].append(("list", f"{tok[2]} {tok[3]} {tok[4]}"))
                else:
                    elements[-1][2].append((tok[2], tok[1]))
        verts = None
        if fmt == "ascii":
            for name, count, props in elements:
                rows = [f.readline().split() for _ in range(count)]
                if name == "vertex":
                    names = [p[0] for p in props]
                    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
                    verts = np.array(
                        [[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows]
                    )
            if verts is None:
                raise ValueError(f"{path}: no vertex element")
            return verts
        endian = "<" if fmt == "binary_little_endian" else ">"
        for name, count, props in elements:
            if any(p[0] == "list" for p in props):
                if name == "vertex":
                    raise ValueError(
                        f"{path}: list properties on vertices are not supported"
                    )
                for _ in range(count):  # variable-length records (faces): walk them
                    for p in props:
                        if p[0] == "list":
                            ct, it, _n = p[1].split()
                            cdt = np.dtype(endian + PLY_TYPES[ct])
                            k = int(np.frombuffer(f.read(cdt.itemsize), cdt)[0])
                            f.read(k * np.dtype(endian + PLY_TYPES[it]).itemsize)
                        else:
                            f.read(np.dtype(endian + PLY_TYPES[p[1]]).itemsize)
                continue
            dt = np.dtype([(p[0], endian + PLY_TYPES[p[1]]) for p in props])
            data = np.frombuffer(f.read(count * dt.itemsize), dt)
            if name == "vertex":
                verts = np.stack([data["x"], data["y"], data["z"]], 1).astype(
                    np.float64
                )
        if verts is None:
            raise ValueError(f"{path}: no vertex element")
        return verts


def load_obj_vertices(name: str) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.obj")
    if not os.path.exists(path):
        print(f"  downloading {name}.obj ...", flush=True)
        _download(MODEL_URL.format(name), path)
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64)


def build_datasets(args) -> list[tuple[str, np.ndarray, str]]:
    """Returns (name, points float64, note).  User-supplied PLY files come first."""
    rng = np.random.default_rng(args.seed)
    ds = []
    for path in getattr(args, "ply", None) or []:
        try:
            pts = load_ply_vertices(path)
            ds.append(
                (
                    os.path.splitext(os.path.basename(path))[0],
                    pts,
                    f"PLY vertices from {os.path.basename(path)}",
                )
            )
        except (OSError, ValueError) as exc:
            print(f"  skipping {path}: {exc}")
    if not args.no_analytic:
        ds += [
            ("cube8", cube8(), "8 co-spherical points: Delaunay not unique"),
            ("cube8+jitter", cube8() + rng.normal(scale=1e-3, size=(8, 3)), "generic"),
            (
                "hollow-cube",
                hollow_cube(2000, rng),
                "2000 points on the faces of [-1,1]^3 (co-planar faces)",
            ),
            (
                "sphere-fibonacci",
                fibonacci_sphere(args.n),
                "co-spherical up to float32 rounding",
            ),
            (
                "sphere-random",
                random_sphere(args.n, rng),
                "co-spherical up to float32 rounding",
            ),
            (
                "torus-grid",
                torus_grid(120, 60),
                "regular grid: many co-circular points",
            ),
            ("torus-random", torus_random(args.n, rng), "generic"),
            ("klein-bottle", klein_bottle(args.n, rng), "generic"),
            ("mobius", mobius(args.n, rng), "generic"),
            ("trefoil-tube", trefoil_tube(args.n, rng), "generic"),
        ]
    for m in args.models:
        try:
            ds.append(
                (
                    m,
                    load_obj_vertices(m),
                    "mesh vertices (symmetric/regular meshes contain co-spherical groups)",
                )
            )
        except (
            OSError,
            ValueError,
        ) as exc:  # network / parse failure should not kill the run
            print(f"  skipping {m}: {exc}")
    return ds


# ----------------------------------------------------------------------------------------
# adjacency + reference triangulation
# ----------------------------------------------------------------------------------------


def _csr_from_edges(src: np.ndarray, dst: np.ndarray, n: int, dev):
    src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    deg = np.bincount(src, minlength=n)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(deg)
    return torch.from_numpy(dst).to(dev, torch.int32), torch.from_numpy(offsets).to(
        dev, torch.int32
    )


def _edges_of_tets(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.sort(tets.astype(np.int64), axis=1)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    n = int(t.max()) + 1
    keys = np.unique(np.concatenate([t[:, i] * n + t[:, j] for i, j in pairs]))
    return keys // n, keys % n


STATUS_NAMES = [
    "success",
    "plane_overflow",
    "vertex_overflow",
    "empty_cell",
    "security_radius_not_reached",
    "inconsistent_boundary",
]


def status_histogram(status: torch.Tensor | None) -> dict[str, int] | None:
    """Per-cell Paragram status counts (see Status enum in voronoi_convex_cell.cuh)."""
    if status is None:
        return None
    h = torch.bincount(
        status.to("cpu", torch.long).clamp(min=0), minlength=len(STATUS_NAMES)
    )
    out = {name: int(h[i]) for i, name in enumerate(STATUS_NAMES)}
    extra = int(h[len(STATUS_NAMES) :].sum())
    if extra:
        out["other"] = extra
    return out


def voronoi_adjacency(
    points32: torch.Tensor,
    backend: str,
    ref_tets: np.ndarray | None,
    bbox_pad: float | None = None,
):
    """(adjacency, offsets, status, backend_name).

    paragram  : the real thing (needs CUDA).
    qhull     : edges of scipy.spatial.Delaunay.  In the generic case this *is* the Voronoi
                face adjacency.  (scipy.spatial.Voronoi.ridge_points was found to silently
                drop edges on sliver-heavy inputs, so it is not used.)
    ref-edges : edges of the reference (CGAL) triangulation -- an exact-arithmetic stand-in
                for Paragram.  In degenerate cases it contains edges (diagonals of the
                co-spherical polytope) that are NOT Voronoi face adjacencies.
    """
    n = points32.shape[0]
    dev = points32.device
    if backend == "paragram":
        import paragram

        kwargs = {}
        if bbox_pad is not None and bbox_pad >= 0:
            kwargs["bbox_pad"] = float(bbox_pad)
        try:
            d = paragram.voronoi_diagram(points32, **kwargs)
        except TypeError as exc:  # unpatched Paragram without the bbox_pad argument
            if kwargs and "bbox_pad" in str(exc):
                d = paragram.voronoi_diagram(points32)
                return (
                    d.adjacency,
                    d.offsets,
                    d.status,
                    "paragram (legacy box: no bbox_pad support)",
                )
            raise
        label = "paragram" + (
            f" (bbox_pad={bbox_pad:g}x extent)"
            if kwargs
            else " (legacy box: +1.0 absolute)"
        )
        return d.adjacency, d.offsets, d.status, label
    if backend == "ref-edges":
        u, v = _edges_of_tets(ref_tets)
        return (*_csr_from_edges(u, v, n, dev), None, "ref-edges")
    u, v = _edges_of_tets(Delaunay(points32.double().cpu().numpy()).simplices)
    return (*_csr_from_edges(u, v, n, dev), None, "qhull-edges")


_CGAL_WORKER = r"""
import sys
import numpy as np
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3

points = np.load(sys.argv[1])
index = {tuple(p): i for i, p in enumerate(points.tolist())}
dt = Delaunay_triangulation_3()
dt.insert([Point_3(*p) for p in points.tolist()])
tets = np.empty((dt.number_of_finite_cells(), 4), dtype=np.int64)
for k, c in enumerate(dt.finite_cells()):
    for i in range(4):
        p = c.vertex(i).point()
        tets[k, i] = index[(p.x(), p.y(), p.z())]
np.save(sys.argv[2], tets)
"""


def _cgal_in_process(points: np.ndarray) -> np.ndarray:
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
    return tets


def _cgal_subprocess(points: np.ndarray, python: str) -> np.ndarray:
    """Run the CGAL reference in another interpreter (env CGAL_PYTHON), e.g. a Python 3.12 venv
    when the main interpreter is a version for which no `cgal` wheel exists."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        pin, pout = os.path.join(d, "points.npy"), os.path.join(d, "tets.npy")
        np.save(pin, points)
        subprocess.run([python, "-c", _CGAL_WORKER, pin, pout], check=True)
        return np.load(pout)


def _cgal_binary(points: np.ndarray, binary: str) -> tuple[np.ndarray, str, dict]:
    """Reference via the compiled `cgal_delaunay` tool (CGAL Parallel_tag + TBB, see
    cgal_delaunay.cpp).  Points/tets are exchanged as raw binary files; the tool reports its own
    insertion and extraction times, which exclude process start-up and file I/O."""
    import tempfile

    binary = os.path.abspath(binary)
    with tempfile.TemporaryDirectory() as d:
        pin, pout = os.path.join(d, "points.f64"), os.path.join(d, "tets.i32")
        np.ascontiguousarray(points, dtype="<f8").tofile(pin)
        t0 = time.perf_counter()
        r = subprocess.run(
            [binary, pin, pout], check=True, capture_output=True, text=True
        )
        wall = time.perf_counter() - t0
        tets = np.fromfile(pout, dtype="<i4").reshape(-1, 4).astype(np.int64)
    info = {"wall_seconds": wall}
    for tok in r.stdout.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                info[k] = float(v) if "." in v else int(v)
            except ValueError:
                info[k] = v
    info["io_seconds"] = max(
        0.0, wall - (info.get("build_seconds", 0.0) + info.get("extract_seconds", 0.0))
    )
    label = (
        f"CGAL {info.get('cgal', '')} "
        + ("parallel" if info.get("parallel") else "sequential")
        + (f" ({info.get('threads')} threads)" if info.get("parallel") else "")
        + " [cgal_delaunay]"
    )
    return tets, label, info


def reference_delaunay(points: np.ndarray) -> tuple[np.ndarray, str, dict]:
    """Returns (tets, label, info).  info["seconds"] is the time to obtain the tetrahedra
    (build + extraction) measured as close to the triangulation as possible.

    Order of preference:
      1. CGAL_DELAUNAY_BIN / --cgal-bin: compiled `cgal_delaunay` tool, CGAL Parallel_tag + TBB.
      2. `cgal` Python bindings in this interpreter (sequential Delaunay_triangulation_3).
      3. CGAL_PYTHON / --cgal-python: the bindings in another interpreter (subprocess).
      4. scipy/Qhull (not exact; used only if nothing else is available).
    """
    binary = os.environ.get("CGAL_DELAUNAY_BIN")
    if binary:
        t0 = time.perf_counter()
        tets, label, info = _cgal_binary(points, binary)
        info["seconds"] = info.get("build_seconds", 0.0) + info.get(
            "extract_seconds", 0.0
        ) or (time.perf_counter() - t0)
        if (
            info.get("parallel")
            and info.get("threads", 1) > 1
            and "CGAL_THREADS" not in os.environ
        ):
            # CGAL's parallel insertion can be *slower* than one thread on surface point clouds
            # (lock-grid contention), so also time a single-threaded build for reference.
            os.environ["CGAL_THREADS"] = "1"
            try:
                _, _, seq = _cgal_binary(points, binary)
            finally:
                del os.environ["CGAL_THREADS"]
            info["sequential_seconds"] = seq.get("build_seconds", 0.0) + seq.get(
                "extract_seconds", 0.0
            )
        return tets, label, info
    t0 = time.perf_counter()
    try:
        import CGAL.CGAL_Triangulation_3  # noqa: F401

        tets = _cgal_in_process(points)
        return (
            tets,
            "CGAL sequential [python bindings]",
            {"seconds": time.perf_counter() - t0},
        )
    except ImportError:
        pass
    cgal_python = os.environ.get("CGAL_PYTHON")
    if cgal_python:
        tets = _cgal_subprocess(points, cgal_python)
        return (
            tets,
            f"CGAL sequential [bindings via {cgal_python}]",
            {"seconds": time.perf_counter() - t0},
        )
    tets = Delaunay(points).simplices.astype(np.int64)
    return (
        tets,
        "scipy.Delaunay (CGAL not installed)",
        {"seconds": time.perf_counter() - t0},
    )


def gdel3d_available() -> bool:
    try:
        import gdel3d  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


class _silence_fds:
    """Redirect the C-level stdout/stderr to /dev/null (gDel3D prints progress from C++)."""

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved = [os.dup(1), os.dup(2)]
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        for fd in (*self._saved, self._null):
            os.close(fd)
        return False


def run_gdel3d(points: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Delaunay tetrahedra with gDel3D (pyGDel3D bindings).

    `Del(N).compute(points)` takes an (N, 3) float64 host array (the transfer to the GPU is
    part of the timing) and returns *all* rows of gDel3D's tetrahedron array: tetrahedra
    incident to the point at infinity (vertex index N) and, after the CPU star-splaying repair,
    also tetrahedra marked dead.  Infinite ones are dropped here; dead ones are dropped with the
    status bytes exposed by `DelOutput.get_tet_info()` (added by patch_pygdel3d.py).  Without
    that patch dead tetrahedra cannot be told apart and `info["dead_filter"]` says so.
    """
    import gc

    from gdel3d import Del

    n = len(points)
    pts = np.ascontiguousarray(points, dtype=np.float64)
    # gDel3D allocates with plain cudaMalloc and its CUDA wrapper exit()s the whole process on any
    # error (no Python traceback).  Release PyTorch's cached GPU memory first, and log the free
    # VRAM so that a silent death of the run is attributable.
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info()
    print(
        f"   gdel3d: computing on {n} points ({free_b / 2**30:.2f} of {total_b / 2**30:.2f} GB VRAM free)",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.perf_counter()
    with _silence_fds():
        tri = Del(n)
        raw, out = tri.compute(pts)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    raw = np.asarray(raw, dtype=np.int64).reshape(-1, 4)
    keep = ((raw >= 0) & (raw < n)).all(1)
    info = {"raw_tets": len(raw), "infinite_tets": int((~keep).sum())}
    if hasattr(out, "get_tet_info"):
        alive = (np.asarray(out.get_tet_info(), dtype=np.int64).reshape(-1) & 1) == 1
        if len(alive) == len(raw):
            info["dead_tets"] = int((~alive & keep).sum())
            keep &= alive
            info["dead_filter"] = "tet_info"
        else:
            info["dead_filter"] = (
                f"tet_info length mismatch ({len(alive)} vs {len(raw)})"
            )
    else:
        info["dead_filter"] = (
            "unavailable (pyGDel3D not patched; dead tets are included)"
        )
    tets = np.unique(np.sort(raw[keep], axis=1), axis=0)
    info["duplicate_tets"] = int(keep.sum() - len(tets))
    if hasattr(
        out, "get_stats"
    ):  # phase timers (ms): GPU init/split/flip/relocate/sort, CPU splaying
        st = {k: float(v) for k, v in dict(out.get_stats()).items()}
        info["stats_ms"] = st
        info["self_reported_total_seconds"] = st.get("totalTime", float("nan")) / 1000.0
        # This build only fills totalTime; the per-phase fields can hold uninitialised values
        # (e.g. initTime = 9.4e4 ms for a 0.4 s run), so accept a phase only if it is consistent
        # with the measured wall time.
        cap = 2.0 * seconds + 0.05
        ok = {
            k: v / 1000.0
            for k, v in st.items()
            if k.endswith("Time") and 0.0 <= v / 1000.0 <= cap
        }
        cpu = ok.get("splayingTime", 0.0) + ok.get("outTime", 0.0)
        gpu_phases = [
            ok[k]
            for k in ("initTime", "splitTime", "flipTime", "relocateTime", "sortTime")
            if k in ok
        ]
        info["phases_seconds"] = {
            k.replace("Time", ""): v for k, v in ok.items() if k != "totalTime"
        }
        if len(gpu_phases) >= 4:
            info["gpu_seconds"] = sum(gpu_phases)
            info["cpu_seconds"] = cpu
        else:
            info["gpu_seconds"] = max(0.0, seconds - cpu)
            info["cpu_seconds"] = cpu
            info["stats_note"] = (
                "per-phase timers not filled by this gDel3D build; GPU = wall - CPU phases"
            )
    try:  # gDel3D's own checker (Euler, adjacency, orientation, empty sphere); it skips dead tets
        with _silence_fds():
            info["self_check"] = bool(out.check_correctness(pts))
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        info["self_check"] = f"failed: {exc}"
    del tri, out  # free the GPU buffers before the next method runs
    return tets, seconds, info


def _read_mat(path: str, kind: str) -> np.ndarray:
    """Local-DeWall matrix file: two uint64 (rows, cols) followed by row-major data."""
    with open(path, "rb") as f:
        rows, cols = np.frombuffer(f.read(16), dtype="<u8")
        payload = f.read()
    rows, cols = int(rows), int(cols)
    if rows * cols == 0:
        return np.zeros((rows, cols), dtype=np.float64 if kind == "f" else np.int64)
    itemsize = len(payload) // (rows * cols)
    dt = {("f", 4): "<f4", ("f", 8): "<f8", ("i", 4): "<i4", ("i", 8): "<i8"}[
        (kind, itemsize)
    ]
    return np.frombuffer(payload, dtype=dt, count=rows * cols).reshape(rows, cols)


def run_dewall(
    points: np.ndarray, binary: str, prenormalized: bool = False
) -> tuple[np.ndarray, float, dict]:
    """Local DeWall (Gao & Chen, CAD 2026) through its command-line tool, built from
    https://github.com/WuhengGao/Local-DeWall with patch_dewall.py.

    The tool reads a text file (N, then x y z per line, parsed as float32), normalises the
    points to (0,1)^3, and writes the *sorted* normalised points and int32 tets indexing them
    (indices >= N are the points at infinity).  The normalisation is replicated here in float32
    and the sorted points are matched back to the input indices with a KD-tree."""
    import re
    import tempfile

    n = len(points)
    p32 = np.ascontiguousarray(points, dtype=np.float32)
    binary = os.path.abspath(
        binary
    )  # the tool runs with cwd set to the temporary directory
    with tempfile.TemporaryDirectory() as d:
        pin, prefix = os.path.join(d, "points.txt"), os.path.join(d, "out_")
        with open(pin, "w") as f:
            f.write(f"{n}\n")
            np.savetxt(
                f, p32, fmt="%.9g"
            )  # 9 significant digits round-trip float32 exactly
        cmd = [binary, prefix, pin] + (["--no-normalize"] if prenormalized else [])
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=d, check=False)
        wall = time.perf_counter() - t0
        if r.returncode != 0 or not os.path.exists(prefix + "t.bin"):
            raise RuntimeError(
                f"Local DeWall exit code {r.returncode}: {(r.stdout + r.stderr)[-1500:]}"
            )
        xs = _read_mat(prefix + "x.bin", "f")
        ts = _read_mat(prefix + "t.bin", "i").astype(np.int64)
    out = r.stdout
    info = {"wall_seconds": wall, "raw_tets": len(ts)}
    m = re.search(r"total time:\s*([0-9.]+)\s*msec", out)
    if m:
        info["gpu_seconds"] = float(m.group(1)) / 1000.0
    phases = {}
    for line in out.splitlines():
        mm = re.match(r"\s*(.+?)\s*(?:time)?\s*[:=]\s*([0-9.]+)\s*msec", line)
        if mm and "total time" not in line:
            phases[mm.group(1).strip()] = float(mm.group(2)) / 1000.0
    info["phases_seconds"] = (
        phases  # every phase runs on the GPU (grid build, body, post-processing)
    )
    # The tool's compute time is what it reports itself; writing/parsing the 100k-line text file and
    # spawning the process are an artefact of its command-line interface, so they are accounted
    # separately as io_seconds instead of being called "CPU work" inside the total.
    info["io_seconds"] = max(0.0, wall - info.get("gpu_seconds", wall))
    info["cpu_seconds"] = 0.0
    status = {}
    for line in out.splitlines():
        mm = re.match(
            r"\s*(success|cycles_overflow|vertex_overflow|need_exact_predict|init_fail)\s+(\d+)\s*$",
            line,
        )
        if mm and mm.group(1) not in status:
            status[mm.group(1)] = int(mm.group(2))
    info["status"] = status
    # replicate Sampler::load_file normalisation (REAL = float): (p - min) / (maxside * 1.001)
    lo, hi = p32.min(0), p32.max(0)
    if prenormalized:  # the tool used the coordinates as given
        ours_norm = p32.astype(np.float64)
        lo = np.zeros(3, np.float32)
        maxside = np.float32(1.0)
    else:
        maxside = np.float32(np.float64((hi - lo).max()) * 1.001)
        ours_norm = ((p32 - lo) / maxside).astype(np.float64)
    finite = (ts >= 0).all(1) & (ts < len(xs)).all(1)
    info["infinite_tets"] = int((~finite).sum())
    dist, orig_of_sorted = cKDTree(ours_norm).query(xs.astype(np.float64))
    info["max_match_dist"] = float(dist.max()) if len(dist) else 0.0
    info["unmatched_points"] = int(len(xs) - len(np.unique(orig_of_sorted)))
    tets = orig_of_sorted[ts[finite]]
    tets = np.unique(np.sort(tets, axis=1), axis=0)
    info["duplicate_tets"] = int(finite.sum() - len(tets))
    # The tool triangulated the normalised float32 copy of the points, a slightly different point
    # set (moved by up to one float32 ulp).  Hand that set back, mapped into the original frame
    # with the exact inverse affine map in float64 (which preserves Delaunay-ness), so that the
    # caller compares against a reference on the same set and reports volumes in the same units.
    if not prenormalized:
        info["_points"] = ours_norm * np.float64(maxside) + lo.astype(np.float64)
        info["reference_on"] = (
            "the tool's float32-renormalised point set (mapped back to the original frame)"
        )
    return tets, info.get("gpu_seconds", wall), info


def _fmt_method_info(label: str, info: dict) -> str:
    if label == "gdel3d":
        return (
            f"gdel3d: {info['raw_tets']} raw tets, {info['infinite_tets']} infinite, "
            f"{info.get('dead_tets', '?')} dead, {info['duplicate_tets']} duplicate, "
            f"dead filter={info['dead_filter']}, self-check={info['self_check']}"
        )
    if label == "dewall":
        st = ", ".join(f"{k}={v}" for k, v in info.get("status", {}).items())
        return (
            f"dewall: {info['raw_tets']} raw tets, {info['infinite_tets']} infinite, {info['duplicate_tets']} duplicate, "
            f"gpu {info.get('gpu_seconds', float('nan')):.3f}s (wall {info['wall_seconds']:.2f}s incl. file I/O), "
            f"index match max dist {info['max_match_dist']:.1e}, unmatched {info['unmatched_points']}"
            + (f", status: {st}" if st else "")
        )
    return f"{label}: {info}"


# ----------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------


@dataclass
class Metrics:
    tets: int
    vertices_used: int
    edges: int
    faces: int
    euler: int  # V - E + F - T, 1 for a triangulated ball
    boundary_faces: int
    hull_triangles: int
    nonmanifold_faces: int  # faces shared by > 2 tets (overlapping tets)
    degenerate_tets: int
    delaunay_violations: int  # tets whose circumsphere strictly contains another point
    total_volume: float
    hull_volume: float
    volume_rel_err: float
    vol_min: float
    vol_max: float
    vol_mean: float
    vol_median: float
    radius_ratio_min: float
    radius_ratio_mean: float
    radius_ratio_p01: float
    slivers: int  # radius ratio < 0.05
    dihedral_min_deg: float
    dihedral_p01_deg: float
    dihedral_below_5deg: int
    seconds: float


def canon_keys(tets: np.ndarray, n: int) -> np.ndarray:
    t = np.sort(tets, axis=1).astype(np.int64)
    return np.unique(((t[:, 0] * n + t[:, 1]) * n + t[:, 2]) * n + t[:, 3])


def _row_keys(t: np.ndarray, n: int) -> np.ndarray:
    """Canonical key per row (rows must be sorted ascending)."""
    return ((t[:, 0] * n + t[:, 1]) * n + t[:, 2]) * n + t[:, 3]


_EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _tet_edge_keys(t: np.ndarray, n: int) -> np.ndarray:
    """(T, 6) undirected edge keys u*n+v (u<v) of sorted tets."""
    return np.stack([t[:, i] * n + t[:, j] for i, j in _EDGE_PAIRS], 1)


def _circumcentres(points: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d = (points[t[:, i]] for i in range(4))
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
    return o, det


def _insphere_margins(points: np.ndarray, t: np.ndarray, k: int = 8) -> np.ndarray:
    """Per tet: min over the k input points nearest to its circumcentre (own vertices excluded) of
    the insphere determinant relative to its rounding-error bound.  < 0: a point strictly inside
    the circumsphere (not Delaunay); ~0: a point ON the sphere (co-spherical tie); > 0: clean.
    The circumcentre of a sliver is unreliable, so it only selects candidates; the decision uses
    the translation-invariant determinant."""
    n = len(points)
    T = len(t)
    a, b, c, d = (points[t[:, i]] for i in range(4))
    o, det = _circumcentres(points, t)
    finite = np.isfinite(o).all(1)
    k = min(k, n)
    cand = np.zeros((T, k), dtype=np.int64)
    if finite.any():
        cand[finite] = cKDTree(points).query(o[finite], k=k)[1].reshape(-1, k)
    sgn = np.sign(det)
    margin = np.full(T, np.inf)
    for j in range(k):
        e = cand[:, j]
        own = (e == t[:, 0]) | (e == t[:, 1]) | (e == t[:, 2]) | (e == t[:, 3])
        pe = points[e]
        A_, B_, C_, D_ = a - pe, b - pe, c - pe, d - pe
        na, nb, nc, nd = (np.linalg.norm(x, axis=1) for x in (A_, B_, C_, D_))
        la, lb, lc, ld = na**2, nb**2, nc**2, nd**2
        t0 = la * np.einsum("ij,ij->i", B_, np.cross(C_, D_))
        t1 = lb * np.einsum("ij,ij->i", A_, np.cross(C_, D_))
        t2 = lc * np.einsum("ij,ij->i", A_, np.cross(B_, D_))
        t3 = ld * np.einsum("ij,ij->i", A_, np.cross(B_, C_))
        bound = (
            la * nb * nc * nd
            + lb * na * nc * nd
            + lc * na * nb * nd
            + ld * na * nb * nc
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = sgn * (-t0 + t1 - t2 + t3) / bound
        rel = np.where(own | ~finite | ~np.isfinite(rel), np.inf, rel)
        margin = np.minimum(margin, rel)
    return margin


INSPHERE_TOL = 1e-12  # relative to the rounding-error bound; float64 noise is ~1e-15


def analyze(
    points: np.ndarray, tets: np.ndarray, seconds: float, hull: ConvexHull
) -> Metrics:
    n = len(points)
    T = len(tets)
    if T == 0:
        return Metrics(
            0,
            0,
            0,
            0,
            0,
            0,
            len(hull.simplices),
            0,
            0,
            0,
            0.0,
            hull.volume,
            1.0,
            *([float("nan")] * 4),
            *([float("nan")] * 3),
            0,
            float("nan"),
            float("nan"),
            0,
            seconds,
        )
    t = np.sort(tets, axis=1)
    a, b, c, d = (points[t[:, i]] for i in range(4))
    B, C, D = b - a, c - a, d - a
    vol = np.abs(np.einsum("ij,ij->i", B, np.cross(C, D))) / 6.0
    bbox = np.ptp(points, axis=0).max()

    faces = np.concatenate(
        [t[:, [1, 2, 3]], t[:, [0, 2, 3]], t[:, [0, 1, 3]], t[:, [0, 1, 2]]]
    )
    fkeys, fcount = np.unique(
        (faces[:, 0] * n + faces[:, 1]) * n + faces[:, 2], return_counts=True
    )
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    ekeys = np.unique(np.concatenate([t[:, i] * n + t[:, j] for i, j in pairs]))
    V = len(np.unique(t))
    E, F = len(ekeys), len(fkeys)

    # circumsphere radius (quality) + empty-sphere check via the insphere determinant
    o, _ = _circumcentres(points, t)
    R = np.linalg.norm(a - o, axis=1)
    margin = _insphere_margins(points, t)
    violations = int(np.sum(margin < -INSPHERE_TOL))

    # quality: radius ratio 3 r_in / R_circ and dihedral angles
    def area(x, y, z):
        return 0.5 * np.linalg.norm(np.cross(y - x, z - x), axis=1)

    surf = area(b, c, d) + area(a, c, d) + area(a, b, d) + area(a, b, c)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = 3 * (3 * vol / surf) / R
    rr = np.nan_to_num(rr, nan=0.0, posinf=0.0)

    normals = [
        np.cross(c - b, d - b),
        np.cross(d - a, c - a),
        np.cross(b - a, d - a),
        np.cross(c - a, b - a),
    ]  # outward-ish per face
    normals = [
        nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-300)
        for nn in normals
    ]
    # orient all normals outward using the opposite vertex
    opp = [a, b, c, d]
    base = [b, a, a, a]
    for k in range(4):
        s = np.sign(np.einsum("ij,ij->i", normals[k], opp[k] - base[k]))
        normals[k] *= -np.where(s == 0, 1, s)[:, None]
    dih = []
    for (
        i,
        j,
    ) in pairs:  # dihedral angle at the edge shared by faces i and j (faces opposite vertices i, j... any pair of faces shares an edge)
        cosang = -np.einsum("ij,ij->i", normals[i], normals[j])
        dih.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    dih = np.stack(dih, 1)
    dmin = dih.min(1)

    return Metrics(
        tets=T,
        vertices_used=V,
        edges=E,
        faces=F,
        euler=V - E + F - T,
        boundary_faces=int((fcount == 1).sum()),
        hull_triangles=len(hull.simplices),
        nonmanifold_faces=int((fcount > 2).sum()),
        degenerate_tets=int((vol <= 1e-14 * bbox**3).sum()),
        delaunay_violations=violations,
        total_volume=float(vol.sum()),
        hull_volume=float(hull.volume),
        volume_rel_err=float(abs(vol.sum() - hull.volume) / hull.volume),
        vol_min=float(vol.min()),
        vol_max=float(vol.max()),
        vol_mean=float(vol.mean()),
        vol_median=float(np.median(vol)),
        radius_ratio_min=float(rr.min()),
        radius_ratio_mean=float(rr.mean()),
        radius_ratio_p01=float(np.percentile(rr, 1)),
        slivers=int((rr < 0.05).sum()),
        dihedral_min_deg=float(dmin.min()),
        dihedral_p01_deg=float(np.percentile(dmin, 1)),
        dihedral_below_5deg=int((dmin < 5).sum()),
        seconds=seconds,
    )


# ----------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------


def fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return (
            f"{v:.4g}"
            if (abs(v) < 1e-3 or abs(v) >= 1e5) and v != 0
            else f"{v:.5f}".rstrip("0").rstrip(".")
        )
    return str(v)


PARAGRAM_BOX_PAD = 1.0  # voronoi_ultra.cu: cells are clipped to the BVH root bounds -/+ 1.0 (absolute) + 1e-5


def _face_outside_box(points, ref_t, edge_keys_all, inv, hull_rays, pad) -> np.ndarray:
    """For every unique edge of the reference triangulation: True if its dual Voronoi face lies
    entirely outside the clipping box.  Bounded faces: all incident circumcentres beyond one box
    plane.  Unbounded faces (hull edges): additionally their rays, whose directions are the outward
    normals of the incident hull triangles (`hull_rays`: (E, 3, 2) booleans = [edge, axis,
    (all rays point to +axis, all rays point to -axis)]), must point away from the box on that
    same side."""
    o, _ = _circumcentres(points, ref_t)
    lo, hi = points.min(0) - pad, points.max(0) + pad
    tid = np.repeat(
        np.arange(len(ref_t)), 6
    )  # edge keys are flattened tet-major (T, 6)
    m = len(edge_keys_all)
    all_hi = np.ones((m, 3), bool)
    all_lo = np.ones((m, 3), bool)
    np.logical_and.at(all_hi, inv, (o > hi)[tid])
    np.logical_and.at(all_lo, inv, (o < lo)[tid])
    return ((all_hi & hull_rays[:, :, 0]).any(1)) | (
        (all_lo & hull_rays[:, :, 1]).any(1)
    )


def _hull_ray_constraints(points, tr, ref_edges) -> tuple[np.ndarray, np.ndarray]:
    """(hull_rays (E,3,2), is_hull_edge (E,)).  For interior edges the ray constraints are True
    (no rays); for hull edges they hold iff every ray (outward normal of an incident hull triangle)
    has a non-negative (resp. non-positive) component along the axis."""
    n = len(points)
    T = len(tr)
    faces = np.concatenate(
        [tr[:, [1, 2, 3]], tr[:, [0, 2, 3]], tr[:, [0, 1, 3]], tr[:, [0, 1, 2]]]
    )
    fk = (faces[:, 0] * n + faces[:, 1]) * n + faces[:, 2]
    _, first, count = np.unique(fk, return_index=True, return_counts=True)
    bidx = first[count == 1]  # one row per boundary face
    bf = faces[bidx]
    tet = bidx % T
    opp = tr[tet, bidx // T]  # vertex of the tet opposite to the face
    p0, p1, p2 = points[bf[:, 0]], points[bf[:, 1]], points[bf[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    inward = np.einsum("ij,ij->i", nrm, points[opp] - p0) > 0
    nrm[inward] *= -1  # outward
    rays_pos = nrm >= 0  # (F, 3)
    rays_neg = nrm <= 0
    E = len(ref_edges)
    hull_rays = np.ones((E, 3, 2), bool)
    is_hull = np.zeros(E, bool)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        ek = bf[:, i] * n + bf[:, j]
        pos = np.searchsorted(ref_edges, ek)
        np.logical_and.at(hull_rays[:, :, 0], pos, rays_pos)
        np.logical_and.at(hull_rays[:, :, 1], pos, rays_neg)
        is_hull[pos] = True
    return hull_rays, is_hull


def classify_tets(points: np.ndarray, t: np.ndarray, hull_volume: float) -> dict:
    """Geometric classification of a set of tets: flat, co-spherical tie, violation, clean."""
    if len(t) == 0:
        return {
            "count": 0,
            "volume_frac": 0.0,
            "flat": 0,
            "tie": 0,
            "violation": 0,
            "clean": 0,
        }
    a, b, c, d = (points[t[:, i]] for i in range(4))
    vol = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
    bbox = np.ptp(points, axis=0).max()
    flat = vol <= 1e-14 * bbox**3
    margin = _insphere_margins(points, t)
    tie = (np.abs(margin) <= INSPHERE_TOL) & ~flat
    viol = (margin < -INSPHERE_TOL) & ~flat
    return {
        "count": len(t),
        "volume_frac": float(vol.sum() / hull_volume) if hull_volume else float("nan"),
        "flat": int(flat.sum()),
        "tie": int(tie.sum()),
        "violation": int(viol.sum()),
        "clean": int((~flat & ~tie & ~viol).sum()),
    }


def explain_difference(
    points: np.ndarray,
    method_tets: np.ndarray,
    ref_tets: np.ndarray,
    hull_volume: float,
    adjacency_edge_keys: np.ndarray | None = None,
    status: np.ndarray | None = None,
    box_pad: float | None = None,
    box_pad_relative: float | None = None,
) -> dict:
    """Break the symmetric difference between a method's tets and the reference down by cause.

    method_only / ref_only: geometric classes (flat, tie, violation, clean).
    If the method is "Paragram adjacency + conversion", pass its edge keys (u*n+v, u<v, sorted):
    every ref-only tet is then attributed to a missing edge (and the edge to a failed cell, to
    bounding-box clipping of its Voronoi face, or to other/float32 causes) or, if all six edges
    exist, to a rejection by the conversion's empty-sphere test (a spurious neighbour).
    """
    n = len(points)
    tm = np.sort(method_tets.astype(np.int64), axis=1)
    tr = np.sort(ref_tets.astype(np.int64), axis=1)
    km, kr = _row_keys(tm, n), _row_keys(tr, n)
    m_only = tm[~np.isin(km, kr)]
    r_only = tr[~np.isin(kr, km)]
    out = {
        "method_only": classify_tets(points, m_only, hull_volume),
        "ref_only": classify_tets(points, r_only, hull_volume),
    }
    if adjacency_edge_keys is None:
        return out

    # --- edge level: adjacency vs reference edges -------------------------------------------
    ek_all = _tet_edge_keys(tr, n).reshape(-1)
    ref_edges, inv = np.unique(ek_all, return_inverse=True)
    adj = np.unique(adjacency_edge_keys)
    missing_edges = ref_edges[~np.isin(ref_edges, adj)]
    spurious = int((~np.isin(adj, ref_edges)).sum())

    hull_rays, hull_mask = _hull_ray_constraints(points, tr, ref_edges)
    pad = None
    if box_pad_relative is not None:
        pad = box_pad_relative * float(np.ptp(points, axis=0).max())
    elif box_pad is not None:
        pad = box_pad
    clipped_all = (
        _face_outside_box(points, tr, ref_edges, inv, hull_rays, pad + 1e-5)
        if pad is not None
        else np.zeros(len(ref_edges), bool)
    )
    clipped = dict(zip(ref_edges.tolist(), clipped_all.tolist()))

    st = (
        np.asarray(status).reshape(-1)
        if status is not None
        else np.zeros(n, dtype=np.int64)
    )
    mu, mv = missing_edges // n, missing_edges % n
    e_failed = (st[mu] != 0) | (st[mv] != 0)
    e_clipped = np.array([clipped[k] for k in missing_edges.tolist()], bool) & ~e_failed
    e_other = ~e_failed & ~e_clipped
    out["edges"] = {
        "reference": len(ref_edges),
        "reference_hull": int(hull_mask.sum()),
        "missing_hull": int(hull_mask[np.searchsorted(ref_edges, missing_edges)].sum()),
        "adjacency": len(adj),
        "missing": len(missing_edges),
        "missing_failed_cell": int(e_failed.sum()),
        "missing_box_clipped": int(e_clipped.sum()),
        "missing_other": int(e_other.sum()),
        "spurious": spurious,
    }

    # --- ref-only tets: attribute to their missing edges -------------------------------------
    cause = dict(
        zip(
            missing_edges.tolist(),
            np.where(e_failed, 1, np.where(e_clipped, 2, 3)).tolist(),
        )
    )
    r_edges = _tet_edge_keys(r_only, n)
    tet_cause = np.zeros(len(r_only), dtype=np.int64)  # 0: all edges present
    for j in range(6):
        cj = np.array([cause.get(k, 0) for k in r_edges[:, j].tolist()], dtype=np.int64)
        # priority: failed cell (1) > box-clipped (2) > other (3); keep the smallest non-zero
        tet_cause = np.where(
            tet_cause == 0, cj, np.where(cj == 0, tet_cause, np.minimum(tet_cause, cj))
        )
    out["ref_only"].update(
        {
            "missing_edge_failed_cell": int((tet_cause == 1).sum()),
            "missing_edge_box_clipped": int((tet_cause == 2).sum()),
            "missing_edge_other": int((tet_cause == 3).sum()),
            "all_edges_present_rejected": int((tet_cause == 0).sum()),
        }
    )
    # --- method-only tets: a violation means the violating point is not a neighbour ---------
    if len(m_only):
        m_edges = _tet_edge_keys(m_only, n)
        out["method_only"]["with_spurious_edge"] = int(
            (~np.isin(m_edges, ref_edges)).any(1).sum()
        )
    else:
        out["method_only"]["with_spurious_edge"] = 0
    return out


def _fmt_diff(label: str, ref_label: str, d: dict) -> list[str]:
    def geo(c):
        return (
            f"{c['count']} tets ({100 * c['volume_frac']:.2f}% of hull volume): flat {c['flat']}, "
            f"tie {c['tie']}, violation {c['violation']}, clean {c['clean']}"
        )

    lines = [
        f"      {ref_label}-only: " + geo(d["ref_only"]),
        f"      {label}-only: " + geo(d["method_only"]),
    ]
    r = d["ref_only"]
    if "missing_edge_failed_cell" in r:
        lines[0] += (
            f" | cause: missing edge -> failed cell {r['missing_edge_failed_cell']}, box-clipped "
            f"{r['missing_edge_box_clipped']}, other {r['missing_edge_other']}; all edges present but "
            f"rejected {r['all_edges_present_rejected']}"
        )
        lines[1] += (
            f" | with an edge absent from {ref_label} {d['method_only']['with_spurious_edge']}"
        )
        e = d["edges"]
        lines.append(
            f"      adjacency edges {e['adjacency']} vs {ref_label} edges {e['reference']}: missing {e['missing']} "
            f"(failed cell {e['missing_failed_cell']}, box-clipped {e['missing_box_clipped']}, other "
            f"{e['missing_other']}; {e['missing_hull']} of them hull edges), spurious {e['spurious']}"
        )
    return lines


def compare_sets(tets_a: np.ndarray, tets_b: np.ndarray, n: int) -> dict:
    """Set comparison of a method's tets (a) against the reference (b)."""
    ka, kb = canon_keys(tets_a, n), canon_keys(tets_b, n)
    common = len(np.intersect1d(ka, kb, assume_unique=True))
    return {
        "common": common,
        "method_only": len(ka) - common,
        "ref_only": len(kb) - common,
        "jaccard": common / max(1, len(ka) + len(kb) - common),
    }


METHOD_LABELS = {
    "paragram": "Paragram + conversion",
    "gdel3d": "gDel3D",
    "dewall": "Local DeWall",
}


def print_block(
    name,
    n,
    note,
    sources: list[str],
    columns: list[tuple[str, Metrics]],
    compares,
    status_hist=None,
    explanations=None,
    timing=None,
):
    """columns: [(label, metrics), ...] with the reference last; compares: {label: cmp} vs reference."""
    print(f"\n== {name}  (N={n}, {note})")
    for s in sources:
        print(f"   {s}")
    if status_hist:
        failed = n - status_hist.get("success", 0)
        detail = ", ".join(
            f"{k}={v}" for k, v in status_hist.items() if v and k != "success"
        )
        print(
            f"   paragram status: {failed} / {n} cells failed"
            + (f" ({detail})" if detail else "")
        )
    labels = [lab.split()[0] for lab, _ in columns]
    print("   " + f"{'metric':24s}" + "".join(f" {lab:>16s}" for lab in labels))
    ref = columns[-1][1]
    for k in asdict(ref):
        vals = [fmt(getattr(m, k)) for _, m in columns]
        flag = "" if all(v == vals[-1] for v in vals) else "  <-"
        print("   " + f"{k:24s}" + "".join(f" {v:>16s}" for v in vals) + flag)
    if timing:
        print(format_timing_table(timing))
    for lab, c in compares.items():
        verdict = (
            "IDENTICAL" if c["method_only"] == 0 and c["ref_only"] == 0 else "DIFFERENT"
        )
        print(
            f"   {lab} vs {labels[-1]}: {verdict}  common={c['common']} {lab}-only={c['method_only']} "
            f"{labels[-1]}-only={c['ref_only']} jaccard={c['jaccard']:.4f}"
        )
        if explanations and lab in explanations and verdict == "DIFFERENT":
            for line in _fmt_diff(lab, labels[-1], explanations[lab]):
                print(line)


def _save_json(path: str, results: dict, complete: bool) -> None:
    """Write results atomically; called after every dataset so partial results survive a crash."""
    results["_env"]["complete"] = complete
    results["_env"]["datasets_done"] = len(
        [k for k in results if not k.startswith("_")]
    )
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=20000, help="samples per analytic surface")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="OBJ names from common-3d-test-models",
    )
    ap.add_argument("--no-analytic", action="store_true", help="skip analytic surfaces")
    ap.add_argument(
        "--ply",
        nargs="*",
        default=None,
        help="additional PLY point clouds / meshes (vertices are used)",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="relative Gaussian jitter added to every dataset",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--adjacency",
        choices=["auto", "paragram", "qhull", "ref-edges"],
        default="auto",
        help="source of the Voronoi adjacency (auto = paragram on CUDA, else qhull)",
    )
    ap.add_argument("--json", default=None, help="write all metrics to this file")
    ap.add_argument(
        "--cgal-python",
        default=None,
        help="interpreter with the CGAL bindings (env CGAL_PYTHON)",
    )
    ap.add_argument(
        "--cgal-bin",
        default=None,
        help="compiled cgal_delaunay tool (env CGAL_DELAUNAY_BIN)",
    )
    ap.add_argument(
        "--paragram-bbox-pad",
        type=float,
        default=10.0,
        help="Paragram clipping-box padding as a multiple of the point-set extent (patched Paragram); "
        "-1 = legacy absolute 1.0 (default 10)",
    )
    ap.add_argument(
        "--repair",
        choices=["on", "off"],
        default="on",
        help="repair Paragram's failed cells on the CPU",
    )
    ap.add_argument(
        "--repair-hull",
        choices=["on", "off"],
        default="on",
        help="with --repair on, also recompute the cells of points on the convex hull exactly",
    )
    ap.add_argument(
        "--gdel3d",
        choices=["auto", "on", "off"],
        default="auto",
        help="also run gDel3D (pyGDel3D)",
    )
    ap.add_argument(
        "--unit-cube",
        action="store_true",
        help="normalise every dataset into the unit cube (Local DeWall's map) before float32 rounding, so that "
        "all methods triangulate exactly the same points and Local DeWall runs with --no-normalize",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="timed repetitions per method (correctness from run 1)",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed warm-up runs per method before timing (excludes JIT compilation, GPU clock ramp-up "
        "and first-touch page faults from the measurements)",
    )
    ap.add_argument(
        "--slow-repeats",
        type=int,
        default=2,
        help="repetitions for a method whose first run is slow",
    )
    ap.add_argument(
        "--slow-threshold",
        type=float,
        default=100.0,
        help="seconds above which a method is slow",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="timestamped progress lines on stderr"
    )
    ap.add_argument(
        "--dewall-bin",
        default=os.environ.get("LOCAL_DEWALL_BIN"),
        help="compiled Local-DeWall tool (see patch_dewall.py); also runs Local DeWall (env LOCAL_DEWALL_BIN)",
    )
    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    if args.cgal_python:
        os.environ["CGAL_PYTHON"] = args.cgal_python
    if args.cgal_bin:
        os.environ["CGAL_DELAUNAY_BIN"] = args.cgal_bin
    use_gdel3d = args.gdel3d == "on" or (args.gdel3d == "auto" and gdel3d_available())
    if args.gdel3d != "off" and not use_gdel3d:
        print(
            "gDel3D not available (pip install pyGDel3D on a CUDA machine); skipping it"
        )
    dewall_bin = args.dewall_bin
    if dewall_bin and not os.path.exists(dewall_bin):
        print(f"Local DeWall binary not found: {dewall_bin}; skipping it")
        dewall_bin = None

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed + 1)
    env = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "jitter": args.jitter,
        "samples_per_surface": args.n,
        "paragram_bbox_pad": args.paragram_bbox_pad,
        "repair": args.repair,
        "repair_hull": args.repair_hull,
        "paragram_max_planes": os.environ.get("PARAGRAM_MAX_PLANES"),
        "paragram_max_verts": os.environ.get("PARAGRAM_MAX_VERTS"),
        "methods": ["paragram"]
        + (["gdel3d"] if use_gdel3d else [])
        + (["dewall"] if dewall_bin else []),
        "unit_cube": args.unit_cube,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "slow_repeats": args.slow_repeats,
        "slow_threshold": args.slow_threshold,
        "argv": sys.argv[1:],
    }
    results = {"_env": env}
    summary = []

    datasets = build_datasets(args)
    print(
        "datasets: " + ", ".join(f"{name} ({len(pts)})" for name, pts, _ in datasets),
        flush=True,
    )
    if args.ply:
        loaded = {name for name, _, note in datasets if note.startswith("PLY")}
        for path in args.ply:
            if os.path.splitext(os.path.basename(path))[0] not in loaded:
                print(f"WARNING: PLY file not loaded: {path}", flush=True)

    for name, pts, note in datasets:
        pts = np.unique(
            pts.astype(np.float64), axis=0
        )  # Delaunay needs distinct points
        if args.jitter > 0:
            pts = pts + rng.normal(
                scale=args.jitter * np.ptp(pts, axis=0).max(), size=pts.shape
            )
        if args.unit_cube:
            pts = unit_cube(pts)
        log(f"{name}: {len(pts)} points" + (" (unit cube)" if args.unit_cube else ""))
        # Paragram consumes float32: use the float32-rounded coordinates everywhere so that all
        # methods triangulate exactly the same point set.
        pts32 = torch.from_numpy(pts.astype(np.float32))
        pts = pts32.double().numpy()
        n = len(pts)
        p_dev = pts32.to(device)

        timing: dict[str, dict] = {}

        def repeat(label, fn):
            """Run fn() args.repeats times (fewer if slow); keep the first result, aggregate timings.
            The first args.warmup runs are discarded so that one-off costs (JIT compilation of
            Paragram's CUDA extension, GPU clock ramp-up, first-touch page faults) are not timed."""
            runs, result, reps, i = [], None, args.repeats, 0
            for w in range(args.warmup):
                log(f"{name}: {label} warm-up {w + 1}/{args.warmup} (not timed)")
                res, tm = fn()
                if result is None:
                    result = res
                log(f"{name}: {label} warm-up took {tm['total']:.3f}s")
            while i < reps:
                log(f"{name}: {label} run {i + 1}/{reps}")
                res, tm = fn()
                if result is None:
                    result = res
                runs.append(tm)
                if i == 0 and tm["total"] > args.slow_threshold:
                    reps = min(reps, args.slow_repeats)
                    log(
                        f"{name}: {label} took {tm['total']:.1f}s > {args.slow_threshold}s: {reps} run(s) only"
                    )
                i += 1
            timing[label] = summarize_runs(runs)
            timing[label]["raw_runs"] = runs
            log(
                f"{name}: {label} mean {timing[label]['mean']:.3f}s over {len(runs)} run(s)"
            )
            return result

        # ---- reference (CGAL parallel; the sequential build is timed alongside) ------------
        def ref_fn():
            r, backend, info = reference_delaunay(pts)
            tm = {
                "total": info["seconds"],
                "cpu": info["seconds"],
                "gpu": 0.0,
                "io": info.get("io_seconds"),
            }
            if "sequential_seconds" in info:
                tm["sequential"] = info["sequential_seconds"]
            return (r, backend, info), tm

        ref, ref_backend, ref_info = repeat("cgal_parallel", ref_fn)
        seq_runs = [
            {"total": r["sequential"], "cpu": r["sequential"], "gpu": 0.0}
            for r in timing["cgal_parallel"]["raw_runs"]
            if "sequential" in r
        ]
        if seq_runs:  # the same tool with CGAL_THREADS=1, timed in every repetition
            timing["cgal_sequential"] = summarize_runs(seq_runs)
            timing["cgal_sequential"]["raw_runs"] = seq_runs
        t_ref = timing["cgal_parallel"]["mean"]

        # ---- Paragram adjacency (+ repair) + conversion ------------------------------------
        backend = args.adjacency
        if backend == "auto":
            backend = "paragram" if device.type == "cuda" else "qhull"

        def paragram_fn():
            def sync():
                if device.type == "cuda":
                    torch.cuda.synchronize()

            t0 = time.perf_counter()
            adjacency, offsets, status, adj_backend = voronoi_adjacency(
                p_dev,
                backend,
                ref,
                bbox_pad=args.paragram_bbox_pad if backend == "paragram" else None,
            )
            sync()
            t_gpu_adj = time.perf_counter() - t0
            status_hist = status_histogram(status)  # pre-repair
            repair_stats, t_cpu_repair = None, 0.0
            if args.repair == "on" and status is not None:
                from paragram_repair import repair_failed_cells

                adjacency, offsets, repair_stats = repair_failed_cells(
                    p_dev,
                    adjacency,
                    offsets,
                    status,
                    include_hull=args.repair_hull == "on",
                    # points must still count as on a hull facet after the jitter moved them
                    hull_tol=max(1e-6, 5.0 * args.jitter),
                )
                t_cpu_repair = repair_stats["seconds"]
                if repair_stats["repaired_cells"] == 0:
                    repair_stats = None
            t0 = time.perf_counter()
            tets, _ = delaunay_from_adjacency(
                p_dev, adjacency, offsets, status, return_circumcentres=False
            )
            sync()
            t_gpu_conv = time.perf_counter() - t0
            tets = tets.cpu().numpy()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            tm = {
                "total": t_gpu_adj + t_cpu_repair + t_gpu_conv,
                "gpu": t_gpu_adj + t_gpu_conv,
                "cpu": t_cpu_repair,
                "adjacency_gpu": t_gpu_adj,
                "repair_cpu": t_cpu_repair,
                "conversion_gpu": t_gpu_conv,
            }
            return (
                adjacency,
                offsets,
                status,
                adj_backend,
                status_hist,
                repair_stats,
                tets,
            ), tm

        adjacency, offsets, status, adj_backend, status_hist, repair_stats, par_tets = (
            repeat("paragram", paragram_fn)
        )
        tp = timing["paragram"]
        tp["breakdown"] = (
            f"adjacency(GPU) {tp.get('adjacency_gpu_mean', 0):.3f} + repair(CPU) "
            f"{tp.get('repair_cpu_mean', 0):.3f} + conversion(GPU) {tp.get('conversion_gpu_mean', 0):.3f}"
        )
        if "cgal_parallel" in timing:
            thr = (ref_info or {}).get("threads", "?")
            timing["cgal_parallel"]["breakdown"] = (
                f"CPU only, {thr} threads (TBB); excl.I/O = binary point/tet file exchange"
            )
        if "cgal_sequential" in timing:
            timing["cgal_sequential"]["breakdown"] = (
                "CPU only, 1 thread (same tool, CGAL_THREADS=1)"
            )
        t_adj = timing["paragram"]["adjacency_gpu_mean"] + timing["paragram"].get(
            "repair_cpu_mean", 0.0
        )

        hull = ConvexHull(pts)
        # one meaning for the "seconds" metric row: the method's mean total (GPU + CPU) per run
        m_par = analyze(pts, par_tets, timing["paragram"]["mean"], hull)
        m_ref = analyze(pts, ref, t_ref, hull)
        cmp = compare_sets(par_tets, ref, n)
        adj_np = adjacency.to("cpu", torch.long).numpy()
        off_np = offsets.to("cpu", torch.long).numpy()
        src_np = np.repeat(np.arange(n), np.diff(off_np))
        lo_, hi_ = np.minimum(src_np, adj_np), np.maximum(src_np, adj_np)
        adj_keys = np.unique((lo_ * n + hi_)[lo_ != hi_])
        is_paragram = adj_backend.startswith("paragram")
        explanations = {
            "paragram": explain_difference(
                pts,
                par_tets,
                ref,
                hull.volume,
                adjacency_edge_keys=adj_keys,
                status=status.to("cpu").numpy()
                if (status is not None and repair_stats is None)
                else None,
                box_pad=(PARAGRAM_BOX_PAD if args.paragram_bbox_pad < 0 else None)
                if is_paragram
                else None,
                box_pad_relative=args.paragram_bbox_pad
                if (is_paragram and args.paragram_bbox_pad >= 0)
                else None,
            )
        }
        sources = [
            f"adjacency: {adj_backend} ({t_adj:.2f}s incl. repair)   reference: {ref_backend}"
        ]
        if repair_stats:
            sources.append(
                f"repair: {repair_stats['failed_cells']} failed cells + {repair_stats['hull_cells']} hull cells "
                f"= {100 * repair_stats['repaired_fraction']:.1f}% of all cells recomputed on the CPU -> "
                f"{repair_stats['local_certified']} from local patches (k={repair_stats['k_hist']}), "
                f"{repair_stats['resolved_globally']} from one global exact triangulation "
                f"({repair_stats['global_backend']}), {repair_stats['seconds']:.2f}s"
                + (
                    f"; hull repair skipped: {repair_stats['hull_cells_skipped']} hull cells exceed the cap"
                    if repair_stats.get("hull_cells_skipped")
                    else ""
                )
            )
        columns = [("paragram", m_par)]
        compares = {"paragram": cmp}
        entry = {
            "n": n,
            "note": note,
            "adjacency": adj_backend,
            "adjacency_seconds": t_adj,
            "paragram_status": status_hist,
            "paragram_bbox_pad": args.paragram_bbox_pad
            if backend == "paragram"
            else None,
            "repair": repair_stats,
            "reference": ref_backend,
            "reference_info": {**ref_info, "seconds": t_ref},
            "paragram": asdict(m_par),
            "ref": asdict(m_ref),
            "compare": cmp,
            "difference": explanations["paragram"],
            "timing": timing,
        }

        # ---- other GPU methods -------------------------------------------------------------
        runners = []
        if use_gdel3d:
            runners.append(("gdel3d", lambda p=pts: run_gdel3d(p)))
        if dewall_bin:
            in_unit = bool(
                pts.min() >= 0.0 and pts.max() < 1.0
            )  # --no-normalize needs [0,1)^3
            runners.append(
                (
                    "dewall",
                    lambda p=pts, u=in_unit: run_dewall(p, dewall_bin, prenormalized=u),
                )
            )
        for label, fn in runners:
            try:

                def method_fn(fn=fn):
                    tets_, secs_, info_ = fn()
                    tm = {
                        "total": secs_,
                        "gpu": info_.get("gpu_seconds"),
                        "cpu": info_.get("cpu_seconds"),
                        "io": info_.get("io_seconds"),
                    }
                    return (tets_, secs_, info_), tm

                m_tets, t_m, m_info = repeat(label, method_fn)
                t_m = timing[label]["mean"]
                pts_m = m_info.pop("_points", None)
                if pts_m is None:  # same point set as everyone else
                    pts_m, ref_m, hull_m = pts, ref, hull
                else:  # the method rescaled its input: compare on ITS coordinates against a reference there
                    ref_m, _, _ = reference_delaunay(pts_m)
                    hull_m = ConvexHull(pts_m)
                mm = analyze(pts_m, m_tets, t_m, hull_m)
                cmp_m = compare_sets(m_tets, ref_m, n)
                columns.append((label, mm))
                compares[label] = cmp_m
                explanations[label] = explain_difference(
                    pts_m, m_tets, ref_m, hull_m.volume
                )
                note_m = ""
                if pts_m is not pts:
                    cmp_o = compare_sets(m_tets, ref, n)
                    m_info["compare_original_coordinates"] = cmp_o
                    m_info["reference_tets_on_own_coordinates"] = len(ref_m)
                    note_m = (
                        f" [compared on {m_info['reference_on']}: reference there has {len(ref_m)} tets; "
                        f"vs the reference on the original coordinates: {cmp_o['method_only']} {label}-only, "
                        f"{cmp_o['ref_only']} ref-only]"
                    )
                sources.append(_fmt_method_info(label, m_info) + note_m)
                if label == "gdel3d":
                    ph = m_info.get("phases_seconds") or {}
                    gpu_ph = " ".join(
                        f"{k} {ph[k]:.3f}"
                        for k in ("init", "split", "flip", "relocate", "sort")
                        if k in ph
                    )
                    cpu_ph = " ".join(
                        f"{k} {ph[k]:.3f}" for k in ("splaying", "out") if k in ph
                    )
                    timing[label]["breakdown"] = (
                        (f"GPU: {gpu_ph} | " if gpu_ph else "")
                        + (f"CPU: {cpu_ph}" if cpu_ph else "CPU: -")
                        + (
                            f"  [{m_info['stats_note']}]"
                            if m_info.get("stats_note")
                            else ""
                        )
                    )
                if label == "dewall":
                    ph = m_info.get("phases_seconds") or {}
                    timing[label]["breakdown"] = (
                        "GPU: "
                        + " ".join(f"{k} {v:.3f}" for k, v in ph.items())
                        + " | excl.I/O = text write+parse and process spawn (its CLI, not the algorithm)"
                    )
                entry.update(
                    {
                        label: asdict(mm),
                        f"compare_{label}": cmp_m,
                        f"{label}_info": m_info,
                        f"difference_{label}": explanations[label],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - keep the run alive if a method crashes on a dataset
                sources.append(f"{label}: FAILED ({exc})")
                entry[f"{label}_error"] = str(exc)
        columns.append((ref_backend, m_ref))
        print_block(
            name, n, note, sources, columns, compares, status_hist, explanations, timing
        )
        results[name] = entry
        if args.json:
            _save_json(args.json, results, complete=False)
        summary.append((name, entry))

    # ---- summary table ----------------------------------------------------------------------
    methods = [m for m in ("gdel3d", "dewall") if any(m in e for _, e in summary)]
    head = (
        f"{'dataset':18s} {'N':>7s} {'tets ref':>10s} {'tets paragram':>13s} {'paragram-only':>13s} {'ref-only':>8s} "
        f"{'viol':>5s} {'volerr':>9s} {'failed cells':>12s}"
    )
    for m in methods:
        head += f" {'tets ' + m:>12s} {m + '-only':>12s} {'ref-only':>8s} {'viol':>5s}"
    print("\n" + "=" * len(head))
    print(head + "  note")
    for name, e in summary:
        failed = (
            (e["n"] - e["paragram_status"].get("success", 0))
            if e.get("paragram_status")
            else 0
        )
        line = (
            f"{name:18s} {e['n']:7d} {e['ref']['tets']:10d} {e['paragram']['tets']:13d} {e['compare']['method_only']:13d} "
            f"{e['compare']['ref_only']:8d} {e['paragram']['delaunay_violations']:5d} {e['paragram']['volume_rel_err']:9.2e} "
            f"{failed:12d}"
        )
        for m in methods:
            if m in e:
                line += (
                    f" {e[m]['tets']:12d} {e[f'compare_{m}']['method_only']:12d} {e[f'compare_{m}']['ref_only']:8d} "
                    f"{e[m]['delaunay_violations']:5d}"
                )
            else:
                line += f" {'-':>12s} {'-':>12s} {'-':>8s} {'-':>5s}"
        print(line + f"  {e['note']}")
    # ---- timing summary: datasets x methods (mean seconds over the repeated runs) ---------
    labels = [
        m
        for m in ("paragram", "gdel3d", "dewall", "cgal_parallel", "cgal_sequential")
        if any(m in (e.get("timing") or {}) for _, e in summary)
    ]
    if labels:
        head2 = f"\n{'dataset':18s} {'N':>7s}" + "".join(
            f" {TIMING_LABELS[m]:>16s}" for m in labels
        )
        print("\n" + "=" * len(head2.strip()))
        print(
            "mean wall time over the repeated runs, seconds (GPU+CPU total per method)"
        )
        print(head2)
        for name, e in summary:
            row = f"{name:18s} {e['n']:7d}"
            for m in labels:
                t = (e.get("timing") or {}).get(m)
                row += f" {t['mean']:16.4f}" if t else f" {'-':>16s}"
            print(row)
        print(
            "(per-method CPU / GPU split and phase breakdown: see the block above each dataset and report.md)"
        )

    print(
        "viol = tetrahedra whose circumsphere strictly contains another input point (should be 0); "
        "volerr = |sum of tet volumes - convex hull volume| / hull volume (should be ~0); "
        "failed cells = Paragram cells with non-zero status (repaired on the CPU when --repair on); "
        "every method is compared against the same reference."
    )

    if args.json:
        _save_json(args.json, results, complete=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
