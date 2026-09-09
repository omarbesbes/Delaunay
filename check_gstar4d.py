"""Smoke-test gStar4D on its own, to tell a broken build from an input it cannot handle.

    python check_gstar4d.py                                  # built-in generator, then random clouds
    python check_gstar4d.py --ply data/*.ply                 # ... and subsamples of real point clouds
    python check_gstar4d.py --grid 256 512 --jitter 0 1e-6    # sweep the grid and the jitter

Three stages, each case bounded by --timeout and reported on one line:

1. **its own generator** -- `gstar4d -n N -d 0`, no input file and none of this benchmark's code.
   If this hangs or crashes, the build is wrong (most likely the port of the PBA stage off the
   texture-reference API, see patch_gstar4d.py); if it passes, the binary itself works.
2. **uniform random points** through `run_gstar4d`, the exact path the benchmark uses, compared
   tetrahedron for tetrahedron against the reference (CGAL if available, else scipy/Qhull).  This
   exercises the input scaling, the PLY parsing and the index matching.
3. **the given point clouds**, at increasing subsample sizes, which is what tells you the size or
   the kind of input where it stops converging.

"loops" is the number of star-consistency iterations gStar4D needed (`-verbose`).  Its consistency
phase is a `do { ... } while (true)` with no iteration cap, so a case that times out with a large
and growing loop count is not converging on that input.  Two knobs are worth sweeping there:
`--jitter`, which removes exact co-spherical / co-planar degeneracies, and `--grid`, the resolution
of the discrete Voronoi diagram that seeds the initial stars (a finer grid separates the points of
a non-uniform cloud better; 512^3 costs about 1 GB of VRAM, 1024^3 about 9 GB).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

import test_delaunay_surfaces as T


def stage1(binary: str, n: int, grid: int, timeout: float) -> bool:
    """Run the tool on its own uniformly distributed points: build check, no benchmark code."""
    cmd = [
        os.path.abspath(binary),
        "-n",
        str(n),
        "-d",
        "0",
        "-g",
        str(grid),
        "-verbose",
        "-check",
    ]
    print(
        f"  {'own generator':<22s} n={n:<8d} g={grid:<5d}             ",
        end="",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        print(
            f"TIMEOUT after {timeout:.0f}s -> the build/port is suspect, not your data"
        )
        return False
    secs = time.perf_counter() - t0
    if r.returncode != 0:
        print(
            f"EXIT {r.returncode} in {secs:.1f}s: {(r.stdout + r.stderr).strip()[-400:]}"
        )
        return False
    loops = re.findall(r"^Loop:\s*(\d+)\s*$", r.stdout, re.MULTILINE)
    total = re.search(r"^\s*Total Time:\s*([0-9.eE+-]+)\s*$", r.stdout, re.MULTILINE)
    # gStar4D's own checks: it always prints "Euler Characteristic:" and adds a "... check failed!"
    # line only when a check fails.
    insphere = "In-sphere check failed" not in r.stdout
    euler = "Euler check failed" not in r.stdout
    orient = "Orientation check failed" not in r.stdout
    print(
        f"OK  {float(total.group(1)) / 1000 if total else secs:7.3f}s  "
        f"loops={int(loops[-1]) + 1 if loops else '?':<4} "
        f"self-check: in-sphere {'ok' if insphere else 'FAILED'}, "
        f"Euler {'ok' if euler else 'FAILED'}, orientation {'ok' if orient else 'FAILED'}"
    )
    return insphere and euler and orient


def case(
    name: str,
    points: np.ndarray,
    binary: str,
    grid: int,
    timeout: float,
    ref: bool,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> bool:
    """Run the benchmark's own runner, then compare against the reference on its point set."""
    if jitter > 0:  # same convention as test_delaunay_surfaces.py --jitter
        rng = rng or np.random.default_rng(0)
        points = points + rng.normal(
            scale=jitter * np.ptp(points, axis=0).max(), size=points.shape
        )
    print(
        f"  {name:<22s} n={len(points):<8d} g={grid:<5d} jit={jitter:<7g} ",
        end="",
        flush=True,
    )
    try:
        tets, secs, info = T.run_gstar4d(
            points, binary, grid_size=grid, timeout=timeout, verbose=True
        )
    except Exception as exc:  # noqa: BLE001 - this is the thing being tested
        print(f"FAILED: {str(exc)[:300]}")
        return False
    msg = (
        f"OK  {secs:7.3f}s  tets={len(tets):<8d} loops={info.get('consistency_loops', '?'):<4} "
        f"match={info['max_match_dist']:.0e} dropped={info['dropped_duplicate_points']}"
        f"+{info['dropped_after_scaling']}"
    )
    if not ref:
        print(msg)
        return True
    reference, backend, _ = T.reference_delaunay(info["_points"])
    cmp = T.compare_sets(tets, reference, len(points))
    verdict = (
        "IDENTICAL"
        if cmp["method_only"] == 0 and cmp["ref_only"] == 0
        else f"differs by {cmp['method_only']} / {cmp['ref_only']} (jaccard {cmp['jaccard']:.4f})"
    )
    print(f"{msg}  vs {backend.split()[0]}: {verdict}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bin", default=os.environ.get("GSTAR4D_BIN", "bin/gstar4d"))
    ap.add_argument(
        "--timeout", type=float, default=120.0, help="per case, seconds (default 120)"
    )
    ap.add_argument(
        "--grid",
        type=int,
        nargs="+",
        default=[256],
        help="gStar4D PBA grid sizes to try (-g, default 256); the grid seeds the initial stars, so "
        "a finer one can help on a non-uniform cloud (512^3 costs ~1 GB of VRAM, 1024^3 ~9 GB)",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        nargs="+",
        default=[0.0],
        help="relative Gaussian jitters to try, as in the benchmark's --jitter (default 0): jitter "
        "removes the exact co-spherical and co-planar degeneracies of a structured cloud",
    )
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 5000, 20000, 100000],
        help="point counts to try (default 1000 5000 20000 100000)",
    )
    ap.add_argument(
        "--ply", nargs="*", default=[], help="PLY point clouds to subsample and try"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-reference", action="store_true", help="skip the comparison, time only"
    )
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(
            f"gStar4D binary not found: {args.bin} (build it with 'bash build_tools.sh')"
        )
        return 2
    print(
        f"binary: {os.path.abspath(args.bin)}   grids: {args.grid}   jitters: {args.jitter}   "
        f"timeout: {args.timeout:.0f}s/case\n"
    )
    rng = np.random.default_rng(args.seed)
    ok = True

    print("[1] the tool's own uniform points (tests the build, not this benchmark)")
    ok &= stage1(args.bin, min(args.sizes[0], 10000), args.grid[0], args.timeout)

    print("\n[2] uniform random points through the benchmark's runner")
    for n in args.sizes:
        pts = rng.random((n, 3))
        for grid in args.grid:
            for jit in args.jitter:
                ok &= case(
                    "uniform in a cube",
                    pts,
                    args.bin,
                    grid,
                    args.timeout,
                    not args.no_reference,
                    jit,
                    rng,
                )

    for path in args.ply:
        name = os.path.splitext(os.path.basename(path))[0]
        cloud = T.unit_cube(T.load_ply_vertices(path))
        print(f"\n[3] {name} ({len(cloud)} points), subsampled")
        for n in [s for s in args.sizes if s < len(cloud)] + [len(cloud)]:
            sub = (
                cloud
                if n == len(cloud)
                else cloud[rng.choice(len(cloud), n, replace=False)]
            )
            for grid in args.grid:
                for jit in args.jitter:
                    ok &= case(
                        name,
                        sub,
                        args.bin,
                        grid,
                        args.timeout,
                        not args.no_reference,
                        jit,
                        rng,
                    )

    print(
        "\nA TIMEOUT in [1] means the build or the CUDA-12 port is broken; a TIMEOUT only in [3] "
        "means gStar4D does not converge on that input (its consistency loop has no iteration "
        "cap),\nwhich is a property of the method, not of this benchmark."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
