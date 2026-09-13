"""Run one method on one point cloud and write the tetrahedra.

    python main.py experiment=triangulate method=gdel3d ply=data/voronoi_jax_068.ply
    python main.py experiment=triangulate method=geodel ply=cloud.ply out=results/mine check=true
    python src/triangulate.py --method cgal_sequential --ply cloud.ply --out results/tri

The studies measure; this produces. It runs a single method once on a PLY file, through the very
same `run_method` the jitter study uses, and writes the result to `<out>/<cloud>_<method>/`:

    points.npy     (N, 3) float64 -- the points the tetrahedra index (see below)
    tets.npy       (T, 4) int64   -- one row per tetrahedron, indices into points.npy
    mesh.vtk       the tetrahedra as a legacy VTK unstructured grid (cells), for ParaView
    mesh.ply       every unique face as a triangle mesh, for CloudCompare and any mesh viewer
    summary.json   method, sizes, seconds, preprocessing, and the CGAL comparison if --check

Preprocessing follows the benchmark, and every step is recorded in summary.json: exact duplicate
points are removed (a Delaunay triangulation is not defined on repeated points), the cloud is
normalised into the unit cube for the method (`--no-unit-cube` to skip that), and an optional
jitter is added (`--jitter 1e-6` is what the benchmark uses; the default here is 0, because a tool
that produces output should not perturb it unasked).

The outputs are written in the **input's coordinate frame**, so `mesh.vtk` overlays the PLY in a
viewer; the normalisation is internal and undone before writing (`--unit-cube-output` keeps it).
`points.npy` is not always the input in a new order: Local DeWall and gStar4D triangulate a
float32, rescaled copy of the points, so for them `points.npy` holds that copy, mapped back into
the input frame, and summary.json says so.

A tetrahedralization fills the convex hull, so from outside every viewer shows only the hull.
CloudCompare reads the VTK grid as points only; load `mesh.ply` there instead and tick *Wireframe*
in its properties, or cut it with Tools > Segmentation > Cross Section. ParaView reads `mesh.vtk`
and shows the cells with a `Clip` filter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import benchmark as T
from jitter_study import METHODS, jittered, run_method


def write_vtk(path: str, points: np.ndarray, tets: np.ndarray) -> None:
    """Legacy ASCII VTK unstructured grid: one cell type (10 = tetrahedron), no attributes."""
    with open(path, "w") as fh:
        fh.write(
            "# vtk DataFile Version 3.0\nDelaunay tetrahedra\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        )
        fh.write(f"POINTS {len(points)} double\n")
        np.savetxt(fh, points, fmt="%.17g")
        fh.write(f"CELLS {len(tets)} {5 * len(tets)}\n")
        np.savetxt(fh, np.hstack([np.full((len(tets), 1), 4), tets]), fmt="%d")
        fh.write(f"CELL_TYPES {len(tets)}\n")
        fh.write("\n".join(["10"] * len(tets)) + "\n")


def write_ply_faces(path: str, points: np.ndarray, tets: np.ndarray) -> int:
    """Binary PLY of the tetrahedra's faces as a triangle mesh -- what CloudCompare can display.

    A tetrahedralization fills the convex hull, so from outside only the hull is visible; in
    CloudCompare tick *Wireframe* in the mesh's properties or use Tools > Segmentation > Cross
    Section to see the tetrahedra. Each interior face is shared by two tetrahedra and is written
    once. Returns the number of faces written.
    """
    faces = np.concatenate(
        [tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]]
    )
    faces = np.unique(np.sort(faces, axis=1), axis=0).astype(np.int32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Delaunay tetrahedra: every unique face, from DelaunayBench triangulate\n"
        f"element vertex {len(points)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_rec = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", (3,))])
    face_rec["n"] = 3
    face_rec["v"] = faces
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(np.ascontiguousarray(points, dtype="<f8").tobytes())
        fh.write(face_rec.tobytes())
    return int(len(faces))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", required=True, choices=METHODS, help="which implementation to run")
    ap.add_argument("--ply", required=True, help="the point cloud (PLY, vertex x y z)")
    ap.add_argument(
        "--out",
        default="results/triangulate",
        help="output root; the result goes to <out>/<cloud>_<method>/ (default results/triangulate)",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Gaussian perturbation relative to the cloud's extent; the benchmark uses 1e-6 "
        "(default 0: the cloud as given)",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed of the jitter")
    ap.add_argument(
        "--no-unit-cube",
        action="store_true",
        help="do not normalise into the unit cube before running the method (the benchmark does)",
    )
    ap.add_argument(
        "--unit-cube-output",
        action="store_true",
        help="write the outputs in the normalised frame instead of mapping them back to the "
        "input's coordinates",
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="keep exact duplicate points (they break most methods)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="also compute the CGAL reference on the same points and compare tetrahedron sets",
    )
    # the same knobs run_method reads in the studies
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK") or 0))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--paragram-bbox-pad", type=float, default=10.0)
    ap.add_argument("--repair", choices=["on", "off"], default="on")
    ap.add_argument("--gstar4d-bin", default=os.environ.get("GSTAR4D_BIN", "bin/gstar4d"))
    ap.add_argument("--gstar4d-grid", type=int, default=512)
    ap.add_argument("--dewall-bin", default=os.environ.get("LOCAL_DEWALL_BIN", "bin/dewall"))
    ap.add_argument("--cgal-bin", default=os.environ.get("CGAL_DELAUNAY_BIN", "bin/cgal_delaunay"))
    ap.add_argument("--tool-timeout", type=float, default=120.0)
    args = ap.parse_args()

    if args.cgal_bin and os.path.exists(args.cgal_bin):
        os.environ["CGAL_DELAUNAY_BIN"] = os.path.abspath(args.cgal_bin)

    # Four of the seven methods run on the GPU.  On a cluster login node there is none, and torch
    # would only say "Found no NVIDIA driver" from deep inside a library call.
    gpu_methods = ("paragram", "gdel3d", "gstar4d", "dewall")
    if args.method in gpu_methods and args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print(
                f"{args.method} runs on the GPU and this machine has none.\n"
                "On the cluster, submit it as a job from the repository root:\n"
                f"    METHOD={args.method} PLY={args.ply} OUT={args.out}"
                + (f" JITTER={args.jitter:g}" if args.jitter else "")
                + (" CHECK=true" if args.check else "")
                + " sbatch script/run_triangulate.sbatch\n"
                "geodel, cgal_parallel and cgal_sequential run on the CPU and work here directly."
            )
            return 2
    for m, path in (("gstar4d", args.gstar4d_bin), ("dewall", args.dewall_bin)):
        if args.method == m and not (path and os.path.exists(path)):
            print(f"{m}: binary not found at {path}; build it with `bash script/build_tools.sh`")
            return 2

    # ---- input, prepared exactly as the benchmark prepares it ------------------------------
    raw = T.load_ply_vertices(args.ply)
    pts = raw.astype(np.float64)
    n_raw = len(pts)
    if not args.no_dedup:
        pts = np.unique(pts, axis=0)
    lo, scale = np.zeros(3), 1.0
    if not args.no_unit_cube:
        lo = pts.min(0)
        scale = float(np.ptp(pts, axis=0).max()) * 1.001
        pts = (pts - lo) / scale
    if args.jitter > 0:
        pts = jittered(pts, args.jitter, args.seed)

    stem = os.path.splitext(os.path.basename(args.ply))[0]
    out = os.path.join(args.out, f"{stem}_{args.method}")
    os.makedirs(out, exist_ok=True)
    print(
        f"{stem}: {n_raw} points"
        + (f", {n_raw - len(pts)} duplicate rows removed" if len(pts) < n_raw else "")
        + (", unit cube" if not args.no_unit_cube else "")
        + (f", jitter {args.jitter:g}" if args.jitter > 0 else "")
        + f"\n{args.method}: running ...",
        flush=True,
    )

    # ---- the method -----------------------------------------------------------------------
    t0 = time.perf_counter()
    tets, info = run_method(args.method, pts, args)
    wall = time.perf_counter() - t0
    tets = np.asarray(tets, dtype=np.int64)
    # Local DeWall and gStar4D return indices into their own float32, rescaled copy of the points
    tool_points = info.pop("_points", None)
    points_out = np.asarray(tool_points if tool_points is not None else pts, dtype=np.float64)
    predicates = info.pop("predicates", None)
    # The method saw the unit cube; the user gave the PLY.  Write what overlays the PLY.
    if not args.no_unit_cube and not args.unit_cube_output:
        points_out = points_out * scale + lo
    frame = "unit cube" if (args.unit_cube_output and not args.no_unit_cube) else "input"

    summary = {
        "method": args.method,
        "input": os.path.abspath(args.ply),
        "points_in_file": n_raw,
        "points_triangulated": int(len(points_out)),
        "tetrahedra": int(len(tets)),
        "seconds": float(info.get("seconds", wall)),
        "wall_seconds_including_io": wall,
        "preprocessing": {
            "duplicates_removed": int(n_raw - len(pts)) if not args.no_dedup else 0,
            "unit_cube": not args.no_unit_cube,
            # the method ran on (p - origin) / scale; the outputs are mapped back unless
            # --unit-cube-output
            "unit_cube_map": {"scale": scale, "origin": lo.tolist()},
            "jitter": args.jitter,
            "seed": args.seed,
        },
        "frame": frame,
        "points": "as triangulated by the tool (float32, rescaled), mapped to the output frame"
        if tool_points is not None
        else "the preprocessed input",
        "method_info": {k: v for k, v in info.items() if not k.startswith("_")},
    }
    if predicates:
        summary["predicates"] = predicates

    # ---- optional check against CGAL on the same points ----------------------------------
    if args.check:
        ref, backend, ref_info = T.reference_delaunay(points_out)
        cmp = T.compare_sets(tets, ref, len(points_out))
        summary["check"] = {
            "reference": backend,
            "reference_seconds": ref_info.get("seconds"),
            "reference_tetrahedra": int(len(ref)),
            **cmp,
            "identical": bool(cmp["ref_only"] == 0 and cmp["method_only"] == 0),
        }

    # ---- outputs --------------------------------------------------------------------------
    np.save(os.path.join(out, "points.npy"), points_out)
    np.save(os.path.join(out, "tets.npy"), tets)
    write_vtk(os.path.join(out, "mesh.vtk"), points_out, tets)
    summary["faces_in_mesh_ply"] = write_ply_faces(os.path.join(out, "mesh.ply"), points_out, tets)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)

    line = f"{args.method}: {len(tets)} tetrahedra in {summary['seconds']:.3f} s"
    if predicates:
        line += f"  ({predicates['exact']}/{predicates['total']} in-sphere tests needed exact arithmetic)"
    if args.check:
        c = summary["check"]
        line += (
            "  -- identical to CGAL"
            if c["identical"]
            else f"  -- vs CGAL: {c['ref_only']} missing, {c['method_only']} extra"
        )
    print(
        line + f"\nwrote {out}/{{points.npy, tets.npy, mesh.vtk, mesh.ply, summary.json}}"
        "\n  mesh.ply is the faces as a triangle mesh (CloudCompare: tick Wireframe, or "
        "Tools > Segmentation > Cross Section); mesh.vtk keeps the cells (ParaView)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
