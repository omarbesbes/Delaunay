# The methods compared, and what is measured

A compact reference for the seven implementations and for every metric the benchmark reports.
The conceptual description of each algorithm -- how it orders the points, what it does in each
phase -- is in the project report.


Benchmarks 3D Delaunay tetrahedralization methods on the same float32 point sets and compares every
result against an exact CGAL reference, tetrahedron for tetrahedron:

| method | what runs | precision |
|---|---|---|
| **Paragram + conversion** | Paragram (GPU Voronoi adjacency, float32) → exact CPU repair of failed / hull cells (`src/paragram_repair.py`) → 4-clique + insphere conversion to tetrahedra (`src/voronoi_to_delaunay.py`) | float32 GPU, float64 conversion |
| **gDel3D** | GPU insertion + flipping, CPU star splaying ([pyGDel3D](https://github.com/half-potato/pyGDel3D), patched) | float64, exact predicates |
| **gStar4D** | GPU star splaying seeded by a discrete Voronoi diagram (PBA) ([gStar4D](https://github.com/ashwin/gStar4D), patched for CUDA 12) | float32 points, exact predicates |
| **Local DeWall** | GPU Delaunay-wall construction ([Local-DeWall](https://github.com/WuhengGao/Local-DeWall), patched for Linux) | float32, exact predicates |
| **GeoDel** | Geogram's `ParallelDelaunay3d` through a Python binding ([GeoDel](https://github.com/Anttwo/GeoDel)), CPU-parallel (OpenMP) | float64, exact predicates |
| **CGAL parallel** | `Delaunay_triangulation_3` with `Parallel_tag` + TBB (`src/cgal_delaunay.cpp`), the reference | exact |
| **CGAL sequential** | same tool with `CGAL_THREADS=1` | exact |

For each dataset the script reports, per method: tetrahedron counts and set differences vs the reference
(Jaccard), empty-circumsphere violations, volume vs convex-hull volume, Euler characteristic, manifoldness,
volume / radius-ratio / dihedral-angle statistics, a breakdown of *why* sets differ (ties on co-spherical
groups, zero-volume tets, missing/spurious adjacency edges), and timings averaged over repeated runs with a
GPU / CPU split. `src/make_report.py` turns the JSON into a Markdown report with charts.

Datasets: the two point clouds in `data/` (≈100k points each), and optionally analytic surfaces (cube,
hollow cube, spheres, tori, Klein bottle, Möbius strip, trefoil) and classic meshes (bunny, spot, teapot,
cow, Suzanne, armadillo). With `--unit-cube` every dataset is normalised into the unit cube in float32 first,
so all methods triangulate *exactly* the same points (Local DeWall and gStar4D would otherwise rescale
internally). Those two tools do rescale and permute their input regardless, so each is compared against a
reference computed on the point set it actually triangulated, and the report says so.

