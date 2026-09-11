# 3-minute presentation

Four slides, ~45 s each.  Spoken text is roughly 400 words; the numbers in brackets are what to
put on the slide, not what to read out.

---

## Slide 1 - Domain (~30 s)

**3D Delaunay triangulation of large point clouds.**

It is the basic geometric structure behind volumetric meshing, surface reconstruction, finite-element
preprocessing and natural-neighbour interpolation.  Our inputs are real photogrammetric point clouds
- two satellite-derived clouds of about 100 000 points each, from the IARPA and JAX datasets - and we
study sizes from 2 000 up to 1 000 000 points.

The practical question is speed: a triangulation of a million points is a routine step in a
pipeline, and CPU implementations cost seconds.  GPUs have been proposed as the answer for over a
decade.

> Slide: one rendered cloud, the tetrahedron count (6.7 per point), the size range 2k-1M.

---

## Slide 2 - Research problem (~45 s)

We started from **Paragram**, a GPU code that computes a *Voronoi diagram* in float32 - not a
Delaunay triangulation.  The first question was therefore a conversion problem: the Voronoi
adjacency is exactly the edge graph of the Delaunay triangulation, so can we recover the tetrahedra
from it, on the GPU, fast enough to be worth it?

That immediately raised the real question.  Every published GPU Delaunay result we could find is
measured on **uniformly random points**.  Real clouds are not uniform and not generic: they come
from sensors on a grid, so they contain thousands of groups of five or more points that are exactly
co-spherical, where the Delaunay triangulation is not even unique.  Nobody had measured what the GPU
methods do on that kind of input.

So the problem we set ourselves: **is a GPU Delaunay triangulation of a real point cloud actually
correct, and actually faster than an exact CPU one?**

> Slide: the Voronoi-to-Delaunay duality in one picture; "published benchmarks: uniform random
> points" vs "our input: gridded sensor data, thousands of exact degeneracies".

---

## Slide 3 - State of the art (~45 s)

On the CPU, the reference is **CGAL** - exact arithmetic, with a parallel mode using TBB - and
**Geogram's ParallelDelaunay3d**, also exact and multithreaded.

On the GPU there are three published methods, and we ran all three:
**gStar4D** (2012), star splaying seeded by a discrete Voronoi diagram on a voxel grid;
**gDel3D** (2014), parallel insertion and bistellar flipping with a CPU star-splaying repair;
and **Local DeWall** (2026), a GPU Delaunay-wall construction.
Plus **Paragram**, which is a GPU Voronoi code rather than a triangulator.

Two things characterise that literature.  The GPU methods are benchmarked against sequential CGAL on
uniform random points, and the codes are old enough that they no longer build: gStar4D uses the CUDA
texture API, removed in CUDA 12.

> Slide: a table of the five methods - CPU/GPU, exact/float32, year.

---

## Slide 4 - Our contribution (~60 s)

**A Voronoi-to-Delaunay converter.**  Tetrahedra are recovered as 4-cliques of the adjacency graph
whose circumsphere is empty, with the in-sphere test in float64 against a Shewchuk-style
rounding-error bound, and streamed so a million points fit in memory.  We verified the conversion is
exact: given exact adjacency it reproduces CGAL tetrahedron for tetrahedron.

**A benchmark that says *why* results differ**, not just how many tetrahedra.  Every difference is
classified - co-spherical tie, zero-volume tetrahedron, empty-sphere violation, or a tetrahedron
lost because an adjacency edge is missing - across 18 datasets, with and without a 1e-6 perturbation.

**Three of the tools had to be repaired to run at all**: gStar4D ported off the removed texture API,
Local DeWall fixed for Linux, and its tetrahedron array - hard-coded at 7 per point - raised, since
it silently truncated four of our datasets.

**And the results contradict the premise.**  The fastest and only fully correct method is **GeoDel on
the CPU**: 18 of 18 datasets identical to CGAL, no failures, 0.9 microseconds per point.  gDel3D is
4x faster at a million points but crashes, hangs or times out on 90 % of the unjittered
measurements.  Paragram matches CGAL on 0 of 18 datasets, with 113 empty-sphere violations, and the
CPU repair that makes it correct is itself a CGAL triangulation.  Finally, **CGAL's parallel mode is
slower than its sequential mode** on this data - so the baseline everyone quotes is the wrong one.

> Slide: the scoreboard - method, datasets matching CGAL, microseconds per point, what differs.

---

## If asked

* **Why is degeneracy so expensive?**  A 1e-6 perturbation - 0.05 % of the point spacing, 0.002 % of
  the hull volume - speeds up gDel3D by 7.7-17.6x and gStar4D by 8.2-23x, and stops gStar4D
  crashing.  Exact predicates on co-spherical groups are the cost.
* **Why cost per point instead of a complexity exponent?**  These curves are a fixed cost plus linear
  work, not power laws.  Fitting one exponent gives values like 0.47, which would mean sub-linear
  complexity - impossible when you must read N points.  Only CGAL sequential is a true power law
  (exponent 1.00, 2 % residual, a flat 8.1 microseconds per point from 2k to 1M).
* **How do you get beyond 100 000 points?**  Two ways, and we did both: tiling translated copies
  (more area, same resolution) and interpolating inside the cloud's own Delaunay tetrahedra (same
  volume, finer spacing).  They agree to within 4 % for five of seven methods, so the scaling result
  does not depend on the choice.
* **Can Paragram be fixed?**  Its clipping box and cell budget we did fix.  What remains is float32
  error inside cells it reports as successful - 74 adjacency edges missing on one cloud with status
  "success" - and that is not reachable without changing the kernel's precision.
