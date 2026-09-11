# 3-minute talk - engineering students, non-specialists, no slides

Four parts, ~45 s each.  About 420 words.

---

## 1. Domain

We work in **computational geometry**, specifically 3D Delaunay triangulation.

The input is a point cloud - unstructured 3D coordinates from a sensor, in our case photogrammetric
reconstructions of terrain, around 100 000 points per tile.  The output is a tetrahedral mesh: the
space between the points partitioned into tetrahedra.  You need that mesh for volume computation,
finite-element simulation, surface reconstruction and interpolation.

Among all possible tetrahedralisations of a point set, the Delaunay triangulation is the canonical
one: it maximises the minimum angle, it is unique when the points are in general position, and it is
characterised by a simple local property - no point lies inside the circumsphere of any tetrahedron.
A million input points give roughly seven million tetrahedra.

## 2. Research problem

Computing it is a bottleneck: a few seconds per million points on a CPU.  The natural idea is to
parallelise it on a GPU, and there is a body of work claiming 5 to 10x speed-ups.

Our problem statement has two parts.

First, a construction problem: we started from a GPU code that computes the **Voronoi diagram**, not
the triangulation.  The two are duals - the Voronoi adjacency graph is exactly the edge graph of the
Delaunay triangulation - so the question was whether the tetrahedra can be reconstructed from that
adjacency efficiently.

Second, and this became the real subject: **all published GPU results are measured on uniformly
random points.**  Sensor data is not uniform and not generic - it is sampled on a grid, which
produces large sets of exactly co-spherical points.  There the Delaunay triangulation is not unique,
and the algorithms must resolve ties with exact arithmetic.  That case was essentially unmeasured.

## 3. What existed before, conceptually

Three algorithmic families:

- **Incremental insertion with flipping** - insert points one at a time, repair local violations by
  flipping tetrahedra.  This is what CGAL does sequentially with exact predicates, and what the main
  GPU method parallelises.
- **Star splaying** - compute each point's neighbourhood independently, then make the neighbourhoods
  mutually consistent.  Naturally parallel, used by the older GPU method.
- **Divide and conquer / Delaunay wall** - build a separating surface and recurse.  Used by the most
  recent GPU method.

On the CPU side the reference is CGAL, exact and trusted, plus a multithreaded exact library.  The
common evaluation protocol: uniform random points, compared against sequential CGAL.

## 4. Our contribution

Three things.

**A Voronoi-to-Delaunay conversion**: tetrahedra recovered as 4-cliques of the adjacency graph whose
circumsphere is empty, with the in-sphere test evaluated against a rounding-error bound so the
result is provably exact for generic input.

**A benchmark protocol that measures correctness, not only time**: every method compared against
CGAL tetrahedron by tetrahedron on 18 datasets, with each difference classified - a legitimate
co-spherical tie, a degenerate zero-volume tetrahedron, a missing tetrahedron, or an actual error.
Two of the three GPU codes no longer compiled; we ported them first.

**The results, which invert the expected conclusion.** On real data: one GPU method silently
truncates its output and loses a third of the mesh volume; another fails on 90 % of our
measurements without jitter; the float32 GPU Voronoi produces empty-sphere violations, so its output
is not a Delaunay triangulation at all.  The fastest method that is always correct is a
multithreaded **CPU** library.  And CGAL's parallel mode is slower than its sequential mode on this
data - so the standard baseline is itself wrong.

Conclusion: the published speed-ups are real for random points, but on structured real-world input
the deciding factors are exactness and robustness, not throughput.
