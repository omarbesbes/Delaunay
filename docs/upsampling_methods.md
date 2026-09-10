# Reaching sizes beyond the data: two ways to upsample a point cloud

The scaling study measures running time against the number of points, from 2 000 to 1 000 000.  The
two clouds hold about 100 000 points each (99 990 and 99 981), so every size above that has to be
manufactured.  There are two defensible ways to do it, they answer different questions, and both are
implemented (`scaling_study.py`, `--upsample tile|densify`):

| | **tile** | **densify** |
|---|---|---|
| question answered | same sensor over a **larger area** | same area at a **finer resolution** |
| scaling regime | weak scaling | strong scaling |
| point spacing | preserved | shrinks as N^(-1/3) |
| bounding box | grows | fixed |
| new points | none: the original points, repeated | interpolated between original points |

Below 100 000 points both modes are identical: a uniform random subsample without replacement.

## Method 1: tiling - more area

The cloud is replicated on a lattice of translated copies and the requested number of points is
drawn from the union.

1. **Choose the lattice.**  For n points, `ceil(n / N)` copies are needed.  The lattice is
   rectangular, with the smallest product of sides that reaches that number, and among those the
   most cube-like: 3 copies as 1x1x3, 6 as 1x2x3, 11 as 1x1x11.
2. **Assign the copy counts to axes** in inverse order of extent, so the most copies go along the
   cloud's *thinnest* axis.  These clouds are slabs (0.89 x 1.00 x 0.13 after normalisation), so
   11 copies stacked along z give a 0.89 x 1.00 x 1.46 domain rather than an 11-long corridor.
3. **Translate.**  Copy (i, j, m) is shifted by `(i, j, m) x (extent + gap)`, an elementwise
   product, so copies sit side by side.  `gap = extent / N^(1/3)` is about one point spacing, which
   keeps them from touching: copy 0 ends at x = 0.886, copy 1 starts at x = 0.905, and the closest
   pair of points across the corridor is 0.019 apart, against a median spacing of 0.003.
4. **Subsample** n points from the union, uniformly and without replacement.

Because the copies are *translated*, not scaled, every copy is congruent to the original: the point
spacing, the local structure and the exact degeneracies are all preserved.  The measured result:

| n | copies | lattice density | tiled domain | median spacing | vs original |
|---|---|---|---|---|---|
| original | 1 | 100 % | 0.89 x 1.00 x 0.13 | 0.00298 | 1.00x |
| 200 000 | 3 | 67 % | 0.89 x 1.00 x 0.40 | 0.00360 | 1.21x |
| 500 000 | 6 | 83 % | 1.79 x 1.00 x 0.40 | 0.00324 | 1.09x |
| 800 000 | 9 | 89 % | 2.70 x 1.00 x 0.40 | 0.00314 | 1.05x |
| 1 000 000 | 11 | 91 % | 0.89 x 1.00 x 1.46 | 0.00311 | 1.04x |

The density is not exactly 100 % because n rarely equals a whole number of copies; drawing 200 000
points from 3 copies (299 970 points) leaves each copy at 67 % of its original density, which
lengthens the spacing by 21 %.  Above 500 000 the effect is under 10 %.

*A note on an earlier version of this method.*  The first implementation used a **cubic** lattice,
which can only supply 1, 8 or 27 copies.  The density then sawtoothed: 25 % at 200 000 points,
rising to 88 % at 700 000, then dropping back to 30 % at 800 000 as the lattice jumped from 8 copies
to 27.  Spacing therefore varied between 1.05x and 1.59x the original *within one sweep*, which is
the opposite of what tiling is for.  Results produced before the switch to rectangular lattices
carry that artefact and their tiled curves should be read with it in mind.

## Method 2: densification - more resolution

The cloud is triangulated once, and new points are interpolated inside its own tetrahedra.

1. **Triangulate the cloud** (one Delaunay per cloud, reused for every size).
2. **Discard the tetrahedra that span voids**: those whose circumradius exceeds
   `--max-circumradius` (default 4) times the median point spacing.  Delaunay fills the convex
   hull, so wherever the cloud is sparse or concave it bridges the gap with a few large
   tetrahedra; sampling those would place points where the sensor saw nothing.
3. **For each new point**, pick a surviving tetrahedron **uniformly** and take a Dirichlet(1,1,1,1)
   convex combination of its four vertices - four non-negative weights summing to 1 - which is
   uniform inside that tetrahedron.
4. **Keep the original points** and add the new ones, so the original degeneracies survive
   alongside the interpolated points.

Two choices in that recipe were made by measurement, not by taste.  Both were tested at 4x the
points on a 20 000-point subsample, where the ideal spacing ratio is 4^(1/3) = 1.59:

| variant | spacing ratio | new points more than 3 spacings from a real one |
|---|---|---|
| volume-weighted tetrahedra | 1.02 | 39.9 % |
| tetrahedron = a point + 3 of its 12 nearest neighbours | 3.53 | 6 % *within 0.2* spacings |
| uniform per tetrahedron, no circumradius filter | 1.62 | 2.6 % |
| **uniform per tetrahedron, circumradius <= 4 spacings** | **1.72** | **0.15 %** |

* **Volume weighting fails.**  63 % of the Delaunay tetrahedra carry 96 % of the hull volume,
  because they are the ones spanning the voids.  Weighting by volume therefore pours 40 % of the
  new points into empty space: at 4x the points the median spacing improved by only 1.02x instead
  of 1.59x.  It fills holes rather than raising resolution.
* **Neighbourhood tetrahedra fail.**  A tetrahedron formed from a point and three of its nearest
  neighbours is anchored on an existing point, so new points pile up next to old ones: the spacing
  over-tightened by 2x and 6 % of the new points landed within 0.2 of a spacing from an existing
  one - manufacturing precisely the near-coincident configurations the benchmark exists to measure.
  Delaunay tetrahedra tile the space instead of being anchored on points, so they carry no such
  bias (1.4 %).
* **The circumradius filter is a trade-off, not a free win.**  A tighter cap keeps new points near
  real ones but concentrates them where the cloud is already dense; no cap tracks the density law
  almost exactly but leaves 2.6 % of the points in voids, up to 11 spacings out.  Cap 4 is the knee.
  Tetrahedra per point stays at 6.55-6.59 for **every** setting including no filter, so the
  workload the benchmark measures is unaffected and this choice is second order.

Note that the interpolation is **volumetric**, not a surface resampling.  A tangent-plane method
(fit a local plane by PCA, sample in it, offset along the normal) would be the standard choice for a
scanned surface, but these clouds are not surfaces: of their 13-point neighbourhoods, 78 % are
isotropic blobs (sqrt(l3/l1) > 0.4) and under 1 % are planar (< 0.15).  They are volumetric point
sets in a thin slab, so points are interpolated inside tetrahedra rather than inside triangles.

## What the two methods showed

For five of the seven methods benchmarked, the two axes agree to within a few percent at 1 000 000
points - densified/tiled ratios of 0.93x to 1.04x for Paragram, GeoDel, gDel3D and both CGAL
variants.  That is the expected outcome given the local geometry: because the neighbourhoods are
isotropic rather than planar, shrinking the spacing does not flatten them, so the work per point is
unchanged.  Tetrahedra per point stayed at 6.5-6.6 under both methods and at every size.

Two methods did distinguish them:

* **gStar4D completed the full range when densified but failed above 162 000 points when tiled.**
  Its stars are seeded from a fixed 512^3 voxel grid over the bounding box; tiling triples the box
  per axis, so voxels become 3x coarser, up to 27x more points share a voxel, and its initial-star
  phase stalls.  Densifying keeps the box fixed, so the grid stays matched to the data.  Concretely:
  its grid must be sized to the extent, not to the point count.
* **Local DeWall is 1.14x slower densified**, and its timing is unstable in either mode - see its
  single-threaded post-processing stage.

The practical consequence for this study is that the tiled curves, which cover the full range for
every method, are also representative of a higher-resolution sensor over the same area - with the
one exception of gStar4D, whose grid makes it sensitive to the extent rather than the count.
