# Is the 1e-6 jitter negligible? A validation on the two point clouds

## What the jitter is

Every dataset is measured twice: as it is, and with a random perturbation added to every point.
The perturbation is one independent Gaussian sample per coordinate,

    sigma = 1e-6 x (largest side of the bounding box)
    p  ->  p + N(0, sigma) per axis

drawn from a fixed seed, so the perturbed cloud is reproducible and identical across methods and
across repeated runs. All coordinates are first normalised into the unit cube by a uniform scale
and a translation, which leaves the Delaunay triangulation mathematically unchanged, so the jitter
is expressed as a fraction of the cloud's own extent rather than in sensor units. On a tile 500 m
across, sigma corresponds to 0.5 mm.

The purpose is diagnostic. Both clouds are gridded products and therefore contain many groups of
five or more exactly co-spherical points, where the Delaunay triangulation is not unique and every
exact-arithmetic method has to resolve a tie. A generic perturbation removes those degeneracies, so
comparing the two passes separates "this method disagrees / is slow because the input is
degenerate" from "this method disagrees / is slow by itself".

## The jitter is four orders of magnitude below the data's own structure

### Table 1 - the jitter against every relevant length scale

| length scale | voronoi_iarpa_001 | voronoi_jax_068 |
|---|---|---|
| points | 99,990 | 99,981 |
| bounding box (normalised) | 0.886 x 0.999 x 0.130 | 0.822 x 0.999 x 0.162 |
| float32 resolution (1 ulp near 1.0) | 1.19e-07 | 1.19e-07 |
| **jitter sigma, per axis** | **9.99e-07** | **9.99e-07** |
| median 3D displacement (1.54 sigma) | 1.54e-06 | 1.54e-06 |
| largest 3D displacement (100k points) | 5.17e-06 | 5.17e-06 |
| closest genuine point pair (p0.1) | 1.23e-04 | 1.07e-04 |
| median point spacing (p50) | 2.98e-03 | 3.01e-03 |
| cloud extent | 1.00 | 1.00 |

The displacement of a point is the length of a 3-vector whose components are each N(0, sigma), so it
follows a chi distribution with three degrees of freedom: median 1.538 sigma, mean sqrt(8/pi) = 1.596
sigma, and a maximum of about 5.2 sigma over 100,000 draws. The measured 1.54 sigma and 5.17 sigma
match that to within sampling error, which is the arithmetic reason the 3D displacement is larger
than the per-axis sigma.

The jitter therefore sits in a wide, deliberate window: **8x above the float32 resolution**, so it
reliably breaks exact ties rather than disappearing into rounding, and **80x below the closest
genuine pair of points** in the data.

### Table 2 - displacement relative to the local point spacing

| nearest-neighbour distance | iarpa: distance / jitter as % of it | jax: distance / jitter as % of it |
|---|---|---|
| p0.1 (the 100 tightest points) | 1.23e-04 / 1.25 % | 1.07e-04 / 1.44 % |
| p1 | 3.95e-04 / 0.39 % | 4.36e-04 / 0.35 % |
| p50 (median) | 2.98e-03 / 0.05 % | 3.01e-03 / 0.05 % |
| p99 | 1.20e-02 / 0.01 % | 1.20e-02 / 0.01 % |
| exact duplicates (distance 0) | 22 points | 26 points |

A typical point moves 1/2000 of the distance to its nearest neighbour. Even among the tightest
0.1 % of pairs - a hundred times closer together than average - the perturbation is only 1.25 % of
the gap, so no pair is reordered and no local structure is altered.

The one exception is the exact duplicates: 22 points in one cloud and 26 in the other are
coincident, and for those the jitter is not small - it separates them into distinct points. This is
visible in the runs as `duplicate points dropped 0+0` in the jittered pass against 15-19 in the
unjittered one. Duplicate points are degenerate input for every method, so separating them is
arguably an improvement, but it is a genuine change to the point set and the only one.

## The geometry is untouched; the combinatorics are not

### Table 3 - effect on the reference (CGAL) triangulation

| quantity | iarpa: no jitter -> jitter | change | jax: no jitter -> jitter | change |
|---|---|---|---|---|
| convex-hull volume | 0.114453 -> 0.114455 | **+0.002 %** | 0.132875 -> 0.132878 | **+0.002 %** |
| sum of tet volumes | 0.114453 -> 0.114455 | +0.002 % | 0.132875 -> 0.132878 | +0.002 % |
| tetrahedra | 660,478 -> 669,003 | +1.29 % | 642,052 -> 665,373 | +3.63 % |
| zero-volume tets | 3 -> 35 | x12 | 5 -> 169 | x34 |
| slivers (radius ratio < 0.05) | 9,801 -> 18,353 | +87 % | 9,704 -> 33,013 | +240 % |
| smallest tet volume | 3.6e-15 -> 1.2e-16 | -97 % | 1.1e-15 -> 1.9e-17 | -98 % |
| min radius ratio | 3.4e-10 -> 6.5e-17 | -100 % | 7.3e-08 -> 3.7e-15 | -100 % |

The shape is preserved to 2 parts in 100,000: the convex hull and the total volume are unchanged for
practical purposes, and no feature of the surface is affected.

The *combinatorics* change substantially, and this is expected rather than a defect. Each
co-spherical group had one ambiguous resolution; the perturbation forces a definite one, which
splits flat configurations into additional, extremely thin tetrahedra. Hence 8,525 extra tetrahedra
on the first cloud, nearly twice the slivers, and a smallest tetrahedron volume two orders of
magnitude lower.

## Conclusion

The 1e-6 jitter is negligible as a change to the data - 0.05 % of the point spacing, 0.002 % of the
hull volume - and decisive as a change to the problem's degeneracy. That combination is what makes
it a valid diagnostic: the two passes triangulate the same surface, but only one of them contains
exact ties.

It is not, however, a preprocessing step to adopt. The jittered mesh has 87-240 % more slivers and a
worse minimum radius ratio, so a pipeline that needs well-shaped tetrahedra is better served by an
exact-arithmetic method on the unperturbed points. What the jitter established here is that the
degeneracies of gridded data, not the methods' asymptotic behaviour, account for most of the
observed differences: with the ties removed, gDel3D runs 7.7-17.6x faster, gStar4D 8.2-23x faster
and stops crashing, and gDel3D's 15,000-49,000 zero-volume tetrahedra disappear entirely.
