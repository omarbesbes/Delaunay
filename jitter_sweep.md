# How much jitter do the LiDAR clouds need, and what does it buy?

*Measured on Ruche (NVIDIA A100-SXM4-40GB, 8 CPU cores), job 1887129, on the two photogrammetric
point clouds `voronoi_iarpa_001` (99 975 points) and `voronoi_jax_068` (99 962 points).
Reproduce with `sbatch run_jitter.sbatch`; raw data in `results/jitter-1887129/`.*

The benchmark perturbs these clouds before triangulating them. This document measures what that
perturbation actually costs and what it actually buys, by sweeping its size over seven orders of
magnitude and recording three things at each value: how far the cloud moves, what each method
costs, and **how often each method's in-sphere test has to fall back to exact arithmetic**.

The last one is the instrument that makes the rest interpretable. A robust Delaunay implementation
evaluates its in-sphere predicate in floating point first, together with an error bound, and only
redoes it in exact arithmetic when the bound says the sign cannot be trusted. Degeneracy is
precisely what makes that filter fail, so the fallback rate is a direct measurement of how degenerate
the input is *as the algorithm sees it* — not a proxy for it.

---

## 0. A methodological result that came first: remove duplicate points

The two clouds contain exactly repeated points: 15 duplicate rows in `iarpa` and 19 in `jax`
(22 and 26 points sitting at 7 and 7 distinct locations). A Delaunay triangulation is not defined on
repeated points, and `test_delaunay_surfaces.py` has always dropped them
(`np.unique(pts.astype(np.float64), axis=0)`). The first version of this sweep did not.

The difference is not cosmetic. With the duplicates left in:

| | with duplicates | duplicates removed |
|---|---|---|
| gDel3D on raw `iarpa` | **segfault** (and OOM-killed at 64 GB) | runs, 0.386 s |
| gStar4D | **8 of 16 runs failed** (timeout / exit 255) | 16/16, all identical to CGAL |
| Local DeWall | **4 of 16 runs failed** (timeout) | 16/16 |
| CGAL 1 thread vs CGAL parallel, raw cloud | disagree by 11 tetrahedra | identical |
| total | 13 failures out of 112 | **0 failures out of 112** |

**Every failure in the sweep was caused by duplicate points, not by degeneracy.** This is worth
stating on its own: a reader who concludes from the first run that "gDel3D crashes on real LiDAR"
would be drawing a conclusion about invalid input handling, not about co-sphericity. Real scanner
output routinely contains repeated returns, so the practical lesson stands — deduplicate before
triangulating — but it is a different lesson.

Everything below is measured on deduplicated clouds.

---

## 1. What the jitter is

An independent Gaussian is added to every coordinate,

```
sigma = jitter x (largest extent of the cloud's bounding box)
```

from a fixed seed, so a given (cloud, jitter) is always the same point set. `jitter = 0` is the
cloud exactly as it came off the sensor. Because the displacement of a point is the length of a
3-vector whose components are each N(0, sigma), its median is 1.538 sigma, not sigma.

---

## 2. How much each jitter deforms the cloud

The absolute displacement means little on its own; what matters is its size relative to the
distance between neighbouring points (median spacing ≈ 0.003 in normalised units).

| jitter | median shift / point spacing | convex-hull volume | points whose nearest neighbour changed |
|---|---|---|---|
| 1e-9 | 0.00005 % | +0.0000 % | 0.00 % |
| 1e-8 | 0.0005 % | +0.0000 % | 0.00 % |
| 1e-7 | 0.005 % | +0.0003 % | 0.00 % |
| **1e-6** | **0.051 %** | **+0.003 %** | **0.05 %** |
| 1e-5 | 0.51 % | +0.036 % | 0.45 % |
| 1e-4 | 5.1 % | +0.40 % | 4.8 % |
| 1e-3 | 51 % | +3.1 % | 41 % |

(`iarpa`; `jax` agrees to within a few percent on every row.)

Up to 1e-5 the cloud is untouched by any standard a downstream user would apply: a shift of half a
percent of the point spacing changes no surface, no normal, no volume. From 1e-4 the perturbation
starts to be a real modification of the data — 5 % of the point spacing, and one point in twenty
acquires a different nearest neighbour.

### What it does to the triangulation is a different question

| jitter | tetrahedra | slivers (radius ratio < 0.05) | flat tetrahedra | smallest tetrahedron volume |
|---|---|---|---|---|
| 0 | 660 465 | 9 788 | 27 | 7.3e-17 |
| 1e-9 | 669 016 | 18 339 | 716 | **1.4e-23** |
| 1e-8 | 669 122 | 18 439 | 125 | 6.3e-21 |
| 1e-7 | 669 063 | 18 370 | 68 | 1.8e-18 |
| 1e-6 | 669 051 | 18 371 | 44 | 1.4e-16 |
| 1e-5 | 669 059 | 18 318 | **0** | 1.1e-14 |
| 1e-4 | 668 922 | 18 059 | 0 | 5.5e-13 |

This is the counter-intuitive part and it is worth being explicit about. **A very small jitter does
not remove degeneracy; it converts exact ties into near-ties, which are numerically worse.** At
jitter 0 the co-spherical groups produce 27 flat tetrahedra and a smallest volume of 7e-17. At
1e-9 there are 716 flat tetrahedra and the smallest is 1.4e-23 — six orders of magnitude thinner.
Only from 1e-5 does the triangulation become genuinely non-degenerate.

The tetrahedron count jumps 1.3 % between jitter 0 and any jitter, and the sliver count nearly
doubles (9 788 → 18 339). Both are the expected signature of co-spherical groups being resolved:
a group of *k* points on one sphere admits many triangulations, and perturbing it commits to one,
which generally has more and thinner tetrahedra than the degenerate configuration reported at
jitter 0.

---

## 3. Exact-arithmetic fallbacks: the central measurement

Share of in-sphere evaluations that the floating-point filter could not decide, so the test had to
be redone in exact arithmetic. Instrumentation: gDel3D via `patch_pygdel3d.py` (counts
`doInSphereFast` against `doInSphereSoS` on the GPU); CGAL via a `-DCGAL_PROFILE` build; Paragram's
converter via `voronoi_to_delaunay.last_insphere_stats()`.

| jitter | gDel3D (iarpa) | gDel3D (jax) | CGAL (both) | Paragram (iarpa) | Paragram (jax) |
|---|---|---|---|---|---|
| 0 | **2.585 %** | **7.786 %** | 0 | 0.0012 % | 0.0035 % |
| 1e-9 | 0 | 0 | 0 | 0.0016 % | 0.0082 % |
| 1e-8 | 0 | 0 | 0 | 0.0021 % | **0.0756 %** |
| 1e-7 | 0 | 0 | 0 | 0.0005 % | 0.0067 % |
| 1e-6 | 0 | 0 | 0 | 0.0001 % | 0.0006 % |
| ≥1e-5 | 0 | 0 | 0 | 0 | 0 |

Three findings, in order of importance.

**gDel3D falls off a cliff, not a slope.** On the raw `jax` cloud, 600 500 of 7 712 270 in-sphere
tests (7.79 %) cannot be decided in floating point and go to exact arithmetic with symbolic
perturbation. At *any* non-zero jitter in this range the number is **exactly zero**. There is no
intermediate regime: the smallest perturbation tested, 1e-9 — five thousandths of one percent of
the point spacing, leaving the convex hull unchanged to four decimal places — removes the problem
completely.

**CGAL never needs exact arithmetic for the in-sphere test, on any input here, including the raw
clouds.** Zero fallbacks out of 4.4 million evaluations, and its static filter never even hands off
to the interval filter. (Its *orientation* predicate does fall back, 16 times on the raw `iarpa`
cloud — so this is not an artefact of the profiling build being inert.) CGAL's semi-static bound is
a rigorous bound on double-precision arithmetic tuned to the actual coordinate magnitudes, and it
resolves configurations that gDel3D's tolerance-based test classifies as ties. Note that gDel3D
routes ties to exact arithmetic *by design*, in order to apply symbolic perturbation; CGAL resolves
a co-spherical result combinatorially instead. So the comparison is not "CGAL's arithmetic is
better" so much as "the two make a different design choice about what to do with a near-tie", and
gDel3D's choice is the one that costs time on this data.

**Paragram's curve is non-monotonic, and the peak is an artefact of float32.** Its undecidable rate
*rises* from jitter 0 to 1e-8 (on `jax`, 0.0035 % → 0.0756 %, a factor of 21) before collapsing.
Paragram is a float32 pipeline, and float32 resolves about 6e-8 in the unit cube: a jitter below
that does not survive the rounding, so at 1e-9 and 1e-8 Paragram is triangulating something very
close to the original degenerate grid, while the partial rounding introduces new near-ties of its
own. From 1e-7 upward the jitter survives and the rate falls monotonically to zero. **A method that
consumes float32 cannot be helped by a jitter finer than float32 resolution** — which is a real
constraint on the Paragram pipeline, not a measurement artefact.

---

## 4. What it costs each method

Seconds, mean of 5 runs. Every one of the 112 measurements succeeded.

| method | jitter 0 (iarpa / jax) | jitter ≥1e-6 (iarpa / jax) | ratio |
|---|---|---|---|
| gDel3D | 0.386 / 0.862 | 0.045 / 0.047 | **8.6× / 18.5×** |
| gStar4D | 1.269 / 4.188 | 0.159 / 0.177 | **8.0× / 23.7×** |
| Local DeWall | 0.125 / 0.260 | 0.086 / 0.107 | 1.4× / 2.4× |
| GeoDel (CPU, MT) | 0.133 / 0.134 | 0.129 / 0.128 | 1.0× |
| CGAL parallel | 0.158 / 0.400 | 0.281 / 0.517 | 0.6× / 0.8× |
| CGAL 1 thread | 0.831 / 0.814 | 0.827 / 0.815 | 1.0× |
| Paragram + conversion | 1.105 / 1.314 | 1.026 / 1.214 | 1.1× |

The two GPU methods pay an order of magnitude for degeneracy; the CPU methods barely notice it.
For gDel3D the mechanism is measured directly — 7.79 % exact fallbacks on raw `jax`, 18.5× slower —
rather than inferred. CGAL sequential varies by 0.9 % (iarpa) and 0.2 % (jax) across the whole sweep,
which is what "the filter never fails" looks like from the outside.

## 5. And whether the answer is right

Tetrahedra against a CGAL reference computed on the same points. "identical" means the two sets of
tetrahedra agree exactly.

| method | 0 | 1e-9 | 1e-8 | 1e-7 | ≥1e-6 |
|---|---|---|---|---|---|
| GeoDel | identical | identical | identical | identical | identical |
| gStar4D | identical | identical | identical | identical | identical |
| CGAL parallel | identical | identical | identical | identical | identical |
| gDel3D | **+14 980 / +48 889 extra** | identical | identical | identical | identical |
| Local DeWall | ±870 / ±740 | −41 / −27 | −6 / −3 | −1 | identical |
| Paragram + conversion | −246 / −151 | −255 / −201 | −248 / −357 | −141 / −125 | −6 405 / −18 374 |

**gDel3D's cost on the raw clouds is not only time: its output is wrong.** It returns every
tetrahedron CGAL found *plus* 14 980 (iarpa) or 48 889 (jax) more — overlapping tetrahedra from the
co-spherical groups, where its symbolic perturbation has committed to more than one triangulation
of the same region. Any non-zero jitter fixes this completely.

**GeoDel and gStar4D are exactly right at every jitter, including zero** — gStar4D pays 8–24× in
time for it, GeoDel pays nothing. GeoDel's record across this study is the strongest of any method:
16/16 identical, 0.13 s, flat.

**Paragram never matches at any jitter**, and its error is *worst* in the middle of the range
(−6 405 and −18 374 missing tetrahedra at 1e-6). Those are the jitters that produce the most
slivers (18 371 at 1e-6 versus 12 083 at 1e-3), so the failure tracks sliver count rather than
degeneracy: Paragram's float32 Voronoi cells cannot resolve very thin tetrahedra. Note these figures
are already the fair comparison — Paragram is compared against a reference computed on the float32
coordinates it actually triangulated, not on the float64 originals.

---

## 6. Conclusion: which jitter to use

**1e-6 is the right default, and this sweep supports it with roughly three orders of magnitude of
margin on either side.**

* It is 1 000× larger than the smallest jitter that eliminates gDel3D's exact-arithmetic fallbacks
  entirely (1e-9), so the choice is not marginal.
* It is the smallest value at which *every* method except Paragram is exactly identical to CGAL —
  Local DeWall still misses one tetrahedron at 1e-7 and converges at 1e-6.
* It deforms the cloud by 0.05 % of the point spacing and 0.003 % of the convex-hull volume, which
  is unmeasurable in any downstream use.
* It is 10× below the value at which the perturbation becomes visible in the data (1e-5 moves
  points by half a percent of the spacing, 1e-4 by five percent).

Two caveats worth carrying into the report:

1. **1e-6 does not produce a non-degenerate triangulation** — 44 flat tetrahedra remain, and the
   thinnest tetrahedron is 1.4e-16 in volume. It removes the *exact* ties that break the
   algorithms, not the near-ties. If the goal were well-conditioned tetrahedra rather than a
   well-defined triangulation, 1e-5 would be the threshold (zero flat tetrahedra), at ten times the
   deformation.
2. **Paragram is not helped by any jitter in this range**, and cannot be helped by one below
   ~6e-8 at all, because of float32. Its disagreement with CGAL is a property of its precision and
   its cell clipping, not of the input's degeneracy.
