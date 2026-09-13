# Notes on interpreting the results


- Identical tets to CGAL is the correctness bar; "tie" differences on co-spherical groups (regular grids,
  symmetric meshes) are legitimate; zero-volume tets (gDel3D on co-planar points) are flagged separately.
- Paragram's known limits: float32 cell clipping fails on sliver-shaped cells (`inconsistent_boundary`),
  loses edges silently on unbounded (hull) cells and on near-co-spherical input; the CPU repair fixes what
  is flagged or on the hull, and the report states what fraction of cells was recomputed on the CPU.
- **gDel3D crashes the interpreter on some inputs** -- `Aborted (core dumped)` during the warm-up
  on `torus-random` at 20 000 points, and a segfault at 22 000 points of `voronoi_iarpa_001` in the
  scaling sweep, both after handling the same clouds at 100 000 points. A crash inside a library is
  not an exception, so no `except` catches it: every pass therefore runs with `--resume` and is
  restarted up to `ATTEMPTS=4` times, which keeps the datasets already measured, records the
  measurement that killed the process as `CRASHED`, and continues with the rest.
- **Local DeWall allocates 7 tetrahedra per point and silently truncates beyond that.**
  `delaunay_solver.cu` had `tet = cuVector<int4>(7 * nv)` with no bound check, so a triangulation
  needing more came back as exactly `7*nv + 1` tetrahedra: `klein-bottle` (140 883 reference tets,
  7.04 per point) -> 140 001, `trefoil-tube` (10.6 per point) -> 140 001, `torus-random` (11.9 per
  point) -> 140 001, while every dataset under 7 per point (spheres 3.0, bunny 6.85, armadillo
  6.84, the two clouds 6.7) was exact. The truncated meshes are badly broken -- 36 % of the hull
  volume missing on `trefoil-tube`, Euler -379, only 17 017 of 20 000 points used.
  `script/patch_dewall.py` raises the factor to 16 (`-DDEWALL_TETS_PER_POINT=<n>` to change it; int4 is
  16 bytes, so 16 per point costs 256 MB at a million points), `run_dewall` reports a result that
  sits exactly at the capacity as `TRUNCATED`, and the verdict calls it INCOMPLETE rather than a
  tie-break -- the volume error is what separates the two, since a genuine tie difference leaves
  the volume exactly right.
- Local DeWall is exact but very slow on near-co-spherical input (hundreds of seconds for 20k sphere
  points), so with the default `--tool-timeout 10` it is reported as failed on those datasets;
  raise the limit (`TOOL_TIMEOUT=120`) if those numbers matter.
- gStar4D is from 2013 and needs three source changes to be usable, all in `script/patch_gstar4d.py`: its discrete
  Voronoi stage uses the texture-reference API, removed in CUDA 12 (replaced by `__ldg` loads through device
  pointers, numerically identical); its PLY writer split each tetrahedron into three triangles at 6 digits
  of precision (now one 4-index line at 9 digits, enough to identify the points exactly); and it built only
  for `sm_35`. It is compiled with `-fmad=false`, without which the GPU-side Shewchuk predicates stop being
  exact. It also drops duplicate points, which the benchmark reports.
- **gStar4D needs `-g 512` on these point clouds**, which is why that is the default. Its stars are
  seeded from a `g x g x g` voxel grid holding *one point per voxel*; the rest become "missing
  points" that go through a slower fix-up path. Uniform points barely collide (78 of 50k at
  g=256), but a LiDAR surface cloud is spaced far more finely along the surface than the voxel
  size, so at g=256 it loses 4 847 of 50 000 points and then stalls *before* the first
  star-consistency iteration. At g=512 the collisions drop to 1 100 and it finishes:
  0.76 s at 50k, 1.53 s at 100k, differing from CGAL only by co-spherical tie-breaks
  (Jaccard 0.9993). `g=1024` is not an option: PBA packs each coordinate into 10 bits
  (`ENCODE(x, y, z) = (x << 20) | (y << 10) | z`, with `0x3ff` reserved as the infinity sentinel),
  so the packed keys go negative and the run dies in a device assert -- the benchmark clamps the
  grid to 512. `--tool-timeout` still bounds the remaining cases: a method that stalls is reported
  as failed for that dataset and the run continues. `script/check_gstar4d.py` is how all of this was
  established, and its `missing=` column is the number to watch.
- GeoDel is the only CPU-parallel method besides CGAL, and gets the same core count as CGAL parallel
  (`--geodel-threads $SLURM_CPUS_PER_TASK`), so the two are directly comparable.

