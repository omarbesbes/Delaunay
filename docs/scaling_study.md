# Scaling study: time versus number of points


`src/scaling_study.py` answers a different question from the benchmark: not "is it correct" but "how
does the time grow with N", for the two point clouds only, with and without jitter. It measures
time only -- no reference triangulation, no metrics -- and writes one record per
(cloud, size, jitter, method) measurement.

```bash
sbatch script/run_scaling.sbatch                                    # the default sweep, ~29 sizes
SIZES="2000:200000:10000 200000" sbatch script/run_scaling.sbatch    # only real subsamples, no tiling
CLOUD=data/voronoi_jax_068.ply REPEATS=1 sbatch script/run_scaling.sbatch
python src/scaling_study.py --plot-only results/scaling-<jobid>/scaling.json --plot mine.png
```

Outputs:

* `scaling.png` -- log-log, one panel per cloud, solid without jitter and dashed with it, the fitted
  exponent alpha of *t ~ N^alpha* in the legend.
* `scaling_breakdown_<cloud>.png` -- **where the time goes as N grows**. Where a method's own
  timers do not add up to the wall clock, the difference is drawn as a grey band: gDel3D reports
  phase timers for its algorithm only, so allocating its device buffers (sized by n) and uploading
  the points fall outside them -- 4 % of the wall time at 2k rising to 36 % at 1M. Paragram, Local
  DeWall and gStar4D have no such gap, since their reported phases cover the whole measurement. one row per jitter, a
  left-hand panel with each method's CPU share, then one stacked panel per method. Paragram is
  broken into its three phases (GPU adjacency, GPU 4-clique conversion, CPU exact repair); the
  others into GPU and CPU totals, with the command-line tools' file exchange drawn as a dotted line
  (measured, excluded from the total).
* `scaling.csv` -- one row per measurement with `gpu`, `cpu`, `io`, `adjacency`, `repair`,
  `conversion` and the tetrahedron count, for your own plots and fits.
* `scaling.json` -- rewritten after every measurement, so a job that is cut short still leaves a
  usable curve.

Two things to know about the sizes:

* **Two ways to exceed the cloud's own point count** (`--upsample`, `UPSAMPLE=`), because the clouds
  hold ~100k points each and the sweep goes to 1M:
  * `tile` (default) -- **more area, same resolution**: the tiling described below.
  * `densify` -- **same area, more resolution**: the cloud is triangulated once, and each new point
    picks one of its Delaunay tetrahedra **uniformly** and takes a Dirichlet(1,1,1,1) convex
    combination of its four vertices (uniform inside that tetrahedron). Two choices, both measured
    at 4x the points on a 20k subsample against the ideal cube-root spacing ratio of 1.59:
    | variant | spacing ratio | new points >3 spacings from a real one |
    |---|---|---|
    | volume-weighted tetrahedra | 1.02 | 39.9 % |
    | point + 3 nearest neighbours | 3.53 | (6 % within 0.2 spacings: manufactured close pairs) |
    | uniform per tetrahedron, no filter | 1.62 | 2.6 % |
    | uniform per tetrahedron, circumradius <= 3 spacings | 1.82 | 0 % |
    | **uniform per tetrahedron, circumradius <= 4 spacings** | **1.72** | **0.15 %** |
    Volume weighting fails because Delaunay fills the convex hull and 63 % of its tetrahedra carry
    96 % of the volume, so the new points pour into the voids. Tetrahedra built from a point and
    its nearest neighbours fail because they are anchored on existing points, so new points pile up
    next to old ones. `--max-circumradius` trades the two defects against each other -- a tighter cap
    keeps points near real ones but concentrates them where the cloud is already dense, no cap
    gets the density law nearly exact but puts 2.6 % of points in empty space -- and the default
    of 4 is the knee. **Tetrahedra per point is 6.55-6.59 for every setting including no filter**,
    so the workload measured is the same either way. Edge length is a poor criterion by
    comparison, discarding 63 % of the tetrahedra and over-tightening by 70 %.
    The interpolation is volumetric rather than a tangent-plane resampling because these clouds are
    not surfaces: 78 % of 13-point neighbourhoods are isotropic blobs and under 1 % are planar.
    Tetrahedra per point holds at 6.5-6.6 from 1x to 10x density.
* **The tiling** (`--upsample tile`): the cloud is replicated on a k x k x k lattice
  of translated copies, so the point spacing -- and with it the local structure and the
  degeneracies -- is preserved while the extent grows, and the requested number of points is drawn
  from that lattice (1M points = 27 copies). Every record says how many tiles were used and the
  plot marks where tiling starts. A tiled input is a fair scaling load but not the same
  distribution as the real cloud, so read the two regimes separately.
* A method is dropped from larger sizes once it exceeds `--skip-above` (default 60 s), since that
  is monotone, or after `--give-up-after` (default 3) *consecutive* failures -- a single crash or
  timeout is input-specific, not a size limit, so the curve continues past it. That keeps CGAL sequential and Paragram's
  global-CGAL repair from consuming the whole job at 1M points.
* **gDel3D is measured in a separate interpreter with a wall-clock limit** (`--isolate gdel3d
  --measure-timeout 300`), because it has failed in all three ways that cannot be caught in
  process: a segfault at 22k points, an abort on `torus-random`, and a 13-minute hang at 42k that
  blocked every method queued behind it. Each such failure now costs one measurement. Both jobs
  additionally wrap every attempt in `timeout` (`ATTEMPT_TIMEOUT`, `PASS_TIMEOUT`) as a backstop.
* **Each method sweeps in its own process** (`scaling-<method>.json`, merged for the plot), because
  a library can crash the interpreter rather than raise: gDel3D has been seen to segfault on a
  22k-point subsample after handling 2k, 12k and 100k fine. Every measurement is marked in the
  JSON before it starts, so the automatic restart (`--resume`, up to `ATTEMPTS=4` per method)
  records the input that killed the process as failed and carries on with the next size instead of
  running into it again. To rebuild the diagram from whatever finished:
  `python src/scaling_study.py --plot-only results/scaling-<jobid>/scaling-*.json --plot mine.png`.

