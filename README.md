# Delaunay benchmark: GPU methods vs CGAL on surface point clouds

Benchmarks 3D Delaunay tetrahedralization methods on the same float32 point sets and compares every
result against an exact CGAL reference, tetrahedron for tetrahedron:

| method | what runs | precision |
|---|---|---|
| **Paragram + conversion** | Paragram (GPU Voronoi adjacency, float32) → exact CPU repair of failed / hull cells (`paragram_repair.py`) → 4-clique + insphere conversion to tetrahedra (`voronoi_to_delaunay.py`) | float32 GPU, float64 conversion |
| **gDel3D** | GPU insertion + flipping, CPU star splaying ([pyGDel3D](https://github.com/half-potato/pyGDel3D), patched) | float64, exact predicates |
| **gStar4D** | GPU star splaying seeded by a discrete Voronoi diagram (PBA) ([gStar4D](https://github.com/ashwin/gStar4D), patched for CUDA 12) | float32 points, exact predicates |
| **Local DeWall** | GPU Delaunay-wall construction ([Local-DeWall](https://github.com/WuhengGao/Local-DeWall), patched for Linux) | float32, exact predicates |
| **GeoDel** | Geogram's `ParallelDelaunay3d` through a Python binding ([GeoDel](https://github.com/Anttwo/GeoDel)), CPU-parallel (OpenMP) | float64, exact predicates |
| **CGAL parallel** | `Delaunay_triangulation_3` with `Parallel_tag` + TBB (`cgal_delaunay.cpp`), the reference | exact |
| **CGAL sequential** | same tool with `CGAL_THREADS=1` | exact |

For each dataset the script reports, per method: tetrahedron counts and set differences vs the reference
(Jaccard), empty-circumsphere violations, volume vs convex-hull volume, Euler characteristic, manifoldness,
volume / radius-ratio / dihedral-angle statistics, a breakdown of *why* sets differ (ties on co-spherical
groups, zero-volume tets, missing/spurious adjacency edges), and timings averaged over repeated runs with a
GPU / CPU split. `make_report.py` turns the JSON into a Markdown report with charts.

Datasets: the two point clouds in `data/` (≈100k points each), and optionally analytic surfaces (cube,
hollow cube, spheres, tori, Klein bottle, Möbius strip, trefoil) and classic meshes (bunny, spot, teapot,
cow, Suzanne, armadillo). With `--unit-cube` every dataset is normalised into the unit cube in float32 first,
so all methods triangulate *exactly* the same points (Local DeWall and gStar4D would otherwise rescale
internally). Those two tools do rescale and permute their input regardless, so each is compared against a
reference computed on the point set it actually triangulated, and the report says so.

## Running on Ruche (Mésocentre, SLURM)

Everything is installed into a conda environment under `$WORKDIR` (Python 3.12, CUDA 12.8 toolkit, GCC 13,
torch cu128, CGAL headers + TBB), so nothing depends on the cluster's module versions except `anaconda3`.

1. Connect and clone (login node):
   ```bash
   ssh <user>@ruche.mesocentre.universite-paris-saclay.fr
   cd $WORKDIR && git clone https://github.com/omarbesbes/Delaunay.git && cd Delaunay
   ```
2. One-time setup (login node, has internet; 15–30 min, mostly downloads and CUDA compilation):
   ```bash
   bash setup_ruche.sh
   ```
   `build_tools.sh` can also be run on its own to (re)build just the standalone tools, but only with
   the environment active — it now says so if it is not:
   ```bash
   module load anaconda3/2023.09-0/none-none && source activate $WORKDIR/envs/delaunay
   export CUDA_HOME=$CONDA_PREFIX CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
   bash build_tools.sh            # or --force to rebuild what is already there
   GPU_ARCHS="8.0" bash build_tools.sh --force   # only the GPU of this node (faster build)
   ```
   It creates `$WORKDIR/envs/delaunay`, installs the Python packages, builds `bin/cgal_delaunay`
   (parallel CGAL), installs the patched Paragram, pyGDel3D and GeoDel, builds `bin/dewall` (Local DeWall)
   and `bin/gstar4d` (gStar4D) for V100 and A100 (`GPU_ARCHS="7.0;8.0"`), and pre-downloads the meshes. Each step prints a
   check line; if one fails, the message says which tool is missing and the benchmark still runs without it.
3. Submit the benchmark. **Use the A100 partition** (`gpua100`, the default in the script): the
   `cu128` torch wheels contain no Volta (sm_70) kernels, so Paragram and gDel3D cannot run on the
   V100 partitions with that build. To use `gpu` / `gpu_test` (V100) instead, reinstall torch with
   Volta support first:
   ```bash
   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash setup_ruche.sh
   ```
   (Local DeWall, gStar4D and CGAL are standalone binaries, and GeoDel is CPU-only, so those four run on
   any of the partitions.)
   ```bash
   sbatch run_ruche.sbatch                 # the two point clouds, 10 timed repeats per method
   FULL=1 BASELINE=0 sbatch run_ruche.sbatch  # everything: clouds + analytic surfaces + meshes,
                                              # corrected Paragram only, with and without jitter
   TOOL_TIMEOUT=120 sbatch run_ruche.sbatch   # allow a slow external tool 2 min (default 10 s)
   REPEATS=5 sbatch run_ruche.sbatch
   JITTER=0 sbatch run_ruche.sbatch           # skip the jittered pass
   squeue -u $USER                         # job state
   tail -f results/delaunay-bench.o<jobid> # live progress ([HH:MM:SS] lines: dataset, method, run i/N)
   ```
   The job runs up to three passes, selected by `BASELINE` and `JITTER`: corrected Paragram
   (10x clipping pad, CPU repair) with gDel3D, gStar4D, Local DeWall, GeoDel and CGAL; the upstream Paragram baseline (legacy clipping box, no repair; 3 repeats); and a jittered
   pass (3 repeats). A method whose first run exceeds 100 s is repeated only twice.
4. Results, in `results/<jobid>/`: `report.md` (+ PNG charts), `results*.json` (all numbers; written after
   every dataset, so partial results survive a crash), `run*.log` (full console output).
   Copy them back with `scp -r <user>@ruche...:$WORKDIR/Delaunay/results/<jobid> .`

Is gStar4D working at all? It is the one method that can hang (see the notes at the end), so it
has its own check — on a GPU node, with the environment active:
```bash
python check_gstar4d.py --ply data/*.ply --timeout 120
python check_gstar4d.py --ply data/*.ply --sizes 50000 --grid 256 512 --jitter 0 1e-6   # what helps?
```
Stage [1] runs the tool's own point generator, with none of this benchmark's code: a timeout there
means the build or the CUDA-12 port is broken. Stages [2] and [3] go through the benchmark's runner
and compare every tetrahedron against the reference, so a timeout only in [3] means gStar4D does
not converge on that input. `loops` is the number of star-consistency iterations it needed.

Quick interactive test on a GPU node (1 h partition):
```bash
srun --partition=gpu_test --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash
module load anaconda3/2023.09-0/none-none && source activate $WORKDIR/envs/delaunay
export CUDA_HOME=$CONDA_PREFIX PARAGRAM_MAX_PLANES=128 PARAGRAM_MAX_VERTS=128
python test_delaunay_surfaces.py --no-analytic --models --ply data/*.ply --unit-cube --repeats 2 --verbose \
  --cgal-bin bin/cgal_delaunay --dewall-bin bin/dewall --gstar4d-bin bin/gstar4d \
  --geodel-threads $SLURM_CPUS_PER_TASK --json results/test.json
```

## Scaling study: time vs number of points

`scaling_study.py` answers a different question from the benchmark: not "is it correct" but "how
does the time grow with N", for the two point clouds only, with and without jitter. It measures
time only -- no reference triangulation, no metrics -- and writes one record per
(cloud, size, jitter, method) measurement.

```bash
sbatch run_scaling.sbatch                                    # the default sweep, ~29 sizes
SIZES="2000:200000:10000 200000" sbatch run_scaling.sbatch    # only real subsamples, no tiling
CLOUD=data/voronoi_jax_068.ply REPEATS=1 sbatch run_scaling.sbatch
python scaling_study.py --plot-only results/scaling-<jobid>/scaling.json --plot mine.png
```

Outputs `scaling.png` (log-log, one panel per cloud, solid without jitter and dashed with it, the
fitted exponent alpha of *t ~ N^alpha* in the legend), `scaling.csv` (for your own plots; it has the
GPU / CPU split, so Paragram's GPU-only curve can be drawn separately from its CPU repair) and
`scaling.json`, rewritten after every size so a job that is cut short still leaves a usable curve.

Two things to know about the sizes:

* **The clouds have ~100k points each** (99 990 and 99 981), so anything above that is not a
  subsample. `points_at()` builds it by **tiling**: the cloud is replicated on a k x k x k lattice
  of translated copies, so the point spacing -- and with it the local structure and the
  degeneracies -- is preserved while the extent grows, and the requested number of points is drawn
  from that lattice (1M points = 27 copies). Every record says how many tiles were used and the
  plot marks where tiling starts. A tiled input is a fair scaling load but not the same
  distribution as the real cloud, so read the two regimes separately.
* A method that fails, times out, or exceeds `--skip-above` (default 60 s) at some size is not
  measured at larger sizes, and its records say so. That keeps CGAL sequential and Paragram's
  global-CGAL repair from consuming the whole job at 1M points.

## Troubleshooting (observed on Ruche)

| symptom | cause | fix |
|---|---|---|
| `no kernel image is available for execution on the device` | the `cu128` torch wheel has kernels for sm_75/80/86/90/100/120 only, and a V100 is sm_70 | use `--partition=gpua100`, or `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 REBUILD=1 bash setup_ruche.sh` |
| `CUDA-capable device(s) is/are busy or unavailable` while `nvidia-smi` shows the GPU idle in Default mode | node-specific driver state | the job records the node in `results/unusable_nodes.txt` and resubmits itself with `--exclude=` (up to `RETRY_MAX=4`) |
| `NVCC_PREPEND_FLAGS: unbound variable` | conda's `cuda-nvcc` activation script under `set -u` | handled: the scripts relax `set -u` around activation |
| `fatal error: cuda_runtime_api.h` in torch's JIT build | conda keeps the CUDA headers in `targets/x86_64-linux/include`, which torch does not add for host code | handled: the scripts export `CPATH` / `LIBRARY_PATH` |
| `ValueError: Unknown CUDA arch (10.1)` | torch's arch auto-detection on cu128 builds | handled: the job pins `TORCH_CUDA_ARCH_LIST` from `nvidia-smi --query-gpu=compute_cap` |

`bash diagnose_gpu.sh` prints the full picture on any GPU node: device files, compute mode, which `libcuda`
is mapped, raw `cuInit` / `cudaMalloc` error codes, a tiny `nvcc`-built CUDA program, and torch's arch list.

## Options worth knowing

```
--unit-cube                   same float32 points for every method (recommended)
--repeats N                   timed repetitions per method (correctness taken from run 1)
--slow-repeats 2 --slow-threshold 100   repetitions for methods slower than the threshold on run 1
--verbose                     timestamped progress on stderr
--paragram-bbox-pad 10 | -1   relative clipping pad (patched Paragram) | upstream absolute 1.0
--repair on|off, --repair-hull on|off   exact CPU repair of failed / convex-hull cells (capped at 20 % of the points)
--tool-timeout 10             kill an external tool (gStar4D, Local DeWall) after N s and report it
                              as failed on that dataset (0 = no limit); gStar4D's star-consistency
                              loop has no iteration cap and can spin forever on hard input, and
                              Local DeWall needs minutes on near-co-spherical input
--gdel3d auto|on|off, --geodel auto|on|off, --geodel-threads N (0 = SLURM_CPUS_PER_TASK, else all cores)
--dewall-bin PATH, --gstar4d-bin PATH, --gstar4d-grid 512 (its PBA grid; max 512), --gstar4d-facet-max N
--cgal-bin PATH, --cgal-python PATH
--n 20000, --models ..., --ply ..., --jitter 1e-6, --adjacency auto|paragram|qhull|ref-edges
```
Build variants of Paragram: `PARAGRAM_MAX_PLANES` / `PARAGRAM_MAX_VERTS` (default 64; the job uses 128),
`NO_FAST_MATH=1`. Paragram's CUDA extension is compiled on first import (a few minutes) into torch's
extension cache; the job does a warm-up call before timing.

## Files

- `test_delaunay_surfaces.py` — the benchmark driver (datasets, methods, metrics, JSON).
- `make_report.py` — Markdown report + charts from one or several JSON files.
- `scaling_study.py`, `run_scaling.sbatch` — time vs number of points (2k .. 1M) for the two point
  clouds, with and without jitter; log-log diagram with fitted exponents, plus CSV.
- `check_gstar4d.py` — smoke-test gStar4D alone: its own generator, then random clouds, then
  subsamples of the PLY clouds, to separate a broken build from an input it cannot handle.
- `voronoi_to_delaunay.py` — Voronoi adjacency → Delaunay tetrahedra (torch, GPU or CPU).
- `paragram_repair.py` — exact CPU fallback for Paragram's failed / hull cells.
- `cgal_delaunay.cpp` — parallel CGAL Delaunay command-line tool.
- `patch_paragram.py`, `patch_pygdel3d.py`, `patch_dewall.py`, `patch_gstar4d.py` — source patches applied
  to the upstream repositories at setup time (documented at the top of each file).
- `setup_ruche.sh`, `run_ruche.sbatch` — cluster setup and SLURM job.
- `data/` — the two point clouds. `third_party/`, `bin/`, `results/` are created by the setup / jobs.

## Notes on interpreting results

- Identical tets to CGAL is the correctness bar; "tie" differences on co-spherical groups (regular grids,
  symmetric meshes) are legitimate; zero-volume tets (gDel3D on co-planar points) are flagged separately.
- Paragram's known limits: float32 cell clipping fails on sliver-shaped cells (`inconsistent_boundary`),
  loses edges silently on unbounded (hull) cells and on near-co-spherical input; the CPU repair fixes what
  is flagged or on the hull, and the report states what fraction of cells was recomputed on the CPU.
- Local DeWall is exact but very slow on near-co-spherical input (hundreds of seconds for 20k sphere
  points), so with the default `--tool-timeout 10` it is reported as failed on those datasets;
  raise the limit (`TOOL_TIMEOUT=120`) if those numbers matter.
- gStar4D is from 2013 and needs three source changes to be usable, all in `patch_gstar4d.py`: its discrete
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
  as failed for that dataset and the run continues. `check_gstar4d.py` is how all of this was
  established, and its `missing=` column is the number to watch.
- GeoDel is the only CPU-parallel method besides CGAL, and gets the same core count as CGAL parallel
  (`--geodel-threads $SLURM_CPUS_PER_TASK`), so the two are directly comparable.
