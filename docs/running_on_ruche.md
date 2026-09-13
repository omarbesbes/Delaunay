# Running on Ruche (Mesocentre, SLURM)

How to set the cluster up, what each job does, and what goes wrong. The commands assume you
are in the repository root; every job is submitted with `sbatch script/<name>.sbatch`.


Everything is installed into a conda environment under `$WORKDIR` (Python 3.12, CUDA 12.8 toolkit, GCC 13,
torch cu128, CGAL headers + TBB), so nothing depends on the cluster's module versions except `anaconda3`.

1. Connect and clone (login node):
   ```bash
   ssh <user>@ruche.mesocentre.universite-paris-saclay.fr
   cd $WORKDIR && git clone https://github.com/omarbesbes/Delaunay.git && cd Delaunay
   ```
2. One-time setup (login node, has internet; 15–30 min, mostly downloads and CUDA compilation):
   ```bash
   bash script/first_install.sh
   ```
   `build_tools.sh` can also be run on its own to (re)build just the standalone tools, but only with
   the environment active — it now says so if it is not:
   ```bash
   module load anaconda3/2023.09-0/none-none && source activate $WORKDIR/envs/delaunay
   export CUDA_HOME=$CONDA_PREFIX CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
   bash script/build_tools.sh            # or --force to rebuild what is already there
   GPU_ARCHS="8.0" bash script/build_tools.sh --force   # only the GPU of this node (faster build)
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
   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash script/first_install.sh
   ```
   (Local DeWall, gStar4D and CGAL are standalone binaries, and GeoDel is CPU-only, so those four run on
   any of the partitions.)
   ```bash
   sbatch script/run_benchmark.sbatch                 # the two point clouds, 10 timed repeats per method
   FULL=1 BASELINE=0 sbatch script/run_benchmark.sbatch  # everything: clouds + analytic surfaces + meshes,
                                              # corrected Paragram only, with and without jitter
   TOOL_TIMEOUT=120 sbatch script/run_benchmark.sbatch   # allow a slow external tool 2 min (default 10 s)
   REPEATS=5 sbatch script/run_benchmark.sbatch
   JITTER=0 sbatch script/run_benchmark.sbatch           # skip the jittered pass
   OUT=results/1832417 sbatch script/run_benchmark.sbatch   # continue an earlier job's results

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
python src/benchmark.py --no-analytic --models --ply data/*.ply --unit-cube --repeats 2 --verbose \
  --cgal-bin bin/cgal_delaunay --dewall-bin bin/dewall --gstar4d-bin bin/gstar4d \
  --geodel-threads $SLURM_CPUS_PER_TASK --json results/test.json
```


## Troubleshooting (observed on Ruche)

| symptom | cause | fix |
|---|---|---|
| `no kernel image is available for execution on the device` | the `cu128` torch wheel has kernels for sm_75/80/86/90/100/120 only, and a V100 is sm_70 | use `--partition=gpua100`, or `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 REBUILD=1 bash script/first_install.sh` |
| `CUDA-capable device(s) is/are busy or unavailable` while `nvidia-smi` shows the GPU idle in Default mode | node-specific driver state | handled by `cuda_wait.sh`, which both jobs run before touching the GPU: it retries for a minute, then records the node in `results/unusable_nodes.txt` and resubmits the job with `--exclude=` (up to `RETRY_MAX=4`) |
| `NVCC_PREPEND_FLAGS: unbound variable` | conda's `cuda-nvcc` activation script under `set -u` | handled: the scripts relax `set -u` around activation |
| `fatal error: cuda_runtime_api.h` in torch's JIT build | conda keeps the CUDA headers in `targets/x86_64-linux/include`, which torch does not add for host code | handled: the scripts export `CPATH` / `LIBRARY_PATH` |
| `ValueError: Unknown CUDA arch (10.1)` | torch's arch auto-detection on cu128 builds | handled: the job pins `TORCH_CUDA_ARCH_LIST` from `nvidia-smi --query-gpu=compute_cap` |

`bash script/diagnose_gpu.sh` prints the full picture on any GPU node: device files, compute mode, which `libcuda`
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


## Jitter study: what the perturbation costs and buys

The benchmark perturbs the LiDAR clouds because their points sit on a sensor grid, which makes
large groups of them exactly co-spherical: the Delaunay triangulation is then not unique and
several methods return overlapping tetrahedra or fail outright. `src/jitter_study.py` sweeps the size
of that perturbation on the two point clouds only, and records, for every value:

* **how much it deforms the cloud** — displacement in absolute terms and as a fraction of the local
  point spacing, convex-hull volume change, points whose nearest neighbour changed, exact
  duplicates removed, and what happened to the reference triangulation (tetrahedra, slivers, flat
  tetrahedra);
* **what each method costs** — seconds, and its tetrahedra against a CGAL reference on the same
  perturbed points;
* **how often the in-sphere predicate falls back to exact arithmetic.** A robust implementation
  evaluates the test in floating point with an error bound and redoes it exactly only when the
  bound says the sign is not trustworthy. Degeneracy is exactly what makes that filter fail, so
  this is the number that says whether a jitter removed the degeneracy or only hid it.

| method | counter | how |
|---|---|---|
| gDel3D | `doInSphereFast` vs `doInSphereSoS` | `script/patch_pygdel3d.py` adds three slots to `_counterVec` and to `Statistics`, read once per flipping loop (one register increment per test, one atomic per block) |
| CGAL | filtered-predicate calls vs filter failures | `-DCGAL_PROFILE` build `bin/cgal_delaunay_profile`, run untimed and single-threaded so the counts are reproducible |
| Paragram + conversion | in-sphere determinants inside the float64 rounding-error bound | `voronoi_to_delaunay.last_insphere_stats()`; there is no exact fallback, so these are tests it cannot decide at all |
| GeoDel | — | would need a `PCK_STATS` build of Geogram, which the wheel is not built with |
| Local DeWall, gStar4D | — | no exact fallback to count |

```bash
sbatch script/run_jitter.sbatch                                     # 0, 1e-9 .. 1e-3 on both clouds
JITTERS="0 1e-8 1e-6 1e-4" REPEATS=1 sbatch script/run_jitter.sbatch
CLOUD=data/voronoi_jax_068.ply sbatch script/run_jitter.sbatch
python src/jitter_study.py --plot-only results/jitter-<jobid>/jitter.json \
    --markdown mine.md --plot mine.png
```

`jitter_sweep.md` is the written-up analysis of the run that settled this (job 1887129): the
deformation table, the exact-fallback measurements, and why 1e-6 is the right default.

Outputs, in `results/jitter-<jobid>/`: `jitter.md` (report-ready tables), `jitter.png` (three rows
per cloud: time, exact-predicate share, deformation), `jitter.csv`, `jitter.json`. One process per
method, all sharing one JSON, so a library that crashes the interpreter only ends its own sweep and
`--resume` carries on; the deformation metrics and CGAL's counters are computed by the first
process and read back by the rest.

The predicate counters need the patched builds: `bash script/first_install.sh` (pyGDel3D) and
`bash script/build_tools.sh` (`bin/cgal_delaunay_profile`). Without them the study still runs and simply
reports no counters for those methods.

